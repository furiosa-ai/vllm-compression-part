# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVFP4 and NVFP4+ fake-quantization kernels for KV cache.

Two ``@torch.library.custom_op`` ops, each appearing by name in inductor's
FX graph dump (``computation_graph.py``):

    vllm_kv_quant::fake_quantize_dequantize_nvfp4(x, global_scale) -> Tensor
    vllm_kv_quant::fake_quantize_dequantize_nvfp4_plus(x) -> Tensor

Plus the matching high-level shape-handling wrappers:

    fake_quantize_nvfp4(x, num_kv_heads, head_dim, global_scale) -> Tensor
    fake_quantize_nvfp4_plus(x, num_kv_heads, head_dim) -> Tensor

NVFP4 = per-tensor FP32 global scale (calibrated) × per-group FP8 E4M3
scale × per-element FP4 E2M1, group_size=16.

NVFP4+ = per-group FP32 scale (NO global scale, NO FP8 rounding on the
scale, NO calibration needed) × per-element FP4 E2M1, group_size=16.

Each custom_op is opaque to torch.compile — the compiler doesn't try to
inline or rewrite its body. Its name appears verbatim in inductor's
``computation_graph.py``, which is what graph-verification tooling greps for.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# FP8 E4M3 round helper (used internally by NVFP4 for the per-group local
# scale on pre-SM89 hardware)
# ---------------------------------------------------------------------------

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max  # 448.0


def _round_to_fp8e4m3(x_fp32: torch.Tensor) -> torch.Tensor:
    """sm_80-compatible E4M3 round (matches hardware cast for normal range).

    On A100, Triton's inductor codegen cannot lower ``torch.float8_e4m3fn``
    casts inside cudagraphs; this fp32-arithmetic emulation does and is
    bit-identical for normals.
    """
    sign = torch.sign(x_fp32)
    abs_x = x_fp32.abs().clamp(max=_FP8_MAX)
    eps_floor = 2.0 ** -9
    abs_safe = abs_x.clamp(min=eps_floor)
    exp = torch.floor(torch.log2(abs_safe))
    exp_clamped = exp.clamp(min=-6.0, max=8.0)
    pow2_exp = torch.pow(2.0, exp_clamped)
    mantissa_q = torch.round(abs_x / pow2_exp * 8.0) / 8.0
    out = sign * mantissa_q * pow2_exp
    return torch.where(x_fp32 == 0, torch.zeros_like(out), out)


def _is_sm89_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


# ---------------------------------------------------------------------------
# NVFP4 quant-dequant (per-tensor static global scale, per-group dynamic
# FP8 E4M3 block scale, per-element FP4 E2M1)
# ---------------------------------------------------------------------------

_FP4_MAX = 6.0     # FP4_E2M1_DATA.max
_NVFP4_GROUP_SIZE = 16


def _round_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest FP4 E2M1 grid value:
        ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

    Bit-identical to ``compressed_tensors.quantization.quant_args.FP4_E2M1_DATA
    .cast_to_fp4`` — including its asymmetric tie-breaking at exact midpoints
    {0.25, 1.25, 2.5, 5.0} which round DOWN to the smaller grid value
    (compressed_tensors closes its smaller-magnitude intervals on both sides:
    [0,0.25], [0.75,1.25], [1.75,2.5], [3.5,5.0]).

    Practically these midpoints are measure-zero in real-valued inputs, but
    matching the canonical reference avoids spurious skew at the few tie cases.
    """
    sign = torch.sign(x)
    a = x.abs()
    # Midpoints between adjacent positive grid values.
    # Inequalities chosen so a==0.25→0, a==1.25→1, a==2.5→2, a==5→4
    # (matches compressed_tensors round-half-toward-smaller).
    out = torch.where(a <= 0.25, torch.zeros_like(a), torch.full_like(a, 0.5))
    out = torch.where(a >= 0.75, torch.full_like(a, 1.0), out)
    out = torch.where(a >  1.25, torch.full_like(a, 1.5), out)
    out = torch.where(a >= 1.75, torch.full_like(a, 2.0), out)
    out = torch.where(a >  2.5,  torch.full_like(a, 3.0), out)
    out = torch.where(a >= 3.5,  torch.full_like(a, 4.0), out)
    out = torch.where(a >  5.0,  torch.full_like(a, 6.0), out)
    return sign * out


@torch.library.custom_op(
    "vllm_kv_quant::fake_quantize_dequantize_nvfp4", mutates_args=()
)
def _fake_quantize_dequantize_nvfp4(
    data: torch.Tensor, global_scale: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 quant-dequant on (B, nh, T, D) input.

    Bit-identical to ``vllm.model_executor.layers.quantization.utils.
    nvfp4_emulation_utils.ref_nvfp4_quant`` for fp32 ``global_scale``.
    Compute order matches the reference:

        scale          = round_fp8(amax_per_group * global_scale / FP4_MAX)
        output_scale   = 1 / (scale * (1 / global_scale))
        x_fp4          = round_fp4(clamp(x * output_scale, -6, 6))
        x_dequantized  = x_fp4 * (scale / global_scale)

    The reciprocal-then-multiply ordering matters at fp32 ULP-level for
    bf16 inputs near the FP4 grid boundaries; using ``x / (scale/gs)``
    directly introduces ~0.1% disagreements with the reference for those
    edge cases.
    """
    B, nh, T, D = data.shape
    G = _NVFP4_GROUP_SIZE
    assert D % G == 0, f"head_dim {D} must be divisible by NVFP4 group_size {G}"
    num_groups = D // G

    grouped = data.view(B, nh, T, num_groups, G).to(torch.float32)
    gs = global_scale.to(torch.float32).reshape(1)

    # 1. per-group amax → FP8 local scale
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    local_scale_unrounded = gs * (amax * (1.0 / _FP4_MAX))
    if _is_sm89_or_newer():
        local_scale = local_scale_unrounded.clamp(max=_FP8_MAX, min=-_FP8_MAX).to(_FP8_DTYPE).to(torch.float32)
    else:
        local_scale = _round_to_fp8e4m3(local_scale_unrounded)

    # 2. quantize using ref's reciprocal-then-multiply ordering (bit-identical
    # to ref_nvfp4_quant: output_scale = 1 / (scale * (1/global_scale)))
    gs_reciprocal = 1.0 / (gs + (gs == 0) * 1e8)
    inner = local_scale * gs_reciprocal
    output_scale = 1.0 / (inner + (inner == 0) * 1e8)

    # 3. quantize: x * output_scale, clamp to ±FP4_MAX, round to FP4 grid
    scaled = grouped * output_scale
    scaled = scaled.clamp(min=-_FP4_MAX, max=_FP4_MAX)
    fp4 = _round_to_fp4_e2m1(scaled)

    # 4. dequantize using ref's order: fp4 * (scale / global_scale)
    # (direct division — must NOT use scale * (1/gs), it differs at fp32 ULP)
    dequant_scale = local_scale / gs
    out = (fp4 * dequant_scale).view(B, nh, T, D)
    # Bit-identical to ref_nvfp4_quant: do NOT sanitize NaN/Inf here. Real
    # K-cache values never contain NaN/Inf in normal inference; if they do,
    # the upstream model is already broken and we want the failure to surface.
    return out.to(data.dtype)


@_fake_quantize_dequantize_nvfp4.register_fake
def _fake_quantize_dequantize_nvfp4_meta(
    data: torch.Tensor, global_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(data)


# ---------------------------------------------------------------------------
# Shape helpers + high-level fake-quant entry point
# ---------------------------------------------------------------------------

def _to_bnhtd(x: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape ``(T, num_kv_heads*head_dim)`` or ``(T, num_kv_heads, head_dim)``
    -> ``(B=1, nh, T, D)``."""
    if x.dim() == 2:
        T = x.shape[0]
        return x.view(1, T, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    if x.dim() == 3:
        return x.unsqueeze(0).transpose(1, 2).contiguous()
    raise ValueError(f"unexpected KV shape {tuple(x.shape)}")


def _from_bnhtd(x4: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    return x4.transpose(1, 2).contiguous().view(*orig_shape)


def fake_quantize_nvfp4(
    x: torch.Tensor, num_kv_heads: int, head_dim: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 round-trip with reshape into the canonical layout.

    ``global_scale`` is a per-tensor (per-layer) FP32 scalar derived offline
    from the K (or V) cache amax (see the calibration utility shipped with
    K-EXAONE-evaluation under ``kv_cache_quantization/``).
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    out = torch.ops.vllm_kv_quant.fake_quantize_dequantize_nvfp4(x4, global_scale)
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# NVFP4+: per-group FP32 scale, per-element FP4 E2M1. No global scale.
# Group size matches NVFP4 (16); the only differences from NVFP4 are:
#   * no per-tensor FP32 global scale (no calibration step required)
#   * per-group scale stays FP32 (NOT rounded to FP8 E4M3)
# ---------------------------------------------------------------------------

_NVFP4_PLUS_GROUP_SIZE = 16


@torch.library.custom_op(
    "vllm_kv_quant::fake_quantize_dequantize_nvfp4_plus", mutates_args=()
)
def _fake_quantize_dequantize_nvfp4_plus(data: torch.Tensor) -> torch.Tensor:
    """NVFP4+ quant-dequant on (B, nh, T, D) input.

    Per-group (16-element) scale, kept in FP32 (no FP8 rounding). No
    per-tensor global scale. Scale derivation per group:

        scale         = amax(|group|) / FP4_MAX
        x_fp4         = round_fp4(clamp(x / scale, -6, 6))
        x_dequantized = x_fp4 * scale

    All-zero groups (amax == 0) are passed through unchanged: we divide
    by a placeholder 1.0 to avoid 0/0, then multiply the rounded result
    by the ORIGINAL (zero) scale, so the output is exactly zero.
    """
    B, nh, T, D = data.shape
    G = _NVFP4_PLUS_GROUP_SIZE
    assert D % G == 0, f"head_dim {D} must be divisible by NVFP4+ group_size {G}"
    num_groups = D // G

    grouped = data.view(B, nh, T, num_groups, G).to(torch.float32)

    # per-group fp32 scale = amax / FP4_MAX. No FP8 round.
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = amax * (1.0 / _FP4_MAX)

    # Avoid 0/0 for all-zero groups by quantizing with a safe scale, then
    # dequantizing with the original (possibly-zero) scale so zero groups
    # stay exactly zero.
    scale_safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    scaled = grouped / scale_safe
    scaled = scaled.clamp(min=-_FP4_MAX, max=_FP4_MAX)
    fp4 = _round_to_fp4_e2m1(scaled)

    out = (fp4 * scale).view(B, nh, T, D)
    # As in NVFP4: do NOT sanitize NaN/Inf — real KV-cache values never
    # contain NaN/Inf during normal inference; surfacing the failure is
    # preferable to masking it.
    return out.to(data.dtype)


@_fake_quantize_dequantize_nvfp4_plus.register_fake
def _fake_quantize_dequantize_nvfp4_plus_meta(data: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(data)


def fake_quantize_nvfp4_plus(
    x: torch.Tensor, num_kv_heads: int, head_dim: int,
) -> torch.Tensor:
    """NVFP4+ round-trip with reshape into the canonical layout.

    No ``global_scale`` argument — per-group FP32 scale is computed on the
    fly inside the kernel, so no calibration ``.pt`` is needed.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    out = torch.ops.vllm_kv_quant.fake_quantize_dequantize_nvfp4_plus(x4)
    return _from_bnhtd(out, orig_shape).to(orig_dtype)
