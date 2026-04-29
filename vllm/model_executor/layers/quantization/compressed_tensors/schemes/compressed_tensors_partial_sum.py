# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Partial-sum QDQ wrappers for emulating TP-communication cost.

Semantics:
  Each matmul kernel (quantized or unquantized Linear, fused MoE) produces
  a **rank-local partial sum** that vLLM's parallel layer subsequently
  passes through ``tensor_model_parallel_all_reduce``. The partial-sum
  emulation inserts a per-group fake-quantize (QDQ) on that rank-local
  partial *before* the all-reduce, as a pure **post-hook** on the inner
  method's output. This models "each rank quantizes its contribution
  before the TP all-reduce" — i.e. emulated low-precision communication.

Design notes:
  * The emulation R = TP (one QDQ per rank per layer). The
    ``partial_sum.num_ranks`` field in the checkpoint config has no effect
    on semantics (the actual shard count is always ``tp_size``).
  * All three layer types (quantized Linear, unquantized Linear, FusedMoE)
    are wrapped uniformly via in-place monkey-patch of the function that
    produces the rank-local partial -- ``scheme.apply_weights`` for
    quantized Linear, ``method.apply`` for the other two.

Entry points:
  * :func:`make_scheme_partial_sum_wrapper` -- quantized Linear scheme.
  * :func:`make_unquant_partial_sum_wrapper` -- UnquantizedLinearMethod.
  * :func:`make_moe_partial_sum_wrapper` -- FusedMoE quant method.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "make_scheme_partial_sum_wrapper",
    "make_unquant_partial_sum_wrapper",
    "make_moe_partial_sum_wrapper",
]


# ─── QDQ helper ────────────────────────────────────────────────────────────
#
# Inline NVFP4+ QDQ (FP4 E2M1 grid + per-group BF16 scale, group_size=16,
# symmetric, no global scale). Properties verified by
# ``test_qdq_nvfp4plus.py``.

# Positive FP4 E2M1 magnitudes.
_FP4_POS_MAGS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX = 6.0

# Bit mask that zeroes the lower 16 bits of an fp32 word -- equivalent to
# truncating the value to the bf16 grid. Used in ``_to_bf16_grid``.
_BF16_TRUNC_MASK = -65536  # = 0xFFFF0000 as a signed int32


def _to_bf16_grid(x: torch.Tensor) -> torch.Tensor:
    """
    Snap each element of ``x`` (fp32) onto the BF16 representable grid via
    bit-mask truncation. The mask is a Python int so no GPU tensor is
    allocated inside the function (required for CUDA-graph capture).
    """
    return (x.contiguous().view(torch.int32) & _BF16_TRUNC_MASK).view(torch.float32)


def _snap_to_fp4_grid(x: torch.Tensor) -> torch.Tensor:
    """
    Round each element of ``x`` (already in [-6, 6]) to the nearest FP4
    E2M1 grid value. Scalar constants are passed as Python floats so no
    temporary tensors are allocated (required for CUDA-graph capture).
    """
    sign = torch.sign(x)
    a = x.abs()
    # Walk from the largest magnitude inward; each torch.where pins down
    # the target magnitude for elements in that band. Boundaries are the
    # midpoints of consecutive FP4 magnitudes.
    out = torch.where(a > 5.0, 6.0, a)
    out = torch.where((a >= 3.5) & (a <= 5.0), 4.0, out)
    out = torch.where((a > 2.5) & (a < 3.5), 3.0, out)
    out = torch.where((a >= 1.75) & (a <= 2.5), 2.0, out)
    out = torch.where((a > 1.25) & (a < 1.75), 1.5, out)
    out = torch.where((a >= 0.75) & (a <= 1.25), 1.0, out)
    out = torch.where((a > 0.25) & (a < 0.75), 0.5, out)
    out = torch.where(a <= 0.25, 0.0, out)
    return sign * out


def _qdq_nvfp4plus_inline(
    partial_results: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Pure-tensor NVFP4+ QDQ over the last dim. Used inside the custom op,
    not directly. ``group_size`` must divide the last dim.
    """
    original_shape = partial_results.shape
    original_dtype = partial_results.dtype
    last_dim = original_shape[-1]

    x_fp32 = partial_results.to(torch.float32)
    grouped = x_fp32.reshape(*original_shape[:-1], last_dim // group_size, group_size)

    amax = grouped.abs().amax(dim=-1, keepdim=True)
    eps = torch.finfo(torch.float32).eps
    scale_fp32 = torch.clamp(amax / _FP4_MAX, min=eps)
    scale_bf16 = _to_bf16_grid(scale_fp32)

    x_scaled = torch.clamp(grouped / scale_bf16, min=-_FP4_MAX, max=_FP4_MAX)
    x_q = _snap_to_fp4_grid(x_scaled)
    x_dq = x_q * scale_bf16

    return x_dq.reshape(original_shape).to(original_dtype)


# ─── Custom op registration ────────────────────────────────────────────────
#
# Expose ``_qdq_nvfp4plus_inline`` to vLLM's compile pipeline as a single
# opaque op so dynamo treats it as a black box (no inductor trace into the
# body, eager and compile bit-identical). ``mutates_args=()`` declares
# purity so inductor can still CSE/DCE around the call.

@torch.library.custom_op(
    "vllm_partial_sum::qdq_nvfp4plus",
    mutates_args=(),
)
def _qdq_nvfp4plus_op(
    partial_results: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return _qdq_nvfp4plus_inline(partial_results, group_size)


@_qdq_nvfp4plus_op.register_fake
def _qdq_nvfp4plus_op_fake(
    partial_results: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return torch.empty_like(partial_results)


def _qdq_partial_sums(
    partial_results: torch.Tensor,
    quant_args: QuantizationArgs,
) -> torch.Tensor:
    """
    Apply NVFP4+ per-group dynamic QDQ along the last dim of
    ``partial_results``. Falls back to passing the input through unchanged
    (with a warning) if ``quant_args`` is not the NVFP4+ shape -- group
    strategy, FP4 type, symmetric, group_size dividing the last dim.
    """
    if not (
        quant_args.strategy == QuantizationStrategy.GROUP
        and quant_args.type == "float"
        and quant_args.num_bits == 4
        and quant_args.symmetric
    ):
        logger.warning_once(
            "PartialSum: torch.compile-safe inline QDQ only handles "
            "symmetric FP4 GROUP. Got strategy=%s type=%s num_bits=%s "
            "symmetric=%s -- skipping QDQ.",
            quant_args.strategy,
            quant_args.type,
            quant_args.num_bits,
            quant_args.symmetric,
        )
        return partial_results

    last_dim = partial_results.shape[-1]
    group_size = int(quant_args.group_size)

    if last_dim % group_size != 0:
        logger.warning_once(
            "PartialSum: last dim %d not divisible by group_size %d; "
            "skipping QDQ.",
            last_dim,
            group_size,
        )
        return partial_results

    return torch.ops.vllm_partial_sum.qdq_nvfp4plus(partial_results, group_size)


# ─── Scheme post-hook (quantized Linear) ───────────────────────────────────

def make_scheme_partial_sum_wrapper(
    inner_scheme: Any,
    num_ranks: int,
    partial_sum_quant_args: QuantizationArgs,
) -> Any:
    """
    In-place decorate a quantized ``CompressedTensorsScheme`` instance so
    its ``apply_weights(layer, x, bias)`` runs a post-hook QDQ. The inner
    matmul (e.g. Marlin kernels, Cutlass) runs unchanged; we only QDQ its
    output before vLLM's all-reduce.
    """
    _orig_apply_weights = inner_scheme.apply_weights

    def apply_weights(layer, x, bias=None):
        # Inner kernel runs without bias (bias is not part of the "quantized
        # communication" payload); we add bias after QDQ.
        out = _orig_apply_weights(layer, x, bias=None)
        out = _qdq_partial_sums(out, partial_sum_quant_args)
        if bias is not None:
            out = out + bias
        return out

    inner_scheme.apply_weights = apply_weights
    inner_scheme._partial_sum_num_ranks = num_ranks
    inner_scheme._partial_sum_quant_args = partial_sum_quant_args
    return inner_scheme


# ─── UnquantizedLinearMethod wrapper ───────────────────────────────────────

def make_unquant_partial_sum_wrapper(
    inner_method: Any,
    num_ranks: int,
    partial_sum_quant_args: QuantizationArgs,
) -> Any:
    """
    In-place decorate an ``UnquantizedLinearMethod`` instance so its
    ``apply(layer, x, bias)`` runs a post-hook QDQ. The inner matmul
    (bf16 × bf16 via torch) runs unchanged; we only QDQ its output
    before vLLM's all-reduce.
    """
    _orig_apply = inner_method.apply

    def _forward(layer, x, bias=None):
        out = _orig_apply(layer, x, bias=None)
        out = _qdq_partial_sums(out, partial_sum_quant_args)
        if bias is not None:
            out = out + bias
        return out

    inner_method.apply = _forward
    inner_method._partial_sum_num_ranks = num_ranks
    inner_method._partial_sum_quant_args = partial_sum_quant_args
    return inner_method


# ─── FusedMoE method wrapper ───────────────────────────────────────────────

def make_moe_partial_sum_wrapper(
    inner_method: Any,
    num_ranks: int,
    partial_sum_quant_args: QuantizationArgs,
) -> Any:
    """
    In-place decorate a FusedMoE quantization method so its output (the
    rank-local partial sum, before the TP all-reduce inside
    ``FusedMoE.forward``) goes through post-hook QDQ.
    """
    _orig_apply = inner_method.apply
    _orig_apply_monolithic = getattr(inner_method, "apply_monolithic", None)

    def _post_qdq(out):
        if isinstance(out, tuple):
            head, body = out
            return (head, _qdq_partial_sums(body, partial_sum_quant_args))
        return _qdq_partial_sums(out, partial_sum_quant_args)

    def apply(*args, **kwargs):
        return _post_qdq(_orig_apply(*args, **kwargs))

    inner_method.apply = apply

    if _orig_apply_monolithic is not None:
        def apply_monolithic(*args, **kwargs):
            return _post_qdq(_orig_apply_monolithic(*args, **kwargs))

        inner_method.apply_monolithic = apply_monolithic

    inner_method._partial_sum_num_ranks = num_ranks
    inner_method._partial_sum_quant_args = partial_sum_quant_args
    return inner_method
