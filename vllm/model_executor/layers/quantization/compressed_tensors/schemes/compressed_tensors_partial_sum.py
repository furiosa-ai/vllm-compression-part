# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Partial-sum NVFP4+ QDQ wrappers (hardcoded for K-EXAONE).

Per-group dynamic fake-quantize on the rank-local matmul output before
vLLM's ``tensor_model_parallel_all_reduce``, on layers matching
``_PARTIAL_SUM_TARGETS``. Quant shape is fixed: symmetric FP4 E2M1,
group_size=16, per-group BF16 scale.
"""

from __future__ import annotations

from typing import Any

import regex as re
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_GROUP_SIZE = 16
_FP4_MAX = 6.0
_BF16_TRUNC_MASK = -65536  # 0xFFFF0000 as a signed int32

# Matched against the layer prefix passed to ``get_quant_method``,
# e.g. ``model.layers.0.self_attn.o_proj``.
_PARTIAL_SUM_TARGETS: tuple[re.Pattern[str], ...] = (
    re.compile(r".*o_proj$"),
    re.compile(r".*mlp\.experts$"),
)


def _is_partial_sum_target(prefix: str | None) -> bool:
    if prefix is None:
        return False
    return any(p.fullmatch(prefix) for p in _PARTIAL_SUM_TARGETS)


def _to_bf16_grid(x: torch.Tensor) -> torch.Tensor:
    # Bit-mask, not ``to(bf16).to(fp32)``: inductor's lowering of the
    # latter diverges from eager by up to one bf16 ULP and amplifies into
    # FP4 grid ticks downstream.
    return (x.contiguous().view(torch.int32) & _BF16_TRUNC_MASK).view(torch.float32)


def _snap_to_fp4_grid(x: torch.Tensor) -> torch.Tensor:
    # Constants are Python floats — ``torch.tensor(...)`` inside the body
    # crashes CUDA-graph capture with ``cudaErrorStreamCaptureUnsupported``.
    sign = torch.sign(x)
    a = x.abs()
    out = torch.where(a > 5.0, 6.0, a)
    out = torch.where((a >= 3.5) & (a <= 5.0), 4.0, out)
    out = torch.where((a > 2.5) & (a < 3.5), 3.0, out)
    out = torch.where((a >= 1.75) & (a <= 2.5), 2.0, out)
    out = torch.where((a > 1.25) & (a < 1.75), 1.5, out)
    out = torch.where((a >= 0.75) & (a <= 1.25), 1.0, out)
    out = torch.where((a > 0.25) & (a < 0.75), 0.5, out)
    out = torch.where(a <= 0.25, 0.0, out)
    return sign * out


@torch.library.custom_op("vllm_partial_sum::qdq_nvfp4plus", mutates_args=())
def _qdq_nvfp4plus(partial_results: torch.Tensor) -> torch.Tensor:
    original_shape = partial_results.shape
    original_dtype = partial_results.dtype
    last_dim = original_shape[-1]

    x_fp32 = partial_results.to(torch.float32)
    grouped = x_fp32.reshape(
        *original_shape[:-1], last_dim // _GROUP_SIZE, _GROUP_SIZE
    )

    amax = grouped.abs().amax(dim=-1, keepdim=True)
    eps = torch.finfo(torch.float32).eps
    scale_fp32 = torch.clamp(amax / _FP4_MAX, min=eps)
    scale_bf16 = _to_bf16_grid(scale_fp32)

    x_scaled = torch.clamp(grouped / scale_bf16, min=-_FP4_MAX, max=_FP4_MAX)
    x_q = _snap_to_fp4_grid(x_scaled)
    x_dq = x_q * scale_bf16

    return x_dq.reshape(original_shape).to(original_dtype)


@_qdq_nvfp4plus.register_fake
def _qdq_nvfp4plus_fake(partial_results: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(partial_results)


def _qdq_partial_sums(partial_results: torch.Tensor) -> torch.Tensor:
    last_dim = partial_results.shape[-1]
    if last_dim % _GROUP_SIZE != 0:
        logger.warning_once(
            "PartialSum: last dim %d not divisible by group_size %d; "
            "skipping QDQ.",
            last_dim,
            _GROUP_SIZE,
        )
        return partial_results
    return torch.ops.vllm_partial_sum.qdq_nvfp4plus(partial_results)


def _maybe_wrap_partial_sum_scheme(scheme: Any, prefix: str | None) -> None:
    if not _is_partial_sum_target(prefix):
        return

    _orig_apply_weights = scheme.apply_weights

    # Bias is added after QDQ — it is not part of the "quantized
    # communication" payload that the all-reduce sees.
    def apply_weights(layer, x, bias=None):
        out = _orig_apply_weights(layer, x, bias=None)
        out = _qdq_partial_sums(out)
        if bias is not None:
            out = out + bias
        return out

    scheme.apply_weights = apply_weights
    logger.info_once(
        "PartialSum[linear]: wrapping %s over %s",
        prefix,
        type(scheme).__name__,
    )


def _maybe_wrap_partial_sum_moe(method: Any, prefix: str | None) -> None:
    if not _is_partial_sum_target(prefix) or method is None:
        return

    _orig_apply = method.apply
    _orig_apply_monolithic = getattr(method, "apply_monolithic", None)

    def _post_qdq(out):
        if isinstance(out, tuple):
            head, body = out
            return (head, _qdq_partial_sums(body))
        return _qdq_partial_sums(out)

    def apply(*args, **kwargs):
        return _post_qdq(_orig_apply(*args, **kwargs))

    method.apply = apply

    if _orig_apply_monolithic is not None:

        def apply_monolithic(*args, **kwargs):
            return _post_qdq(_orig_apply_monolithic(*args, **kwargs))

        method.apply_monolithic = apply_monolithic

    logger.info_once(
        "PartialSum[moe]: wrapping %s over %s",
        prefix,
        type(method).__name__,
    )
