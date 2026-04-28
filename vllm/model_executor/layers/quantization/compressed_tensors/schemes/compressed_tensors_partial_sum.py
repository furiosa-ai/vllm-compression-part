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
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
)
from compressed_tensors.quantization.utils import (
    compute_dynamic_scales_and_zp,
    generate_gparam,
    is_fp4,
)

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "make_scheme_partial_sum_wrapper",
    "make_unquant_partial_sum_wrapper",
    "make_moe_partial_sum_wrapper",
]


# ─── QDQ helper ────────────────────────────────────────────────────────────

def _qdq_partial_sums(
    partial_results: torch.Tensor,
    quant_args: QuantizationArgs,
) -> torch.Tensor:
    """
    Apply per-group dynamic QDQ along the last dim of ``partial_results``.

    Shape-agnostic (flattens all but last dim for scale computation).

    NOTE: the compressed_tensors helper calls ``.item()`` internally, which
    Dynamo cannot trace in fullgraph mode. Callers should run with
    ``TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`` or ``--enforce-eager``.
    """
    original_shape = partial_results.shape
    original_dtype = partial_results.dtype

    flat = partial_results.reshape(-1, original_shape[-1])

    global_scale = None
    if quant_args.strategy == QuantizationStrategy.TENSOR_GROUP:
        if is_fp4(quant_args):
            global_scale = generate_gparam(
                updated_min_val=flat.min(),
                updated_max_val=flat.max(),
                scale_data=FP8_E4M3_DATA,
                quant_data=FP4_E2M1_DATA,
            )
        else:
            logger.warning_once(
                "PartialSum: TENSOR_GROUP with %d-bit %s not supported for "
                "global_scale computation; only FP4 is.",
                quant_args.num_bits,
                quant_args.type,
            )

    scale, zero_point = compute_dynamic_scales_and_zp(
        value=flat,
        args=quant_args,
        module=None,
        global_scale=global_scale,
    )
    qdq = fake_quantize(
        x=flat,
        scale=scale,
        zero_point=zero_point,
        args=quant_args,
        global_scale=global_scale,
    )
    return qdq.reshape(original_shape).to(original_dtype)


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
