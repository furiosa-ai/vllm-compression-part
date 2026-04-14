# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
NVFP8 W8A8 scheme for compressed-tensors models.

Handles models with:
  - format: float-quantized
  - weight strategy: tensor_group (group_size=16)
  - weight scale_dtype: float8_e4m3fn  (local per-16 scale)
  - weight global_scale: float32       (per-tensor global scale)
  - input strategy: tensor_group (dynamic=True)

The weight_scale stored in the checkpoint is already the *combined*
scale: round_to_fp8(global_scale * local_pre).  The global_scale
parameter is registered (to absorb checkpoint keys) but is not needed
for dequantization.

At load time weights are dequantized to the model dtype.
At inference time activations are dynamically fake-quantized per group
before a standard torch.matmul with the dequantized weights.
"""

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A8NVFp8"]

_NVFP8_GROUP_SIZE = 16


class CompressedTensorsW8A8NVFp8(CompressedTensorsScheme):
    """
    Emulation-mode NVFP8 W8A8 scheme.

    Weights are stored as float8_e4m3fn with float8_e4m3fn per-group-16
    combined scales and a float32 global scale.  At load time weights are
    dequantized to the model dtype.  At inference time activations undergo
    dynamic per-group-16 fake-quantize (TENSOR_GROUP) before a standard
    matmul with the dequantized weights.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
    ):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.group_size = weight_quant.group_size or _NVFP8_GROUP_SIZE

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # FP8 weight data
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # FP8 combined local scale: (out, in // group_size)
        # Already encodes global_scale * local_pre, rounded to fp8
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # FP32 global scale — registered to absorb checkpoint keys;
        # not used during dequantization (already baked into weight_scale).
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Convert fp8 scale to float32 first (fp8 arithmetic not supported on CPU,
        # and dequantize expects a floating-point scale tensor)
        float_scale = layer.weight_scale.data.to(torch.float32)
        # Expand combined scale: (out, in//16) → (out, in)
        scale_expanded = float_scale.repeat_interleave(self.group_size, dim=1)
        # Dequantize: weight_fp8 → float32, then multiply by combined scale
        # (global_scale is already baked into weight_scale at quantization time)
        dq_weight = layer.weight.data.to(torch.float32) * scale_expanded
        layer.weight = torch.nn.Parameter(
            dq_weight.to(layer.params_dtype), requires_grad=False
        )
        del layer.weight_scale
        del layer.weight_global_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Dynamic TENSOR_GROUP activation quantization
        # compute_dynamic_scales_and_zp computes global_scale on the fly
        # when global_scale is None (our patch to helpers.py)
        scale, zero_point = compute_dynamic_scales_and_zp(
            value=x, args=self.input_quant, module=None, global_scale=None
        )
        qdq_input = fake_quantize(
            x=x,
            scale=scale,
            zero_point=zero_point,
            args=self.input_quant,
        )
        qdq_input = qdq_input.to(x.dtype)

        out = torch.matmul(qdq_input, layer.weight.to(qdq_input.dtype).t())
        if bias is not None:
            out = out + bias
        return out
