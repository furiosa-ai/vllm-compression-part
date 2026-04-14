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

import math
from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from compressed_tensors.quantization.utils.helpers import (
    FP8_E4M3_DATA,
    FP4_E2M1_DATA,
    generate_gparam,
)

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

        # Combined local scale: (out, in // group_size)
        # Stored as float8_e4m3fn in checkpoint (matching NVFP8 spec)
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
        # weight_scale stores combined = global_scale * local_raw (in bfloat16)
        # weight_fp8 was quantized as: round(w / combined) → subnormals ~0.002
        # Reconstruction: w = w_fp8 * combined / global_scale = w_fp8 * local_raw
        float_scale = layer.weight_scale.data.to(torch.float32)
        # Expand combined scale: (out, in//group_size) → (out, in)
        scale_expanded = float_scale.repeat_interleave(self.group_size, dim=1)
        # global_scale: scalar fp32 (max across partitions)
        global_s = layer.weight_global_scale.data.to(torch.float32).max()
        # Dequantize: divide by global_scale to cancel it out of combined scale
        dq_weight = layer.weight.data.to(torch.float32) * scale_expanded / global_s
        layer.weight = torch.nn.Parameter(
            dq_weight.to(layer.params_dtype), requires_grad=False
        )
        del layer.weight_scale
        del layer.weight_global_scale

    def _compute_activation_global_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dynamic global_scale for TENSOR_GROUP activation quantization.

        Must match the logic in compute_dynamic_scales_and_zp so the same
        global_scale that was baked into the combined scale is recovered.
        """
        gs = self.group_size
        reshaped = x.unflatten(-1, (math.ceil(x.shape[-1] / gs), gs))
        min_val = torch.amin(reshaped, dim=-1)
        max_val = torch.amax(reshaped, dim=-1)
        quant_data = (
            FP8_E4M3_DATA if self.input_quant.num_bits == 8 else FP4_E2M1_DATA
        )
        return generate_gparam(
            min_val.amin().reshape(1),
            max_val.amax().reshape(1),
            quant_data=quant_data,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_quant is not None:
            # W8A8: dynamic TENSOR_GROUP activation quantization
            # compute_dynamic_scales_and_zp bakes global_scale into returned
            # scale (combined = global * local).  fake_quantize needs
            # global_scale so it can divide it back out before quantizing.
            scale, zero_point = compute_dynamic_scales_and_zp(
                value=x, args=self.input_quant, module=None, global_scale=None
            )
            # Recompute the same global_scale that was baked into scale
            global_scale = self._compute_activation_global_scale(x)
            qdq_input = fake_quantize(
                x=x,
                scale=scale,
                zero_point=zero_point,
                args=self.input_quant,
                global_scale=global_scale,
            ).to(x.dtype)
        else:
            # W8A16: weights already dequantized at load time, pass through
            qdq_input = x

        out = torch.matmul(qdq_input, layer.weight.to(qdq_input.dtype).t())
        if bias is not None:
            out = out + bias
        return out
