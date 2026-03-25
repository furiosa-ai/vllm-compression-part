# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Group FP8 (MXFP8+) scheme for compressed-tensors models.

Handles models with:
  - format: float-quantized
  - weight strategy: group (group_size=32)
  - scale_dtype: bfloat16 / null  (NOT uint8)

Similar to MXFP8 but uses higher-precision bfloat16 scales instead of
E8M0 uint8 exponents, giving better quantization fidelity.
"""

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import (
    dequantize,
    fake_quantize,
)
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A8GrpFp8"]

_GRP_FP8_GROUP_SIZE = 32


class CompressedTensorsW8A8GrpFp8(CompressedTensorsScheme):
    """
    Emulation-mode group-FP8 (MXFP8+) W8A8 scheme.

    Weights are stored as float8_e4m3fn with bfloat16 per-group scales.
    At load time, weights are dequantized to the model dtype.
    At inference time, activations undergo dynamic per-group fake-quantize
    before a standard matmul with the dequantized weights.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
    ):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.group_size = weight_quant.group_size or _GRP_FP8_GROUP_SIZE

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

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.bfloat16,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dq_weight = dequantize(
            x_q=layer.weight.data,
            scale=layer.weight_scale.data,
            zero_point=None,
        )

        layer.weight = torch.nn.Parameter(dq_weight, requires_grad=False)
        del layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
