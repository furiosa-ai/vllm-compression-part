# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from compressed_tensors_furiosa_extension.kernels import custom_extensions as extensions
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from compressed_tensors_furiosa_extension.quantization.quant_scheme import (
    create_mxfp8_scheme,
)
from compressed_tensors.quantization import QuantizationArgs, QuantizationType, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import (
    fake_quantize,
    dequantize,
)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           ModelWeightParameter)


logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A8MXFp8Plus"]


def fake_quantize_mxfp8_plus(
    input: torch.Tensor,
    group_size: int = 32,
    ebits: int = 4,
    mbits: int = 5,
    max_norm: float = 448.0,
    scale_bits: int = 8,
    axis: int = -1,
    flush_fp32_subnorms: bool = False,
    rounding_mode: int = 2,
    quantization_args: QuantizationArgs = None,
) -> torch.Tensor:
    
    scale, zero_point = compute_dynamic_scales_and_zp(
        value=input, args=quantization_args, module=None, global_scale=None
    )

    mxfp8_qdq_input = fake_quantize(
        x=input,
        scale=scale,
        zero_point=zero_point,
        args=quantization_args,
        g_idx=None,
        global_scale=None,
    )
    
    return mxfp8_qdq_input.to(input.dtype)


class CompressedTensorsW8A8MXFp8Plus(CompressedTensorsScheme):

    def __init__(self, weight_quant: QuantizationArgs,
                 input_quant: QuantizationArgs):
        self.group_size = 32
        self.ebits = 4
        self.mbits = 5
        self.max_norm = 448.0
        self.scale_bits = 16
        self.weight_quant = weight_quant
        self.input_quant = input_quant

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // self.group_size,
            dtype=torch.bfloat16,
        ),
                                                input_dim=1,
                                                output_dim=0,
                                                weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        # compressed-tensors 의 group fp8 dequantize 함수 사용
        dq_w = dequantize(
            x_q=layer.weight,          # FP8 weight
            scale=layer.weight_scale,  # group scale: [out, in/32]
            zero_point=None,
            args=None,                 # <- GROUP/32 자동 추론
            dtype=None,                # 원하는 dtype 있으면 torch.bfloat16 등으로 지정
        )
        layer.weight = torch.nn.Parameter(dq_w, requires_grad=False)

    def apply_weights(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        axis = x.dim() - 1
        qdq_input = fake_quantize_mxfp8_plus(
            input=x,
            group_size=self.group_size,
            ebits=self.ebits,
            mbits=self.mbits,
            max_norm=self.max_norm,
            scale_bits=self.scale_bits,
            axis=axis,
            flush_fp32_subnorms=False,
            rounding_mode=2,  # rd_away
            quantization_args=self.input_quant
        )
        
        dq_weight = layer.weight.to(qdq_input.dtype)
        
        out = torch.matmul(qdq_input, dq_weight.t())
        del qdq_input, dq_weight

        if bias is not None:
            out = out + bias
        return out
