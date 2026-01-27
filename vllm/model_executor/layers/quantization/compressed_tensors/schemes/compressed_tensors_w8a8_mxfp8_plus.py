# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from compressed_tensors_furiosa_extension.kernels import custom_extensions as extensions
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp

from compressed_tensors.quantization import QuantizationArgs, QuantizationType, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import (
    fake_quantize,
    dequantize,
)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           ModelWeightParameter)


logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A8MXFp8Plus"]

torch.library.define(
    "furiosa::quantize_mxfp8_plus",
    "(Tensor input, int group_size, int axis, int rounding_mode) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)

@torch.library.impl("furiosa::quantize_mxfp8_plus", "cuda")
def quantize_mxfp8_plus_cuda(
    input: torch.Tensor,
    group_size: int = 32,
    axis: int = -1,
    rounding_mode: int = 2,
) -> torch.Tensor:
    input_contig = input.contiguous() if not input.is_contiguous() else input
    return extensions.quantize_mxfp8_plus_by_tile_func_cuda(
        input_contig, group_size, axis, rounding_mode
    )

@torch.library.register_fake("furiosa::quantize_mxfp8_plus")
def quantize_mxfp8_plus_fake(
    input: torch.Tensor,
    group_size: int = 32,
    axis: int = -1,
    rounding_mode: int = 2,
) -> torch.Tensor:
    return torch.empty_like(input)

@torch.library.impl("furiosa::quantize_mxfp8_plus", "cpu")
def quantize_mxfp8_plus_cpu(
    input: torch.Tensor,
    rounding_mode: int = 2,
    group_size: int = 32,
    axis: int = -1,
) -> torch.Tensor:
    from compressed_tensors_furiosa_extension.quantization.quant_scheme import create_mxfp8_plus_scheme
    quantization_args = create_mxfp8_plus_scheme().input_activations
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

def fake_quantize_mxfp8_plus(
    input: torch.Tensor,
    group_size: int = 32,
    axis: int = -1,
    rounding_mode: int = 2,
) -> torch.Tensor:
    
    return torch.ops.furiosa.quantize_mxfp8_plus(
        input, group_size, axis, rounding_mode
    )


class CompressedTensorsW8A8MXFp8Plus(CompressedTensorsScheme):

    def __init__(self):
        self.group_size = 32

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
            axis=axis,
            rounding_mode=2,  # rd_away
        )
        
        dq_weight = layer.weight.to(qdq_input.dtype)
        
        out = torch.matmul(qdq_input, dq_weight.t())
        del qdq_input, dq_weight

        if bias is not None:
            out = out + bias
        return out
