# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV-cache NVFP4 fake-quantization config.

Picklable dataclass that flows through ``VllmConfig`` to every worker. The
per-layer NVFP4 global scales are NOT stored here -- only the file path is.
Each worker lazy-loads the scales (with caching) inside
``attach_kv_quant_to_layer``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheQuantConfig:
    """Configuration for KV-cache NVFP4 fake-quantization.

    Pass to ``LLM(...)`` via ``kv_cache_quant_config=KVCacheQuantConfig(...)``,
    or specify on the ``vllm serve`` CLI:

        --kv-cache-quant-method nvfp4 \\
        --kv-cache-quant-global-scales-path <path>

    Attributes:
        method: Only ``"nvfp4"`` is supported on this branch. The field is
            kept for forward-extensibility.
        global_scales_path: Required. Path to a ``.pt`` file containing
            per-layer NVFP4 global scales. The ``.pt`` must hold:

                'gs_K': fp32 tensor of shape ``(num_layers,)``
                'gs_V': fp32 tensor of shape ``(num_layers,)``

            Per the NVFP4 spec, global scales are FP32 per tensor (per layer
            here) and are the same across all TP ranks.
    """

    method: str = "nvfp4"
    global_scales_path: str | None = None

    def __post_init__(self) -> None:
        if self.method != "nvfp4":
            raise ValueError(
                f"Only method='nvfp4' is supported on this branch; "
                f"got {self.method!r}"
            )
        if not self.global_scales_path:
            raise ValueError(
                "method='nvfp4' requires global_scales_path "
                "(a .pt file with per-layer 'gs_K' / 'gs_V' fp32 tensors)"
            )

    def is_active(self) -> bool:
        """Returns True if this config requires per-step quantization work."""
        return True
