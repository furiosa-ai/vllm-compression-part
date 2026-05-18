# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV-cache NVFP4 fake-quantization config.

Picklable dataclass that flows through ``VllmConfig`` to every worker. The
per-layer NVFP4 global scales are NOT stored here -- only the file path is.
Each worker lazy-loads the scales (with caching) inside
``attach_kv_quant_to_layer``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


_SUPPORTED_METHODS = ("nvfp4", "nvfp4_plus")


@dataclass
class KVCacheQuantConfig:
    """Configuration for KV-cache fake-quantization.

    Pass to ``LLM(...)`` via ``kv_cache_quant_config=KVCacheQuantConfig(...)``,
    or specify on the ``vllm serve`` CLI:

        # NVFP4 (requires offline calibration .pt):
        --kv-cache-quant-method nvfp4 \\
        --kv-cache-quant-global-scales-path <path>

        # NVFP4+ (no calibration; per-group FP32 scale computed dynamically):
        --kv-cache-quant-method nvfp4_plus

    Attributes:
        method: One of ``"nvfp4"`` or ``"nvfp4_plus"``.
            * ``"nvfp4"`` — per-tensor FP32 global × per-group FP8 E4M3 ×
              per-element FP4 E2M1; group_size=16; requires
              ``global_scales_path``.
            * ``"nvfp4_plus"`` — per-group FP32 × per-element FP4 E2M1;
              group_size=16; NO global scale, NO calibration step.
              ``global_scales_path`` must be unset.
        global_scales_path: Required iff ``method == "nvfp4"``. Path to a
            ``.pt`` file containing per-layer NVFP4 global scales:

                'gs_K': fp32 tensor of shape ``(num_layers,)``
                'gs_V': fp32 tensor of shape ``(num_layers,)``

            For ``method == "nvfp4_plus"`` this MUST be left unset; passing
            a path raises so silent misuse can't fall back to NVFP4 behavior.
    """

    method: str = "nvfp4"
    global_scales_path: str | None = None

    def __post_init__(self) -> None:
        if self.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {_SUPPORTED_METHODS}; "
                f"got {self.method!r}"
            )
        if self.method == "nvfp4":
            if not self.global_scales_path:
                raise ValueError(
                    "method='nvfp4' requires global_scales_path "
                    "(a .pt file with per-layer 'gs_K' / 'gs_V' fp32 tensors). "
                    "Pass --kv-cache-quant-global-scales-path <path> on the CLI, "
                    "or kv_cache_quant_config=KVCacheQuantConfig(..., "
                    "global_scales_path=...) to LLM()."
                )
            # Fail fast at config-construction time rather than later when a
            # worker tries to torch.load(); the runtime error path bubbles
            # through engine startup and is harder to debug.
            if not os.path.isfile(self.global_scales_path):
                raise FileNotFoundError(
                    f"global_scales_path {self.global_scales_path!r} does not "
                    f"exist (or is not a regular file). Run the calibration "
                    f"utility under K-EXAONE-evaluation/kv_cache_quantization/ "
                    f"first, or fix the path."
                )
        elif self.method == "nvfp4_plus":
            if self.global_scales_path:
                raise ValueError(
                    "method='nvfp4_plus' does NOT use a global-scales .pt "
                    "file — per-group FP32 scale is computed dynamically "
                    "inside the kernel and no calibration step is required. "
                    f"Got global_scales_path={self.global_scales_path!r}. "
                    "Remove the --kv-cache-quant-global-scales-path flag "
                    "(or set method='nvfp4' if you intended to use the "
                    "calibrated NVFP4 path)."
                )

    def is_active(self) -> bool:
        """Returns True if this config requires per-step quantization work."""
        return True
