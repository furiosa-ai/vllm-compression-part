# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KV-cache NVFP4 fake-quantization for accuracy studies.

The NVFP4 kernel runs quant -> dequant in the same step. KV cache *storage*
is unchanged (still BF16); only the *values* are constrained to the FP4 grid.
So this is an accuracy study, not a memory/speedup study.

Wired into vLLM via two direct calls. Neither is a runtime monkey-patch:

    USER                                     OUR CODE                       UPSTREAM EDIT
    ────                                     ────────                       ─────────────
    LLM(kv_cache_quant_config=...)
      └─ stored on VllmConfig
                                                                            (none)
    Attention.__init__:
      └─ attach_kv_quant_to_layer ───────────> layer_hooks.attach_*         attention.py:391
                                                  │
                                                  ├─ reads VllmConfig
                                                  └─ attaches LayerKVQuantState
                                                     (with per-layer gs_k/gs_v)

    Attention.forward:
      └─ apply_kv_quant ─────────────────────> layer_hooks.apply_*          attention.py:488
                                                  │
                                                  └─ dispatches to
                                                     kernels.fake_quantize_nvfp4

Public API (re-exported below):

    attach_kv_quant_to_layer(layer, prefix)         called from Attention.__init__
    apply_kv_quant(layer, key, value) -> (K, V)     called from Attention.forward

    LayerKVQuantState                               per-layer state (nn.Module)
    fake_quantize_nvfp4                             direct test/utility access
"""

from .kernels import fake_quantize_nvfp4, fake_quantize_nvfp4_plus
from .layer_hooks import (
    LayerKVQuantState,
    apply_kv_quant,
    attach_kv_quant_to_layer,
)

__all__ = [
    "attach_kv_quant_to_layer",
    "apply_kv_quant",
    "LayerKVQuantState",
    "fake_quantize_nvfp4",
    "fake_quantize_nvfp4_plus",
]
