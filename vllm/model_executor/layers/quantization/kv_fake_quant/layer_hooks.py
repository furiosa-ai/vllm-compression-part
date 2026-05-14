# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-Attention-layer KV-quant state + the two hot-path entry points.

``LayerKVQuantState`` is an ``nn.Module`` so the per-layer NVFP4 global
scales (``gs_k``, ``gs_v``) auto-migrate when ``module.to(device)`` is
called on the parent Attention layer.

  attach_kv_quant_to_layer(layer, prefix)        called from Attention.__init__
                                                 builds + attaches state
  apply_kv_quant(layer, key, value) -> (K, V)    called from Attention.forward
                                                 reads state + dispatches to kernels
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.config import KVCacheQuantConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger

from .kernels import fake_quantize_nvfp4

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Per-layer state object
# ---------------------------------------------------------------------------

class LayerKVQuantState(nn.Module):
    """Per-Attention-layer NVFP4 KV-quant state.

    Subclassing ``nn.Module`` (rather than using a plain dataclass) makes the
    per-layer global-scale buffers auto-migrate with the parent's
    ``module.to(device)``.

    Attributes:
        method: Currently always ``"nvfp4"``. Stored for potential future
            multi-method dispatch.
        gs_k / gs_v: NVFP4 per-tensor (per-layer) FP32 global scales,
            registered as non-persistent buffers.
    """

    def __init__(
        self,
        method: str,
        gs_k: torch.Tensor,
        gs_v: torch.Tensor,
    ) -> None:
        super().__init__()
        self.method = method
        # Non-persistent buffers auto-migrate with module.to(device); won't be
        # saved with state_dict.
        self.register_buffer("gs_k", gs_k, persistent=False)
        self.register_buffer("gs_v", gs_v, persistent=False)


# ---------------------------------------------------------------------------
# Global-scales cache (one entry per file, populated lazily inside the worker
# the first time an nvfp4 layer is constructed)
# ---------------------------------------------------------------------------

_GLOBAL_SCALES_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _load_global_scales_cached(
    path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-layer NVFP4 global scales from the derived ``.pt``. Kept FP32
    on CPU and cached by path so workers only pay the IO cost once.

    The ``.pt`` must hold:
        'gs_K': fp32 tensor of shape (num_layers,)
        'gs_V': fp32 tensor of shape (num_layers,)
    """
    cached = _GLOBAL_SCALES_CACHE.get(path)
    if cached is not None:
        return cached
    blob = torch.load(path, weights_only=True)
    if "gs_K" not in blob or "gs_V" not in blob:
        raise KeyError(
            f"NVFP4 global scales file {path!r} is missing required keys "
            f"'gs_K' and/or 'gs_V'. Got top-level keys: {list(blob.keys())}"
        )
    gk = blob["gs_K"].to(torch.float32)
    gv = blob["gs_V"].to(torch.float32)
    _GLOBAL_SCALES_CACHE[path] = (gk, gv)
    return gk, gv


def get_active_kv_quant_config() -> KVCacheQuantConfig | None:
    """Read the active KV-quant config from the current VllmConfig, or None
    if no LLM is constructed / no kv_cache_quant_config was set."""
    vllm_cfg = get_current_vllm_config_or_none()
    if vllm_cfg is None:
        return None
    cfg = vllm_cfg.kv_cache_quant_config
    if cfg is None or not cfg.is_active():
        return None
    return cfg


# ---------------------------------------------------------------------------
# Per-layer registration (called from Attention.__init__)
# ---------------------------------------------------------------------------

def attach_kv_quant_to_layer(layer, prefix: str) -> None:
    """Build a ``LayerKVQuantState`` from the active config and attach it as
    ``layer.kv_quant_state``. No-op unless ``LLM(kv_cache_quant_config=...)``
    was set (or the equivalent ``vllm serve`` CLI flags were passed).
    """
    cfg = get_active_kv_quant_config()
    if cfg is None:
        return

    gs_k, gs_v = _resolve_nvfp4_global_scales(layer, prefix, cfg)
    layer.kv_quant_state = LayerKVQuantState(
        method=cfg.method,
        gs_k=gs_k,
        gs_v=gs_v,
    )


def _resolve_nvfp4_global_scales(
    layer, prefix: str, cfg: KVCacheQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the per-layer NVFP4 global scale tensor ``(num_layers,)`` down
    to this layer. Returns FP32 scalars (shape ``[1]``) on the layer's
    device.

    Global scale is per-tensor (per-layer) by NVFP4 spec, so it is the same
    across all TP ranks and head shards -- no kv-head splitting required.
    """
    from vllm.model_executor.models.utils import extract_layer_index
    try:
        layer_idx = extract_layer_index(prefix)
    except Exception as e:
        raise RuntimeError(
            f"[kv_fake_quant] NVFP4 could not extract layer_idx "
            f"from prefix={prefix!r}"
        ) from e
    gs_k_full, gs_v_full = _load_global_scales_cached(cfg.global_scales_path)
    try:
        device = next(layer.parameters()).device
    except StopIteration:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() else "cpu"
        )
    return (gs_k_full[layer_idx].clone().reshape(1).to(device),
            gs_v_full[layer_idx].clone().reshape(1).to(device))


# ---------------------------------------------------------------------------
# Forward dispatch (called from Attention.forward)
# ---------------------------------------------------------------------------

def apply_kv_quant(
    layer, key: torch.Tensor, value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return NVFP4 fake-quantized ``(K, V)``.

    Reads ``layer.kv_quant_state`` for the per-layer global scales.
    """
    state: LayerKVQuantState = layer.kv_quant_state
    if state.method != "nvfp4":
        raise ValueError(
            f"Only method='nvfp4' is supported; got {state.method!r}"
        )
    nh = layer.num_kv_heads
    hd = layer.head_size
    key = fake_quantize_nvfp4(key, nh, hd, state.gs_k)
    value = fake_quantize_nvfp4(value, nh, hd, state.gs_v)
    return key, value
