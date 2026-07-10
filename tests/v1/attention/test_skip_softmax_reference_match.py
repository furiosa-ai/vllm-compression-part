# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-match tests for BLASST skip-softmax in the FlashInfer TRTLLM path.

Proves the TRTLLM kernels (``trtllm_batch_context_with_kv_cache`` for prefill,
``trtllm_batch_decode_with_kv_cache`` for decode) drop the *correct* KV score
blocks when ``skip_softmax_threshold_scale_factor_{prefill,decode}`` is set —
the skip-softmax analog of the NVFP4+ bit-identity test
(``tests/kernels/quantization/test_nvfp4_plus_kv_bitidentity.py``).

Reference design
----------------
The reference is an INDEPENDENT PyTorch implementation derived from the
documented rule in ``tests/skip_softmax_tuning_guide.md`` (the same rule as
ModelOpt ``FlashSkipSoftmax.calc_correction_factor_and_p``); it is NOT copied
from the kernel:

    per (head, q-tile) the kv-tiles are scanned left to right;
    drop tile  iff  max_tile_score - running_cummax < log(threshold)
    threshold = scale_factor / seq_k

where ``max_tile_score`` is the max of the causally-masked QK^T * sm_scale
scores inside the (q-tile, kv-tile) and ``running_cummax`` is the running max
of ``max_tile_score`` over kv-tiles.  Surviving entries go through a plain
fp32 softmax; dropped tiles contribute exactly nothing.

Notes on the reference (each of these is a deliberate, documented choice):

* ``running_cummax`` is implemented INCLUSIVE of the current tile.  For any
  threshold < 1 (i.e. log(threshold) < 0 — the only sane regime, since
  scale_factor << seq_k in practice) inclusive vs. exclusive cummax yield
  IDENTICAL drop decisions: when the current tile raises the max, the margin
  is 0 (inclusive) or positive (exclusive), and neither is < log(threshold).
  This also guarantees every query row keeps at least one surviving tile, so
  the post-drop softmax is always well defined.
* ``KV_TILE_SIZE = 128`` is the assumed kernel kv-tile granularity.  This
  MUST match the TRTLLM kernel's internal kv tile for the comparison to be
  exact; 128 is the ModelOpt calibration default (``bc=128`` in the tuning
  guide) and the typical trtllm-gen tile.  It is a named constant / parameter
  so it can be swept if the kernel generation changes.  On a mismatch the
  aggressive-threshold tests embed a per-tile-size diagnostic in the failure
  message (see ``_diagnose_tile_configs``).
* ``Q_TILE_SIZE = 128`` with prefill ``query_len = 32`` means the whole new
  query chunk of a request lands in ONE q-tile, so the per-q-tile drop
  decision is shared by all its rows regardless of the kernel's actual q-tile
  size (as long as it is >= 32).  Decode has query_len = 1, where q-tile
  granularity is irrelevant.
* ``seq_k`` per phase mirrors the fork's call sites in
  ``vllm/v1/attention/backends/flashinfer.py``: prefill passes
  ``max_kv_len = attn_metadata.prefill.max_seq_len`` and decode passes
  ``max_seq_len = attn_metadata.decode.max_seq_len`` — i.e. the BATCH max in
  both phases.  The tests use batch specs with UNIFORM seq_lens so that the
  batch max equals every request's own length: the reference is then correct
  even if the kernel internally derives the threshold from per-request
  lengths instead of the scalar it is handed.  The spy on the kernel call
  additionally asserts the scalar actually passed.

What the tests assert
---------------------
For both a PREFILL and a DECODE batch spec:

1. skip-off (threshold None)   -> kernel output ~= exact fp32 attention
   (sanity: harness + backend + reference base are right), and the TRTLLM
   kernel was actually called with ``skip_softmax_threshold_scale_factor=None``.
2. aggressive threshold        -> kernel output ~= the block-dropped
   REFERENCE (proves the RIGHT tiles were skipped) AND meaningfully different
   from exact attention (proves a real effect, not a silent no-op).
3. a monotonic threshold sweep -> divergence from exact attention grows as
   the threshold gets more aggressive (corroboration).

The exact skip FRACTION is not observable from the kernel, so no % sparsity
is asserted — correctness is established through the reference match alone.

Robustness against decision-boundary flakiness: the aggressive scale factor
is not hardcoded.  It is derived from the reference margins of the (seeded,
deterministic) test data by placing log(threshold) in the middle of the
widest gap of the sorted margin distribution near the target drop fraction
(``_choose_robust_scale_factor``).  Kernel-vs-reference score arithmetic
differs only by fp32 accumulation order (~1e-4 in margin), orders of
magnitude below the enforced gap, so no tile decision can flip between the
kernel and the reference.

Hardware note (deliberately NOT gated)
--------------------------------------
These tests REQUIRE the FlashInfer TRTLLM skip-softmax path to actually run:
a CUDA device on which ``supports_trtllm_attention()`` is true (SM100 /
Blackwell family with the NVIDIA artifactory reachable for cubins) and a
flashinfer build whose trtllm kernels accept
``skip_softmax_threshold_scale_factor``.  There is intentionally NO platform
gating/skip: the config forces ``attention_config.use_trtllm_attention=True``
and the spy asserts the TRTLLM kernel fired.  Where the path cannot run
(e.g. A100/SM80, where vLLM silently falls back to native FlashInfer), the
spy assertion fails — a TRUE NEGATIVE, not a skip.  This is by design: a
silent fallback is precisely the failure mode this test exists to expose.

Run:
    .venv/bin/python -m pytest tests/v1/attention/test_skip_softmax_reference_match.py -v
    # or standalone (no pytest needed):
    .venv/bin/python tests/v1/attention/test_skip_softmax_reference_match.py
"""

import functools
import math
import sys
import unittest.mock
from pathlib import Path

import torch

try:
    import pytest
except ImportError:
    # pytest is optional — file also runs as a standalone script.
    class _PytestStub:
        class mark:  # noqa: N801
            @staticmethod
            def parametrize(*args, **kwargs):
                return lambda fn: fn

    pytest = _PytestStub()  # type: ignore

# Repo root on sys.path so `import tests...` works when run standalone
# (pytest adds it automatically via rootdir/conftest).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.v1.attention.test_attention_backends import (  # noqa: E402
    _convert_dtype_to_torch,
    create_and_prepopulate_kv_cache,
    run_attention_backend,
)
from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm.v1.attention.backends.utils import set_kv_cache_layout  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (see module docstring for the justification of each)
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen3-0.6B"  # same tiny model the existing harness tests use

# ASSUMPTION: kernel kv-tile granularity. Must match the TRTLLM kernel's
# internal kv tile (ModelOpt calibration default bc=128). Parameterized in
# the reference so it can be swept; _diagnose_tile_configs reports
# alternatives on a mismatch.
KV_TILE_SIZE = 128
# Prefill q-tile. Chosen >= the prefill query_len (32) so the whole query
# chunk shares one drop decision per kv-tile, independent of the kernel's
# real q-tile size (any value >= 32 behaves identically here).
Q_TILE_SIZE = 128

# vLLM paged-KV page size (unrelated to the skip tile: pages are gathered
# into contiguous kv tiles by the kernel; the skip decision granularity is
# the kv tile over absolute sequence positions).
PAGE_SIZE = 16

# Uniform seq_lens (see docstring: makes batch-max seq_k == per-request
# seq_k, removing the only ambiguity in the threshold denominator).
BATCH_SPECS = {
    "uniform_prefill": BatchSpec(
        seq_lens=[2048] * 4, query_lens=[32] * 4, name="uniform_prefill"
    ),
    "uniform_decode": BatchSpec(
        seq_lens=[2048] * 8, query_lens=[1] * 8, name="uniform_decode"
    ),
}

# Tolerances (bf16 I/O, fp32 accumulation on both sides):
#   * bf16 has ~8 bits of mantissa; attention outputs here are O(1), so one
#     bf16 ulp is ~4e-3-8e-3. The existing harness passes at atol=rtol=1e-2
#     against a flex-attention reference; we double it to absorb the extra
#     accumulation-order noise of the online-softmax TRTLLM kernel vs. a
#     monolithic fp32 softmax (split-K, tile-ordered exp/rescale).
#   * The engineered block-drop effect is >= MIN_EFFECT = 5e-2 at the output
#     max-abs level (asserted on the reference itself), i.e. >2x the match
#     tolerance — the "right blocks dropped" match cannot pass by accident
#     against the wrong (exact / differently-dropped) output.
TOL_EXACT = dict(atol=2e-2, rtol=2e-2)  # skip-off vs exact fp32 reference
TOL_MATCH = dict(atol=2e-2, rtol=2e-2)  # aggressive vs dropped reference
MIN_EFFECT = 5e-2  # min max-abs(exact_ref - dropped_ref): "a real effect"

# Aggressive-threshold selection: target drop fraction of (finite) tile
# margins, and the minimum decision-boundary gap we insist on so that no
# tile decision can flip between kernel and reference arithmetic (~1e-4
# margin noise from fp32 accumulation order).
TARGET_DROP_FRAC = {"prefill": 0.35, "decode": 0.5}
MIN_DECISION_GAP = 2e-3


# ---------------------------------------------------------------------------
# Independent reference: exact attention + documented block-drop rule
# ---------------------------------------------------------------------------


def _masked_scores(
    q: torch.Tensor,  # (q_len, num_q_heads, head_dim)
    k: torch.Tensor,  # (s_len, num_kv_heads, head_dim)
    *,
    sm_scale: float,
    context_len: int,
) -> torch.Tensor:
    """Causally-masked fp32 scores, shape (num_q_heads, q_len, s_len).

    GQA: kv heads are repeat_interleaved to q heads (same convention as the
    harness's flex-attention reference).
    """
    q_len, num_q_heads, _ = q.shape
    s_len, num_kv_heads, _ = k.shape
    assert num_q_heads % num_kv_heads == 0
    rep = num_q_heads // num_kv_heads

    qf = q.transpose(0, 1).float()  # (Hq, q_len, D)
    kf = k.transpose(0, 1).float().repeat_interleave(rep, dim=0)  # (Hq, s_len, D)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale

    q_pos = torch.arange(q_len, device=q.device) + context_len
    kv_pos = torch.arange(s_len, device=q.device)
    visible = q_pos[:, None] >= kv_pos[None, :]  # causal
    return scores.masked_fill(~visible, float("-inf"))


def _tile_margins(
    scores: torch.Tensor,  # (H, q_len, s_len), masked, fp32
    *,
    q_tile_size: int,
    kv_tile_size: int,
    cummax_split_size: int | None = None,
) -> torch.Tensor:
    """Per-(head, q-tile, kv-tile) margin = max_tile_score - running_cummax.

    Returns (H, n_qt, n_kt) fp32. -inf entries correspond to fully-masked
    tiles (always droppable — they contribute nothing anyway).

    ``cummax_split_size`` (in TILES) optionally restarts the running cummax
    every N kv-tiles — a knob to mirror split-KV kernels that scan kv chunks
    independently. Default None = one global left-to-right scan, matching
    the documented rule.
    """
    H, q_len, s_len = scores.shape
    n_qt = math.ceil(q_len / q_tile_size)
    n_kt = math.ceil(s_len / kv_tile_size)
    pad_q = n_qt * q_tile_size - q_len
    pad_k = n_kt * kv_tile_size - s_len
    padded = torch.nn.functional.pad(
        scores, (0, pad_k, 0, pad_q), value=float("-inf")
    )
    tiles = padded.view(H, n_qt, q_tile_size, n_kt, kv_tile_size)
    tile_max = tiles.amax(dim=4).amax(dim=2)  # (H, n_qt, n_kt)

    if cummax_split_size is None:
        run_max = tile_max.cummax(dim=-1).values
    else:
        chunks = []
        for start in range(0, n_kt, cummax_split_size):
            chunk = tile_max[..., start : start + cummax_split_size]
            chunks.append(chunk.cummax(dim=-1).values)
        run_max = torch.cat(chunks, dim=-1)

    # (-inf) - (-inf) = nan can only arise if an entire q-tile has no visible
    # key in the leading kv-tiles, which cannot happen for causal attention
    # (kv position 0 is visible to every query row). Map defensively to -inf
    # (= "droppable"), never to "keep".
    margins = torch.nan_to_num(tile_max - run_max, nan=float("-inf"))
    return margins


def reference_skip_softmax_attention(
    q: torch.Tensor,  # (q_len, num_q_heads, head_dim), model dtype
    k: torch.Tensor,  # (s_len, num_kv_heads, head_dim)
    v: torch.Tensor,  # (s_len, num_kv_heads, head_dim)
    *,
    sm_scale: float,
    context_len: int,
    scale_factor: float | None,
    seq_k: int,
    q_tile_size: int = Q_TILE_SIZE,
    kv_tile_size: int = KV_TILE_SIZE,
    cummax_split_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fp32 attention + the documented skip-softmax block-drop mask.

    drop tile  iff  max_tile_score - running_cummax < log(scale_factor/seq_k)

    ``scale_factor=None`` or ``0.0`` disables dropping (0.0 -> threshold 0,
    log -> -inf, nothing is below it — mirrors the tuning guide's "a scale
    factor of 0.0 must reproduce the dense baseline").

    Returns:
        out:     (q_len, num_q_heads, head_dim) fp32
        drop:    (num_q_heads, n_qt, n_kt) bool — which tiles were dropped
        margins: (num_q_heads, n_qt, n_kt) fp32 — max_tile_score - cummax

    Written loop-light but fully materialized so every intermediate
    (scores / tile_max / margins / drop) is inspectable under a debugger.
    """
    q_len, num_q_heads, _ = q.shape
    s_len = k.shape[0]
    scores = _masked_scores(q, k, sm_scale=sm_scale, context_len=context_len)
    margins = _tile_margins(
        scores,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
        cummax_split_size=cummax_split_size,
    )

    if scale_factor is None or scale_factor == 0.0:
        drop = torch.zeros_like(margins, dtype=torch.bool)
    else:
        log_thr = math.log(scale_factor / seq_k)
        drop = margins < log_thr
        # Expand tile decisions to element level and mask.
        n_qt, n_kt = drop.shape[1], drop.shape[2]
        drop_el = (
            drop[:, :, None, :, None]
            .expand(num_q_heads, n_qt, q_tile_size, n_kt, kv_tile_size)
            .reshape(num_q_heads, n_qt * q_tile_size, n_kt * kv_tile_size)
        )[:, :q_len, :s_len]
        scores = scores.masked_fill(drop_el, float("-inf"))

    # Inclusive cummax guarantees the running-max tile survives (margin 0 is
    # never < log_thr for threshold < 1), so every row has >= 1 finite score.
    probs = torch.softmax(scores, dim=-1)
    rep = num_q_heads // v.shape[1]
    vf = v.transpose(0, 1).float().repeat_interleave(rep, dim=0)
    out = torch.einsum("hqk,hkd->hqd", probs, vf).transpose(0, 1)
    return out, drop, margins


def _choose_robust_scale_factor(
    margins_list: list[torch.Tensor],
    *,
    seq_k: int,
    target_drop_frac: float,
    window_frac: float = 0.15,
    min_gap: float = MIN_DECISION_GAP,
) -> tuple[float, float, float]:
    """Pick an aggressive scale factor whose decision boundary is robust.

    Places log(threshold) in the middle of the WIDEST gap of the sorted
    finite-margin distribution within +-window_frac of the target drop
    fraction, so no tile margin sits near the boundary and ULP-level
    kernel-vs-reference score differences (~1e-4) cannot flip a decision.

    Returns (scale_factor, log_thr, gap). Raises AssertionError with a
    "test data" message (NOT a kernel bug) if no adequately wide gap exists
    — fix by changing the seed or the target fraction, not the kernel.
    """
    m = torch.cat(
        [x[torch.isfinite(x)].flatten().float().cpu() for x in margins_list]
    )
    vals, _ = m.sort()
    n = vals.numel()
    assert n > 16, "not enough tiles to choose a threshold from"
    center = int(target_drop_frac * n)
    half = max(2, int(window_frac * n))
    lo = max(1, center - half)
    hi = min(n - 1, center + half)
    gaps = vals[lo : hi + 1] - vals[lo - 1 : hi]
    j = int(torch.argmax(gaps))
    gap = float(gaps[j])
    log_thr = float((vals[lo - 1 + j] + vals[lo + j]) / 2)
    assert gap >= min_gap, (
        f"TEST-DATA issue (not a kernel bug): widest margin gap near the "
        f"{target_drop_frac:.0%} drop target is {gap:.2e} < {min_gap:.0e}; "
        f"decisions would be flaky. Change the seed or target fraction."
    )
    assert log_thr < 0, (
        f"chosen log(threshold)={log_thr:.3f} >= 0 implies threshold >= 1 "
        f"(degenerate: would drop even running-max tiles); lower the target."
    )
    return seq_k * math.exp(log_thr), log_thr, gap


# ---------------------------------------------------------------------------
# CPU self-checks of the reference (no GPU, no vLLM server, no kernel)
# ---------------------------------------------------------------------------


def _tiny_qkv(q_len=32, s_len=512, hq=4, hkv=2, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(q_len, hq, d, generator=g)
    k = torch.randn(s_len, hkv, d, generator=g)
    v = torch.randn(s_len, hkv, d, generator=g)
    return q, k, v, 1.0 / math.sqrt(d), s_len - q_len


def test_reference_none_threshold_is_exact_softmax():
    """scale_factor=None must reduce to plain causal fp32 attention."""
    q, k, v, sm_scale, ctx = _tiny_qkv()
    out, drop, _ = reference_skip_softmax_attention(
        q, k, v, sm_scale=sm_scale, context_len=ctx, scale_factor=None, seq_k=512
    )
    assert not drop.any()
    # Independent exact path: torch SDPA with an explicit causal mask.
    rep = q.shape[1] // k.shape[1]
    qf = q.transpose(0, 1).float()
    kf = k.transpose(0, 1).float().repeat_interleave(rep, dim=0)
    vf = v.transpose(0, 1).float().repeat_interleave(rep, dim=0)
    q_pos = torch.arange(q.shape[0]) + ctx
    mask = q_pos[:, None] >= torch.arange(k.shape[0])[None, :]
    expected = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, attn_mask=mask, scale=sm_scale
    ).transpose(0, 1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_reference_zero_scale_factor_drops_nothing():
    """Tuning guide sanity bound: scale factor 0.0 == dense baseline."""
    q, k, v, sm_scale, ctx = _tiny_qkv(seed=1)
    out0, drop0, _ = reference_skip_softmax_attention(
        q, k, v, sm_scale=sm_scale, context_len=ctx, scale_factor=0.0, seq_k=512
    )
    out_none, _, _ = reference_skip_softmax_attention(
        q, k, v, sm_scale=sm_scale, context_len=ctx, scale_factor=None, seq_k=512
    )
    assert not drop0.any()
    torch.testing.assert_close(out0, out_none, atol=0.0, rtol=0.0)


def test_reference_drop_set_grows_with_scale_factor():
    """Monotonicity: a larger scale factor's drop set contains a smaller's."""
    q, k, v, sm_scale, ctx = _tiny_qkv(seed=2)
    prev = None
    for log_thr in (-6.0, -4.0, -2.0, -1.0):
        sf = 512 * math.exp(log_thr)
        _, drop, _ = reference_skip_softmax_attention(
            q, k, v, sm_scale=sm_scale, context_len=ctx, scale_factor=sf, seq_k=512
        )
        if prev is not None:
            assert (prev & ~drop).sum() == 0, "drop set must grow monotonically"
        prev = drop


def test_reference_running_max_tile_never_dropped():
    """For threshold < 1 the tile holding the running max (margin 0) always
    survives, so every query row keeps at least one visible kv-tile."""
    q, k, v, sm_scale, ctx = _tiny_qkv(seed=3)
    sf = 512 * math.exp(-0.5)  # very aggressive, threshold still < 1
    out, drop, margins = reference_skip_softmax_attention(
        q, k, v, sm_scale=sm_scale, context_len=ctx, scale_factor=sf, seq_k=512
    )
    assert not drop[margins == 0].any()
    assert torch.isfinite(out).all()
    assert drop.any(), "this scale factor should drop most tiles"


# ---------------------------------------------------------------------------
# GPU-side machinery: build data, run the real TRTLLM kernel with a spy
# ---------------------------------------------------------------------------


class _Case:
    """Everything needed to run the kernel and the reference for one spec."""

    def __init__(self, batch_spec: BatchSpec):
        self.batch_spec = batch_spec
        set_random_seed(42)
        device = torch.device("cuda:0")
        self.device = device

        vllm_config = create_vllm_config(
            model_name=MODEL,
            max_model_len=max(batch_spec.seq_lens),
            block_size=PAGE_SIZE,
            num_gpu_blocks=8192,
        )
        # Force the TRTLLM path. NOTE: this is NOT a hardware gate — on
        # platforms where supports_trtllm_attention() is false vLLM still
        # falls back to native FlashInfer, and the spy assertion below then
        # fails (a true negative, by design).
        vllm_config.attention_config.use_trtllm_attention = True
        self.vllm_config = vllm_config
        self.kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

        mc, pc = vllm_config.model_config, vllm_config.parallel_config
        self.num_q_heads = mc.get_num_attention_heads(pc)
        self.num_kv_heads = mc.get_num_kv_heads(pc)
        self.head_size = mc.get_head_size()
        self.sm_scale = 1.0 / math.sqrt(self.head_size)
        self.dtype = _convert_dtype_to_torch(mc.dtype)

        # Per-request Q / full K / full V (contexts feed the paged cache,
        # trailing query_len tokens are the "new" K/V) — mirrors the harness.
        self.per_req: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        all_q, all_k_new, all_v_new, k_ctx, v_ctx = [], [], [], [], []
        for s_len, q_len in zip(batch_spec.seq_lens, batch_spec.query_lens):
            ctx = s_len - q_len
            q = torch.randn(
                q_len, self.num_q_heads, self.head_size, dtype=self.dtype,
                device=device,
            )
            kf = torch.randn(
                s_len, self.num_kv_heads, self.head_size, dtype=self.dtype,
                device=device,
            )
            vf = torch.randn(
                s_len, self.num_kv_heads, self.head_size, dtype=self.dtype,
                device=device,
            )
            self.per_req.append((q, kf, vf, ctx))
            all_q.append(q)
            all_k_new.append(kf[ctx:])
            all_v_new.append(vf[ctx:])
            k_ctx.append(kf[:ctx])
            v_ctx.append(vf[:ctx])
        self.query = torch.cat(all_q, dim=0)
        self.key = torch.cat(all_k_new, dim=0)
        self.value = torch.cat(all_v_new, dim=0)

        self.common_attn_metadata = create_common_attn_metadata(
            batch_spec, PAGE_SIZE, device
        )
        kv_cache = create_and_prepopulate_kv_cache(
            k_contexts=k_ctx,
            v_contexts=v_ctx,
            block_size=PAGE_SIZE,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.dtype,
            device=device,
            num_blocks=vllm_config.cache_config.num_gpu_blocks or 8192,
            common_attn_metadata=self.common_attn_metadata,
            randomize_blocks=True,
        )
        # FlashInfer layout: (num_blocks, 2, ...) + HND (harness convention;
        # the TRTLLM path asserts get_kv_cache_layout() == "HND").
        kv_cache = kv_cache.transpose(0, 1)
        self.kv_cache = kv_cache.transpose(2, 3).contiguous().transpose(2, 3)

        # seq_k the kernel call sites use (batch max; uniform specs make it
        # equal to every request's own length — see module docstring).
        self.seq_k = max(batch_spec.seq_lens)

        # Exact reference + threshold-independent margins, per request.
        outs, margins = [], []
        for q, kf, vf, ctx in self.per_req:
            out, _, m = reference_skip_softmax_attention(
                q, kf, vf, sm_scale=self.sm_scale, context_len=ctx,
                scale_factor=None, seq_k=self.seq_k,
            )
            outs.append(out)
            margins.append(m)
        self.exact_ref = torch.cat(outs, dim=0)  # fp32, (tokens, Hq, D)
        self.margins = margins

    def reference(
        self,
        scale_factor: float | None,
        *,
        q_tile_size: int = Q_TILE_SIZE,
        kv_tile_size: int = KV_TILE_SIZE,
        cummax_split_size: int | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        """Dropped reference output (fp32) + (#dropped, #total finite) tiles."""
        outs, dropped, total = [], 0, 0
        for q, kf, vf, ctx in self.per_req:
            out, drop, m = reference_skip_softmax_attention(
                q, kf, vf, sm_scale=self.sm_scale, context_len=ctx,
                scale_factor=scale_factor, seq_k=self.seq_k,
                q_tile_size=q_tile_size, kv_tile_size=kv_tile_size,
                cummax_split_size=cummax_split_size,
            )
            outs.append(out)
            finite = torch.isfinite(m)
            dropped += int((drop & finite).sum())
            total += int(finite.sum())
        return torch.cat(outs, dim=0), dropped, total

    def run_kernel(
        self, sf_prefill: float | None, sf_decode: float | None
    ) -> tuple[torch.Tensor, dict[str, list[dict]]]:
        """Run the real FlashInfer backend, spying on the TRTLLM entry points.

        The spies (a) prove the TRTLLM kernel actually fired (no silent
        fallback), (b) capture the skip_softmax_threshold_scale_factor and
        the max_kv_len / max_seq_len the fork passed as seq_k.
        """
        self.vllm_config.attention_config.skip_softmax_threshold_scale_factor_prefill = (  # noqa: E501
            sf_prefill
        )
        self.vllm_config.attention_config.skip_softmax_threshold_scale_factor_decode = (  # noqa: E501
            sf_decode
        )

        import vllm.v1.attention.backends.flashinfer as fi_mod

        calls: dict[str, list[dict]] = {"prefill": [], "decode": []}
        real_prefill = fi_mod.trtllm_batch_context_with_kv_cache
        real_decode = fi_mod.trtllm_batch_decode_with_kv_cache

        def spy_prefill(*args, **kwargs):
            calls["prefill"].append(dict(kwargs))
            return real_prefill(*args, **kwargs)

        def spy_decode(*args, **kwargs):
            calls["decode"].append(dict(kwargs))
            return real_decode(*args, **kwargs)

        # set_current_vllm_config wraps the WHOLE call (not just impl
        # creation, which run_attention_backend already wraps) because the
        # metadata builder's __init__ reads attention_config.
        # use_trtllm_attention via get_current_vllm_config() to decide the
        # decode TRTLLM dispatch.
        set_kv_cache_layout("HND")
        try:
            with (
                set_current_vllm_config(self.vllm_config),
                unittest.mock.patch.object(
                    fi_mod, "trtllm_batch_context_with_kv_cache", spy_prefill
                ),
                unittest.mock.patch.object(
                    fi_mod, "trtllm_batch_decode_with_kv_cache", spy_decode
                ),
            ):
                output = run_attention_backend(
                    AttentionBackendEnum.FLASHINFER,
                    self.kv_cache_spec,
                    ["placeholder"],
                    self.vllm_config,
                    self.device,
                    self.common_attn_metadata,
                    self.query,
                    self.key,
                    self.value,
                    self.kv_cache,
                )
        finally:
            set_kv_cache_layout(None)
        return output, calls


@functools.lru_cache(maxsize=None)
def _get_case(spec_name: str) -> _Case:
    return _Case(BATCH_SPECS[spec_name])


def _assert_trtllm_fired(
    calls: dict[str, list[dict]],
    phase: str,
    expected_scale_factor: float | None,
    expected_seq_k: int,
):
    other = "decode" if phase == "prefill" else "prefill"
    assert len(calls[phase]) == 1 and len(calls[other]) == 0, (
        f"TRTLLM {phase} kernel did not fire exactly once "
        f"(prefill calls={len(calls['prefill'])}, "
        f"decode calls={len(calls['decode'])}). On platforms without TRTLLM "
        f"support (needs SM100-family + flashinfer cubins) vLLM silently "
        f"falls back to native FlashInfer — that fallback is exactly what "
        f"this assertion is meant to expose. This test has no hardware skip "
        f"by design."
    )
    kw = calls[phase][0]
    got_sf = kw.get("skip_softmax_threshold_scale_factor")
    assert got_sf == expected_scale_factor, (
        f"kernel received skip_softmax_threshold_scale_factor={got_sf}, "
        f"expected {expected_scale_factor}"
    )
    # seq_k plumbing: prefill passes max_kv_len, decode passes max_seq_len.
    seq_k_key = "max_kv_len" if phase == "prefill" else "max_seq_len"
    got_seq_k = kw.get(seq_k_key)
    assert got_seq_k == expected_seq_k, (
        f"kernel {seq_k_key}={got_seq_k}, but the reference assumed "
        f"seq_k={expected_seq_k}; the threshold denominators disagree"
    )


def _diagnose_tile_configs(
    case: _Case, kernel_out: torch.Tensor, scale_factor: float
) -> str:
    """On a reference mismatch, report errors for alternative tile configs.

    Helps distinguish "kernel drops WRONG blocks" from "our KV_TILE_SIZE /
    cummax-scan assumption doesn't match this kernel generation".
    """
    lines = ["per-config max|kernel - dropped_ref| (fp32):"]
    for kv_tile in (32, 64, 128, 256):
        for split in (None, 4, 8):
            ref, ndrop, ntot = case.reference(
                scale_factor, kv_tile_size=kv_tile, cummax_split_size=split
            )
            err = (kernel_out.float() - ref).abs().max().item()
            lines.append(
                f"  kv_tile={kv_tile:<4d} cummax_split={str(split):<4s} "
                f"drop={ndrop}/{ntot:<5d} max_err={err:.4f}"
            )
    return "\n".join(lines)


def _phase_of(spec_name: str) -> str:
    return "decode" if spec_name.endswith("decode") else "prefill"


def _sf_args(phase: str, scale_factor: float | None):
    return (scale_factor, None) if phase == "prefill" else (None, scale_factor)


# ---------------------------------------------------------------------------
# Kernel tests: skip-off sanity, aggressive reference-match, sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", ["uniform_prefill", "uniform_decode"])
def test_skip_off_matches_exact_attention(spec_name: str):
    """threshold=None -> TRTLLM output ~= exact fp32 attention (tight tol).

    Sanity for everything downstream: harness plumbing, paged cache
    simulation, HND layout, the TRTLLM dispatch, and our exact-reference
    base all agree before any dropping enters the picture.
    """
    case = _get_case(spec_name)
    phase = _phase_of(spec_name)
    output, calls = case.run_kernel(None, None)
    _assert_trtllm_fired(calls, phase, None, case.seq_k)
    torch.testing.assert_close(
        output.float(),
        case.exact_ref,
        **TOL_EXACT,
        msg=lambda m: f"[{spec_name}] skip-off TRTLLM output != exact "
        f"attention — backend/harness broken independent of skip-softmax.\n{m}",
    )


@pytest.mark.parametrize("spec_name", ["uniform_prefill", "uniform_decode"])
def test_aggressive_threshold_matches_dropped_reference(spec_name: str):
    """The core claim: with an aggressive threshold the kernel output equals
    the reference that drops exactly the tiles the documented rule selects —
    and that output is meaningfully different from exact attention."""
    case = _get_case(spec_name)
    phase = _phase_of(spec_name)

    # Derive a boundary-robust aggressive scale factor from the reference
    # margins of this exact (seeded) data.
    scale_factor, log_thr, gap = _choose_robust_scale_factor(
        case.margins, seq_k=case.seq_k, target_drop_frac=TARGET_DROP_FRAC[phase]
    )
    dropped_ref, n_drop, n_total = case.reference(scale_factor)

    # Self-check the setting is in the interesting regime: some but not all
    # tiles drop, and the effect on the output clearly exceeds tolerance.
    frac = n_drop / n_total
    assert 0.05 < frac < 0.95, (
        f"[{spec_name}] engineered drop fraction {frac:.2f} out of range; "
        f"test data/threshold selection needs adjustment (not a kernel bug)"
    )
    ref_effect = (dropped_ref - case.exact_ref).abs().max().item()
    assert ref_effect > MIN_EFFECT, (
        f"[{spec_name}] reference drop effect {ref_effect:.3f} <= "
        f"{MIN_EFFECT}; dropped-vs-exact would be indistinguishable at the "
        f"match tolerance (test-data issue, not a kernel bug)"
    )

    output, calls = case.run_kernel(*_sf_args(phase, scale_factor))
    _assert_trtllm_fired(calls, phase, scale_factor, case.seq_k)

    out_f = output.float()
    # (a) Real effect: the kernel did NOT silently run dense softmax.
    kernel_vs_exact = (out_f - case.exact_ref).abs().max().item()
    assert kernel_vs_exact > MIN_EFFECT / 2, (
        f"[{spec_name}] scale_factor={scale_factor:.4g} "
        f"(log_thr={log_thr:.3f}): kernel output is within "
        f"{kernel_vs_exact:.4f} of EXACT attention while the reference "
        f"predicts a {ref_effect:.3f} effect — skip-softmax was a no-op "
        f"(threshold ignored or dropped nothing)."
    )
    # (b) Right blocks: kernel matches the block-dropped reference.
    try:
        torch.testing.assert_close(out_f, dropped_ref, **TOL_MATCH)
    except AssertionError as e:
        raise AssertionError(
            f"[{spec_name}] kernel disagrees with the documented-rule "
            f"reference (scale_factor={scale_factor:.4g}, "
            f"log_thr={log_thr:.3f}, boundary gap={gap:.3g}, "
            f"drop={n_drop}/{n_total}).\n"
            f"If max_err below is small only for a different kv_tile/split, "
            f"the kernel's tile granularity differs from KV_TILE_SIZE="
            f"{KV_TILE_SIZE} — update the constant, don't loosen tolerance.\n"
            + _diagnose_tile_configs(case, output, scale_factor)
            + f"\noriginal failure:\n{e}"
        ) from e
    # (c) Discrimination: the kernel is strictly closer to the dropped
    # reference than to exact attention.
    kernel_vs_ref = (out_f - dropped_ref).abs().max().item()
    assert kernel_vs_ref < kernel_vs_exact, (
        f"[{spec_name}] kernel is closer to exact ({kernel_vs_exact:.4f}) "
        f"than to the dropped reference ({kernel_vs_ref:.4f})"
    )


@pytest.mark.parametrize("spec_name", ["uniform_prefill", "uniform_decode"])
def test_threshold_sweep_divergence_is_monotonic(spec_name: str):
    """Corroboration: divergence from exact attention grows (within noise
    slack) as the threshold gets more aggressive, and spans no-op -> clearly
    non-trivial. Uses only the kernel + the exact reference, so it is
    insensitive to the tile-size assumption."""
    case = _get_case(spec_name)
    phase = _phase_of(spec_name)

    log_thrs = [-8.0, -4.0, -2.5, -1.5, -0.8]
    diffs = []
    for log_thr in log_thrs:
        scale_factor = case.seq_k * math.exp(log_thr)
        output, calls = case.run_kernel(*_sf_args(phase, scale_factor))
        _assert_trtllm_fired(calls, phase, scale_factor, case.seq_k)
        diffs.append((output.float() - case.exact_ref).abs().max().item())

    # Non-decreasing within a small noise slack (block decisions are
    # discrete; equal consecutive values are fine).
    slack = 5e-3
    for i in range(1, len(diffs)):
        assert diffs[i] >= diffs[i - 1] - slack, (
            f"[{spec_name}] divergence not monotonic in threshold: "
            f"log_thrs={log_thrs} -> diffs={[f'{d:.4f}' for d in diffs]}"
        )
    assert diffs[-1] > diffs[0] + MIN_EFFECT / 2, (
        f"[{spec_name}] most aggressive threshold barely moved the output "
        f"({diffs[0]:.4f} -> {diffs[-1]:.4f}): skip-softmax looks inert "
        f"across the whole sweep."
    )


# ---------------------------------------------------------------------------
# Standalone runner — works without pytest installed.
#   .venv/bin/python tests/v1/attention/test_skip_softmax_reference_match.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failed_names = []
    passed = failed = 0

    # Definition order: CPU reference self-checks first, then GPU tests.
    cpu_tests = [
        test_reference_none_threshold_is_exact_softmax,
        test_reference_zero_scale_factor_drops_nothing,
        test_reference_drop_set_grows_with_scale_factor,
        test_reference_running_max_tile_never_dropped,
    ]
    gpu_tests = [
        (test_skip_off_matches_exact_attention, "uniform_prefill"),
        (test_skip_off_matches_exact_attention, "uniform_decode"),
        (test_aggressive_threshold_matches_dropped_reference, "uniform_prefill"),
        (test_aggressive_threshold_matches_dropped_reference, "uniform_decode"),
        (test_threshold_sweep_divergence_is_monotonic, "uniform_prefill"),
        (test_threshold_sweep_divergence_is_monotonic, "uniform_decode"),
    ]

    print("skip-softmax reference self-checks (CPU):")
    for fn in cpu_tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
            failed_names.append(fn.__name__)

    print("skip-softmax TRTLLM kernel vs reference (GPU, TRTLLM required):")
    for fn, spec in gpu_tests:
        name = f"{fn.__name__}[{spec}]"
        try:
            fn(spec)
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
            failed_names.append(name)

    print(f"\n{passed} passed, {failed} failed")
    if failed_names:
        print("failed:", *failed_names, sep="\n  ")
    sys.exit(0 if failed == 0 else 1)
