# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speedup test for BLASST skip-softmax: prove the TRTLLM kernels actually
AVOID work (get faster) as the skip threshold gets more aggressive.

Why a latency test (and why an output test can never replace it)
-----------------------------------------------------------------
True-skip and "compute-then-mask-to-zero" produce byte-identical outputs (and
identical LSE), so "computation was skipped" is observable ONLY through a work
side-channel.  Kernel latency is that side-channel here: with the SAME cubin
and only the threshold scalar changing, a kernel that truly skips gets faster
as more tiles drop; a compute-then-mask kernel stays flat.

Everything asserted here is EMPIRICAL — no assumption about the kernel's
internal drop rule.  (An earlier reference-match test asserted the kernel
drops exactly the tiles of a documented cummax rule; it was removed because
the rule was our own description, not kernel ground truth, and the flashinfer
0.6.8.post1 kernels demonstrably follow a different, more conservative rule —
probed: ln/log2 threshold domains, cummax restarts, GQA-grouped decisions,
tile sizes 32-256 all fail to reproduce the kernel's drop set.  End-to-end
accuracy-vs-sparsity evals are the correctness authority instead.)

What is asserted (per phase: prefill and decode)
------------------------------------------------
1. SANITY — threshold off => TRTLLM kernel output ~= exact fp32 attention,
   and the spy proves the TRTLLM kernel actually fired (no silent fallback).
   Validates the whole harness before any skipping enters the picture.
2. SPEEDUP — median kernel latency TRENDS DOWN across the skip rungs
   (Spearman rank correlation <= -0.6; flat/rising = compute-then-mask =
   FAIL) and falls >= 5% at the most aggressive rung vs the skip-cubin
   no-drop baseline.
3. LIVENESS — at the most aggressive rung the output measurably diverges
   from exact attention (the threshold demonstrably alters what the kernel
   computes; the speedup is not a scheduling artifact).

Design notes
------------
* Long-context sweep specs (skipping needs room to matter):
  prefill = one request, 2048-token query chunk attending a 32768-token
  sequence; decode = 32 requests at seq_len 32768.  Uniform seq_lens so the
  batch-max seq_k the kernel receives equals every request's own length.
* Measurement isolates the kernel: the TRTLLM entry point is spied on to
  capture the exact kwargs of a real backend invocation, then the kernel is
  REPLAYED directly and timed with ``torch.cuda.Event`` — median of >= 100
  iters, measured ROUND-ROBIN across all threshold rungs so multi-percent
  clock/thermal drift cannot fake or hide a trend.
* Two baselines (important): enabling the threshold selects a different
  ("SkipsSoftmax") cubin with its own fixed overhead vs the dense kernel
  (measured +3.5%/+8.7% on B200).  Reductions are therefore computed against
  a skip-cubin rung whose threshold drops ~nothing (log_thr=-3), which
  isolates SKIPPED WORK from cubin overhead; the dense (threshold=None) rung
  is measured and reported as the net user-visible effect.
* Threshold ladder is a fixed log-spaced sweep of log(threshold) (threshold =
  scale_factor / seq_k) spanning the regime where the kernel demonstrably
  engages (visible effects start around threshold ~ 0.4 on this build).

Hardware gate (raises, never skips)
-----------------------------------
BLASST skip-softmax is consumed only by FlashInfer's TRTLLM kernels, available
on the SM100 (Blackwell) family with the NVIDIA artifactory reachable.  On any
other GPU vLLM silently falls back to native FlashInfer and skip-softmax is a
no-op.  Running this test there would measure noise, so incompatible hardware
raises ``RuntimeError`` — a hard error, not a skip, by design: the silent
fallback is precisely the failure mode this file exists to expose.

Run:
    .venv/bin/python -m pytest tests/v1/attention/test_skip_softmax_speedup.py -v -s
    # or standalone (no pytest needed):
    .venv/bin/python tests/v1/attention/test_skip_softmax_speedup.py
"""

import functools
import math
import statistics
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

# Repo root on sys.path so `import tests...` works when run standalone.
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
# Configuration
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen3-0.6B"  # same tiny model the harness tests use
PAGE_SIZE = 16  # vLLM paged-KV page size

# Long-context sweep specs + paged-KV pool sized to hold them
# (decode: 32 * 32768 / 16 = 65536 pages, plus slack).
SWEEP_SPECS = {
    "sweep_prefill": BatchSpec(
        seq_lens=[32768], query_lens=[2048], name="sweep_prefill"
    ),
    "sweep_decode": BatchSpec(
        seq_lens=[32768] * 32, query_lens=[1] * 32, name="sweep_decode"
    ),
}
NUM_GPU_BLOCKS = {"sweep_prefill": 8192, "sweep_decode": 66560}

# Fixed log-spaced log(threshold) rungs; the first (-3.0) is the skip-cubin
# no-drop WORK BASELINE, the rest get progressively more aggressive up to
# threshold ~ 0.98 (see module docstring).
LOG_THRESHOLDS = (-3.0, -1.5, -0.8, -0.4, -0.2, -0.1, -0.05, -0.02)

# Latency measurement.
LAT_WARMUP_ITERS = 20
LAT_ITERS = 100
# Required strength of the decreasing latency-vs-threshold trend across the
# skip rungs (Spearman rho of rung index vs latency; -1 = perfectly
# decreasing).  Rank-based so single-rung measurement wiggles don't fail the
# test, while a flat (rho ~ 0) or rising curve — the compute-then-mask
# signature — does.
TREND_MIN_RHO = 0.6
# At the most aggressive rung latency must have fallen at least this much vs
# the no-drop skip baseline.  Any true-skip implementation clears this
# easily; compute-then-mask cannot.
MIN_REDUCTION = 0.05

# Skip-off sanity tolerance (bf16 I/O, fp32 accumulation both sides) and the
# minimum output divergence at the most aggressive rung that counts as "the
# threshold demonstrably did something".
TOL_EXACT = dict(atol=2e-2, rtol=2e-2)
MIN_OUTPUT_EFFECT = 2.5e-2


# ---------------------------------------------------------------------------
# Hardware gate — RAISES on BLASST-incompatible hardware (never skips)
# ---------------------------------------------------------------------------


def _require_blasst_capable_gpu() -> None:
    """Hard-error unless the FlashInfer TRTLLM skip-softmax path can run here.

    Deliberately a RuntimeError and not a pytest skip: on unsupported hardware
    (anything outside the SM100/Blackwell family, or with the NVIDIA
    artifactory unreachable) vLLM silently falls back to native FlashInfer and
    the skip threshold is never consumed — every number this file would
    produce is noise.  A skip would hide exactly the no-op this test exists
    to expose.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "BLASST skip-softmax speedup test needs a CUDA GPU (SM100/"
            "Blackwell family); no CUDA device is available. This test "
            "raises instead of skipping by design."
        )
    from vllm.utils.flashinfer import supports_trtllm_attention

    if not supports_trtllm_attention():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        raise RuntimeError(
            f"BLASST-incompatible hardware: {name} (compute capability "
            f"{cap[0]}.{cap[1]}). FlashInfer TRTLLM attention — the only "
            f"consumer of skip_softmax_threshold_scale_factor — requires the "
            f"SM100 (Blackwell) family with the NVIDIA artifactory reachable "
            f"for cubins. On this GPU skip-softmax is a silent no-op, so this "
            f"test FAILS rather than skips (a skip would mask the no-op)."
        )


# ---------------------------------------------------------------------------
# Exact fp32 attention reference (rule-free — used only for the sanity and
# liveness checks)
# ---------------------------------------------------------------------------


def _exact_attention(
    q: torch.Tensor,  # (q_len, num_q_heads, head_dim)
    k: torch.Tensor,  # (s_len, num_kv_heads, head_dim)
    v: torch.Tensor,  # (s_len, num_kv_heads, head_dim)
    *,
    sm_scale: float,
    context_len: int,
) -> torch.Tensor:
    """Plain causal fp32 attention with GQA repeat (harness convention)."""
    q_len, num_q_heads, _ = q.shape
    s_len, num_kv_heads, _ = k.shape
    rep = num_q_heads // num_kv_heads
    qf = q.transpose(0, 1).float()
    kf = k.transpose(0, 1).float().repeat_interleave(rep, dim=0)
    vf = v.transpose(0, 1).float().repeat_interleave(rep, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale
    q_pos = torch.arange(q_len, device=q.device) + context_len
    kv_pos = torch.arange(s_len, device=q.device)
    scores = scores.masked_fill(
        q_pos[:, None] < kv_pos[None, :], float("-inf")
    )
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,hkd->hqd", probs, vf).transpose(0, 1)


# ---------------------------------------------------------------------------
# Case: real paged-KV inputs + the real FlashInfer backend with a spy
# (adapted from the earlier reference-match harness)
# ---------------------------------------------------------------------------


class _Case:
    """Everything needed to run the kernel and the exact reference for one
    batch spec."""

    def __init__(self, batch_spec: BatchSpec, num_gpu_blocks: int):
        self.batch_spec = batch_spec
        set_random_seed(42)
        device = torch.device("cuda:0")
        self.device = device

        vllm_config = create_vllm_config(
            model_name=MODEL,
            max_model_len=max(batch_spec.seq_lens),
            block_size=PAGE_SIZE,
            num_gpu_blocks=num_gpu_blocks,
        )
        # Force the TRTLLM path. NOTE: this is NOT a hardware gate — on
        # platforms where supports_trtllm_attention() is false vLLM still
        # falls back to native FlashInfer, and the spy assertion then fails.
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
        all_q, all_k_new, all_v_new, k_ctx, v_ctx, exact = [], [], [], [], [], []
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
            all_q.append(q)
            all_k_new.append(kf[ctx:])
            all_v_new.append(vf[ctx:])
            k_ctx.append(kf[:ctx])
            v_ctx.append(vf[:ctx])
            exact.append(
                _exact_attention(
                    q, kf, vf, sm_scale=self.sm_scale, context_len=ctx
                )
            )
        self.query = torch.cat(all_q, dim=0)
        self.key = torch.cat(all_k_new, dim=0)
        self.value = torch.cat(all_v_new, dim=0)
        self.exact_ref = torch.cat(exact, dim=0)  # fp32, (tokens, Hq, D)

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
            num_blocks=num_gpu_blocks,
            common_attn_metadata=self.common_attn_metadata,
            randomize_blocks=True,
        )
        # FlashInfer layout: (num_blocks, 2, ...) + HND (harness convention;
        # the TRTLLM path asserts get_kv_cache_layout() == "HND").
        kv_cache = kv_cache.transpose(0, 1)
        self.kv_cache = kv_cache.transpose(2, 3).contiguous().transpose(2, 3)

        # seq_k the kernel call sites use (batch max; uniform specs make it
        # equal to every request's own length).
        self.seq_k = max(batch_spec.seq_lens)

    def run_kernel(
        self, sf_prefill: float | None, sf_decode: float | None
    ) -> tuple[torch.Tensor, dict[str, list[dict]]]:
        """Run the real FlashInfer backend, spying on the TRTLLM entry points.

        The spies (a) prove the TRTLLM kernel actually fired (no silent
        fallback), (b) capture the skip_softmax_threshold_scale_factor and
        the max_kv_len / max_seq_len the fork passed as seq_k.
        """
        ac = self.vllm_config.attention_config
        ac.skip_softmax_threshold_scale_factor_prefill = sf_prefill
        ac.skip_softmax_threshold_scale_factor_decode = sf_decode

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
    return _Case(SWEEP_SPECS[spec_name], NUM_GPU_BLOCKS[spec_name])


def _phase_of(spec_name: str) -> str:
    return "decode" if spec_name.endswith("decode") else "prefill"


def _sf_args(phase: str, scale_factor: float | None):
    return (scale_factor, None) if phase == "prefill" else (None, scale_factor)


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
        f"support vLLM silently falls back to native FlashInfer — that "
        f"fallback is exactly what this assertion is meant to expose."
    )
    kw = calls[phase][0]
    got_sf = kw.get("skip_softmax_threshold_scale_factor")
    assert got_sf == expected_scale_factor, (
        f"kernel received skip_softmax_threshold_scale_factor={got_sf}, "
        f"expected {expected_scale_factor}"
    )
    seq_k_key = "max_kv_len" if phase == "prefill" else "max_seq_len"
    got_seq_k = kw.get(seq_k_key)
    assert got_seq_k == expected_seq_k, (
        f"kernel {seq_k_key}={got_seq_k}, expected {expected_seq_k}; the "
        f"threshold denominator is not what this test assumes"
    )


# ---------------------------------------------------------------------------
# Kernel capture + interleaved replay timing
# ---------------------------------------------------------------------------


class _KernelCapture:
    """Patch the TRTLLM entry points to record the exact (args, kwargs) of
    each call while calling through to the real kernel.

    Composes with _Case.run_kernel's own spy: entering this context BEFORE
    run_kernel makes run_kernel snapshot our wrapper as its "real" function,
    so its kwargs-assertion spy still works and we additionally get the full
    argument set for replay.  ``real[phase]`` is the true kernel binding.
    """

    _ATTRS = {
        "prefill": "trtllm_batch_context_with_kv_cache",
        "decode": "trtllm_batch_decode_with_kv_cache",
    }

    def __init__(self):
        import vllm.v1.attention.backends.flashinfer as fi_mod

        self._fi_mod = fi_mod
        self.captured: dict[str, list[tuple[tuple, dict]]] = {
            "prefill": [],
            "decode": [],
        }
        self.real = {p: getattr(fi_mod, a) for p, a in self._ATTRS.items()}
        self._patchers = []

    def __enter__(self):
        for phase, attr in self._ATTRS.items():
            real_fn = self.real[phase]

            def wrapper(*args, _phase=phase, _real=real_fn, **kwargs):
                self.captured[_phase].append((args, kwargs))
                return _real(*args, **kwargs)

            p = unittest.mock.patch.object(self._fi_mod, attr, wrapper)
            p.start()
            self._patchers.append(p)
        return self

    def __exit__(self, *exc):
        for p in self._patchers:
            p.stop()
        return False


def _capture_kernel_call(case, phase: str, scale_factor: float | None):
    """Run the backend once with the given threshold; return the replayable
    (real_fn, args, kwargs) of the single TRTLLM call it made plus the
    backend output for divergence reporting."""
    with _KernelCapture() as cap:
        output, calls = case.run_kernel(*_sf_args(phase, scale_factor))
    _assert_trtllm_fired(calls, phase, scale_factor, case.seq_k)
    assert len(cap.captured[phase]) == 1
    args, kwargs = cap.captured[phase][0]
    return (cap.real[phase], args, kwargs), output


def _time_replays_interleaved(captures) -> list[float]:
    """Median kernel latency (ms) per capture, measured ROUND-ROBIN.

    Sequential per-rung timing is vulnerable to multi-percent clock/thermal
    drift between rungs (observed on B200) — interleaving spreads any drift
    evenly across all rungs so their relative ordering is trustworthy.
    """
    for fn, args, kwargs in captures:
        for _ in range(LAT_WARMUP_ITERS):
            fn(*args, **kwargs)
    torch.cuda.synchronize()
    rounds = max(1, LAT_ITERS // 10)
    samples: list[list[float]] = [[] for _ in captures]
    for _ in range(rounds):
        for idx, (fn, args, kwargs) in enumerate(captures):
            for _ in range(10):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(*args, **kwargs)
                end.record()
                end.synchronize()
                samples[idx].append(start.elapsed_time(end))
    return [statistics.median(s) for s in samples]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (ties get average ranks); no scipy."""

    def _ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return num / den if den else 0.0


def _fmt_sf(sf: float | None) -> str:
    return "off" if sf is None else f"{sf:.4g}"


# ---------------------------------------------------------------------------
# Test 1 — sanity: skip off => kernel == exact attention (TRTLLM verified)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", ["sweep_prefill", "sweep_decode"])
def test_skip_off_matches_exact_attention(spec_name: str):
    """threshold=None -> TRTLLM output ~= exact fp32 attention.

    Sanity for everything downstream: harness plumbing, paged cache
    simulation, HND layout, the TRTLLM dispatch, and the exact reference all
    agree before any skipping enters the picture.
    """
    _require_blasst_capable_gpu()
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


# ---------------------------------------------------------------------------
# Test 2 — THE speedup test: kernel gets faster as the threshold rises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", ["sweep_prefill", "sweep_decode"])
def test_kernel_latency_decreases_with_sparsity(spec_name: str):
    """Median TRTLLM kernel latency must trend down as the threshold gets
    more aggressive, fall substantially at the most aggressive rung, and the
    output there must measurably differ from exact attention (liveness)."""
    _require_blasst_capable_gpu()
    case = _get_case(spec_name)
    phase = _phase_of(spec_name)

    # rungs: dense baseline (None) first, then the skip-cubin ladder.
    scale_factors: list[float | None] = [None] + [
        case.seq_k * math.exp(lt) for lt in LOG_THRESHOLDS
    ]
    captures, out_divs = [], []
    for sf in scale_factors:
        cap, output = _capture_kernel_call(case, phase, sf)
        captures.append(cap)
        out_divs.append(
            (output.float() - case.exact_ref).abs().max().item()
        )
    medians = _time_replays_interleaved(captures)

    # Report. rows[0] = dense cubin (report-only); rows[1] = skip cubin at a
    # threshold that drops ~nothing (the work baseline).
    base = medians[1]
    print(
        f"\n[{spec_name}] kernel latency vs threshold "
        f"(reduction vs the skip-on/no-drop baseline rung):"
    )
    print(
        f"  {'scale_factor':>14} {'thr=sf/seq_k':>13} {'latency_ms':>12} "
        f"{'reduction':>10} {'max|out-exact|':>15}"
    )
    for sf, med, div in zip(scale_factors, medians, out_divs):
        thr = "-" if sf is None else f"{sf / case.seq_k:.4f}"
        print(
            f"  {_fmt_sf(sf):>14} {thr:>13} {med:>12.4f} "
            f"{(base - med) / base:>10.3f} {div:>15.4f}"
        )
    print(
        f"  (skip-cubin overhead vs dense kernel: "
        f"{(base - medians[0]) / medians[0]:+.1%})"
    )

    # (a) SPEEDUP TREND across the skip rungs.
    skip_lats = medians[1:]
    rho = _spearman_rho([float(i) for i in range(len(skip_lats))], skip_lats)
    assert rho <= -TREND_MIN_RHO, (
        f"[{spec_name}] kernel latency does not decrease with threshold "
        f"aggressiveness across the skip rungs (Spearman rho={rho:+.2f}, "
        f"need <= -{TREND_MIN_RHO}): values="
        f"{[f'{v:.4f}' for v in skip_lats]} — a flat or rising curve means "
        f"the kernel computes everything and merely masks "
        f"(compute-then-mask), i.e. skip-softmax does not actually skip."
    )
    # (b) SUBSTANTIAL reduction at the most aggressive rung.
    red_last = (base - skip_lats[-1]) / base
    assert red_last >= MIN_REDUCTION, (
        f"[{spec_name}] at the most aggressive threshold "
        f"(~{scale_factors[-1] / case.seq_k:.2f}) kernel latency only fell "
        f"{red_last:.1%} (< floor {MIN_REDUCTION:.0%}): the kernel is not "
        f"skipping a meaningful amount of work."
    )
    # (c) LIVENESS: the aggressive threshold demonstrably changed the output.
    assert out_divs[-1] >= MIN_OUTPUT_EFFECT, (
        f"[{spec_name}] the most aggressive threshold left the output "
        f"within {out_divs[-1]:.4f} of exact attention "
        f"(< {MIN_OUTPUT_EFFECT}) — the observed latency change is not "
        f"attributable to skipping (threshold looks inert)."
    )


# ---------------------------------------------------------------------------
# Standalone runner — works without pytest installed.
#   .venv/bin/python tests/v1/attention/test_skip_softmax_speedup.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failed_names = []
    passed = failed = 0
    tests = [
        (test_skip_off_matches_exact_attention, "sweep_prefill"),
        (test_skip_off_matches_exact_attention, "sweep_decode"),
        (test_kernel_latency_decreases_with_sparsity, "sweep_prefill"),
        (test_kernel_latency_decreases_with_sparsity, "sweep_decode"),
    ]
    print("skip-softmax speedup (GPU, SM100/TRTLLM required):")
    for fn, spec in tests:
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
