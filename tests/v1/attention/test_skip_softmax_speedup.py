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
as more tiles drop; a compute-then-mask kernel stays flat.  Correctness
("were the RIGHT tiles dropped?") is the separate, existing
``test_skip_softmax_reference_match.py``.

Design
------
* Long-context sweep specs (skipping needs room to matter):
  prefill = one request, 2048-token query chunk attending a 32768-token
  sequence; decode = 32 requests at seq_len 32768.
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
  engages.  Empirical note (B200, flashinfer 0.6.8.post1): the kernel's drop
  rule is much more conservative than the documented ModelOpt rule (see the
  xfail in the reference-match test) — visible effects start at log_thr >~ -1.
  The documented-rule drop fraction is printed per rung as context only.

Assertions (per phase)
----------------------
1. Median kernel latency TRENDS DOWN across the skip rungs (Spearman rank
   correlation <= -0.6).  Flat or rising = compute-then-mask = FAIL.
2. At the most aggressive rung (threshold ~ 0.98) latency fell >= 5% vs the
   skip-cubin no-drop baseline.  A real skip clears this easily.

Hardware gate (raises, never skips)
-----------------------------------
BLASST skip-softmax is consumed only by FlashInfer's TRTLLM kernels, available
on the SM100 (Blackwell) family with the NVIDIA artifactory reachable.  On any
other GPU vLLM silently falls back to native FlashInfer and skip-softmax is a
no-op.  Running this test there would measure noise, so incompatible hardware
raises ``RuntimeError`` — a hard error, not a skip, by design (same philosophy
as test_skip_softmax_reference_match.py: the silent fallback is the failure
mode this file exists to expose).

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

import tests.v1.attention.test_skip_softmax_reference_match as rm  # noqa: E402
from tests.v1.attention.test_skip_softmax_reference_match import (  # noqa: E402
    _assert_trtllm_fired,
    _phase_of,
    _sf_args,
)
from tests.v1.attention.utils import BatchSpec  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# Long-context specs. Uniform seq_lens for the same reason as the
# reference-match test (batch-max seq_k == every request's own length).
SWEEP_SPECS = {
    "sweep_prefill": BatchSpec(
        seq_lens=[32768], query_lens=[2048], name="sweep_prefill"
    ),
    "sweep_decode": BatchSpec(
        seq_lens=[32768] * 32, query_lens=[1] * 32, name="sweep_decode"
    ),
}
# Paged-KV pool sizing: decode needs 32 * 32768 / 16 = 65536 pages (+slack);
# the reference-match default of 8192 covers the single-request prefill spec.
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
# Case + ladder construction (reuses the reference-match _Case wholesale)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _get_case(spec_name: str) -> "rm._Case":
    """Build the reference-match _Case for a sweep spec.

    _Case hardcodes num_gpu_blocks=8192, which cannot hold the decode sweep's
    32 x 32768 tokens — patch create_vllm_config in the reference-match module
    to raise the pool size instead of duplicating the whole class.
    """
    orig = rm.create_vllm_config

    def patched(**kwargs):
        kwargs["num_gpu_blocks"] = NUM_GPU_BLOCKS[spec_name]
        return orig(**kwargs)

    with unittest.mock.patch.object(rm, "create_vllm_config", patched):
        return rm._Case(SWEEP_SPECS[spec_name])


@functools.lru_cache(maxsize=None)
def _get_ladder(spec_name: str) -> tuple[tuple[float | None, float], ...]:
    """(scale_factor, documented_rule_drop_frac) rungs: dense baseline first.

    The drop fraction comes from the documented-rule reference's drop mask on
    this exact seeded data (finite tiles only) — printed as CONTEXT only; the
    kernel's real rule is more conservative (see module docstring).
    """
    case = _get_case(spec_name)
    ladder: list[tuple[float | None, float]] = [(None, 0.0)]
    for log_thr in LOG_THRESHOLDS:
        sf = case.seq_k * math.exp(log_thr)
        _, n_drop, n_total = case.reference(sf)
        ladder.append((sf, n_drop / n_total))
    return tuple(ladder)


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
    (real_fn, args, kwargs) of the single TRTLLM call it made."""
    with _KernelCapture() as cap:
        _output, calls = case.run_kernel(*_sf_args(phase, scale_factor))
    _assert_trtllm_fired(calls, phase, scale_factor, case.seq_k)
    assert len(cap.captured[phase]) == 1
    args, kwargs = cap.captured[phase][0]
    return cap.real[phase], args, kwargs


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


def _skip_rows(rows):
    """rows[0] is the dense (threshold=None) rung — report-only.  rows[1:]
    are the skip-cubin rungs; rows[1] (log_thr=-3, drops ~nothing) is the
    work baseline all reductions are computed against."""
    return rows[1:]


def _fmt_sf(sf: float | None) -> str:
    return "off" if sf is None else f"{sf:.4g}"


def _print_table(spec_name: str, rows) -> None:
    print(
        f"\n[{spec_name}] kernel latency vs documented-rule drop fraction "
        f"(reduction vs the skip-on/no-drop baseline rung):"
    )
    print(
        f"  {'scale_factor':>14} {'pred_drop':>10} {'latency_ms':>12} "
        f"{'reduction':>10}"
    )
    base = _skip_rows(rows)[0][2]
    for sf, pred, val in rows:
        red = (base - val) / base
        print(f"  {_fmt_sf(sf):>14} {pred:>10.3f} {val:>12.4f} {red:>10.3f}")
    dense = rows[0][2]
    print(
        f"  (skip-cubin overhead vs dense kernel: {(base - dense) / dense:+.1%})"
    )


def _reductions(rows) -> list[float]:
    """Fractional latency reduction of each skip rung vs the skip-on/no-drop
    baseline rung (index 0 of the returned list is that baseline, = 0.0)."""
    skip = _skip_rows(rows)
    base = skip[0][2]
    return [(base - val) / base for _sf, _pred, val in skip]


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


# ---------------------------------------------------------------------------
# THE test — kernel gets faster as the threshold drops more tiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", ["sweep_prefill", "sweep_decode"])
def test_kernel_latency_decreases_with_sparsity(spec_name: str):
    """Median TRTLLM kernel latency must trend down as the threshold gets
    more aggressive, and fall substantially at the most aggressive rung —
    the work side-channel proof that skip-softmax actually skips."""
    _require_blasst_capable_gpu()
    case = _get_case(spec_name)
    phase = _phase_of(spec_name)
    ladder = _get_ladder(spec_name)

    captures = [_capture_kernel_call(case, phase, sf) for sf, _ in ladder]
    medians = _time_replays_interleaved(captures)
    rows = [(sf, pred, med) for (sf, pred), med in zip(ladder, medians)]
    _print_table(spec_name, rows)

    vals = [val for _sf, _pred, val in _skip_rows(rows)]
    rho = _spearman_rho([float(i) for i in range(len(vals))], vals)
    assert rho <= -TREND_MIN_RHO, (
        f"[{spec_name}] kernel latency does not decrease with threshold "
        f"aggressiveness across the skip rungs (Spearman rho={rho:+.2f}, "
        f"need <= -{TREND_MIN_RHO}): values={[f'{v:.4f}' for v in vals]} — "
        f"a flat or rising curve means the kernel computes everything and "
        f"merely masks (compute-then-mask), i.e. skip-softmax does not "
        f"actually skip."
    )

    red_last = _reductions(rows)[-1]
    pred_last = ladder[-1][1]
    assert red_last >= MIN_REDUCTION, (
        f"[{spec_name}] at the most aggressive threshold (documented rule "
        f"would drop {pred_last:.1%} of kv-tiles) kernel latency only fell "
        f"{red_last:.1%} (< floor {MIN_REDUCTION:.0%}): the kernel is not "
        f"skipping a meaningful amount of work."
    )


# ---------------------------------------------------------------------------
# Standalone runner — works without pytest installed.
#   .venv/bin/python tests/v1/attention/test_skip_softmax_speedup.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failed_names = []
    passed = failed = 0
    print("skip-softmax speedup-vs-sparsity (GPU, SM100/TRTLLM required):")
    for spec in ("sweep_prefill", "sweep_decode"):
        name = f"test_kernel_latency_decreases_with_sparsity[{spec}]"
        try:
            test_kernel_latency_decreases_with_sparsity(spec)
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
