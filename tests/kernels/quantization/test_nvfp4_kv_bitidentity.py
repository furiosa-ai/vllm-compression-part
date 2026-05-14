"""Bit-identity tests for the KV-cache NVFP4 fake-quant kernel.

Verifies ``vllm.model_executor.layers.quantization.kv_fake_quant.kernels``:
  - ``_round_to_fp4_e2m1``  (FP4 grid round, against compressed_tensors)
  - ``_fake_quantize_dequantize_nvfp4``  (full quant+dequant pipeline,
                                          against vllm ``ref_nvfp4_quant``)

Locks in:
  1. FP4 round tie-breaking matches compressed_tensors (round half toward
     smaller magnitude at exact midpoints {0.25, 1.25, 2.5, 5.0}).
  2. Full quant+dequant is byte-equal to ``ref_nvfp4_quant`` across:
     fp32/bf16 inputs, clamp boundaries, subnormals, NaN/Inf, FP8 over/
     underflow, all-zero, and realistic K-cache shapes.
  3. Reference's FP4 outputs are confined to the canonical 8-value grid.

If any of these break, downstream NVFP4 KV-cache evaluation runs are not
bit-equivalent to the reference and accuracy results may diverge in
non-obvious ways.

Run:
    .venv/bin/python -m pytest tests/kernels/quantization/test_nvfp4_kv_bitidentity.py -v
    # or standalone (no pytest needed):
    .venv/bin/python tests/kernels/quantization/test_nvfp4_kv_bitidentity.py
"""

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


from vllm.model_executor.layers.quantization.kv_fake_quant.kernels import (
    _fake_quantize_dequantize_nvfp4 as quant_dequant,
    _round_to_fp4_e2m1 as round_fp4,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ct_cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Inlined copy of ``compressed_tensors`` ``FP4_E2M1_DATA.cast_to_fp4``.

    Bypasses the @torch.compile decorator on the original (which hangs with
    our test inputs) but is otherwise byte-identical.
    """
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _ref_full_pipeline(data: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """``ref_nvfp4_quant`` + manual dequant — analogue of our kernel's
    combined quant+dequant in one shot. Returns fp32 output cast to
    ``data.dtype``.
    """
    assert gs.dtype == torch.float32
    assert data.ndim == 4
    B, nh, T, D = data.shape
    G = 16
    flat = data.view(B * nh * T, D).to(torch.float32)
    fp4, scale_fp8 = ref_nvfp4_quant(flat, gs, G)
    fp4 = fp4.view(B * nh * T, D // G, G)
    eff = (scale_fp8 / gs).unsqueeze(-1)
    deq = (fp4 * eff).view(B, nh, T, D).to(data.dtype)
    return deq


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-equal even when both contain NaN.

    ``torch.equal`` returns False on NaN, so compare NaN masks separately.
    """
    a_nan = torch.isnan(a)
    b_nan = torch.isnan(b)
    if not torch.equal(a_nan, b_nan):
        return False
    mask = ~a_nan & ~b_nan
    return torch.equal(a[mask], b[mask])


# ---------------------------------------------------------------------------
# 1. FP4 round-to-grid: ours vs compressed_tensors (inlined)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x,expected", [
    # Exact grid points → no-op
    (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0),
    (3.0, 3.0), (4.0, 4.0), (6.0, 6.0),
    # Midpoints (round HALF toward smaller magnitude — compressed_tensors style)
    (0.25, 0.0), (1.25, 1.0), (2.5, 2.0), (5.0, 4.0),
    (-0.25, -0.0), (-1.25, -1.0), (-2.5, -2.0), (-5.0, -4.0),
    # Near-misses (asymmetric intervals)
    (0.75, 1.0), (1.75, 2.0), (3.5, 4.0),
])
def test_fp4_round_at_boundary(x, expected):
    out = round_fp4(torch.tensor([x], dtype=torch.float32))
    assert torch.equal(out, torch.tensor([expected], dtype=torch.float32)), (
        f"input {x} -> ours {out.item()}, expected {expected}"
    )


def test_fp4_round_matches_compressed_tensors_random():
    """ours vs inlined compressed_tensors over 100k random uniform[-6, 6]."""
    torch.manual_seed(0)
    big = torch.empty(100_000).uniform_(-6, 6)
    ours = round_fp4(big.clone())
    ref = _ct_cast_to_fp4(big.clone())
    n_diff = (ours != ref).sum().item()
    assert n_diff == 0, f"{n_diff} / 100000 mismatches"


# ---------------------------------------------------------------------------
# 2. Full quant+dequant pipeline: ours vs vllm ref_nvfp4_quant
# ---------------------------------------------------------------------------

GS_ONES = torch.tensor([1.0], dtype=torch.float32)


def _check_pipeline(name, data, gs):
    ours = quant_dequant(data, gs)
    ref = _ref_full_pipeline(data, gs)
    assert _bit_equal(ours, ref), (
        f"[{name}] full-pipeline mismatch  shape={tuple(data.shape)} "
        f"dtype={data.dtype}  max|diff|="
        f"{(ours.float() - ref.float()).abs().max().item():.4g}"
    )


def test_pipeline_midpoints_fp32():
    data = torch.tensor([[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.25, -1.25, -2.5, -5.0, 0.25, 1.25, 2.5, 5.0]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("midpoints fp32", data, GS_ONES)


def test_pipeline_clamp_path():
    """Inputs > FP4_MAX must be clamped before round."""
    data = torch.tensor([[7.0, 10.0, 100.0, 1e6, -7.0, -10.0, -100.0, -1e6,
                          6.0, 6.0001, 5.9999, 4.0, 3.5, 5.0, 5.001, 4.999]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("clamp path", data, GS_ONES)


def test_pipeline_round_to_zero_boundary():
    """Tiny values at the |x| <= 0.125 round-to-zero boundary."""
    data = torch.tensor([[0.0, 1e-7, 1e-10, 1e-30, 0.124, 0.125, 0.126,
                          0.249, 0.250, 0.251, 0.7499, 0.75, 0.7501,
                          1e-4, 0.1, 0.01]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("round-to-zero boundary", data, GS_ONES)


def test_pipeline_subnormals_and_neg_zero():
    data = torch.tensor([[1e-40, -1e-40, 1e-42, 5e-39, 1e-38, 1e-37,
                          2 ** -126, -2 ** -126, 0.0, -0.0, 1e-44, 5e-45,
                          1.4e-45, 1e-43, 1e-41, 1e-39]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("subnormals + -0", data, GS_ONES)


def test_pipeline_nan_inf():
    """NaN / +Inf / -Inf must propagate identically to ref (no nan_to_num)."""
    data = torch.tensor([[float('nan'), float('inf'), float('-inf'), 0.0,
                          1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0,
                          -3.0, -4.0, -5.0, -6.0]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("nan + inf", data, GS_ONES)


def test_pipeline_fp8_overflow_path():
    """Large global_scale -> local_scale > FP8_MAX=448 -> must clamp."""
    torch.manual_seed(7)
    data = torch.randn(1, 1, 1, 16, dtype=torch.float32) * 5
    gs = torch.tensor([1e6], dtype=torch.float32)
    _check_pipeline("fp8 overflow", data, gs)


def test_pipeline_fp8_underflow_path():
    """Tiny global_scale -> local_scale rounds to FP8 zero."""
    torch.manual_seed(8)
    data = torch.randn(1, 1, 1, 16, dtype=torch.float32) * 5
    gs = torch.tensor([1e-6], dtype=torch.float32)
    _check_pipeline("fp8 underflow", data, gs)


def test_pipeline_all_zeros():
    data = torch.zeros(1, 1, 1, 16, dtype=torch.float32)
    _check_pipeline("all zeros", data, GS_ONES)


def test_pipeline_bf16_extremes():
    data = torch.tensor([[7.0, -7.0, 0.0, 0.125, 5.999, 6.001, 4.0, 1e-3,
                          1e-30, 100.0, -100.0, 0.249, 1.249, 2.499, 4.999, -5.001]],
                        dtype=torch.bfloat16).view(1, 1, 1, 16)
    _check_pipeline("bf16 extremes", data, GS_ONES)


def test_pipeline_realistic_kcache_bf16():
    """Realistic K-cache shape (1, 8, 128, 128) bf16 with model-amax-derived gs."""
    torch.manual_seed(42)
    data = torch.randn(1, 8, 128, 128, dtype=torch.bfloat16) * 3
    gs = torch.tensor([2688.0 / data.abs().amax().float()], dtype=torch.float32)
    _check_pipeline("realistic K-cache bf16", data, gs)


def test_pipeline_realistic_fp32():
    torch.manual_seed(43)
    data = torch.randn(1, 8, 64, 128, dtype=torch.float32) * 0.5
    gs = torch.tensor([2688.0 / data.abs().amax()], dtype=torch.float32)
    _check_pipeline("realistic fp32", data, gs)


# ---------------------------------------------------------------------------
# 3. Reference sanity: ref's FP4 outputs land in the canonical 8-value grid
# ---------------------------------------------------------------------------

def test_ref_fp4_outputs_in_canonical_grid():
    torch.manual_seed(11)
    data = torch.randn(8, 128, dtype=torch.float32) * 4
    gs = torch.tensor([2688.0 / data.abs().amax()], dtype=torch.float32)
    fp4, scale = ref_nvfp4_quant(data, gs, 16)

    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    seen = torch.unique(fp4.abs())
    for v in seen:
        ok = bool((torch.abs(grid - v.item()) < 1e-6).any())
        assert ok, f"FP4 output {v.item()} not in canonical grid"

    assert scale.dtype == torch.float32
    assert scale.min().item() >= 0.0
    assert scale.max().item() <= 448.0  # FP8 E4M3 max


# ---------------------------------------------------------------------------
# Standalone runner — works without pytest installed.
#   .venv/bin/python tests/kernels/quantization/test_nvfp4_kv_bitidentity.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    import sys

    BOUNDARY_CASES = [
        (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0),
        (3.0, 3.0), (4.0, 4.0), (6.0, 6.0),
        (0.25, 0.0), (1.25, 1.0), (2.5, 2.0), (5.0, 4.0),
        (-0.25, -0.0), (-1.25, -1.0), (-2.5, -2.0), (-5.0, -4.0),
        (0.75, 1.0), (1.75, 2.0), (3.5, 4.0),
    ]

    passed = failed = 0
    print("FP4 boundary points:")
    for x, expected in BOUNDARY_CASES:
        try:
            test_fp4_round_at_boundary(x, expected)
            print(f"  PASS  fp4({x:>6}) = {expected}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1

    fns = [
        (n, f)
        for n, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if n.startswith("test_") and n != "test_fp4_round_at_boundary"
    ]
    print("\nFull-pipeline + reference-sanity:")
    for n, f in fns:
        try:
            f()
            print(f"  PASS  {n}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {n}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
