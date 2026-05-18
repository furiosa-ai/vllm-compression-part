"""Bit-identity tests for the KV-cache NVFP4+ fake-quant kernel.

Verifies ``vllm.model_executor.layers.quantization.kv_fake_quant.kernels
._fake_quantize_dequantize_nvfp4_plus`` against a small inline reference.

Reference design
----------------
compressed_tensors has NO per-group fp32-scale quant function (its
``ref_nvfp4_quant`` always rounds the group scale to FP8 E4M3 and applies
a global scale). For NVFP4+ the only piece worth an independent reference
is the FP4 grid round; we reuse ``_ct_cast_to_fp4`` (inlined copy of
``compressed_tensors.quantization.quant_args.FP4_E2M1_DATA.cast_to_fp4``,
same one used by the NVFP4 bit-identity test) so the test's rounding step
is independent of our kernel's ``_round_to_fp4_e2m1``.

Locks in:
  1. Per-group FP32 scale: ``scale = amax(|group|) / 6.0`` (no FP8 round).
  2. No per-tensor global scale.
  3. All-zero group → output exactly zero (no 0/0 NaN leak).
  4. FP4 grid is the canonical ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} (inherited
     via ``_ct_cast_to_fp4``).
  5. NaN / Inf propagate identically between kernel and reference.

Run:
    .venv/bin/python -m pytest tests/kernels/quantization/test_nvfp4_plus_kv_bitidentity.py -v
    # or standalone (no pytest needed):
    .venv/bin/python tests/kernels/quantization/test_nvfp4_plus_kv_bitidentity.py
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
    _fake_quantize_dequantize_nvfp4_plus as quant_dequant_plus,
    _NVFP4_PLUS_GROUP_SIZE as G,
)


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def _ct_cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Inlined ``compressed_tensors`` ``FP4_E2M1_DATA.cast_to_fp4``.

    Used as the FP4 grid round in the reference, so the test's rounding
    step is independent of our kernel's ``_round_to_fp4_e2m1``.
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


_FP4_MAX = 6.0


def _ref_nvfp4_plus(data: torch.Tensor) -> torch.Tensor:
    """Reference NVFP4+ quant-dequant.

    Per-group(16) FP32 scale, no global scale. Compute order matches the
    kernel exactly so any ULP-level disagreement would be a kernel bug.
    """
    assert data.ndim == 4
    B, nh, T, D = data.shape
    assert D % G == 0
    num_groups = D // G
    grouped = data.view(B, nh, T, num_groups, G).to(torch.float32)

    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = amax * (1.0 / _FP4_MAX)

    # Same all-zero-group guard as the kernel.
    scale_safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    scaled = (grouped / scale_safe).clamp(min=-_FP4_MAX, max=_FP4_MAX)

    # Independent FP4 round (compressed_tensors-equivalent).
    flat = scaled.reshape(-1).clone()
    fp4 = _ct_cast_to_fp4(flat).view_as(scaled)

    out = (fp4 * scale).view(B, nh, T, D)
    return out.to(data.dtype)


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


def _check_pipeline(name, data):
    ours = quant_dequant_plus(data)
    ref = _ref_nvfp4_plus(data)
    assert _bit_equal(ours, ref), (
        f"[{name}] full-pipeline mismatch  shape={tuple(data.shape)} "
        f"dtype={data.dtype}  max|diff|="
        f"{(ours.float() - ref.float()).abs().max().item():.4g}"
    )


# ---------------------------------------------------------------------------
# Coverage: shape boundaries, special inputs, realistic shapes
# ---------------------------------------------------------------------------

def test_pipeline_grid_points_fp32():
    """One group containing exact FP4 grid points + signed midpoints."""
    data = torch.tensor([[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.25, -1.25, -2.5, -5.0, 0.25, 1.25, 2.5, 5.0]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("grid points fp32", data)


def test_pipeline_clamp_path():
    """Group whose amax pushes scale up; all values still land in [-6, 6]
    after division because scale = amax/6."""
    data = torch.tensor([[7.0, 10.0, 100.0, 1e6, -7.0, -10.0, -100.0, -1e6,
                          6.0, 6.0001, 5.9999, 4.0, 3.5, 5.0, 5.001, 4.999]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("clamp path", data)


def test_pipeline_round_to_zero_boundary():
    """Tiny values where rounding pushes things to zero."""
    data = torch.tensor([[0.0, 1e-7, 1e-10, 1e-30, 0.124, 0.125, 0.126,
                          0.249, 0.250, 0.251, 0.7499, 0.75, 0.7501,
                          1e-4, 0.1, 0.01]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("round-to-zero boundary", data)


def test_pipeline_subnormals_and_neg_zero():
    data = torch.tensor([[1e-40, -1e-40, 1e-42, 5e-39, 1e-38, 1e-37,
                          2 ** -126, -2 ** -126, 0.0, -0.0, 1e-44, 5e-45,
                          1.4e-45, 1e-43, 1e-41, 1e-39]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("subnormals + -0", data)


def test_pipeline_nan_inf():
    """NaN / +Inf / -Inf must propagate identically to ref (no nan_to_num)."""
    data = torch.tensor([[float('nan'), float('inf'), float('-inf'), 0.0,
                          1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0,
                          -3.0, -4.0, -5.0, -6.0]],
                        dtype=torch.float32).view(1, 1, 1, 16)
    _check_pipeline("nan + inf", data)


def test_pipeline_all_zero_group():
    """All-zero group must yield exactly zero (no 0/0 NaN leak)."""
    data = torch.zeros(1, 1, 1, 16, dtype=torch.float32)
    _check_pipeline("all zeros", data)
    out = quant_dequant_plus(data)
    assert torch.equal(out, torch.zeros_like(out)), (
        f"all-zero group must dequant to exactly 0, got {out}"
    )


def test_pipeline_mixed_zero_and_nonzero_groups():
    """Two groups in one head_dim=32: one all-zero, one normal."""
    g_zero = torch.zeros(16, dtype=torch.float32)
    g_normal = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.25, -1.25, -2.5, -5.0, 0.25, 1.25, 2.5, 5.0],
                            dtype=torch.float32)
    data = torch.cat([g_zero, g_normal]).view(1, 1, 1, 32)
    _check_pipeline("mixed zero + normal groups", data)


def test_pipeline_bf16_extremes():
    data = torch.tensor([[7.0, -7.0, 0.0, 0.125, 5.999, 6.001, 4.0, 1e-3,
                          1e-30, 100.0, -100.0, 0.249, 1.249, 2.499, 4.999, -5.001]],
                        dtype=torch.bfloat16).view(1, 1, 1, 16)
    _check_pipeline("bf16 extremes", data)


def test_pipeline_realistic_kcache_bf16():
    """Realistic K-cache shape (1, 8, 128, 128) bf16."""
    torch.manual_seed(42)
    data = torch.randn(1, 8, 128, 128, dtype=torch.bfloat16) * 3
    _check_pipeline("realistic K-cache bf16", data)


def test_pipeline_realistic_fp32():
    torch.manual_seed(43)
    data = torch.randn(1, 8, 64, 128, dtype=torch.float32) * 0.5
    _check_pipeline("realistic fp32", data)


def test_pipeline_head_dim_64():
    """head_dim=64 → 4 groups per (h, t)."""
    torch.manual_seed(44)
    data = torch.randn(1, 4, 16, 64, dtype=torch.float32) * 2
    _check_pipeline("head_dim=64", data)


def test_pipeline_head_dim_256():
    """head_dim=256 → 16 groups per (h, t)."""
    torch.manual_seed(45)
    data = torch.randn(1, 2, 8, 256, dtype=torch.float32) * 2
    _check_pipeline("head_dim=256", data)


def test_pipeline_random_large_batch():
    """100k random elements covering most of [-12, 12]."""
    torch.manual_seed(46)
    data = torch.empty(1, 8, 128, 64, dtype=torch.float32).uniform_(-12, 12)
    _check_pipeline("random uniform[-12, 12]", data)


# ---------------------------------------------------------------------------
# Standalone runner — works without pytest installed.
#   .venv/bin/python tests/kernels/quantization/test_nvfp4_plus_kv_bitidentity.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    import sys

    fns = [
        (n, f)
        for n, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if n.startswith("test_")
    ]
    passed = failed = 0
    print("NVFP4+ full-pipeline:")
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
