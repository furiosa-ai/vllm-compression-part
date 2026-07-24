"""Calibrate skip-softmax threshold scale factors with NVIDIA Model-Optimizer.

Runs RULER-based calibration of the ``flash_skip_softmax`` method from
``nvidia-modelopt`` against a HuggingFace causal-LM, then writes a JSON
file containing the ``(a, b)`` exponential-model parameters per phase
plus the concrete scale factors to pass to vLLM:

    scale_factor(target) = a * exp(b * target)

which is exactly the value of
``--attention-config.skip_softmax_threshold_scale_factor_{prefill,decode}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--out",
        default="calibration/skip_softmax.json",
        help="Path to write calibration JSON",
    )
    p.add_argument(
        "--target-sparsity-prefill",
        type=float,
        default=0.7,
        help="Target prefill sparsity (0.0 skips prefill calibration)",
    )
    p.add_argument(
        "--target-sparsity-decode",
        type=float,
        default=0.7,
        help="Target decode sparsity (0.0 skips decode calibration)",
    )
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--max-seqlen", type=int, default=16384)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--num-decode-tokens", type=int, default=10)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache generated RULER samples (optional)",
    )
    p.add_argument(
        "--also-evaluate",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Additional target sparsities to report scale factors for "
            "(does not re-run calibration). Example: --also-evaluate 0.3 0.5 0.7"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    t0 = time.time()
    print(f"[calibrate] Loading {args.model} with attn_implementation='eager' …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[calibrate] Loaded in {time.time() - t0:.1f}s")

    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from modelopt.torch.sparsity.attention_sparsity.calibration import calibrate as _cal
    from modelopt.torch.sparsity.attention_sparsity.utils import (
        get_named_sparse_attention_modules,
    )

    # Monkey-patch the decode-phase fast-prefill attn impl: transformers 5.6
    # ships a buggy flash_attention path for Qwen3-MoE (s_aux can be None).
    # SDPA avoids F.softmax too, so it still bypasses measurement.
    _orig = _cal.create_decode_calibration_forward_loop

    def _patched_decode_loop(*a, **kw):
        inner = _orig(*a, **kw)

        def wrapped(model):
            orig_cfg_value = getattr(model.config, "_attn_implementation", "eager")
            # Replace the function's closure-written "flash_attention_2" with
            # "sdpa" by substituting the model's config right before inner runs.
            try:
                return inner(model)
            finally:
                model.config._attn_implementation = orig_cfg_value

        return wrapped

    # Simpler: directly patch the hardcoded backend string in the factory.
    import modelopt.torch.sparsity.attention_sparsity.calibration.calibrate as _calmod

    _src_fn = _calmod.create_decode_calibration_forward_loop

    def _safer_decode_factory(calibration_data, tokenizer_name_or_path, num_decode_tokens=10):
        # Mirror the original implementation but use "sdpa" for fast prefill.
        from modelopt.torch.utils import get_module_device

        tok = _calmod._load_tokenizer(tokenizer_name_or_path)

        def forward_loop(model):
            device = get_module_device(model)
            for sample in calibration_data:
                inputs = tok(
                    sample["input"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=sample["length"],
                )
                input_ids = inputs["input_ids"].to(device)
                original = getattr(model.config, "_attn_implementation", "eager")
                with torch.no_grad():
                    try:
                        model.config._attn_implementation = "sdpa"
                        outputs = model(input_ids, use_cache=True)
                        past_kv = outputs.past_key_values
                        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                        del outputs

                        model.config._attn_implementation = "eager"
                        for _ in range(num_decode_tokens):
                            outputs = model(
                                next_token,
                                past_key_values=past_kv,
                                use_cache=True,
                            )
                            past_kv = outputs.past_key_values
                            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                            del outputs
                    finally:
                        model.config._attn_implementation = original
                del past_kv
                torch.cuda.empty_cache()

        return forward_loop

    _calmod.create_decode_calibration_forward_loop = _safer_decode_factory

    sparse_cfg: dict = {
        "*": {
            "method": "flash_skip_softmax",
            "backend": "pytorch",
            "enable": True,
            "br": 128,
            "bc": 128,
            "is_causal": True,
        },
        "calibration": {
            "target_sparse_ratio": {
                "prefill": args.target_sparsity_prefill,
                "decode": args.target_sparsity_decode,
            },
            "samples": args.samples,
            "max_seqlen": args.max_seqlen,
            "chunk_size": args.chunk_size,
            "num_decode_tokens": args.num_decode_tokens,
        },
    }
    if args.cache_dir:
        sparse_cfg["calibration"]["cache_dir"] = args.cache_dir

    config = {"sparse_cfg": sparse_cfg}

    # ----------------------------------------------------------------------------------
    # GROUP-CONSENSUS PATCH (GQA-aware sparsity measurement)
    #
    # WHY: the modelopt-0.45 `flash_skip_softmax` calibrator decides skips INDEPENDENTLY
    # PER QUERY HEAD and reports the per-head mean sparsity. The deployed FlashInfer
    # kernel (fmha_v2/trtllm_gen) skips a KV tile only by a UNANIMOUS vote across the
    # whole GQA group (__all_sync + atomicAnd(skip_softmax_vote), verified in
    # flashinfer 0.6.8.post1 csrc/fmha_v2/fmha/warpspec/epilogue.h): a tile is skipped
    # only if EVERY query head sharing that KV read votes to skip. So per-head calibration
    # fits SF to an upper bound the fused kernel cannot realize, and calibrated targets
    # under-deliver as *group* sparsity.
    #
    # FIX: reduce the per-head KEEP mask group-wise (a KV tile is KEPT if ANY head in the
    # group keeps it == OR over the group) before counting, and divide by num_kv_heads.
    # This makes stats["sparsity"] the group-consensus sparsity the kernel realizes, so
    # the fitted (a,b) map GROUP-sparsity -> SF. No-op for MHA (rep == 1).
    #
    # Applied by rebinding the modelopt method (kept in this furiosa-owned driver rather
    # than editing the pip-installed package). num_kv_heads is bound from the model config
    # via closure BEFORE sparsify(), because calibration runs INSIDE sparsify() — setting
    # it on the modules afterwards would be too late. block_mask is kept per-head and the
    # group OR-reduction is applied only to the sparsity COUNT, so the per-head
    # element_mask path can never shape-mismatch.
    # ----------------------------------------------------------------------------------
    import numpy as _np
    import math as _math
    from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import (
        FlashSkipSoftmax as _FSS,
    )

    _NKV = int(getattr(model.config, "num_key_value_heads", 0) or 0)
    _NQH = int(getattr(model.config, "num_attention_heads", 0) or 0)
    print(
        f"[patch] group-consensus: num_kv_heads={_NKV} num_attention_heads={_NQH} "
        f"rep={(_NQH // _NKV) if _NKV else 'n/a'} (rep==1 => MHA no-op)"
    )

    def _patched_calc(self, attn_weights, phase):  # noqa: C901  (faithful copy of 0.45 body)
        batch_size, num_heads, seq_q, seq_k = attn_weights.shape

        # --- group-consensus helper (GQA-aware) ---
        num_kv_heads = _NKV or num_heads
        rep = max(num_heads // num_kv_heads, 1)  # GQA group size; 1 == MHA (no-op)

        def _group_keep(keep):
            # keep: per-head KEEP mask [B, Hq, ...] -> group KEEP [B, Hkv, ...] via OR
            # over the rep heads that share one KV read (contiguous, repeat_kv order:
            # q-head = kv_idx*rep + r). A KV tile is kept if ANY head in the group keeps it.
            return keep if rep == 1 else keep.unflatten(1, (num_kv_heads, rep)).any(dim=2)

        calibration_params = self.calibration_params
        target_sparse_ratio = self.target_sparse_ratio
        use_calibration_params = (
            calibration_params is not None
            and phase in calibration_params
            and target_sparse_ratio is not None
        )

        if use_calibration_params:
            a = calibration_params[phase]["a"]
            b = calibration_params[phase]["b"]
            target_sparsity = target_sparse_ratio.get(phase, 0.5)
            scale_factor = a * _np.exp(b * target_sparsity)
            log_thresholds = [_np.log(scale_factor / seq_k)]
        else:
            log_thresholds = [_np.log(t) for t in self.thresholds]

        if phase == "prefill":
            blocked_attn, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k = (
                self._reshape_to_blocks(attn_weights, self.br, self.bc)
            )
            block_max = blocked_attn.max(dim=-1)[0]
            del blocked_attn
            block_max_cummax = block_max.cummax(dim=-1)[0]

            block_max_larger = torch.ones_like(block_max)
            block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
            correction_factor = (block_max_larger.sum() / block_max_larger.numel()).item()
            del block_max_larger

            if self.is_causal:
                num_causal_blocks = num_block_rows * (2 * num_block_cols - num_block_rows + 1) // 2
                total_valid_blocks = batch_size * num_kv_heads * num_causal_blocks  # group-wise
                total_blocks = num_causal_blocks
            else:
                total_valid_blocks = batch_size * num_kv_heads * num_block_rows * num_block_cols
                total_blocks = num_block_rows * num_block_cols

            dense_blocks_list = []
            block_mask_0 = None
            block_diff = block_max - block_max_cummax
            for i, log_threshold in enumerate(log_thresholds):
                block_mask = (block_diff > log_threshold).any(dim=-2)  # per-head KEEP
                dense_blocks_list.append(_group_keep(block_mask).sum().item())  # group count
                if i == 0 and not self._calibration_mode:
                    block_mask_0 = block_mask
                del block_mask

            del block_max, block_max_cummax

            if not self._calibration_mode and block_mask_0 is not None:
                element_mask = (
                    block_mask_0.unsqueeze(-2)
                    .unsqueeze(-1)
                    .expand(batch_size, num_heads, num_block_rows, self.br, num_block_cols, self.bc)
                )
                del block_mask_0
                element_mask = element_mask.reshape(
                    batch_size, num_heads, padded_seq_q, padded_seq_k
                )
                element_mask = element_mask[:, :, :seq_q, :seq_k]
            else:
                element_mask = None

        else:  # decode
            blocked_attn, _, num_block_cols, _, padded_seq_k = self._reshape_to_blocks(
                attn_weights, 1, self.bc
            )
            block_max = blocked_attn.max(dim=-1)[0]
            del blocked_attn
            block_max_cummax = block_max.cummax(dim=-1)[0]

            block_max_larger = torch.ones_like(block_max)
            block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
            correction_factor = (block_max_larger.sum() / block_max_larger.numel()).item()
            del block_max_larger

            total_valid_blocks = batch_size * num_kv_heads * num_block_cols  # group-wise
            total_blocks = num_block_cols

            dense_blocks_list = []
            block_mask_0 = None
            for i, log_threshold in enumerate(log_thresholds):
                block_mask = block_max - block_max_cummax > log_threshold  # per-head KEEP
                dense_blocks_list.append(_group_keep(block_mask).sum().item())  # group count
                if i == 0 and not self._calibration_mode:
                    block_mask_0 = block_mask
                del block_mask

            del block_max, block_max_cummax

            if not self._calibration_mode and block_mask_0 is not None:
                element_mask = block_mask_0[..., None].expand(
                    batch_size, num_heads, 1, 1, num_block_cols, self.bc
                )
                del block_mask_0
                element_mask = element_mask.reshape(batch_size, num_heads, 1, padded_seq_k)
                element_mask = element_mask[:, :, :seq_q, :seq_k]
            else:
                element_mask = None

        sparsity_list = [1.0 - d / total_valid_blocks for d in dense_blocks_list]
        sparsity_out = sparsity_list
        sparse_blocks_out = [int(s * total_blocks) for s in sparsity_list]

        stats = {
            "correction_factor": correction_factor,
            "sparsity": sparsity_out,
            "phase": phase,
            "total_blocks": total_blocks,
            "sparse_blocks": sparse_blocks_out,
            "sample_length": seq_k,
        }
        return element_mask, stats

    _FSS.calc_correction_factor_and_p = _patched_calc
    print("[patch] rebound FlashSkipSoftmax.calc_correction_factor_and_p (group-consensus)")

    print("[calibrate] Starting calibration …")
    t1 = time.time()
    mtsa.sparsify(model, config)
    print(f"[calibrate] Calibration finished in {time.time() - t1:.1f}s")

    named = get_named_sparse_attention_modules(model)
    if not named:
        print("[calibrate] ERROR: no sparse attention modules registered", file=sys.stderr)
        return 2
    _, any_mod = named[0]
    method = any_mod._sparse_method_instance
    params = getattr(method, "calibration_params", None)
    targets = getattr(method, "target_sparse_ratio", None)

    if not params:
        print("[calibrate] ERROR: calibration produced no parameters", file=sys.stderr)
        return 3

    def scale(phase: str, target: float) -> float:
        p_ = params[phase]
        return float(p_["a"] * math.exp(p_["b"] * target))

    result: dict = {
        "model": args.model,
        "calibration_params": {
            phase: {k: float(v) for k, v in params[phase].items()}
            for phase in params
        },
        "target_sparse_ratio": {k: float(v) for k, v in targets.items()},
        "formula": "scale_factor = a * exp(b * target_sparsity)",
        "vllm_flags": {},
        "additional_operating_points": {},
    }

    if "prefill" in params and args.target_sparsity_prefill > 0:
        result["vllm_flags"]["skip_softmax_threshold_scale_factor_prefill"] = scale(
            "prefill", args.target_sparsity_prefill
        )
    if "decode" in params and args.target_sparsity_decode > 0:
        result["vllm_flags"]["skip_softmax_threshold_scale_factor_decode"] = scale(
            "decode", args.target_sparsity_decode
        )

    for extra in args.also_evaluate or []:
        result["additional_operating_points"][f"target={extra}"] = {
            phase: scale(phase, extra) for phase in params
        }

    result["sparsity_granularity"] = "group_consensus"
    result["num_kv_heads"] = _NKV
    result["note"] = (
        "GROUP-CONSENSUS calibration: sparsity measured GQA-group-wise (a KV tile is "
        "kept if ANY head in the group keeps it) to match the served FlashInfer kernel's "
        "unanimous skip vote. SFs differ from (are larger than) per-head calibration. "
        "Report accuracy vs realized (block-count) group sparsity."
    )

    out_path.write_text(json.dumps(result, indent=2))
    print(f"[calibrate] Wrote {out_path}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
