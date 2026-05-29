"""Calculate anchoring metrics using the paper's stated formulas.

This module intentionally keeps the original ``anchoring_measure.metrics``
implementation intact. It reuses the model/prompt plumbing from that module,
but replaces the two implementation-specific proxy scores with the formulas
from the paper:

* A_ent = sqrt(G_unif * L_non-unif)
* A_prob = (log2 P(A | Q, R) - log2 P(A | Q)) / |A|
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from anchoring_measure import metrics as base


TAU_G = 0.1
TORCH_DTYPE = base.TORCH_DTYPE
STEP_MICROBATCH = base.STEP_MICROBATCH
ENTROPY_BATCH_SIZE = 512


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def compute_entropy_anchoring_paper(
    token_entropies: List[float],
    step_boundaries: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute Equation 5 with Equations 3 and 4 from the paper."""
    default = {
        "EntropyAnchoring": 0.0,
        "PaperGlobalUniformity": 0.0,
        "PaperLocalNonUniformity": 0.0,
        "PaperStepCount": 0.0,
    }
    if len(token_entropies) < 2:
        return default

    h_arr = np.array(token_entropies, dtype=float)
    if step_boundaries is None:
        window_size = 50
        step_boundaries = [0] + list(range(window_size, len(h_arr), window_size))
        if step_boundaries[-1] != len(h_arr):
            step_boundaries.append(len(h_arr))

    step_ids = []
    for i in range(len(step_boundaries) - 1):
        start, end = int(step_boundaries[i]), int(step_boundaries[i + 1])
        if end > start:
            step_ids.append(float(np.mean(h_arr[start:end])))

    if len(step_ids) < 2:
        return default

    step_arr = np.array(step_ids, dtype=float)
    span = float(np.max(step_arr) - np.min(step_arr))
    if span > 1e-12:
        u = (step_arr - np.min(step_arr)) / span
    else:
        u = np.zeros_like(step_arr)

    global_uniformity = 1.0 / (1.0 + float(np.var(u)) / TAU_G)

    deltas = np.abs(np.diff(u))
    mean_delta = float(np.mean(deltas)) if len(deltas) else 0.0
    if mean_delta > 1e-12:
        cv_delta = float(np.std(deltas) / mean_delta)
        local_non_uniformity = cv_delta / (1.0 + cv_delta)
    else:
        cv_delta = 0.0
        local_non_uniformity = 0.0

    entropy_anchoring = math.sqrt(global_uniformity * local_non_uniformity)
    return {
        "EntropyAnchoring": float(entropy_anchoring),
        "PaperGlobalUniformity": float(global_uniformity),
        "PaperLocalNonUniformity": float(local_non_uniformity),
        "PaperStepCount": float(len(step_ids)),
        "PaperDeltaCV": float(cv_delta),
    }


@torch.no_grad()
def prepare_prefix_past_and_last_logits(
    model: torch.nn.Module,
    prefix_ids_1d: torch.LongTensor,
) -> Tuple[Optional[Any], int, torch.Tensor]:
    device = next(model.parameters()).device
    inp = prefix_ids_1d.unsqueeze(0).to(device)
    out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=True)
    return getattr(out, "past_key_values", None), int(prefix_ids_1d.numel()), out.logits[:, -1, :].detach()


@torch.no_grad()
def compute_reasoning_token_entropies_paper(
    model: torch.nn.Module,
    flat_reason_ids: torch.LongTensor,
    prefix_ids: torch.LongTensor,
    batch_size: int = ENTROPY_BATCH_SIZE,
) -> List[float]:
    """Entropy of the predictive distribution for each reasoning token.

    For token r_t, this uses the distribution before r_t is consumed. That is
    the direct autoregressive reading of the paper's predictive distribution
    p_t over the vocabulary.
    """
    device = next(model.parameters()).device
    total = int(flat_reason_ids.numel())
    if total == 0:
        return []

    if not base.USE_SHARED_PREFIX_PAST:
        prompt_ids = torch.cat([prefix_ids.to(device), flat_reason_ids.to(device)], dim=0)
        prompt_len = int(prefix_ids.numel())
        entropies: List[float] = []
        for i in range(0, total, batch_size):
            reason_end = min(i + batch_size, total)
            pred_start = prompt_len + i - 1
            pred_end = prompt_len + reason_end - 1
            input_ids = prompt_ids[: prompt_len + reason_end].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, pred_start:pred_end, :]
            entropies.extend(entropy_from_logits(logits).squeeze(0).cpu().tolist())
            del logits, outputs, input_ids, attention_mask
        return entropies

    current_past, current_seq_len, next_logits = prepare_prefix_past_and_last_logits(model, prefix_ids)
    entropies = []
    for i in range(0, total, batch_size):
        chunk_ids = flat_reason_ids[i : min(i + batch_size, total)]
        chunk_len = int(chunk_ids.numel())
        if chunk_len <= 0:
            continue

        entropies.append(float(entropy_from_logits(next_logits).squeeze(0).cpu().item()))

        input_ids = chunk_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones((1, current_seq_len + chunk_len), dtype=torch.long, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=base._repeat_past(current_past, 1),
            use_cache=True,
        )

        if chunk_len > 1:
            logits = outputs.logits[:, : chunk_len - 1, :]
            entropies.extend(entropy_from_logits(logits).squeeze(0).cpu().tolist())

        next_logits = outputs.logits[:, -1, :].detach()
        current_past = outputs.past_key_values
        current_seq_len += chunk_len
        del outputs, input_ids, attention_mask

    return entropies


def compute_probabilistic_anchoring_paper(ys_log10_sum: List[float], ans_len: int) -> Dict[str, float]:
    """Compute Equation 6 as per-answer-token log2 probability gain."""
    default = {
        "ProbabilisticAnchoring": 0.0,
        "PaperBaseLog2PerToken": 0.0,
        "PaperFinalLog2PerToken": 0.0,
        "PaperBitGainRate": 0.0,
    }
    if len(ys_log10_sum) < 2 or ans_len <= 0:
        return default

    base_log2_per_token = (float(ys_log10_sum[0]) / ans_len) * base.LOG10_TO_BITS
    final_log2_per_token = (float(ys_log10_sum[-1]) / ans_len) * base.LOG10_TO_BITS
    bit_gain_rate = final_log2_per_token - base_log2_per_token
    return {
        "ProbabilisticAnchoring": float(bit_gain_rate),
        "PaperBaseLog2PerToken": float(base_log2_per_token),
        "PaperFinalLog2PerToken": float(final_log2_per_token),
        "PaperBitGainRate": float(bit_gain_rate),
    }


def has_non_finite(metrics_record: Dict[str, Any]) -> bool:
    for value in metrics_record.values():
        if isinstance(value, float) and not math.isfinite(value):
            return True
    return False


def process_single_method_paper(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_ctx: Optional[int],
    think_open_ids: torch.LongTensor,
    think_close_ids: torch.LongTensor,
    reason_sep_ids: torch.LongTensor,
    method: str,
    obj: Dict[str, Any],
    line_idx: int,
    debug_errors: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        question = obj.get("questions", {}).get(method)
        answer = obj.get("answers", {}).get(method)
        reasoning = obj.get("reasonings", {}).get(method, "")
        context = obj.get("contexts", {}).get(method, "")
        if not (question and answer and reasoning):
            return None

        ans_ids = base.tok_1d(tokenizer, answer, device)
        if int(ans_ids.numel()) <= 0:
            return None

        chat_prefix_ids_q = base.tok_1d(tokenizer, base.apply_chat_prefix(tokenizer, question), device)
        prefix_ids_q = torch.cat([chat_prefix_ids_q, think_open_ids], dim=0)
        past_prefix_q, past_len_q = (
            base.prepare_prefix_past(model, prefix_ids_q)
            if base.USE_SHARED_PREFIX_PAST
            else (None, int(prefix_ids_q.numel()))
        )

        n_parts, flat_reason_ids, end_pos = base.tokenize_reasoning_to_flat_ids(
            tokenizer, reasoning, reason_sep_ids, base.MAX_STEPS_PER_SAMPLE, device
        )
        if int(flat_reason_ids.numel()) <= 0:
            return None

        step_idxs = list(base.step_idx_generator(n_parts)) if base.COMPUTE_ONLY_SAMPLED_STEPS else list(range(n_parts + 1))
        _, _, ys_log10_sum, ans_len = base.compute_absolute_curve_for_R_sampled(
            model,
            tokenizer,
            prefix_ids_q,
            past_prefix_q,
            past_len_q,
            think_close_ids,
            ans_ids,
            max_ctx,
            flat_reason_ids,
            end_pos,
            step_idxs,
            STEP_MICROBATCH,
        )
        if not ys_log10_sum:
            return None

        lexical_score = base.calculate_token_lcs_recall(flat_reason_ids.cpu().tolist(), ans_ids.cpu().tolist())
        prob_metrics = compute_probabilistic_anchoring_paper(ys_log10_sum, ans_len)

        prompt_content = context if context and context.strip() else question
        chat_prefix_ids_c = base.tok_1d(tokenizer, base.apply_chat_prefix(tokenizer, prompt_content), device)
        entropy_prefix_ids = torch.cat([chat_prefix_ids_c, think_open_ids], dim=0)
        token_entropies = compute_reasoning_token_entropies_paper(
            model, flat_reason_ids, entropy_prefix_ids, ENTROPY_BATCH_SIZE
        )
        entropy_metrics = compute_entropy_anchoring_paper(token_entropies, end_pos)

        metrics_record = {
            "method": method,
            "sample_idx": line_idx,
            "lexical_anchoring": float(lexical_score),
            "entropy_anchoring": entropy_metrics["EntropyAnchoring"],
            "ProbabilisticAnchoring": prob_metrics["ProbabilisticAnchoring"],
            **entropy_metrics,
            **prob_metrics,
        }

        if has_non_finite(metrics_record):
            return None
        return metrics_record
    except Exception as exc:
        if debug_errors:
            print(f"[debug] sample={line_idx} method={method} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate paper-formula anchoring metrics")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--scoring-model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--debug-errors", action="store_true")
    parser.add_argument("--disable-shared-prefix-past", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional global row limit for smoke tests")
    args = parser.parse_args()

    if args.disable_shared_prefix_past:
        base.USE_SHARED_PREFIX_PAST = False

    rank, world_size, local_rank = base.get_rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, f"metrics_rank{rank}.jsonl")
    valid_data_file = os.path.join(args.output_dir, f"valid_data_rank{rank}.jsonl")

    print(f"[rank{rank}] Loading model from: {args.scoring_model}")
    cfg = AutoConfig.from_pretrained(args.scoring_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.scoring_model,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.scoring_model,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
        )

    model.to(device)
    model.eval()
    max_ctx = getattr(model.config, "max_position_embeddings", None)

    think_open_ids = base.tok_1d(tokenizer, "<think>\n", device)
    think_close_ids = base.tok_1d(tokenizer, "</think>\n\n", device)
    reason_sep_ids = base.tok_1d(tokenizer, base.SEQ_SEP, device)

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            if i % world_size == rank:
                samples.append((i, json.loads(line)))

    total_samples = len(samples)
    valid_samples = 0
    discarded_samples = 0

    with open(metrics_file, "w", encoding="utf-8") as fw_metrics, open(valid_data_file, "w", encoding="utf-8") as fw_valid:
        for line_idx, obj in tqdm(samples, desc=f"Processing paper metrics (rank{rank})", disable=(rank != 0)):
            methods = list(obj.get("questions", {}).keys())
            if not methods:
                continue

            sample_metrics = []
            sample_valid = True
            for method in methods:
                record = process_single_method_paper(
                    model,
                    tokenizer,
                    device,
                    max_ctx,
                    think_open_ids,
                    think_close_ids,
                    reason_sep_ids,
                    method,
                    obj,
                    line_idx,
                    args.debug_errors,
                )
                if record is None:
                    sample_valid = False
                    break
                sample_metrics.append(record)

            if sample_valid and len(sample_metrics) == len(methods):
                for record in sample_metrics:
                    fw_metrics.write(json.dumps(record) + "\n")
                fw_metrics.flush()
                fw_valid.write(json.dumps(obj) + "\n")
                fw_valid.flush()
                valid_samples += 1
            else:
                discarded_samples += 1

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[rank{rank}] Paper-metric processing complete:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid samples (all methods finite): {valid_samples}")
    print(f"  Discarded samples: {discarded_samples}")
    print(f"  Metrics saved to: {metrics_file}")
    print(f"  Valid data saved to: {valid_data_file}")


if __name__ == "__main__":
    main()
