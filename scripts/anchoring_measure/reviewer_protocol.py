#!/usr/bin/env python3
"""Reviewer-protocol anchoring metrics, aggregation, and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


LOGE_TO_BITS = 1.0 / math.log(2.0)
SEQ_SEP = "\n\n"
TORCH_DTYPE = torch.bfloat16
STEP_MICROBATCH = 4
ENTROPY_BATCH = 384
METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
CONTROL_ORDER = ["Blind CoT", "Response-as-CoT", "+Prob Anchor", "+Entropy Anchor"]
ZONE_KEYS = ["Reason", "Encode", "Cloze", "Copy"]
WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\W\s]", flags=re.UNICODE)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for", "with",
    "as", "by", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "from", "at", "into", "about", "we", "you", "i", "he", "she", "they", "them", "our",
    "your", "their", "not", "no", "yes", "do", "does", "did", "can", "could", "would", "should",
    "will", "may", "might", "must", "have", "has", "had", "there", "here", "which", "what", "when",
    "where", "why", "how", "also", "than", "therefore", "thus", "because",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_metric_dir(path: Path) -> List[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("metrics_rank*.jsonl"))
    rows = []
    for file in files:
        rows.extend(read_jsonl(file))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def split_reasoning(text: str) -> List[str]:
    parts = [p.strip() for p in text.strip().split(SEQ_SEP)]
    parts = [p for p in parts if p]
    if len(parts) > 1:
        return parts
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    numbered = [line for line in lines if re.match(r"^\s*(?:\d+[\.)]|[-*])\s+", line)]
    if len(numbered) > 1:
        return lines
    sentence_parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    sentence_parts = [p.strip() for p in sentence_parts if p.strip()]
    if len(sentence_parts) > 1:
        return sentence_parts
    return parts


def trace_prefix(text: str, tokenizer: Any, keep_ratio: float = 0.85) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return ""
    keep = max(1, int(math.floor(len(ids) * keep_ratio)))
    return tokenizer.decode(ids[:keep], skip_special_tokens=False)


def tokenize_words(text: str) -> List[str]:
    return [tok.lower() for tok in WORD_RE.findall(text)]


def lcs_len(a: Sequence[Any], b: Sequence[Any]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[-1]


def lcs_recall(candidate: str, reference: str) -> float:
    cand = tokenize_words(candidate)
    ref = tokenize_words(reference)
    if not cand or not ref:
        return 0.0
    return float(lcs_len(cand, ref) / len(ref))


def apply_chat_prefix(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tok_1d(tokenizer: Any, text: str, device: Optional[torch.device] = None) -> torch.LongTensor:
    return torch.tensor(tokenizer(text, add_special_tokens=False).input_ids, dtype=torch.long, device=device)


def tokenize_reasoning(tokenizer: Any, text: str, sep_ids: torch.LongTensor, device: torch.device) -> Tuple[int, torch.LongTensor, List[int]]:
    parts = split_reasoning(text)
    if not parts:
        return 0, torch.empty((0,), dtype=torch.long, device=device), []
    flat: List[int] = []
    ends: List[int] = []
    for idx, part in enumerate(parts):
        if idx > 0:
            flat.extend(sep_ids.tolist())
        flat.extend(tokenizer(part, add_special_tokens=False).input_ids)
        ends.append(len(flat))
    return len(parts), torch.tensor(flat, dtype=torch.long, device=device), ends


def step_idx_generator(n: int) -> List[int]:
    if n <= 0:
        return [0]
    if n <= 50:
        return list(range(n + 1))
    out = list(range(0, 51))
    out.extend(range(54, n + 1, 4))
    if out[-1] != n:
        out.append(n)
    return out


@torch.no_grad()
def batch_answer_logprob(model: Any, prompt_ids_list: List[torch.LongTensor], ans_ids: torch.LongTensor, max_ctx: Optional[int], batch_size: int) -> List[float]:
    device = next(model.parameters()).device
    ans_ids = ans_ids.to(device)
    ans_len = int(ans_ids.numel())
    scores = []
    for i0 in range(0, len(prompt_ids_list), batch_size):
        chunk = prompt_ids_list[i0:i0 + batch_size]
        seqs, prompt_lens = [], []
        for prompt in chunk:
            full = torch.cat([prompt.to(device), ans_ids], dim=0)
            if max_ctx is not None and int(full.numel()) > max_ctx:
                full = full[-int(max_ctx):]
            seqs.append(full)
            prompt_lens.append(int(full.numel()) - ans_len)
        max_len = max(int(s.numel()) for s in seqs)
        input_ids = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for b, seq in enumerate(seqs):
            length = int(seq.numel())
            input_ids[b, :length] = seq
            attn[b, :length] = 1
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logp = torch.log_softmax(out.logits, dim=-1)
        for b, prompt_len in enumerate(prompt_lens):
            pred_pos = torch.arange(prompt_len - 1, prompt_len + ans_len - 1, device=device)
            target = input_ids[b, prompt_len:prompt_len + ans_len]
            selected = logp[b, pred_pos, :].gather(1, target.unsqueeze(1)).squeeze(1)
            scores.append(float(selected.sum().item()))
        del out, logp, input_ids, attn
    return scores


@torch.no_grad()
def reasoning_entropies(model: Any, flat_reason_ids: torch.LongTensor, prefix_ids: torch.LongTensor, batch_size: int) -> List[float]:
    device = next(model.parameters()).device
    full = torch.cat([prefix_ids.to(device), flat_reason_ids.to(device)], dim=0)
    prefix_len = int(prefix_ids.numel())
    total = int(flat_reason_ids.numel())
    entropies: List[float] = []
    for i in range(0, total, batch_size):
        reason_start = prefix_len + i
        reason_end = prefix_len + min(i + batch_size, total)
        input_ids = full[:reason_end].unsqueeze(0)
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits[:, reason_start - 1:reason_end - 1, :]
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropies.extend((-torch.sum(probs * log_probs, dim=-1)).squeeze(0).cpu().tolist())
        del out, logits, probs, log_probs, input_ids, attn
    return entropies


def entropy_anchoring(token_entropies: List[float], boundaries: List[int], vocab_size: int, tau_g: float) -> float:
    if len(token_entropies) < 2 or not boundaries:
        return 0.0
    arr = np.array(token_entropies, dtype=float) / max(1e-12, math.log(vocab_size))
    step_vals = []
    start = 0
    for end in boundaries:
        end = int(min(end, len(arr)))
        if end > start:
            step_vals.append(float(np.mean(arr[start:end])))
        start = end
    if len(step_vals) < 2:
        return 0.0
    u = np.array(step_vals, dtype=float)
    g_unif = 1.0 / (1.0 + float(np.var(u)) / tau_g)
    deltas = np.abs(np.diff(u))
    mu = float(np.mean(deltas))
    if mu <= 1e-12:
        l_non = 0.0
    else:
        cv = float(np.std(deltas)) / mu
        l_non = cv / (1.0 + cv)
    return float(np.sqrt(g_unif * l_non))


def load_paraphrases(path: Path) -> Dict[int, List[str]]:
    out = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        out[int(row["sample_idx"])] = [ensure_text(x) for x in row.get("paraphrases", [])[:2]]
    return out


def rank_info() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        return rank, world, local
    return 0, 1, 0


def score(args: argparse.Namespace) -> None:
    rank, world, local = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))

    rows = read_jsonl(args.input)
    paras = load_paraphrases(args.paraphrases)
    sep_ids = tok_1d(tokenizer, SEQ_SEP, device)
    think_close_ids = tok_1d(tokenizer, "</think>\n\n", device)
    methods_filter = [m for m in args.methods.split(",") if m] if args.methods else None

    metrics_path = args.output_dir / f"metrics_rank{rank}.jsonl"
    valid_path = args.output_dir / f"valid_data_rank{rank}.jsonl"
    invalid_path = args.output_dir / f"invalid_rank{rank}.jsonl"
    valid = invalid = 0
    with metrics_path.open("w", encoding="utf-8") as fw, valid_path.open("w", encoding="utf-8") as fv, invalid_path.open("w", encoding="utf-8") as fi:
        for sample_idx, row in tqdm(list(enumerate(rows))[rank::world], desc=f"rank{rank}", disable=(rank != 0)):
            methods = methods_filter or list(row.get("reasonings", {}).keys())
            sample_records = []
            ok = True
            paraphrases = paras.get(sample_idx, [])
            if len(paraphrases) < 2:
                fi.write(json.dumps({"sample_idx": sample_idx, "method": "ALL", "error": "missing_valid_paraphrases"}, ensure_ascii=False) + "\n")
                invalid += 1
                fi.flush()
                continue
            for method in methods:
                try:
                    q = ensure_text(row.get("questions", {}).get(method))
                    a = ensure_text(row.get("answers", {}).get(method))
                    r = ensure_text(row.get("reasonings", {}).get(method))
                    if not (q.strip() and a.strip() and r.strip()):
                        raise ValueError("missing Q/A/R")
                    ans_ids = tok_1d(tokenizer, a, device)
                    if int(ans_ids.numel()) == 0:
                        raise ValueError("empty answer ids")
                    n_steps, flat_reason, end_pos = tokenize_reasoning(tokenizer, r, sep_ids, device)
                    if int(flat_reason.numel()) == 0:
                        raise ValueError("empty reasoning ids")

                    prefix_q = tok_1d(tokenizer, apply_chat_prefix(tokenizer, q), device)
                    base_prompt = torch.cat([prefix_q, think_close_ids], dim=0)
                    step_prompts = [base_prompt]
                    for s in step_idx_generator(n_steps)[1:]:
                        r_end = int(end_pos[s - 1]) if s - 1 < len(end_pos) else int(end_pos[-1])
                        step_prompts.append(torch.cat([prefix_q, flat_reason[:r_end], think_close_ids], dim=0))
                    logps = batch_answer_logprob(model, step_prompts, ans_ids, max_ctx, STEP_MICROBATCH)
                    base_log2_per_token = float(logps[0] * LOGE_TO_BITS / int(ans_ids.numel()))
                    final_log2_per_token = float(logps[-1] * LOGE_TO_BITS / int(ans_ids.numel()))
                    bit_gain = final_log2_per_token - base_log2_per_token

                    ent_prefix = prefix_q
                    token_h = reasoning_entropies(model, flat_reason, ent_prefix, ENTROPY_BATCH)
                    aent = entropy_anchoring(token_h, end_pos, vocab_size, args.tau_g)

                    prefix_r = trace_prefix(r, tokenizer, args.lexical_prefix_ratio)
                    raw_lcs = lcs_recall(prefix_r, a)
                    par_scores = [lcs_recall(prefix_r, p) for p in paraphrases[:2]]
                    delta_lex = raw_lcs - float(np.mean(par_scores))

                    rec = {
                        "sample_idx": sample_idx,
                        "id": row.get("id", sample_idx),
                        "method": method,
                        "n_steps": n_steps,
                        "reasoning_tokens": int(flat_reason.numel()),
                        "answer_tokens": int(ans_ids.numel()),
                        "logp_base_per_token_bits": base_log2_per_token,
                        "logp_final_per_token_bits": final_log2_per_token,
                        "B": bit_gain,
                        "Aent_raw": aent,
                        "A_lex_raw": raw_lcs,
                        "A_lex_paraphrase_mean": float(np.mean(par_scores)),
                        "Delta_lex": delta_lex,
                    }
                    if any(isinstance(v, float) and not np.isfinite(v) for v in rec.values()):
                        raise ValueError("non-finite metric")
                    sample_records.append(rec)
                except Exception as exc:
                    ok = False
                    fi.write(json.dumps({"sample_idx": sample_idx, "method": method, "error": repr(exc)}, ensure_ascii=False) + "\n")
                    if args.debug_errors:
                        traceback.print_exc()
                    break
            if ok:
                for rec in sample_records:
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fv.write(json.dumps(row, ensure_ascii=False) + "\n")
                valid += 1
            else:
                invalid += 1
            fw.flush()
            fv.flush()
            fi.flush()
            torch.cuda.empty_cache()
    print(json.dumps({"rank": rank, "valid": valid, "invalid": invalid, "metrics": str(metrics_path)}, indent=2))


def bucket_key(n_steps: int) -> str:
    if n_steps <= 4:
        return "1-4"
    if n_steps <= 8:
        return "5-8"
    if n_steps <= 16:
        return "9-16"
    if n_steps <= 32:
        return "17-32"
    return "33+"


def mean_by_bucket(records: List[Dict[str, Any]], field: str) -> tuple[Dict[str, float], float]:
    grouped = defaultdict(list)
    all_vals = []
    for rec in records:
        val = float(rec[field])
        grouped[bucket_key(int(rec.get("n_steps", 0)))].append(val)
        all_vals.append(val)
    global_mean = float(np.mean(all_vals)) if all_vals else 0.0
    return {k: float(np.mean(v)) for k, v in grouped.items()}, global_mean


def baseline_value(rec: Dict[str, Any], bucket_means: Dict[str, float], global_mean: float) -> float:
    return float(bucket_means.get(bucket_key(int(rec.get("n_steps", 0))), global_mean))


def bootstrap_ci(vals: np.ndarray, rng: random.Random, n_boot: int) -> tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    means = []
    for _ in range(n_boot):
        idx = [rng.randrange(vals.size) for _ in range(vals.size)]
        means.append(float(np.mean(vals[idx])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace) -> None:
    method_records = read_metric_dir(args.method_metrics)
    blind_records = read_metric_dir(args.blind_metrics)
    control_records = read_metric_dir(args.control_metrics) if args.control_metrics else []
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    method_order = [m for m in args.method_order.split(",") if m] if args.method_order else METHOD_ORDER

    base_blind = [r for r in blind_records if int(r["sample_idx"]) % 2 == 0]
    heldout_blind = [r for r in blind_records if int(r["sample_idx"]) % 2 == 1]
    if not base_blind or not heldout_blind:
        raise ValueError("Blind records must contain both base and heldout split samples")

    b_lex, g_lex = mean_by_bucket(base_blind, "Delta_lex")
    b_ent, g_ent = mean_by_bucket(base_blind, "Aent_raw")
    b_prob, g_prob = mean_by_bucket(base_blind, "B")

    def add_excess(rec: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(rec)
        item["A_lex_excess"] = float(item["Delta_lex"]) - baseline_value(item, b_lex, g_lex)
        item["A_ent_excess"] = float(item["Aent_raw"]) - baseline_value(item, b_ent, g_ent)
        item["A_prob_excess"] = float(item["B"]) - baseline_value(item, b_prob, g_prob)
        return item

    method_ex = [add_excess(r) for r in method_records]
    heldout_ex = [add_excess(r) for r in heldout_blind]
    control_ex = [add_excess(r) for r in control_records]

    y_thr = float(np.quantile([r["A_prob_excess"] for r in heldout_ex], args.zone_quantile))
    x_thr = float(np.quantile([r["A_ent_excess"] for r in heldout_ex], args.zone_quantile))
    uninformative_thr = float(np.quantile([r["A_prob_excess"] for r in heldout_ex], args.uninformative_quantile))

    write_jsonl(out_dir / "method_excess_metrics.jsonl", method_ex)
    write_jsonl(out_dir / "heldout_blind_excess_metrics.jsonl", heldout_ex)
    if control_ex:
        write_jsonl(out_dir / "control_excess_metrics.jsonl", control_ex)

    rng = random.Random(args.bootstrap_seed)
    table_rows = []
    zone_rows = []
    for method in method_order:
        rows = [r for r in method_ex if r["method"] == method]
        if not rows:
            continue
        metric_row: Dict[str, Any] = {"Method": method, "N": len(rows)}
        for field, name in [("A_lex_excess", "A_lex_excess"), ("A_ent_excess", "A_ent_excess"), ("A_prob_excess", "A_prob_excess")]:
            vals = np.array([float(r[field]) for r in rows], dtype=float)
            lo, hi = bootstrap_ci(vals, rng, args.bootstrap)
            metric_row[name] = 100.0 * float(np.mean(vals))
            metric_row[name + "_ci95"] = f"[{100.0 * lo:.2f}, {100.0 * hi:.2f}]"
        vals_prob = np.array([float(r["A_prob_excess"]) for r in rows], dtype=float)
        metric_row["UninformativeRate"] = 100.0 * float(np.mean(vals_prob < uninformative_thr))
        table_rows.append(metric_row)

        x = np.array([float(r["A_ent_excess"]) for r in rows], dtype=float)
        y = np.array([float(r["A_prob_excess"]) for r in rows], dtype=float)
        zone_rows.append({
            "Method": method,
            "N": len(rows),
            "Reason": 100.0 * float(np.mean((x < x_thr) & (y < y_thr))),
            "Encode": 100.0 * float(np.mean((x < x_thr) & (y >= y_thr))),
            "Cloze": 100.0 * float(np.mean((x >= x_thr) & (y < y_thr))),
            "Copy": 100.0 * float(np.mean((x >= x_thr) & (y >= y_thr))),
        })

    metric_fields = [
        "Method", "N",
        "A_lex_excess", "A_lex_excess_ci95",
        "A_ent_excess", "A_ent_excess_ci95",
        "A_prob_excess", "A_prob_excess_ci95",
        "UninformativeRate",
    ]
    write_csv(out_dir / "table1_metrics.csv", table_rows, metric_fields)
    write_csv(out_dir / "table2_zones.csv", zone_rows, ["Method", "N", *ZONE_KEYS])
    (out_dir / "table1_metrics.md").write_text(
        markdown_table(metric_fields, [[r.get(f, "") for f in metric_fields] for r in table_rows]),
        encoding="utf-8",
    )
    (out_dir / "table2_zones.md").write_text(
        markdown_table(["Method", "N", *ZONE_KEYS], [[r["Method"], r["N"], *(r[k] for k in ZONE_KEYS)] for r in zone_rows]),
        encoding="utf-8",
    )

    if control_ex:
        sanity = []
        for method in CONTROL_ORDER:
            rows = [r for r in control_ex if r["method"] == method]
            if not rows:
                continue
            sanity.append({
                "Method": method,
                "N": len(rows),
                "A_lex_excess": 100.0 * float(np.mean([r["A_lex_excess"] for r in rows])),
                "A_ent_excess": 100.0 * float(np.mean([r["A_ent_excess"] for r in rows])),
                "A_prob_excess": 100.0 * float(np.mean([r["A_prob_excess"] for r in rows])),
            })
        write_csv(out_dir / "controlled_sanity.csv", sanity, ["Method", "N", "A_lex_excess", "A_ent_excess", "A_prob_excess"])
        (out_dir / "controlled_sanity.md").write_text(
            markdown_table(["Method", "N", "A_lex_excess", "A_ent_excess", "A_prob_excess"], [[r["Method"], r["N"], r["A_lex_excess"], r["A_ent_excess"], r["A_prob_excess"]] for r in sanity]),
            encoding="utf-8",
        )

    plot(method_ex, method_order, out_dir / "figure_methods.png", x_thr, y_thr)
    if control_ex:
        plot(control_ex, CONTROL_ORDER, out_dir / "figure_controlled.png", x_thr, y_thr)

    report = {
        "method_metrics": str(args.method_metrics),
        "blind_metrics": str(args.blind_metrics),
        "control_metrics": str(args.control_metrics) if args.control_metrics else None,
        "method_order": method_order,
        "blind_base_rule": "sample_idx % 2 == 0",
        "blind_heldout_rule": "sample_idx % 2 == 1",
        "zone_thresholds": {"A_ent_excess": x_thr, "A_prob_excess": y_thr, "quantile": args.zone_quantile},
        "uninformative_threshold": {"A_prob_excess": uninformative_thr, "quantile": args.uninformative_quantile},
        "baseline_means": {
            "Delta_lex": {"bucket": b_lex, "global": g_lex},
            "Aent_raw": {"bucket": b_ent, "global": g_ent},
            "B": {"bucket": b_prob, "global": g_prob},
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    parts = [
        "# Reviewer Protocol Final Results",
        "",
        "## Table 1",
        (out_dir / "table1_metrics.md").read_text(encoding="utf-8"),
        "## Table 2",
        (out_dir / "table2_zones.md").read_text(encoding="utf-8"),
    ]
    if control_ex:
        parts.extend(["## Controlled Sanity", (out_dir / "controlled_sanity.md").read_text(encoding="utf-8")])
    parts.extend([
        "## Figures",
        f"- Methods: `{out_dir / 'figure_methods.png'}`",
        f"- Controlled: `{out_dir / 'figure_controlled.png'}`" if control_ex else "",
    ])
    (out_dir / "report.md").write_text("\n".join(p for p in parts if p), encoding="utf-8")
    print(out_dir / "report.md")


def plot(records: List[Dict[str, Any]], order: List[str], path: Path, x_thr: float, y_thr: float) -> None:
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for r in records:
        grouped[str(r["method"])].append(r)
    fig, axes = plt.subplots(1, len(order), figsize=(4.3 * len(order), 4.3))
    if len(order) == 1:
        axes = [axes]
    cmap, norm = plt.cm.coolwarm, plt.Normalize(-0.5, 0.5)
    for idx, method in enumerate(order):
        ax = axes[idx]
        rows = grouped.get(method, [])
        x = np.array([float(r["A_ent_excess"]) for r in rows], dtype=float)
        y = np.array([float(r["A_prob_excess"]) for r in rows], dtype=float)
        c = np.array([float(r["A_lex_excess"]) for r in rows], dtype=float)
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=34, alpha=0.75, rasterized=True)
        ax.axvline(x_thr, color="black", ls="--", lw=1.5)
        ax.axhline(y_thr, color="black", ls="--", lw=1.5)
        ax.set_title(method, fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        if idx == 0:
            ax.set_ylabel("Probabilistic Anchoring Excess", fontsize=13, fontweight="bold")
        ax.set_xlabel("Entropic Anchoring Excess", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_controlled(args: argparse.Namespace) -> None:
    method_rows = read_jsonl(args.methods_input)
    blind_rows = read_jsonl(args.blind_input)
    blind_by_idx = {int(r["sample_idx"]) if "sample_idx" in r else i: r for i, r in enumerate(blind_rows)}
    out = []
    for idx, row in enumerate(method_rows):
        q = ensure_text(row.get("questions", {}).get("NEU"))
        a = ensure_text(row.get("answers", {}).get("NEU"))
        blind = blind_by_idx.get(idx)
        blind_r = ensure_text(blind.get("reasonings", {}).get(args.blind_method)) if blind else ensure_text(row.get("reasonings", {}).get("NEU"))
        entropy_anchor = function_word_skeleton(a)
        methods = {
            "Blind CoT": blind_r,
            "Response-as-CoT": a,
            "+Prob Anchor": f"{blind_r}\n\n{a}",
            "+Entropy Anchor": entropy_anchor,
        }
        out.append({
            "id": row.get("id", idx),
            "questions": {m: q for m in methods},
            "answers": {m: a for m in methods},
            "contexts": {m: q for m in methods},
            "reasonings": methods,
        })
    write_jsonl(args.output, out)
    print(json.dumps({"rows": len(out), "output": str(args.output)}, indent=2))


def function_word_skeleton(text: str) -> str:
    toks = tokenize_words(text)
    words = [t for t in toks if t in STOPWORDS]
    if not words:
        words = toks[:80]
    chunks = [" ".join(words[i:i + 18]) for i in range(0, len(words), 18)]
    return "\n\n".join(chunks[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--paraphrases", type=Path, required=True)
    p.add_argument("--scoring-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", default="")
    p.add_argument("--tau-g", type=float, default=0.1)
    p.add_argument("--lexical-prefix-ratio", type=float, default=0.85)
    p.add_argument("--debug-errors", action="store_true")
    p.set_defaults(func=score)

    p = sub.add_parser("build-controlled")
    p.add_argument("--methods-input", type=Path, required=True)
    p.add_argument("--blind-input", type=Path, required=True)
    p.add_argument("--blind-method", default="Blind CoT")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=build_controlled)

    p = sub.add_parser("aggregate")
    p.add_argument("--method-metrics", type=Path, required=True)
    p.add_argument("--blind-metrics", type=Path, required=True)
    p.add_argument("--control-metrics", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method-order", default=",".join(METHOD_ORDER))
    p.add_argument("--zone-quantile", type=float, default=0.90)
    p.add_argument("--uninformative-quantile", type=float, default=0.10)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
