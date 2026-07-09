#!/usr/bin/env python3
"""Compute/aggregate the claude.traj.md three anchoring metrics.

Definitions:
- A_lex: IDF-weighted recall of answer content words in reasoning.
- A_prob: clipped normalized answer-surprisal reduction, multiplied by 100 in
  summary tables.
- A_traj: 100 * ConfidenceGap_mean from the mismatch diagnostic run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from anchoring_measure import metrics as base_metrics  # noqa: E402


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for", "with",
    "as", "by", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "from", "at", "into", "about", "we", "you", "i", "he", "she", "they", "them", "our",
    "your", "their", "not", "no", "yes", "do", "does", "did", "can", "could", "would", "should",
    "will", "may", "might", "must", "have", "has", "had", "there", "here", "which", "what", "when",
    "where", "why", "how", "also", "than", "therefore", "thus", "because",
}
LEXICAL_METRIC_KEYS = [
    "A_lex",
    "A_lex_E25",
    "A_lex_E50",
    "A_lex_QF",
    "A_lex_QF_E25",
    "A_lex_QF_E50",
]
METHOD_ORDER = [
    "R0 / Blind CoT",
    "NEU",
    "SUP",
    "AUG-SUP",
    "QA-SUP",
    "PG-SUP",
    "SSR",
    "SSR_PLUS",
    "SSR_PLUS_STRUCT",
    "SSR_PLUS_STRUCT_BALANCED",
    "SSR_PLUS_STRUCT_C1_STEPS",
    "SSR_PLUS_STRUCT_LONG",
    "SSR_QSKEL",
    "SSR_2STEP_QA",
    "A1-FDB",
    "A6-Gist",
    "B11-NGramBlock",
    "B13-BoN",
    "B12-Contrastive-g0.5",
    "CV-SUP",
    "DL-SUP",
    "FS-SUP",
    "SSR_PLUS_STRUCT_COMPACT",
    "SSR_PLUS_STRUCT_MID",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl_files(path: Path, pattern: str = "*.jsonl") -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        yield from sorted(path.glob(pattern))


def is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
    )


def is_word_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category[0] in {"L", "N", "M"} or ch == "_"


def has_letter_or_number(token: str) -> bool:
    return any(unicodedata.category(ch)[0] in {"L", "N"} for ch in token)


def tokenize_words(text: str) -> List[str]:
    tokens: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            tokens.append("".join(buf).lower())
            buf.clear()

    for ch in text:
        if ch.isspace():
            flush()
        elif is_cjk_char(ch):
            flush()
            tokens.append(ch.lower())
        elif is_word_char(ch):
            buf.append(ch)
        else:
            flush()
            if not unicodedata.category(ch).startswith(("P", "S", "Z", "C")):
                tokens.append(ch.lower())
    flush()
    return tokens


def content_words(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if tok in STOPWORDS or not has_letter_or_number(tok):
            continue
        if len(tok) == 1 and not (tok.isdigit() or is_cjk_char(tok)):
            continue
        out.append(tok)
    return out


def prefix_fraction(tokens: Sequence[str], fraction: float | None) -> Sequence[str]:
    if fraction is None:
        return tokens
    if not tokens or fraction <= 0:
        return []
    limit = max(1, math.ceil(len(tokens) * fraction))
    return tokens[:limit]


def build_idf(rows: Sequence[Dict[str, Any]], methods: Sequence[str]) -> Dict[str, float]:
    docs = []
    for row in rows:
        for method in methods:
            answer = str(row.get("answers", {}).get(method, ""))
            if answer.strip():
                docs.append(set(content_words(tokenize_words(answer))))
    df = Counter(tok for doc in docs for tok in doc)
    n_docs = max(1, len(docs))
    return {tok: math.log((n_docs + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}


def content_idf_recall(
    reasoning: str,
    answer: str,
    idf: Dict[str, float],
    *,
    question: str = "",
    reasoning_fraction: float | None = None,
    filter_question: bool = False,
) -> float:
    answer_tokens = content_words(tokenize_words(answer))
    if filter_question:
        question_tokens = set(content_words(tokenize_words(question)))
        answer_tokens = [tok for tok in answer_tokens if tok not in question_tokens]
    reasoning_tokens = content_words(tokenize_words(reasoning))
    reasoning_counts = Counter(prefix_fraction(reasoning_tokens, reasoning_fraction))
    answer_counts = Counter(answer_tokens)
    denom = sum(idf.get(tok, 1.0) * count for tok, count in answer_counts.items())
    if denom <= 0:
        return 0.0
    numer = 0.0
    for tok, count in answer_counts.items():
        numer += idf.get(tok, 1.0) * min(count, reasoning_counts.get(tok, 0))
    return numer / denom


def method_order(methods: Iterable[str]) -> List[str]:
    seen = set(methods)
    ordered = [method for method in METHOD_ORDER if method in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def get_rank_info() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", int(os.environ["RANK"])))
    return 0, 1, 0


def compute_aprob_for_method(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_ctx: int | None,
    think_open_ids: torch.LongTensor,
    think_close_ids: torch.LongTensor,
    reason_sep_ids: torch.LongTensor,
    method: str,
    row: Dict[str, Any],
    sample_idx: int,
) -> Dict[str, Any] | None:
    q = row.get("questions", {}).get(method)
    a = row.get("answers", {}).get(method)
    reasoning = row.get("reasonings", {}).get(method, "")
    if not (q and a and reasoning):
        return None

    ans_ids = base_metrics.tok_1d(tokenizer, str(a), device)
    if int(ans_ids.numel()) <= 0:
        return None

    chat_prefix_ids_q = base_metrics.tok_1d(tokenizer, base_metrics.apply_chat_prefix(tokenizer, str(q)), device)
    prefix_ids_q = torch.cat([chat_prefix_ids_q, think_open_ids], dim=0)
    past_prefix_q, past_len_q = base_metrics.prepare_prefix_past(model, prefix_ids_q) if base_metrics.USE_SHARED_PREFIX_PAST else (None, int(prefix_ids_q.numel()))
    n_parts, flat_r, end_pos = base_metrics.tokenize_reasoning_to_flat_ids(
        tokenizer,
        str(reasoning),
        reason_sep_ids,
        base_metrics.MAX_STEPS_PER_SAMPLE,
        device,
    )
    if int(flat_r.numel()) <= 0:
        return None

    # A_prob only needs the baseline P(A|Q) and final P(A|Q,R). The old
    # anchoring script samples many intermediate prefixes because it also
    # supports trajectory/curve analyses; doing that here would waste most of
    # the scorer time.
    step_idxs = [0, n_parts]
    _, xs, ys_total, ans_len = base_metrics.compute_absolute_curve_for_R_sampled(
        model,
        tokenizer,
        prefix_ids_q,
        past_prefix_q,
        past_len_q,
        think_close_ids,
        ans_ids,
        max_ctx,
        flat_r,
        end_pos,
        step_idxs,
        base_metrics.STEP_MICROBATCH,
    )
    if not xs or ans_len <= 0:
        return None
    ys_rate = [(y / ans_len) * base_metrics.LOG10_TO_BITS for y in ys_total]
    aprob = base_metrics.compute_probabilistic_anchoring(ys_rate)["Aprob"]
    if not math.isfinite(float(aprob)):
        return None
    return {
        "sample_idx": sample_idx,
        "method": method,
        "Aprob": float(aprob),
        "B_100": float((ys_rate[-1] - ys_rate[0]) * 100.0) if len(ys_rate) >= 2 else 0.0,
        "answer_tokens": int(ans_len),
        "reasoning_tokens": int(flat_r.numel()),
    }


def score(args: argparse.Namespace) -> None:
    if args.disable_shared_prefix_past:
        base_metrics.USE_SHARED_PREFIX_PAST = False

    rank, world_size, local_rank = get_rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    all_rows = read_jsonl(args.input)
    start_index = int(getattr(args, "start_index", 0) or 0)
    end_index = int(args.limit) if args.limit is not None else len(all_rows)
    rows = all_rows[start_index:end_index]
    methods = [m for m in args.methods.split(",") if m] if args.methods else method_order({m for row in rows for m in row.get("questions", {})})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_tag = f"_s{start_index}_e{end_index}" if start_index or args.limit is not None else ""
    metrics_path = args.output_dir / f"prob_metrics_rank{rank}{chunk_tag}.jsonl"
    invalid_path = args.output_dir / f"invalid_rank{rank}{chunk_tag}.jsonl"
    print(f"[rank{rank}] loading model: {args.scoring_model}", flush=True)
    cfg = AutoConfig.from_pretrained(args.scoring_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.scoring_model,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=base_metrics.TORCH_DTYPE,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.scoring_model,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=base_metrics.TORCH_DTYPE,
            low_cpu_mem_usage=True,
        )
    model.to(device)
    model.eval()
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    think_open_ids = base_metrics.tok_1d(tokenizer, "<think>\n", device)
    think_close_ids = base_metrics.tok_1d(tokenizer, "</think>\n\n", device)
    reason_sep_ids = base_metrics.tok_1d(tokenizer, base_metrics.SEQ_SEP, device)

    total = valid = invalid = 0
    with metrics_path.open("w", encoding="utf-8") as fw, invalid_path.open("w", encoding="utf-8") as ferr:
        for sample_idx, row in tqdm(
            [(i, r) for i, r in enumerate(rows, start=start_index) if i % world_size == rank],
            desc=f"prob-score rank{rank}",
            disable=(rank != 0),
        ):
            for method in methods:
                if not str(row.get("questions", {}).get(method, "")).strip():
                    continue
                total += 1
                try:
                    record = compute_aprob_for_method(
                        model,
                        tokenizer,
                        device,
                        max_ctx,
                        think_open_ids,
                        think_close_ids,
                        reason_sep_ids,
                        method,
                        row,
                        sample_idx,
                    )
                    if record is None:
                        invalid += 1
                        ferr.write(json.dumps({"sample_idx": sample_idx, "method": method, "reason": "empty_or_failed"}) + "\n")
                    else:
                        valid += 1
                        fw.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    invalid += 1
                    ferr.write(json.dumps({"sample_idx": sample_idx, "method": method, "reason": type(exc).__name__, "message": str(exc)}) + "\n")
                    if args.debug_errors:
                        traceback.print_exc()
                fw.flush()
                ferr.flush()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(json.dumps({"rank": rank, "total": total, "valid": valid, "invalid": invalid, "output": str(metrics_path)}, ensure_ascii=False), flush=True)


def read_prob_metrics(path: Path) -> Dict[tuple[int, str], Dict[str, Any]]:
    out: Dict[tuple[int, str], Dict[str, Any]] = {}
    for file in iter_jsonl_files(path, "prob_metrics_rank*.jsonl"):
        for row in read_jsonl(file):
            out[(int(row["sample_idx"]), str(row["method"]))] = row
    return out


def read_traj(path: Path) -> Dict[str, tuple[int, float]]:
    out: Dict[str, tuple[int, float]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == "ConfidenceGap_mean":
                out[str(row["method"])] = (int(float(row["n"])), 100.0 * float(row["mean"]))
    return out


def aggregate(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    methods = [m for m in args.methods.split(",") if m] if args.methods else method_order({m for row in rows for m in row.get("questions", {})})
    idf = build_idf(rows, methods)
    prob = read_prob_metrics(args.prob_metrics)
    traj = read_traj(args.traj_csv)

    per_record: List[Dict[str, Any]] = []
    for sample_idx, row in enumerate(rows):
        for method in methods:
            q = str(row.get("questions", {}).get(method, ""))
            a = str(row.get("answers", {}).get(method, ""))
            r = str(row.get("reasonings", {}).get(method, ""))
            if not (q.strip() and a.strip() and r.strip()):
                continue
            p = prob.get((sample_idx, method))
            if not p:
                continue
            lexical_metrics = {
                "A_lex": content_idf_recall(r, a, idf),
                "A_lex_E25": content_idf_recall(r, a, idf, reasoning_fraction=0.25),
                "A_lex_E50": content_idf_recall(r, a, idf, reasoning_fraction=0.50),
                "A_lex_QF": content_idf_recall(r, a, idf, question=q, filter_question=True),
                "A_lex_QF_E25": content_idf_recall(
                    r,
                    a,
                    idf,
                    question=q,
                    reasoning_fraction=0.25,
                    filter_question=True,
                ),
                "A_lex_QF_E50": content_idf_recall(
                    r,
                    a,
                    idf,
                    question=q,
                    reasoning_fraction=0.50,
                    filter_question=True,
                ),
            }
            per_record.append(
                {
                    "sample_idx": sample_idx,
                    "method": method,
                    **lexical_metrics,
                    "A_prob": float(p["Aprob"]),
                    "B_100": float(p.get("B_100", float("nan"))),
                    "reasoning_tokens": int(p.get("reasoning_tokens", 0)),
                }
            )

    summary_rows: List[Dict[str, Any]] = []
    for method in method_order({r["method"] for r in per_record}):
        vals = [r for r in per_record if r["method"] == method]
        if not vals:
            continue
        n_traj, a_traj = traj.get(method, (0, float("nan")))
        summary_rows.append(
            {
                "Method": method,
                "N": len(vals),
                **{
                    key: 100.0 * float(np.mean([float(r[key]) for r in vals]))
                    for key in LEXICAL_METRIC_KEYS
                },
                "A_traj": a_traj,
                "A_traj_N": n_traj,
                "A_prob": 100.0 * float(np.mean([float(r["A_prob"]) for r in vals])),
                "B_100": float(np.mean([float(r["B_100"]) for r in vals if math.isfinite(float(r["B_100"]))])),
                "Avg_R_tokens": float(np.mean([int(r["reasoning_tokens"]) for r in vals])),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_fields = ["sample_idx", "method", *LEXICAL_METRIC_KEYS, "A_prob", "B_100", "reasoning_tokens"]
    with (args.output_dir / "qwen3_4b_traj_three_metrics_per_record.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields)
        writer.writeheader()
        writer.writerows(per_record)
    fields = ["Method", "N", *LEXICAL_METRIC_KEYS, "A_traj", "A_traj_N", "A_prob", "B_100", "Avg_R_tokens"]
    with (args.output_dir / "qwen3_4b_traj_three_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    md_lines = [
        "| Method | N | A_lex | A_lex_E50 | A_lex_QF | A_lex_QF_E50 | A_traj | A_traj N | A_prob | B_100 | Avg R tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['Method']} | {row['N']} | {row['A_lex']:.3f} | {row['A_lex_E50']:.3f} | "
            f"{row['A_lex_QF']:.3f} | {row['A_lex_QF_E50']:.3f} | {row['A_traj']:.6f} | "
            f"{row['A_traj_N']} | {row['A_prob']:.3f} | {row['B_100']:.3f} | {row['Avg_R_tokens']:.1f} |"
        )
    (args.output_dir / "qwen3_4b_traj_three_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    lex_lines = [
        "| Method | N | A_lex | A_lex_E25 | A_lex_E50 | A_lex_QF | A_lex_QF_E25 | A_lex_QF_E50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lex_lines.append(
            f"| {row['Method']} | {row['N']} | {row['A_lex']:.3f} | {row['A_lex_E25']:.3f} | "
            f"{row['A_lex_E50']:.3f} | {row['A_lex_QF']:.3f} | {row['A_lex_QF_E25']:.3f} | "
            f"{row['A_lex_QF_E50']:.3f} |"
        )
    (args.output_dir / "qwen3_4b_lexical_diagnostics.md").write_text("\n".join(lex_lines) + "\n", encoding="utf-8")

    manifest = {
        "definition_alignment": {
            "source_plan": "/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/insights/claude.traj.md",
            "A_lex": "Content_IDF: IDF-weighted answer-content recall in reasoning, reported as 100 * mean. Unicode alphabetic spans are tokenized as words rather than single characters.",
            "A_lex_E25": "Same lexical recall restricted to the first 25% of reasoning content tokens.",
            "A_lex_E50": "Same lexical recall restricted to the first 50% of reasoning content tokens.",
            "A_lex_QF": "Question-filtered lexical recall after removing answer content terms that already appear in the question.",
            "A_lex_QF_E25": "Question-filtered lexical recall restricted to the first 25% of reasoning content tokens.",
            "A_lex_QF_E50": "Question-filtered lexical recall restricted to the first 50% of reasoning content tokens.",
            "A_traj": "100 * ConfidenceGap_mean from Qwen3-4B-Thinking-2507 mismatch diagnostics, matching the percentage-point reporting scale used for A_lex and A_prob.",
            "A_prob": "clipped normalized answer-surprisal reduction from Qwen3-4B-Thinking-2507 scorer, reported as 100 * mean.",
            "B_100": "Appendix robustness value: raw bit gain * 100, unnormalized.",
        },
        "input": str(args.input),
        "prob_metrics": str(args.prob_metrics),
        "traj_csv": str(args.traj_csv),
        "records": len(per_record),
        "methods": [row["Method"] for row in summary_rows],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "records": len(per_record), "methods": len(summary_rows)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score-prob")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--scoring-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", default="")
    p.add_argument("--start-index", type=int, default=0, help="Start row offset for resumable chunk scoring; sample_idx keeps this global offset.")
    p.add_argument("--limit", type=int)
    p.add_argument("--disable-shared-prefix-past", action="store_true")
    p.add_argument("--debug-errors", action="store_true")
    p.set_defaults(func=score)

    p = sub.add_parser("aggregate")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--prob-metrics", type=Path, required=True)
    p.add_argument("--traj-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", default="")
    p.set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
