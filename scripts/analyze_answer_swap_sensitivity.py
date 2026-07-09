#!/usr/bin/env python3
"""Build and score answer-swap sensitivity inputs for RCoT traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import LCSseq


METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]
DELTA_PAIRS = [("SUP", "NEU"), ("AUG-SUP", "NEU"), ("SSR", "NEU")]
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


def methods_from_rows(rows: Sequence[Dict[str, Any]], requested: Sequence[str]) -> List[str]:
    out = []
    for method in requested:
        if any(method in row.get("questions", {}) for row in rows):
            out.append(method)
    return out


def rebuild_context(question: str, answer: str) -> str:
    return f"User: {question}\nAssistant: {answer}".strip()


def choose_swap_indices(n_rows: int, seed: int) -> List[int]:
    if n_rows <= 1:
        return [0] * n_rows
    rng = random.Random(seed)
    indices = list(range(n_rows))
    swapped = indices[:]
    for _ in range(1000):
        rng.shuffle(swapped)
        if all(i != j for i, j in zip(indices, swapped)):
            return swapped
    return [(i + 1) % n_rows for i in indices]


def parse_delta_pairs(text: str) -> List[Tuple[str, str]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "|" in item:
            method, base = item.split("|", 1)
        elif ":" in item:
            method, base = item.split(":", 1)
        elif "-" in item:
            method, base = item.split("-", 1)
        else:
            raise ValueError(f"Delta pair must be METHOD|BASE or METHOD-BASE, got: {item}")
        out.append((method.strip(), base.strip()))
    return out


def build_swap_input(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    methods = methods_from_rows(rows, [m for m in args.methods.split(",") if m])
    swap_indices = choose_swap_indices(len(rows), args.seed)
    swapped_rows = []
    for idx, row in enumerate(rows):
        swap_idx = swap_indices[idx]
        source = rows[swap_idx]
        questions: Dict[str, str] = {}
        answers: Dict[str, str] = {}
        contexts: Dict[str, str] = {}
        for method in methods:
            q = ensure_text(row.get("questions", {}).get(method))
            a = ensure_text(source.get("answers", {}).get(method))
            if not q.strip() or not a.strip():
                continue
            questions[method] = q
            answers[method] = a
            contexts[method] = rebuild_context(q, a)
        swapped_rows.append(
            {
                "id": f"swap::{idx}::{swap_idx}::{row.get('id', idx)}",
                "questions": questions,
                "answers": answers,
                "contexts": contexts,
                "swap_source_id": source.get("id", swap_idx),
            }
        )
    n = write_jsonl(args.output, swapped_rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": n,
        "methods": methods,
        "seed": args.seed,
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def tokenize_words(text: str) -> List[str]:
    return [tok.lower() for tok in WORD_RE.findall(text)]


def content_words(tokens: Sequence[str]) -> List[str]:
    return [tok for tok in tokens if tok not in STOPWORDS and (len(tok) > 1 or not tok.isascii())]


def ngrams(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def f1(overlap: float, a_total: float, b_total: float) -> float:
    if a_total <= 0 or b_total <= 0 or overlap <= 0:
        return 0.0
    precision = overlap / a_total
    recall = overlap / b_total
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def rouge_l_f1(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    overlap = float(LCSseq.similarity(a, b))
    return f1(overlap, float(len(a)), float(len(b)))


def rouge_n_f1(a: Sequence[str], b: Sequence[str], n: int) -> float:
    ca = ngrams(a, n)
    cb = ngrams(b, n)
    if not ca or not cb:
        return 0.0
    overlap = float(sum((ca & cb).values()))
    return f1(overlap, float(sum(ca.values())), float(sum(cb.values())))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def parse_swap_id(value: Any) -> int | None:
    text = ensure_text(value)
    if not text.startswith("swap::"):
        return None
    parts = text.split("::", 3)
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def collect_texts_for_embeddings(records: Sequence[Dict[str, Any]]) -> Tuple[List[str], Dict[Tuple[int, str, str], int]]:
    texts: List[str] = []
    keys: Dict[Tuple[int, str, str], int] = {}
    for rec in records:
        idx = int(rec["sample_idx"])
        method = str(rec["method"])
        for kind in ("orig", "swap"):
            key = (idx, method, kind)
            keys[key] = len(texts)
            texts.append(ensure_text(rec[f"{kind}_reasoning"]))
    return texts, keys


def add_embedding_similarity(records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if args.no_embeddings or not records:
        return
    from sentence_transformers import SentenceTransformer

    kwargs: Dict[str, Any] = {}
    if args.device:
        kwargs["device"] = args.device
    model = SentenceTransformer(str(args.embedding_model), **kwargs)
    model.max_seq_length = args.max_length
    texts, keys = collect_texts_for_embeddings(records)
    vecs = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    for rec in records:
        idx = int(rec["sample_idx"])
        method = str(rec["method"])
        orig = vecs[keys[(idx, method, "orig")]]
        swap = vecs[keys[(idx, method, "swap")]]
        sim = float(np.dot(orig, swap))
        rec["embedding_cosine"] = sim
        rec["embedding_sensitivity"] = 1.0 - sim


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        return math.nan
    pos = (len(vals) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else math.nan


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int) -> Tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    boots = [statistics.fmean(rng.choice(vals) for _ in vals) for _ in range(n_boot)]
    return percentile(boots, 0.025), percentile(boots, 0.975)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, (int, float)):
                cells.append("NA" if not math.isfinite(float(value)) else f"{float(value):.4f}")
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def score_swap(args: argparse.Namespace) -> None:
    orig_rows = read_jsonl(args.original)
    if args.limit is not None:
        orig_rows = orig_rows[: args.limit]
    swap_rows = read_jsonl(args.swap)
    methods = [m for m in args.methods.split(",") if m]

    swap_by_idx: Dict[int, Dict[str, Any]] = {}
    for row in swap_rows:
        idx = parse_swap_id(row.get("id"))
        if idx is not None:
            swap_by_idx[idx] = row

    records: List[Dict[str, Any]] = []
    for idx, orig in enumerate(orig_rows):
        swap = swap_by_idx.get(idx)
        if not swap:
            continue
        for method in methods:
            orig_r = ensure_text(orig.get("reasonings", {}).get(method))
            swap_r = ensure_text(swap.get("reasonings", {}).get(method))
            if not orig_r.strip() or not swap_r.strip():
                continue
            orig_tokens = tokenize_words(orig_r)
            swap_tokens = tokenize_words(swap_r)
            orig_content = content_words(orig_tokens)
            swap_content = content_words(swap_tokens)
            rec = {
                "sample_idx": idx,
                "id": orig.get("id", idx),
                "swap_id": swap.get("id"),
                "method": method,
                "orig_words": len(orig_tokens),
                "swap_words": len(swap_tokens),
                "orig_reasoning": orig_r,
                "swap_reasoning": swap_r,
                "word_rouge_l_f1": rouge_l_f1(orig_tokens, swap_tokens),
                "word_rouge_2_f1": rouge_n_f1(orig_tokens, swap_tokens, 2),
                "word_rouge_3_f1": rouge_n_f1(orig_tokens, swap_tokens, 3),
                "content_jaccard": jaccard(orig_content, swap_content),
            }
            for key in ["word_rouge_l_f1", "word_rouge_2_f1", "word_rouge_3_f1", "content_jaccard"]:
                rec[f"{key}_sensitivity"] = 1.0 - float(rec[key])
            records.append(rec)

    add_embedding_similarity(records, args)

    metrics = [
        "word_rouge_l_f1_sensitivity",
        "word_rouge_2_f1_sensitivity",
        "word_rouge_3_f1_sensitivity",
        "content_jaccard_sensitivity",
    ]
    if records and "embedding_sensitivity" in records[0]:
        metrics.append("embedding_sensitivity")

    summary_rows = []
    for metric in metrics:
        for method in methods:
            vals = [float(r[metric]) for r in records if r["method"] == method and math.isfinite(float(r.get(metric, math.nan)))]
            summary_rows.append({"metric": metric, "method": method, "n": len(vals), "mean": mean(vals)})

    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_sample[int(rec["sample_idx"])][str(rec["method"])] = rec
    rng = random.Random(args.seed)
    delta_pairs = parse_delta_pairs(args.delta_pairs)
    delta_rows = []
    for metric in metrics:
        for method, base in delta_pairs:
            vals = []
            for sample_methods in by_sample.values():
                if method not in sample_methods or base not in sample_methods:
                    continue
                a = sample_methods[method].get(metric)
                b = sample_methods[base].get(metric)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(float(a)) and math.isfinite(float(b)):
                    vals.append(float(a) - float(b))
            lo, hi = bootstrap_ci(vals, rng, args.bootstrap)
            delta_rows.append(
                {
                    "metric": metric,
                    "delta": f"{method}-{base}",
                    "n": len(vals),
                    "mean_delta": mean(vals),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "ci_excludes_zero": bool(vals) and (lo > 0 or hi < 0),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_fields = [
        "sample_idx", "id", "swap_id", "method", "orig_words", "swap_words",
        "word_rouge_l_f1", "word_rouge_l_f1_sensitivity",
        "word_rouge_2_f1", "word_rouge_2_f1_sensitivity",
        "word_rouge_3_f1", "word_rouge_3_f1_sensitivity",
        "content_jaccard", "content_jaccard_sensitivity",
        "embedding_cosine", "embedding_sensitivity",
    ]
    write_csv(args.output_dir / "per_sample_swap_sensitivity.csv", records, per_sample_fields)
    write_csv(args.output_dir / "summary_by_method.csv", summary_rows, ["metric", "method", "n", "mean"])
    write_csv(args.output_dir / "paired_deltas_vs_neu.csv", delta_rows, ["metric", "delta", "n", "mean_delta", "ci95_low", "ci95_high", "ci_excludes_zero"])

    lines = [f"# Answer-Swap Sensitivity (n={len(by_sample)})", ""]
    lines.append("Higher sensitivity means the regenerated trace changed more after swapping the answer while keeping the question fixed.")
    lines.append("")
    lines.append("## Mean Sensitivity")
    lines.append("")
    table_rows = []
    for metric in metrics:
        row = [metric]
        for method in methods:
            row.append(mean(r["mean"] for r in summary_rows if r["metric"] == metric and r["method"] == method))
        table_rows.append(row)
    lines.append(markdown_table(["metric", *methods], table_rows))
    lines.append("")
    lines.append("## Paired Deltas vs NEU")
    lines.append("")
    lines.append(markdown_table(
        ["metric", "delta", "mean", "95% CI", "excludes 0"],
        [
            [
                row["metric"],
                row["delta"],
                row["mean_delta"],
                f"[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}]",
                row["ci_excludes_zero"],
            ]
            for row in delta_rows
        ],
    ))
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "samples": len(by_sample), "output_dir": str(args.output_dir)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--summary-output", type=Path)
    build.add_argument("--methods", default=",".join(METHODS))
    build.add_argument("--limit", type=int)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(func=build_swap_input)

    score = sub.add_parser("score")
    score.add_argument("--original", type=Path, required=True)
    score.add_argument("--swap", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--methods", default=",".join(METHODS))
    score.add_argument("--delta-pairs", default=",".join(f"{m}-{b}" for m, b in DELTA_PAIRS))
    score.add_argument("--limit", type=int)
    score.add_argument("--embedding-model", type=Path, default=Path("/home/pengguangyue/workspace/models/xlm-roberta-large"))
    score.add_argument("--device", default=None)
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--max-length", type=int, default=512)
    score.add_argument("--no-embeddings", action="store_true")
    score.add_argument("--bootstrap", type=int, default=2000)
    score.add_argument("--seed", type=int, default=13)
    score.set_defaults(func=score_swap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
