#!/usr/bin/env python3
"""Pilot indicators for suppression-induced hidden anchoring.

This script analyzes existing metric-format RCoT traces. It is intentionally
CPU-only and does not call a model: the goal is to cheaply screen for behavioral
signals that suppression prompts may induce answer monitoring even when standard
excess metrics are flat.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]
DELTA_PAIRS = [("SUP", "NEU"), ("AUG-SUP", "NEU"), ("SSR", "NEU")]

WORD_RE = re.compile(r"\b[\w.-]+\b", flags=re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|[a-zA-Z]+)?(?!\w)")
CODE_RE = re.compile(r"`([^`]+)`|(?<!\w)[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?!\w)")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "may",
    "might", "more", "most", "must", "not", "of", "on", "or", "our", "she",
    "should", "so", "such", "than", "that", "the", "their", "then", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your",
}

SUPPRESSION_PATTERNS = [
    r"\bavoid(?:ing)?\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bshould not\b",
    r"\bmust not\b",
    r"\bwithout (?:revealing|stating|mentioning|saying|giving)\b",
    r"\bnot (?:reveal|state|mention|say|give|include|copy)\b",
    r"\bno (?:need to )?(?:reveal|state|mention|directly say)\b",
    r"\bwithhold(?:ing)?\b",
    r"\bhidden\b",
    r"\bleak(?:age|ing)?\b",
    r"\bpeek(?:ed|ing)?\b",
    r"\bsolution section\b",
    r"\bfinal answer\b",
    r"\banswer itself\b",
    r"\bdirectly (?:state|mention|copy|output)\b",
]

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bwon't\b",
    r"\bshouldn't\b",
    r"\bmustn't\b",
]

SELF_CORRECTION_PATTERNS = [
    r"\bwait\b",
    r"\bactually\b",
    r"\brather\b",
    r"\binstead\b",
    r"\bhowever\b",
    r"\bbut\b",
    r"\bon the other hand\b",
    r"\breconsider\b",
    r"\brecheck\b",
    r"\bcorrection\b",
    r"\bwrong\b",
    r"\bmistake\b",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def content_words(text: str) -> List[str]:
    toks = []
    for tok in words(text):
        stripped = tok.strip("._-")
        if not stripped:
            continue
        if stripped in STOPWORDS:
            continue
        if len(stripped) < 3 and not any(ch.isdigit() for ch in stripped):
            continue
        toks.append(stripped)
    return toks


def unique_recall(source: Iterable[str], target: Iterable[str]) -> float:
    src = set(source)
    tgt = set(target)
    if not tgt:
        return math.nan
    return len(src & tgt) / len(tgt)


def idf_recall(reasoning: str, answer: str, idf: Dict[str, float]) -> float:
    answer_toks = set(content_words(answer))
    if not answer_toks:
        return math.nan
    reasoning_toks = set(content_words(reasoning))
    denom = sum(idf.get(tok, 1.0) for tok in answer_toks)
    if denom <= 0:
        return math.nan
    numer = sum(idf.get(tok, 1.0) for tok in answer_toks if tok in reasoning_toks)
    return numer / denom


def idf_overlap_against_tokens(reasoning_tokens: Iterable[str], answer: str, idf: Dict[str, float]) -> float:
    answer_toks = set(content_words(answer))
    if not answer_toks:
        return math.nan
    reasoning_set = set(reasoning_tokens)
    denom = sum(idf.get(tok, 1.0) for tok in answer_toks)
    if denom <= 0:
        return math.nan
    numer = sum(idf.get(tok, 1.0) for tok in answer_toks if tok in reasoning_set)
    return numer / denom


def decoy_answers(rows: Sequence[Dict[str, Any]], sample_idx: int, method: str, k: int) -> List[str]:
    if len(rows) <= 1 or k <= 0:
        return []
    offsets = list(range(1, min(k, len(rows) - 1) + 1))
    out = []
    for offset in offsets:
        decoy_row = rows[(sample_idx + offset) % len(rows)]
        answer = ensure_text(decoy_row.get("answers", {}).get(method))
        if answer.strip():
            out.append(answer)
    return out


def contrastive_specificity(
    reasoning: str,
    answer: str,
    decoys: Sequence[str],
    idf: Dict[str, float],
) -> Dict[str, float]:
    if not decoys:
        return {}
    reasoning_tokens = content_words(reasoning)
    true_plain = unique_recall(reasoning_tokens, content_words(answer))
    true_idf = idf_overlap_against_tokens(reasoning_tokens, answer, idf)
    decoy_plain_vals = [unique_recall(reasoning_tokens, content_words(decoy)) for decoy in decoys]
    decoy_idf_vals = [idf_overlap_against_tokens(reasoning_tokens, decoy, idf) for decoy in decoys]
    decoy_plain = mean(decoy_plain_vals)
    decoy_idf = mean(decoy_idf_vals)
    return {
        "contrastive_content_recall": true_plain - decoy_plain if not math.isnan(true_plain) and not math.isnan(decoy_plain) else math.nan,
        "contrastive_content_idf_recall": true_idf - decoy_idf if not math.isnan(true_idf) and not math.isnan(decoy_idf) else math.nan,
        "decoy_content_recall_mean": decoy_plain,
        "decoy_content_idf_recall_mean": decoy_idf,
    }


def prefix_by_word_fraction(text: str, frac: float) -> str:
    toks = list(WORD_RE.finditer(text))
    if not toks:
        return ""
    n = max(1, math.ceil(len(toks) * frac))
    return text[: toks[min(n, len(toks)) - 1].end()]


def count_patterns(text: str, patterns: Sequence[str]) -> int:
    lower = text.lower()
    return sum(len(re.findall(pattern, lower)) for pattern in patterns)


def per_1k(count: int, n_words: int) -> float:
    return 1000.0 * count / max(n_words, 1)


def zdict_compress_len(payload: bytes, zdict: bytes) -> int:
    if not payload:
        return 0
    if not zdict:
        return len(zlib.compress(payload, level=9))
    zdict = zdict[-32768:]
    compressor = zlib.compressobj(level=9, zdict=zdict)
    return len(compressor.compress(payload) + compressor.flush())


def compression_savings(reasoning: str, answer: str) -> Tuple[float, float]:
    r_bytes = reasoning.encode("utf-8", errors="ignore")
    a_bytes = answer.encode("utf-8", errors="ignore")
    if not r_bytes:
        return math.nan, math.nan
    plain = len(zlib.compress(r_bytes, level=9))
    cond = zdict_compress_len(r_bytes, a_bytes)
    savings = plain - cond
    return savings / len(r_bytes), savings / max(len(a_bytes), 1)


def extract_numbers(text: str) -> List[str]:
    return [m.group(0).lower() for m in NUMBER_RE.finditer(text)]


def extract_code_terms(text: str) -> List[str]:
    out = []
    for match in CODE_RE.finditer(text):
        out.append((match.group(1) or match.group(0)).lower())
    return out


def build_idf(rows: Sequence[Dict[str, Any]], methods: Sequence[str]) -> Dict[str, float]:
    docs = []
    for row in rows:
        for method in methods:
            answer = ensure_text(row.get("answers", {}).get(method))
            reasoning = ensure_text(row.get("reasonings", {}).get(method))
            docs.append(set(content_words(answer)))
            docs.append(set(content_words(reasoning)))
    df = Counter(tok for doc in docs for tok in doc)
    n_docs = max(len(docs), 1)
    return {tok: math.log((n_docs + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}


def load_excess(path: Path | None) -> Dict[Tuple[int, str], Dict[str, float]]:
    if not path:
        return {}
    loaded: Dict[Tuple[int, str], Dict[str, float]] = {}
    for row in read_jsonl(path):
        try:
            sample_idx = int(row["sample_idx"])
            method = str(row["method"])
        except (KeyError, TypeError, ValueError):
            continue
        loaded[(sample_idx, method)] = {
            key: float(row[key])
            for key in ["A_lex_excess", "A_ent_excess", "A_prob_excess", "B"]
            if key in row and row[key] is not None
        }
    return loaded


def analyze_row(
    sample_idx: int,
    row: Dict[str, Any],
    method: str,
    idf: Dict[str, float],
    excess: Dict[Tuple[int, str], Dict[str, float]],
    all_rows: Sequence[Dict[str, Any]],
    n_decoys: int,
) -> Dict[str, Any]:
    answer = ensure_text(row.get("answers", {}).get(method))
    reasoning = ensure_text(row.get("reasonings", {}).get(method))
    wtoks = words(reasoning)
    n_words = len(wtoks)
    ans_content = content_words(answer)
    rea_content = content_words(reasoning)

    sup_count = count_patterns(reasoning, SUPPRESSION_PATTERNS)
    neg_count = count_patterns(reasoning, NEGATION_PATTERNS)
    corr_count = count_patterns(reasoning, SELF_CORRECTION_PATTERNS)
    comp_norm_r, comp_norm_a = compression_savings(reasoning, answer)

    out: Dict[str, Any] = {
        "sample_idx": sample_idx,
        "id": row.get("id", sample_idx),
        "method": method,
        "reasoning_words": n_words,
        "answer_words": len(words(answer)),
        "suppression_marker_count": sup_count,
        "suppression_marker_per_1k": per_1k(sup_count, n_words),
        "negation_count": neg_count,
        "negation_per_1k": per_1k(neg_count, n_words),
        "self_correction_count": corr_count,
        "self_correction_per_1k": per_1k(corr_count, n_words),
        "answer_content_recall": unique_recall(rea_content, ans_content),
        "answer_content_idf_recall": idf_recall(reasoning, answer, idf),
        "answer_number_recall": unique_recall(extract_numbers(reasoning), extract_numbers(answer)),
        "answer_code_recall": unique_recall(extract_code_terms(reasoning), extract_code_terms(answer)),
        "zdict_savings_per_r_byte": comp_norm_r,
        "zdict_savings_per_a_byte": comp_norm_a,
    }
    out.update(contrastive_specificity(reasoning, answer, decoy_answers(all_rows, sample_idx, method, n_decoys), idf))
    for frac in [0.10, 0.25, 0.50]:
        prefix = prefix_by_word_fraction(reasoning, frac)
        tag = str(int(frac * 100))
        out[f"early{tag}_content_recall"] = unique_recall(content_words(prefix), ans_content)
        out[f"early{tag}_content_idf_recall"] = idf_recall(prefix, answer, idf)
    out.update(excess.get((sample_idx, method), {}))
    content_idf = out.get("answer_content_idf_recall")
    content_plain = out.get("answer_content_recall")
    for anchor_metric in ["A_prob_excess", "A_ent_excess", "B"]:
        anchor_val = out.get(anchor_metric)
        if isinstance(anchor_val, (int, float)) and not math.isnan(float(anchor_val)):
            if isinstance(content_idf, (int, float)) and not math.isnan(float(content_idf)):
                out[f"{anchor_metric}_per_content_idf"] = float(anchor_val) / max(float(content_idf), 1e-4)
            if isinstance(content_plain, (int, float)) and not math.isnan(float(content_plain)):
                out[f"{anchor_metric}_per_content_recall"] = float(anchor_val) / max(float(content_plain), 1e-4)
    return out


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return math.nan
    return statistics.fmean(vals)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def paired_deltas(records: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[Tuple[str, str, str], List[float]]:
    return paired_deltas_for_pairs(records, metrics, DELTA_PAIRS)


def paired_deltas_for_pairs(
    records: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    delta_pairs: Sequence[Tuple[str, str]],
) -> Dict[Tuple[str, str, str], List[float]]:
    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_sample[int(rec["sample_idx"])][str(rec["method"])] = rec
    deltas: Dict[Tuple[str, str, str], List[float]] = {}
    for metric in metrics:
        for method, base in delta_pairs:
            vals = []
            for method_rows in by_sample.values():
                if method not in method_rows or base not in method_rows:
                    continue
                a = method_rows[method].get(metric)
                b = method_rows[base].get(metric)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not math.isnan(a) and not math.isnan(b):
                    vals.append(float(a) - float(b))
            deltas[(metric, method, base)] = vals
    return deltas


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


def add_paired_tradeoff_metrics(records: List[Dict[str, Any]]) -> None:
    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_sample[int(rec["sample_idx"])][str(rec["method"])] = rec

    anchor_metrics = ["A_prob_excess", "A_ent_excess", "B"]
    lexical_metrics = ["answer_content_idf_recall", "answer_content_recall"]
    for method_rows in by_sample.values():
        neu = method_rows.get("NEU")
        if not neu:
            continue
        for method, rec in method_rows.items():
            for anchor in anchor_metrics:
                a = rec.get(anchor)
                a0 = neu.get(anchor)
                if not (isinstance(a, (int, float)) and isinstance(a0, (int, float))):
                    continue
                if math.isnan(float(a)) or math.isnan(float(a0)):
                    continue
                for lex in lexical_metrics:
                    l = rec.get(lex)
                    l0 = neu.get(lex)
                    if not (isinstance(l, (int, float)) and isinstance(l0, (int, float))):
                        continue
                    if math.isnan(float(l)) or math.isnan(float(l0)):
                        continue
                    anchor_retention = float(a) / max(abs(float(a0)), 1e-4)
                    lexical_retention = float(l) / max(abs(float(l0)), 1e-4)
                    rec[f"{anchor}_retention_minus_{lex}_retention"] = anchor_retention - lexical_retention
                    rec[f"{anchor}_retention_over_{lex}_retention"] = anchor_retention / max(lexical_retention, 1e-4)


def add_within_sample_gap_metrics(records: List[Dict[str, Any]]) -> None:
    by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_sample[int(rec["sample_idx"])].append(rec)

    pairs = [
        ("A_prob_excess", "answer_content_idf_recall"),
        ("A_prob_excess", "answer_content_recall"),
        ("A_ent_excess", "answer_content_idf_recall"),
        ("A_ent_excess", "answer_content_recall"),
        ("B", "answer_content_idf_recall"),
        ("B", "answer_content_recall"),
    ]
    for sample_records in by_sample.values():
        for anchor, lex in pairs:
            anchor_vals = [float(r[anchor]) for r in sample_records if isinstance(r.get(anchor), (int, float)) and not math.isnan(float(r[anchor]))]
            lex_vals = [float(r[lex]) for r in sample_records if isinstance(r.get(lex), (int, float)) and not math.isnan(float(r[lex]))]
            if len(anchor_vals) < 2 or len(lex_vals) < 2:
                continue
            anchor_mean = statistics.fmean(anchor_vals)
            lex_mean = statistics.fmean(lex_vals)
            anchor_sd = statistics.pstdev(anchor_vals) or 1.0
            lex_sd = statistics.pstdev(lex_vals) or 1.0
            for rec in sample_records:
                a = rec.get(anchor)
                l = rec.get(lex)
                if not (isinstance(a, (int, float)) and isinstance(l, (int, float))):
                    continue
                if math.isnan(float(a)) or math.isnan(float(l)):
                    continue
                rec[f"{anchor}_minus_{lex}_within_sample_zgap"] = (
                    (float(a) - anchor_mean) / anchor_sd
                    - (float(l) - lex_mean) / lex_sd
                )


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    boots = []
    for _ in range(n_boot):
        boots.append(statistics.fmean(rng.choice(vals) for _ in vals))
    return percentile(boots, 0.025), percentile(boots, 0.975)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--excess", type=Path)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delta-pairs", default=",".join(f"{m}-{b}" for m, b in DELTA_PAIRS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--decoys", type=int, default=8)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    idf = build_idf(rows, methods)
    excess = load_excess(args.excess)
    delta_pairs = parse_delta_pairs(args.delta_pairs)

    records: List[Dict[str, Any]] = []
    for sample_idx, row in enumerate(rows):
        for method in methods:
            if ensure_text(row.get("reasonings", {}).get(method)).strip():
                records.append(analyze_row(sample_idx, row, method, idf, excess, rows, args.decoys))
    add_paired_tradeoff_metrics(records)
    add_within_sample_gap_metrics(records)

    metrics = [
        "reasoning_words",
        "suppression_marker_per_1k",
        "negation_per_1k",
        "self_correction_per_1k",
        "answer_content_recall",
        "answer_content_idf_recall",
        "contrastive_content_recall",
        "contrastive_content_idf_recall",
        "decoy_content_recall_mean",
        "decoy_content_idf_recall_mean",
        "early10_content_idf_recall",
        "early25_content_idf_recall",
        "early50_content_idf_recall",
        "answer_number_recall",
        "answer_code_recall",
        "zdict_savings_per_r_byte",
        "zdict_savings_per_a_byte",
        "A_prob_excess_per_content_idf",
        "A_ent_excess_per_content_idf",
        "B_per_content_idf",
        "A_prob_excess_per_content_recall",
        "A_ent_excess_per_content_recall",
        "B_per_content_recall",
        "A_prob_excess_retention_minus_answer_content_idf_recall_retention",
        "A_prob_excess_retention_minus_answer_content_recall_retention",
        "A_ent_excess_retention_minus_answer_content_idf_recall_retention",
        "A_ent_excess_retention_minus_answer_content_recall_retention",
        "B_retention_minus_answer_content_idf_recall_retention",
        "B_retention_minus_answer_content_recall_retention",
        "A_prob_excess_retention_over_answer_content_idf_recall_retention",
        "A_prob_excess_retention_over_answer_content_recall_retention",
        "A_ent_excess_retention_over_answer_content_idf_recall_retention",
        "A_ent_excess_retention_over_answer_content_recall_retention",
        "B_retention_over_answer_content_idf_recall_retention",
        "B_retention_over_answer_content_recall_retention",
        "A_prob_excess_minus_answer_content_idf_recall_within_sample_zgap",
        "A_prob_excess_minus_answer_content_recall_within_sample_zgap",
        "A_ent_excess_minus_answer_content_idf_recall_within_sample_zgap",
        "A_ent_excess_minus_answer_content_recall_within_sample_zgap",
        "B_minus_answer_content_idf_recall_within_sample_zgap",
        "B_minus_answer_content_recall_within_sample_zgap",
        "A_lex_excess",
        "A_ent_excess",
        "A_prob_excess",
        "B",
    ]
    metrics = [m for m in metrics if any(m in rec for rec in records)]

    summary_rows = []
    for metric in metrics:
        for method in methods:
            vals = [float(rec[metric]) for rec in records if rec["method"] == method and metric in rec and isinstance(rec[metric], (int, float)) and not math.isnan(float(rec[metric]))]
            summary_rows.append({"metric": metric, "method": method, "n": len(vals), "mean": mean(vals)})

    rng = random.Random(args.seed)
    delta_values = paired_deltas_for_pairs(records, metrics, delta_pairs)
    delta_rows = []
    for (metric, method, base), vals in delta_values.items():
        lo, hi = bootstrap_ci(vals, rng, args.bootstrap)
        delta_rows.append(
            {
                "metric": metric,
                "delta": f"{method}-{base}",
                "n": len(vals),
                "mean_delta": mean(vals),
                "ci95_low": lo,
                "ci95_high": hi,
                "support_direction": "positive" if mean(vals) > 0 else "negative_or_flat",
                "ci_excludes_zero": bool(vals) and (lo > 0 or hi < 0),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_fields = sorted({k for rec in records for k in rec})
    write_csv(args.output_dir / "per_sample_indicators.csv", records, all_fields)
    write_csv(args.output_dir / "summary_by_method.csv", summary_rows, ["metric", "method", "n", "mean"])
    write_csv(args.output_dir / "paired_deltas_vs_neu.csv", delta_rows, ["metric", "delta", "n", "mean_delta", "ci95_low", "ci95_high", "support_direction", "ci_excludes_zero"])

    interesting = [
        "suppression_marker_per_1k",
        "negation_per_1k",
        "self_correction_per_1k",
        "contrastive_content_idf_recall",
        "contrastive_content_recall",
        "early25_content_idf_recall",
        "zdict_savings_per_r_byte",
        "A_prob_excess_per_content_idf",
        "A_ent_excess_per_content_idf",
        "B_per_content_idf",
        "A_prob_excess_retention_minus_answer_content_idf_recall_retention",
        "A_ent_excess_retention_minus_answer_content_idf_recall_retention",
        "B_retention_minus_answer_content_idf_recall_retention",
        "A_prob_excess_minus_answer_content_idf_recall_within_sample_zgap",
        "A_ent_excess_minus_answer_content_idf_recall_within_sample_zgap",
        "B_minus_answer_content_idf_recall_within_sample_zgap",
        "A_prob_excess",
    ]
    interesting = [m for m in interesting if m in metrics]
    md = [
        f"# Suppression-support indicators (n={len(rows)})",
        "",
        "## Means by method",
        markdown_table(
            ["metric", *methods],
            [
                [
                    metric,
                    *[
                        next((row["mean"] for row in summary_rows if row["metric"] == metric and row["method"] == method), math.nan)
                        for method in methods
                    ],
                ]
                for metric in interesting
            ],
        ),
        "",
        "## Paired deltas vs NEU",
        markdown_table(
            ["metric", "delta", "mean_delta", "ci95_low", "ci95_high", "ci_excludes_zero"],
            [
                [row["metric"], row["delta"], row["mean_delta"], row["ci95_low"], row["ci95_high"], row["ci_excludes_zero"]]
                for row in delta_rows
                if row["metric"] in interesting
            ],
        ),
        "",
        "Positive SUP-NEU or AUG-SUP-NEU deltas on monitoring/compression/early-leakage indicators are candidate support for suppression-induced hidden anchoring.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "records": len(records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
