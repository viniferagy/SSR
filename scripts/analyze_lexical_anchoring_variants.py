#!/usr/bin/env python3
"""Compare lexical anchoring variants for metric-format SSR rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from tqdm import tqdm
from rapidfuzz.distance import LCSseq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
DEFAULT_EMBED_MODEL = "/home/pengguangyue/workspace/models/xlm-roberta-large"

NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?")
CODE_RE = re.compile(r"`([^`]+)`|<[^>\s]+>|[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for", "with",
    "as", "by", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "from", "at", "into", "about", "we", "you", "i", "he", "she", "they", "them", "our",
    "your", "their", "not", "no", "yes", "do", "does", "did", "can", "could", "would", "should",
    "will", "may", "might", "must", "have", "has", "had", "there", "here", "which", "what", "when",
    "where", "why", "how", "also", "than", "therefore", "thus", "because",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def prf(overlap: float, pred_total: float, ref_total: float) -> tuple[float, float, float]:
    p = overlap / pred_total if pred_total > 0 else 0.0
    r = overlap / ref_total if ref_total > 0 else 0.0
    f = 2.0 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def ngrams(tokens: Sequence[str], n: int) -> List[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(candidate: Sequence[str], reference: Sequence[str], n: int) -> tuple[float, float, float]:
    cand = Counter(ngrams(candidate, n))
    ref = Counter(ngrams(reference, n))
    if not cand or not ref:
        return 0.0, 0.0, 0.0
    overlap = sum((cand & ref).values())
    return prf(float(overlap), float(sum(cand.values())), float(sum(ref.values())))


def rouge_l(candidate: Sequence[str], reference: Sequence[str]) -> tuple[float, float, float]:
    overlap = float(LCSseq.similarity(candidate, reference))
    return prf(overlap, float(len(candidate)), float(len(reference)))


def char_lcs_recall(candidate: str, reference: str) -> float:
    max_chars = 4096
    if len(candidate) > max_chars:
        candidate = candidate[-max_chars:]
    if len(reference) > max_chars:
        reference = reference[-max_chars:]
    cand = list(candidate.lower())
    ref = list(reference.lower())
    return lcs_len(cand, ref) / len(ref) if ref else 0.0


def char_ngram_recall(candidate: str, reference: str, n: int = 3) -> float:
    cand = candidate.lower()
    ref = reference.lower()
    cand_ngrams = Counter(cand[i : i + n] for i in range(max(0, len(cand) - n + 1)))
    ref_ngrams = Counter(ref[i : i + n] for i in range(max(0, len(ref) - n + 1)))
    if not ref_ngrams:
        return 0.0
    return sum((cand_ngrams & ref_ngrams).values()) / sum(ref_ngrams.values())


def containment(candidate_norm: str, reference_norm: str) -> float:
    if not reference_norm:
        return 0.0
    return 1.0 if reference_norm in candidate_norm else 0.0


def unique_recall(candidate: Sequence[str], reference: Sequence[str]) -> float:
    ref = set(reference)
    if not ref:
        return 0.0
    return len(set(candidate) & ref) / len(ref)


def weighted_recall(candidate: Sequence[str], reference: Sequence[str], idf: Dict[str, float]) -> float:
    ref = Counter(reference)
    cand = Counter(candidate)
    denom = sum(idf.get(tok, 1.0) * count for tok, count in ref.items())
    if denom <= 0:
        return 0.0
    overlap = 0.0
    for tok, count in ref.items():
        overlap += idf.get(tok, 1.0) * min(count, cand.get(tok, 0))
    return overlap / denom


def content_idf_recall(
    reasoning: str,
    answer: str,
    idf: Dict[str, float],
    *,
    question: str = "",
    reasoning_fraction: float | None = None,
    filter_question: bool = False,
) -> float:
    answer_content = content_words(tokenize_words(answer))
    if filter_question:
        question_content = set(content_words(tokenize_words(question)))
        answer_content = [tok for tok in answer_content if tok not in question_content]
    reasoning_content = prefix_fraction(content_words(tokenize_words(reasoning)), reasoning_fraction)
    return weighted_recall(reasoning_content, answer_content, idf)


def extract_numbers(text: str) -> List[str]:
    return [m.group(0).lower() for m in NUMBER_RE.finditer(text)]


def extract_code_terms(text: str) -> List[str]:
    out = []
    for match in CODE_RE.finditer(text):
        out.append((match.group(1) or match.group(0)).lower())
    return out


def build_idf(rows: List[Dict[str, Any]], methods: Sequence[str]) -> Dict[str, float]:
    docs = []
    for row in rows:
        for method in methods:
            answer = str(row.get("answers", {}).get(method, ""))
            docs.append(set(content_words(tokenize_words(answer))))
    df = Counter(tok for doc in docs for tok in doc)
    n_docs = max(1, len(docs))
    return {tok: math.log((n_docs + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}


def load_existing_lexical(metrics_dir: Path) -> Dict[tuple[int, str], float]:
    values: Dict[tuple[int, str], float] = {}
    if not metrics_dir.exists():
        return values
    files = sorted(metrics_dir.glob("metrics_rank*.jsonl"))
    for path in files:
        for line in path.open("r", encoding="utf-8"):
            row = json.loads(line)
            if "sample_idx" in row and "method" in row and "lexical_anchoring" in row:
                values[(int(row["sample_idx"]), str(row["method"]))] = float(row["lexical_anchoring"])
    return values


def load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def token_lcs_recall(tokenizer, reasoning: str, answer: str) -> float:
    if tokenizer is None:
        return float("nan")
    reasoning_ids = tokenizer(reasoning, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not reasoning_ids or not answer_ids:
        return 0.0
    return float(LCSseq.similarity(reasoning_ids, answer_ids) / len(answer_ids))


def token_lengths(tokenizer, reasoning: str, answer: str) -> tuple[float, float]:
    if tokenizer is None:
        return float("nan"), float("nan")
    return (
        float(len(tokenizer(reasoning, add_special_tokens=False).input_ids)),
        float(len(tokenizer(answer, add_special_tokens=False).input_ids)),
    )


def pair_records(
    rows: List[Dict[str, Any]],
    methods: Sequence[str],
    existing_lex: Dict[tuple[int, str], float],
    idf: Dict[str, float],
    tokenizer=None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        for method in methods:
            answer = str(row.get("answers", {}).get(method, ""))
            question = str(row.get("questions", {}).get(method, ""))
            reasoning = str(row.get("reasonings", {}).get(method, ""))
            if not answer.strip() or not reasoning.strip():
                continue

            ans_tok = tokenize_words(answer)
            rea_tok = tokenize_words(reasoning)
            ans_content = content_words(ans_tok)
            rea_content = content_words(rea_tok)
            norm_answer = " ".join(ans_tok)
            norm_reason = " ".join(rea_tok)

            rl_p, rl_r, rl_f = rouge_l(rea_tok, ans_tok)
            r1_p, r1_r, r1_f = rouge_n(rea_tok, ans_tok, 1)
            r2_p, r2_r, r2_f = rouge_n(rea_tok, ans_tok, 2)
            r3_p, r3_r, r3_f = rouge_n(rea_tok, ans_tok, 3)
            c2_p, c2_r, c2_f = rouge_n(rea_content, ans_content, 2)
            c3_p, c3_r, c3_f = rouge_n(rea_content, ans_content, 3)

            numbers_answer = extract_numbers(answer)
            numbers_reasoning = extract_numbers(reasoning)
            code_answer = extract_code_terms(answer)
            code_reasoning = extract_code_terms(reasoning)

            reasoning_tokens, answer_tokens = token_lengths(tokenizer, reasoning, answer)
            computed_token_lcs = token_lcs_recall(tokenizer, reasoning, answer)
            if not np.isfinite(computed_token_lcs):
                computed_token_lcs = existing_lex.get((idx, method), float("nan"))

            record = {
                "sample_idx": idx,
                "id": row.get("id", idx),
                "method": method,
                "answer_words": len(ans_tok),
                "reasoning_words": len(rea_tok),
                "answer_tokens": answer_tokens,
                "reasoning_tokens": reasoning_tokens,
                "current_token_lcs_recall": computed_token_lcs,
                "char_3gram_recall": char_ngram_recall(reasoning, answer, 3),
                "word_rouge_l_precision": rl_p,
                "word_rouge_l_recall": rl_r,
                "word_rouge_l_f1": rl_f,
                "rouge_1_precision": r1_p,
                "rouge_1_recall": r1_r,
                "rouge_1_f1": r1_f,
                "rouge_2_precision": r2_p,
                "rouge_2_recall": r2_r,
                "rouge_2_f1": r2_f,
                "rouge_3_precision": r3_p,
                "rouge_3_recall": r3_r,
                "rouge_3_f1": r3_f,
                "content_unigram_recall": unique_recall(rea_content, ans_content),
                "content_idf_recall": content_idf_recall(reasoning, answer, idf),
                "content_idf_recall_early25": content_idf_recall(reasoning, answer, idf, reasoning_fraction=0.25),
                "content_idf_recall_early50": content_idf_recall(reasoning, answer, idf, reasoning_fraction=0.50),
                "content_idf_recall_q_filtered": content_idf_recall(reasoning, answer, idf, question=question, filter_question=True),
                "content_idf_recall_q_filtered_early25": content_idf_recall(
                    reasoning,
                    answer,
                    idf,
                    question=question,
                    reasoning_fraction=0.25,
                    filter_question=True,
                ),
                "content_idf_recall_q_filtered_early50": content_idf_recall(
                    reasoning,
                    answer,
                    idf,
                    question=question,
                    reasoning_fraction=0.50,
                    filter_question=True,
                ),
                "content_rouge_2_recall": c2_r,
                "content_rouge_2_f1": c2_f,
                "content_rouge_3_recall": c3_r,
                "content_rouge_3_f1": c3_f,
                "token_set_jaccard": len(set(rea_tok) & set(ans_tok)) / len(set(rea_tok) | set(ans_tok)) if set(rea_tok) | set(ans_tok) else 0.0,
                "answer_substring_containment": containment(norm_reason, norm_answer),
                "number_recall": unique_recall(numbers_reasoning, numbers_answer),
                "code_term_recall": unique_recall(code_reasoning, code_answer),
            }
            records.append(record)
    return records


def add_embedding_metrics(records: List[Dict[str, Any]], rows: List[Dict[str, Any]], methods: Sequence[str], args: argparse.Namespace) -> None:
    if args.no_embeddings:
        return
    model_kwargs: Dict[str, Any] = {}
    if args.embedding_device:
        model_kwargs["device"] = args.embedding_device
    model = SentenceTransformer(args.embedding_model, **model_kwargs)
    model.max_seq_length = args.embedding_max_length

    answers = []
    reasonings = []
    for record in records:
        row = rows[int(record["sample_idx"])]
        method = str(record["method"])
        answers.append(str(row.get("answers", {}).get(method, "")))
        reasonings.append(str(row.get("reasonings", {}).get(method, "")))

    ans_vecs = model.encode(
        answers,
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    rea_vecs = model.encode(
        reasonings,
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    cos = np.sum(ans_vecs * rea_vecs, axis=1)
    cos01 = (cos + 1.0) / 2.0
    for record, raw, scaled in zip(records, cos, cos01):
        record["sentence_embedding_cosine"] = float(raw)
        record["sentence_embedding_cosine_01"] = float(np.clip(scaled, 0.0, 1.0))


def summarize(records: List[Dict[str, Any]], metric_names: List[str], methods: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for metric in metric_names:
        rows = []
        for method in methods:
            vals = np.array([float(r[metric]) for r in records if r["method"] == method and np.isfinite(float(r.get(metric, float("nan"))))], dtype=float)
            if vals.size == 0:
                continue
            rows.append({
                "metric": metric,
                "method": method,
                "N": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p25": float(np.quantile(vals, 0.25)),
                "p75": float(np.quantile(vals, 0.75)),
                "p90": float(np.quantile(vals, 0.90)),
            })
        out[metric] = rows
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def correlations(records: List[Dict[str, Any]], metric_names: List[str], baseline: str) -> List[Dict[str, Any]]:
    rows = []
    base = np.array([float(r.get(baseline, float("nan"))) for r in records], dtype=float)
    for metric in metric_names:
        vals = np.array([float(r.get(metric, float("nan"))) for r in records], dtype=float)
        mask = np.isfinite(base) & np.isfinite(vals)
        if not np.any(mask):
            continue
        rows.append({
            "metric": metric,
            "N": int(mask.sum()),
            "pearson_vs_current": pearson(base[mask], vals[mask]),
            "spearman_vs_current": pearson(rankdata(base[mask]), rankdata(vals[mask])),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{100 * v:.1f}" if 0.0 <= v <= 1.0 else f"{v:.3f}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/qwen3_4b_thinking_2507_examples/inputs/example.metric.jsonl"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("runs/qwen3_4b_thinking_2507_examples/metrics/methods"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/lexical_metric_variants"))
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--tokenizer-model", default=None, help="Tokenizer used for current_token_lcs_recall")
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    rows = list(read_jsonl(args.input))
    idf = build_idf(rows, methods)
    existing = load_existing_lexical(args.metrics_dir)
    tokenizer = load_tokenizer(args.tokenizer_model)
    records = pair_records(rows, methods, existing, idf, tokenizer)
    add_embedding_metrics(records, rows, methods, args)

    metric_names = [
        "current_token_lcs_recall",
        "char_3gram_recall",
        "word_rouge_l_recall",
        "word_rouge_l_f1",
        "rouge_1_recall",
        "rouge_2_recall",
        "rouge_3_recall",
        "content_unigram_recall",
        "content_idf_recall",
        "content_idf_recall_early25",
        "content_idf_recall_early50",
        "content_idf_recall_q_filtered",
        "content_idf_recall_q_filtered_early25",
        "content_idf_recall_q_filtered_early50",
        "content_rouge_2_recall",
        "content_rouge_3_recall",
        "token_set_jaccard",
        "answer_substring_containment",
        "number_recall",
        "code_term_recall",
    ]
    if records and "sentence_embedding_cosine_01" in records[0]:
        metric_names.extend(["sentence_embedding_cosine", "sentence_embedding_cosine_01"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    write_csv(args.out_dir / "per_pair_metrics.csv", records, fieldnames)

    summary = summarize(records, metric_names, methods)
    summary_rows = [row for metric in metric_names for row in summary.get(metric, [])]
    write_csv(args.out_dir / "summary_by_method.csv", summary_rows, ["metric", "method", "N", "mean", "median", "p25", "p75", "p90"])

    corr_rows = correlations(records, metric_names, "current_token_lcs_recall")
    write_csv(args.out_dir / "correlation_vs_current.csv", corr_rows, ["metric", "N", "pearson_vs_current", "spearman_vs_current"])

    report = [
        "# Lexical Anchoring Metric Variants",
        "",
        f"Input: `{args.input}`",
        f"Pairs: `{len(records)}`",
        f"Embedding model: `{args.embedding_model if not args.no_embeddings else 'disabled'}`",
        "",
        "## Mean by Method",
        "",
    ]
    for metric in metric_names:
        rows_for_metric = summary.get(metric, [])
        if not rows_for_metric:
            continue
        report.append(f"### {metric}")
        report.append(markdown_table(["Method", "N", "Mean", "Median", "P75", "P90"], [
            [r["method"], r["N"], r["mean"], r["median"], r["p75"], r["p90"]] for r in rows_for_metric
        ]))
    report.append("## Correlation with current_token_lcs_recall")
    report.append(markdown_table(["Metric", "N", "Pearson", "Spearman"], [
        [r["metric"], r["N"], r["pearson_vs_current"], r["spearman_vs_current"]] for r in corr_rows
    ]))
    (args.out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({
        "input": str(args.input),
        "out_dir": str(args.out_dir),
        "pairs": len(records),
        "metrics": metric_names,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
