#!/usr/bin/env python3
"""Three-layer blur decomposition for probabilistic answer anchoring.

The decomposition follows ``insights/claude.blur.md``:

    B_full   = log p(A | Q, R) - log p(A | Q)
    B_lex    = log p(A | Q, R_lex_blur) - log p(A | Q)
    B_sem    = log p(A | Q, R_sem_blur) - log p(A | Q)

    C_lex    = B_full - B_lex
    C_sem    = B_lex - B_sem
    C_struct = B_sem

All quantities are per-answer-token bits, matching reviewer-protocol ``B``.
This script is deliberately separate from ``reviewer_protocol.py`` so blur
experiments cannot silently change official reported metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


LOGE_TO_BITS = 1.0 / math.log(2.0)
TORCH_DTYPE = torch.bfloat16
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*%?|[^\W\s]", re.UNICODE)
SEGMENT_RE = re.compile(r"\n+|[^.!?。！？\n]+[.!?。！？]?", re.UNICODE)
STEP_SEP = "\n\n"
DEFAULT_SCORING_MODEL = Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B")
DEFAULT_EMBEDDING_MODEL = Path("/home/pengguangyue/workspace/models/xlm-roberta-large")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for",
    "with", "as", "by", "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "from", "at", "into", "about", "we", "you", "i", "he", "she",
    "they", "them", "our", "your", "their", "not", "no", "yes", "do", "does", "did", "can",
    "could", "would", "should", "will", "may", "might", "must", "have", "has", "had", "there",
    "here", "which", "what", "when", "where", "why", "how", "also", "than", "therefore",
    "thus", "because", "each", "any", "all", "some", "most", "more", "less", "very", "just",
    "only", "same", "other", "another", "such", "within", "without", "across", "between",
}

DEFAULT_SOURCE_SPECS = [
    {
        "path": "runs/final_reviewer_protocol_1k/full/inputs/blind_1k.metric.jsonl",
        "methods": {"Blind CoT": "R0 / Blind CoT"},
        "split": "full-1k",
        "family": "blind",
    },
    {
        "path": "runs/enhanced_suppression_variants_qwen3_8b_1k/inputs/methods_with_enhanced.metric.jsonl",
        "methods": {
            "NEU": "NEU",
            "SUP": "SUP",
            "AUG-SUP": "AUG-SUP",
            "SSR": "SSR",
            "SSR_PLUS": "SSR_PLUS",
            "SSR_PLUS_STRUCT": "SSR_PLUS_STRUCT",
            "QA-SUP": "QA-SUP",
            "PG-SUP": "PG-SUP",
        },
        "split": "full-1k",
        "family": "main-enhanced",
    },
    {
        "path": "runs/ssr_plus_struct_variants_1k_newmetrics/stable_20260618_170144/inputs/methods_1k_plus_struct_variants_drop712.metric.jsonl",
        "methods": {
            "SSR_PLUS_STRUCT_C1_STEPS": "SSR_PLUS_STRUCT_C1_STEPS",
            "SSR_PLUS_STRUCT_LONG": "SSR_PLUS_STRUCT_LONG",
        },
        "split": "full-1k-drop712",
        "family": "struct-variants",
        "allow_missing_records": True,
    },
    {
        "path": "runs/qskel_answer_masked_1k/full/inputs/qskel_1k.metric.jsonl",
        "methods": {"SSR_QSKEL": "SSR_QSKEL"},
        "split": "full-1k",
        "family": "answer-masked",
    },
    {
        "path": "runs/tier1_baselines/1k/inputs/tier1_final.metric.jsonl",
        "methods": {
            "A1-FDB": "A1-FDB",
            "A6-Gist": "A6-Gist",
            "B11-NGramBlock": "B11-NGramBlock",
            "B13-BoN": "B13-BoN",
        },
        "split": "full-1k",
        "family": "tier1-baselines",
    },
    {
        "path": "runs/tier1_baselines/1k/inputs/b12_pilot_100.metric.jsonl",
        "methods": {"B12-Contrastive-g0.5": "B12-Contrastive-g0.5"},
        "split": "pilot-100",
        "family": "tier1-baselines",
    },
    {
        "path": "runs/entropy_targeted_suppression_smoke/inputs/methods_with_plus.metric.jsonl",
        "methods": {"CV-SUP": "CV-SUP", "DL-SUP": "DL-SUP"},
        "split": "smoke-128",
        "family": "entropy-targeted",
    },
    {
        "path": "runs/entropy_targeted_suppression_fs_smoke/inputs/methods_with_plus.metric.jsonl",
        "methods": {"FS-SUP": "FS-SUP"},
        "split": "smoke-128",
        "family": "entropy-targeted",
    },
    {
        "path": "runs/ssr_plus_struct_compact_smoke/inputs/plus_methods.metric.jsonl",
        "methods": {"SSR_PLUS_STRUCT_COMPACT": "SSR_PLUS_STRUCT_COMPACT"},
        "split": "smoke-128",
        "family": "struct-smoke",
    },
    {
        "path": "runs/ssr_plus_struct_mid_smoke/inputs/plus_methods.metric.jsonl",
        "methods": {"SSR_PLUS_STRUCT_MID": "SSR_PLUS_STRUCT_MID"},
        "split": "smoke-128",
        "family": "struct-smoke",
    },
    {
        "path": "runs/struct_prompt_ablation_32/inputs/c2_c3.metric.jsonl",
        "methods": {
            "SSR_PLUS_STRUCT_C2_COMPACT": "SSR_PLUS_STRUCT_C2_COMPACT",
            "SSR_PLUS_STRUCT_C3_EXACT2": "SSR_PLUS_STRUCT_C3_EXACT2",
        },
        "split": "ablation-32",
        "family": "struct-ablation",
    },
    {
        "path": "runs/struct_prompt_ablation_32/inputs/c4.metric.jsonl",
        "methods": {"SSR_PLUS_STRUCT_C4_BLANKLINE": "SSR_PLUS_STRUCT_C4_BLANKLINE"},
        "split": "ablation-32",
        "family": "struct-ablation",
    },
    {
        "path": "runs/struct_prompt_ablation_32/inputs/c5.metric.jsonl",
        "methods": {"SSR_PLUS_STRUCT_C5_SEPARATE_RULE": "SSR_PLUS_STRUCT_C5_SEPARATE_RULE"},
        "split": "ablation-32",
        "family": "struct-ablation",
    },
    {
        "path": "runs/struct_prompt_ablation_32/inputs/c6_c7.metric.jsonl",
        "methods": {
            "SSR_PLUS_STRUCT_C6_LONG_STEPS": "SSR_PLUS_STRUCT_C6_LONG_STEPS",
            "SSR_PLUS_STRUCT_C7_DEVELOPED": "SSR_PLUS_STRUCT_C7_DEVELOPED",
        },
        "split": "ablation-32",
        "family": "struct-ablation",
    },
    {
        "path": "runs/struct_prompt_ablation_32/inputs/c8_c9.metric.jsonl",
        "methods": {
            "SSR_PLUS_STRUCT_C8_LONG_REASON": "SSR_PLUS_STRUCT_C8_LONG_REASON",
            "SSR_PLUS_STRUCT_C9_LONG_TEMPLATE": "SSR_PLUS_STRUCT_C9_LONG_TEMPLATE",
        },
        "split": "ablation-32",
        "family": "struct-ablation",
    },
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def token_items(text: str) -> Iterator[Tuple[str, int, int]]:
    for match in WORD_RE.finditer(text):
        yield match.group(0), match.start(), match.end()


def is_wordlike(tok: str) -> bool:
    return any(ch.isalnum() for ch in tok)


def normalized(tok: str) -> str:
    return tok.strip().lower()


def token_class(tok: str) -> str:
    raw = tok.strip()
    low = raw.lower()
    if re.fullmatch(r"\d{4}", raw):
        return "year"
    if re.fullmatch(r"\d+(?:[.,]\d+)*%?", raw):
        return "number"
    if re.fullmatch(r"[A-Z]", raw):
        return "choice"
    if re.fullmatch(r"[A-Z]{2,}", raw):
        return "acronym"
    if raw[:1].isupper() and raw[1:].islower():
        return "proper"
    if low in {"true", "false"}:
        return "boolean"
    return "word"


def content_keys(answer: str) -> Dict[str, str]:
    toks = [tok for tok, _, _ in token_items(answer) if is_wordlike(tok)]
    keys: Dict[str, str] = {}
    if not toks:
        return keys
    for tok in toks:
        low = normalized(tok)
        cls = token_class(tok)
        keep = (
            cls in {"number", "year", "choice", "acronym", "proper", "boolean"}
            or (len(low) > 2 and low not in STOPWORDS)
        )
        if keep:
            key = f"CASE:{tok}" if cls == "choice" else low
            keys[key] = cls
    if not keys and len(toks) <= 3:
        for tok in toks:
            if is_wordlike(tok):
                cls = token_class(tok)
                key = f"CASE:{tok}" if cls == "choice" else normalized(tok)
                keys[key] = cls
    return keys


def token_key(tok: str, answer_keys: Dict[str, str]) -> Optional[str]:
    case_key = f"CASE:{tok}"
    if case_key in answer_keys:
        return case_key
    low = normalized(tok)
    if low in answer_keys:
        return low
    return None


DEFAULT_REPLACEMENTS = {
    "year": ["1984", "2007", "2019", "2025"],
    "number": ["17", "42", "3.5", "64"],
    "choice": ["D", "C", "B", "E"],
    "acronym": ["NLP", "CPU", "EU", "UN"],
    "proper": ["Tokyo", "Jordan", "Mercury", "Riverton", "Atlas"],
    "boolean": ["false", "true"],
    "word": ["object", "factor", "concept", "detail", "element"],
}


def build_replacement_pools(records: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    pools: Dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        text = f"{rec['answer']}\n{rec['reasoning']}"
        for tok, _, _ in token_items(text):
            if not is_wordlike(tok):
                continue
            low = normalized(tok)
            cls = token_class(tok)
            if cls == "word" and (len(low) <= 2 or low in STOPWORDS):
                continue
            pools[cls][tok] += 1
    out: Dict[str, List[str]] = {}
    for cls, counts in pools.items():
        out[cls] = [tok for tok, _ in counts.most_common()]
    return out


def replacement_for(
    original_key: str,
    cls: str,
    answer_norms: set[str],
    pools: Dict[str, List[str]],
    offset: int,
) -> str:
    candidates = list(pools.get(cls, [])) + DEFAULT_REPLACEMENTS.get(cls, DEFAULT_REPLACEMENTS["word"])
    original_norm = original_key.replace("CASE:", "").lower()
    for idx in range(len(candidates)):
        cand = candidates[(idx + offset) % len(candidates)]
        cand_norm = normalized(cand)
        if cand_norm and cand_norm not in answer_norms and cand_norm != original_norm:
            return cand
    return DEFAULT_REPLACEMENTS.get(cls, DEFAULT_REPLACEMENTS["word"])[0]


def match_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement.lower() if replacement.isalpha() else replacement


def lexical_blur(reasoning: str, answer: str, pools: Dict[str, List[str]], record_idx: int) -> Tuple[str, Dict[str, Any]]:
    keys = content_keys(answer)
    answer_norms = {key.replace("CASE:", "").lower() for key in keys}
    mapping: Dict[str, str] = {}
    for offset, (key, cls) in enumerate(sorted(keys.items())):
        mapping[key] = replacement_for(key, cls, answer_norms, pools, record_idx + offset)

    pieces: List[str] = []
    last = 0
    replacement_count = 0
    replaced_unique: set[str] = set()
    for tok, start, end in token_items(reasoning):
        key = token_key(tok, keys)
        if key is None:
            continue
        pieces.append(reasoning[last:start])
        repl = match_case(mapping[key], tok)
        pieces.append(repl)
        last = end
        replacement_count += 1
        replaced_unique.add(key)
    pieces.append(reasoning[last:])
    blurred = "".join(pieces)
    meta = {
        "answer_content_tokens": len(keys),
        "lex_replacements": replacement_count,
        "lex_replaced_unique": len(replaced_unique),
        "lex_mapping": mapping,
    }
    return blurred, meta


def segment_text(text: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for match in SEGMENT_RE.finditer(text):
        raw = match.group(0)
        is_ws = not raw.strip()
        segments.append({"text": raw, "start": match.start(), "end": match.end(), "is_ws": is_ws})
    if not segments:
        segments.append({"text": text, "start": 0, "end": len(text), "is_ws": not text.strip()})
    return segments


def segment_word_count(text: str) -> int:
    return sum(1 for tok, _, _ in token_items(text) if is_wordlike(tok))


def neutral_sentence(segment: str) -> str:
    low = segment.lower()
    stripped = segment.strip()
    punct = "."
    if stripped.endswith("?"):
        punct = "."
    elif stripped.endswith(("!", "。", "！", "？")):
        punct = "."
    if any(w in low for w in ["compare", "than", "difference", "alternative", "option"]):
        base = "This step compares the relevant alternatives at a high level"
    elif any(w in low for w in ["rule out", "exclude", "eliminate", "not ", "cannot", "can't"]):
        base = "This step removes an incompatible possibility without naming it"
    elif any(w in low for w in ["calculate", "compute", "equation", "sum", "ratio", "percent", "number"]):
        base = "This step performs the needed quantitative check in neutral terms"
    elif any(w in low for w in ["because", "since", "implies", "therefore", "thus", "so "]):
        base = "This step links a premise to a general implication"
    elif any(w in low for w in ["conclude", "answer", "result", "final"]):
        base = "This step records an intermediate conclusion without naming the target"
    else:
        base = "This step continues the reasoning using answer-neutral information"
    prefix_ws = segment[: len(segment) - len(segment.lstrip())]
    suffix_ws = segment[len(segment.rstrip()):]
    return f"{prefix_ws}{base}{punct}{suffix_ws}"


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def semantic_blur_from_similarities(
    lex_text: str,
    similarities: Sequence[float],
    threshold: float,
    top_frac: float,
    min_replace: int,
    min_words: int,
) -> Tuple[str, Dict[str, Any]]:
    segments = segment_text(lex_text)
    sim_values = [float(x) for x in similarities if math.isfinite(float(x))]
    selected_positions: set[int] = set()
    if sim_values:
        indexed = sorted(enumerate(sim_values), key=lambda item: item[1], reverse=True)
        if top_frac > 0:
            target = int(math.ceil(len(indexed) * top_frac))
            target = max(min_replace, target)
            target = min(len(indexed), target)
        else:
            target = 0
        for pos, sim in indexed[:target]:
            if sim >= threshold:
                selected_positions.add(pos)
        if not selected_positions and min_replace > 0:
            for pos, _sim in indexed[: min(min_replace, len(indexed))]:
                selected_positions.add(pos)

    pieces: List[str] = []
    repl_count = 0
    sim_idx = 0
    replaced_sims: List[float] = []
    max_sim = float("nan")
    for seg in segments:
        text = str(seg["text"])
        if seg["is_ws"] or segment_word_count(text) < min_words:
            pieces.append(text)
            continue
        sim = float(similarities[sim_idx])
        current_sim_idx = sim_idx
        sim_idx += 1
        max_sim = sim if not math.isfinite(max_sim) else max(max_sim, sim)
        if current_sim_idx in selected_positions:
            pieces.append(neutral_sentence(text))
            repl_count += 1
            replaced_sims.append(sim)
        else:
            pieces.append(text)
    meta = {
        "semantic_segments": sim_idx,
        "semantic_replacements": repl_count,
        "semantic_threshold": threshold,
        "semantic_top_frac": top_frac,
        "semantic_min_replace": min_replace,
        "semantic_max_similarity": finite_or_none(max_sim),
        "semantic_replaced_similarity_mean": statistics.fmean(replaced_sims) if replaced_sims else None,
    }
    return "".join(pieces), meta


def load_source_specs(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return list(DEFAULT_SOURCE_SPECS)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        specs = data.get("sources", [])
    else:
        specs = data
    if not isinstance(specs, list):
        raise ValueError("Source manifest must be a list or {'sources': [...]}")
    return specs


def collect_records(
    root: Path,
    specs: Sequence[Dict[str, Any]],
    strict: bool = True,
    limit_per_method: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    kept_by_method: Counter[str] = Counter()
    for spec_idx, spec in enumerate(specs):
        path = Path(spec["path"])
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            if strict:
                raise FileNotFoundError(path)
            continue
        rows = read_jsonl(path)
        method_map = spec.get("methods", {})
        if isinstance(method_map, list):
            method_map = {m: m for m in method_map}
        if not isinstance(method_map, dict):
            raise ValueError(f"Invalid methods in source spec {spec_idx}: {method_map!r}")
        limit = spec.get("limit")
        if limit is not None:
            rows = rows[: int(limit)]
        allow_missing_records = bool(spec.get("allow_missing_records", False))
        for source_sample_idx, row in enumerate(rows):
            for input_method, output_method in method_map.items():
                if limit_per_method is not None and kept_by_method[str(output_method)] >= limit_per_method:
                    continue
                q = ensure_text(row.get("questions", {}).get(input_method))
                a = ensure_text(row.get("answers", {}).get(input_method))
                r = ensure_text(row.get("reasonings", {}).get(input_method))
                if not (q.strip() and a.strip() and r.strip()):
                    skipped.append(
                        {
                            "source_path": str(path.relative_to(root) if path.is_relative_to(root) else path),
                            "source_sample_idx": source_sample_idx,
                            "input_method": input_method,
                            "method": output_method,
                            "reason": "missing Q/A/R",
                            "allowed": allow_missing_records or not strict,
                        }
                    )
                    if allow_missing_records:
                        continue
                    if strict:
                        raise ValueError(f"Missing Q/A/R for {input_method} in {path} row {source_sample_idx}")
                    continue
                records.append(
                    {
                        "record_idx": len(records),
                        "source_sample_idx": source_sample_idx,
                        "id": row.get("id", source_sample_idx),
                        "method": output_method,
                        "input_method": input_method,
                        "source_path": str(path.relative_to(root) if path.is_relative_to(root) else path),
                        "split": spec.get("split", ""),
                        "family": spec.get("family", ""),
                        "question": q,
                        "answer": a,
                        "reasoning": r,
                    }
                )
                kept_by_method[str(output_method)] += 1
    return records, skipped


def build_blur_input(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    specs = load_source_specs(args.sources_manifest)
    records, skipped = collect_records(root, specs, strict=not args.skip_missing, limit_per_method=args.limit_per_method)
    if not records:
        raise ValueError("No records collected")

    pools = {} if args.fixed_replacements else build_replacement_pools(records)
    for rec in tqdm(records, desc="lexical-blur"):
        lex, meta = lexical_blur(str(rec["reasoning"]), str(rec["answer"]), pools, int(rec["record_idx"]))
        rec["reasoning_lex_blur"] = lex
        rec["blur_meta"] = meta

    if not args.no_semantic_blur:
        from sentence_transformers import SentenceTransformer

        model_kwargs: Dict[str, Any] = {}
        if args.embedding_device:
            model_kwargs["device"] = args.embedding_device
        embedder = SentenceTransformer(str(args.embedding_model), **model_kwargs)
        embedder.max_seq_length = args.embedding_max_length

        texts: List[str] = []
        answer_text_indices: List[int] = []
        segment_slices: List[Tuple[int, int]] = []
        segment_counts: List[int] = []
        for rec in tqdm(records, desc="collect-segments"):
            answer_text_indices.append(len(texts))
            texts.append(str(rec["answer"]))
            start = len(texts)
            count = 0
            for seg in segment_text(str(rec["reasoning_lex_blur"])):
                if seg["is_ws"] or segment_word_count(str(seg["text"])) < args.semantic_min_words:
                    continue
                texts.append(str(seg["text"]).strip())
                count += 1
            segment_slices.append((start, len(texts)))
            segment_counts.append(count)

        vecs = embedder.encode(
            texts,
            batch_size=args.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for rec_idx, rec in enumerate(tqdm(records, desc="semantic-blur")):
            a_vec = vecs[answer_text_indices[rec_idx]]
            start, end = segment_slices[rec_idx]
            sims = [float(np.dot(a_vec, vecs[i])) for i in range(start, end)]
            sem, meta = semantic_blur_from_similarities(
                str(rec["reasoning_lex_blur"]),
                sims,
                args.semantic_threshold,
                args.semantic_top_frac,
                args.semantic_min_replace,
                args.semantic_min_words,
            )
            rec["reasoning_sem_blur"] = sem
            rec["blur_meta"].update(meta)
    else:
        for rec in records:
            rec["reasoning_sem_blur"] = rec["reasoning_lex_blur"]
            rec["blur_meta"].update(
                {
                    "semantic_segments": 0,
                    "semantic_replacements": 0,
                    "semantic_threshold": args.semantic_threshold,
                    "semantic_top_frac": args.semantic_top_frac,
                    "semantic_min_replace": args.semantic_min_replace,
                    "semantic_max_similarity": None,
                    "semantic_replaced_similarity_mean": None,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)
    manifest = {
        "input": str(args.output),
        "records": len(records),
        "skipped_records": len(skipped),
        "skipped": skipped[:200],
        "semantic_threshold": args.semantic_threshold,
        "semantic_top_frac": args.semantic_top_frac,
        "semantic_min_replace": args.semantic_min_replace,
        "semantic_min_words": args.semantic_min_words,
        "sources": specs,
        "method_counts": dict(Counter(str(r["method"]) for r in records)),
        "split_counts": dict(Counter(str(r["split"]) for r in records)),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output), "manifest": str(manifest_path)}, indent=2))


def rank_info() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        return rank, world, local
    return 0, 1, 0


def apply_chat_prefix(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tok_1d(tokenizer: Any, text: str, device: Optional[torch.device] = None) -> torch.LongTensor:
    return torch.tensor(tokenizer(text, add_special_tokens=False).input_ids, dtype=torch.long, device=device)


@torch.no_grad()
def batch_answer_logprob(
    model: Any,
    prompt_ids_list: List[torch.LongTensor],
    ans_ids: torch.LongTensor,
    max_ctx: Optional[int],
    batch_size: int,
) -> List[float]:
    device = next(model.parameters()).device
    ans_ids = ans_ids.to(device)
    ans_len = int(ans_ids.numel())
    scores: List[float] = []
    for i0 in range(0, len(prompt_ids_list), batch_size):
        chunk = prompt_ids_list[i0:i0 + batch_size]
        seqs: List[torch.LongTensor] = []
        prompt_lens: List[int] = []
        for prompt in chunk:
            full = torch.cat([prompt.to(device), ans_ids], dim=0)
            if max_ctx is not None and int(full.numel()) > max_ctx:
                full = full[-int(max_ctx):]
            seqs.append(full)
            prompt_lens.append(int(full.numel()) - ans_len)
        max_len = max(int(seq.numel()) for seq in seqs)
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


def prompt_ids(tokenizer: Any, prefix_q: torch.LongTensor, reasoning: str, think_close_ids: torch.LongTensor, device: torch.device) -> torch.LongTensor:
    reason_ids = tok_1d(tokenizer, reasoning, device)
    return torch.cat([prefix_q, reason_ids, think_close_ids], dim=0)


def score_blurs(args: argparse.Namespace) -> None:
    rank, world, local = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.scoring_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
    load_kwargs = {
        "config": cfg,
        "trust_remote_code": True,
        "torch_dtype": TORCH_DTYPE,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.scoring_model, attn_implementation="flash_attention_2", **load_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.scoring_model, **load_kwargs)
    model.to(device)
    model.eval()

    max_ctx = getattr(model.config, "max_position_embeddings", None)
    think_close_ids = tok_1d(tokenizer, "</think>\n\n", device)
    rows = read_jsonl(args.input)
    method_filter = set(m for m in args.methods.split(",") if m) if args.methods else None
    if method_filter:
        rows = [r for r in rows if str(r["method"]) in method_filter]
    if args.limit is not None:
        rows = rows[: args.limit]
    shard = rows[rank::world]

    metrics_path = args.output_dir / f"blur_metrics_rank{rank}.jsonl"
    invalid_path = args.output_dir / f"invalid_rank{rank}.jsonl"
    valid = invalid = 0
    with metrics_path.open("w", encoding="utf-8") as fw, invalid_path.open("w", encoding="utf-8") as fi:
        for rec in tqdm(shard, desc=f"rank{rank}", disable=(rank != 0)):
            try:
                q = ensure_text(rec["question"])
                a = ensure_text(rec["answer"])
                r_full = ensure_text(rec["reasoning"])
                r_lex = ensure_text(rec["reasoning_lex_blur"])
                r_sem = ensure_text(rec["reasoning_sem_blur"])
                ans_ids = tok_1d(tokenizer, a, device)
                if int(ans_ids.numel()) == 0:
                    raise ValueError("empty answer ids")
                prefix_q = tok_1d(tokenizer, apply_chat_prefix(tokenizer, q), device)
                base_prompt = torch.cat([prefix_q, think_close_ids], dim=0)
                prompts = [
                    base_prompt,
                    prompt_ids(tokenizer, prefix_q, r_full, think_close_ids, device),
                    prompt_ids(tokenizer, prefix_q, r_lex, think_close_ids, device),
                    prompt_ids(tokenizer, prefix_q, r_sem, think_close_ids, device),
                ]
                logps = batch_answer_logprob(model, prompts, ans_ids, max_ctx, args.batch_size)
                denom = int(ans_ids.numel())
                base = float(logps[0] * LOGE_TO_BITS / denom)
                full = float(logps[1] * LOGE_TO_BITS / denom)
                lex = float(logps[2] * LOGE_TO_BITS / denom)
                sem = float(logps[3] * LOGE_TO_BITS / denom)
                b_full = full - base
                b_lex = lex - base
                b_sem = sem - base
                c_lex = b_full - b_lex
                c_sem = b_lex - b_sem
                c_struct = b_sem
                residual = b_full - (c_lex + c_sem + c_struct)
                out = {
                    "record_idx": rec["record_idx"],
                    "source_sample_idx": rec["source_sample_idx"],
                    "id": rec.get("id"),
                    "method": rec["method"],
                    "input_method": rec.get("input_method"),
                    "source_path": rec.get("source_path"),
                    "split": rec.get("split"),
                    "family": rec.get("family"),
                    "answer_tokens": denom,
                    "full_reasoning_tokens": int(tok_1d(tokenizer, r_full).numel()),
                    "lex_reasoning_tokens": int(tok_1d(tokenizer, r_lex).numel()),
                    "sem_reasoning_tokens": int(tok_1d(tokenizer, r_sem).numel()),
                    "logp_base_per_token_bits": base,
                    "logp_full_per_token_bits": full,
                    "logp_lex_blur_per_token_bits": lex,
                    "logp_sem_blur_per_token_bits": sem,
                    "B_full": b_full,
                    "B_lex_blur": b_lex,
                    "B_sem_blur": b_sem,
                    "C_lex": c_lex,
                    "C_sem": c_sem,
                    "C_struct": c_struct,
                    "additive_residual": residual,
                }
                out.update({k: finite_or_none(v) for k, v in rec.get("blur_meta", {}).items()})
                required_metric_fields = [
                    "logp_base_per_token_bits",
                    "logp_full_per_token_bits",
                    "logp_lex_blur_per_token_bits",
                    "logp_sem_blur_per_token_bits",
                    "B_full",
                    "B_lex_blur",
                    "B_sem_blur",
                    "C_lex",
                    "C_sem",
                    "C_struct",
                    "additive_residual",
                ]
                if any(not math.isfinite(float(out[field])) for field in required_metric_fields):
                    raise ValueError("non-finite metric")
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                fw.flush()
                valid += 1
            except Exception as exc:
                invalid += 1
                fi.write(json.dumps({"record_idx": rec.get("record_idx"), "method": rec.get("method"), "error": repr(exc)}, ensure_ascii=False) + "\n")
                fi.flush()
                if args.debug_errors:
                    traceback.print_exc()
            torch.cuda.empty_cache()
    print(json.dumps({"rank": rank, "valid": valid, "invalid": invalid, "metrics": str(metrics_path)}, indent=2))


def read_metric_dir(path: Path) -> List[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("blur_metrics_rank*.jsonl"))
    rows: List[Dict[str, Any]] = []
    for file in files:
        rows.extend(read_jsonl(file))
    return rows


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return math.nan
    pos = (len(vals) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int) -> Tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    means = [statistics.fmean(rng.choice(vals) for _ in vals) for _ in range(n_boot)]
    return percentile(means, 0.025), percentile(means, 0.975)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else math.nan


def fmt(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.2f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(args: argparse.Namespace) -> None:
    records = read_metric_dir(args.metrics)
    if not records:
        raise ValueError(f"No blur metrics found under {args.metrics}")
    order = [m for m in args.method_order.split(",") if m] if args.method_order else []
    seen_order = []
    for rec in records:
        method = str(rec["method"])
        if method not in seen_order:
            seen_order.append(method)
    if not order:
        order = seen_order
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["method"])].append(rec)

    rng = random.Random(args.bootstrap_seed)
    fields_raw = [
        "B_full",
        "B_lex_blur",
        "B_sem_blur",
        "C_lex",
        "C_sem",
        "C_struct",
        "additive_residual",
        "lex_replacements",
        "semantic_replacements",
        "full_reasoning_tokens",
    ]
    table_rows: List[Dict[str, Any]] = []
    for method in order:
        rows = grouped.get(method, [])
        if not rows:
            continue
        item: Dict[str, Any] = {
            "Method": method,
            "N": len(rows),
            "Split": ",".join(sorted({str(r.get("split", "")) for r in rows if r.get("split")})),
            "Family": ",".join(sorted({str(r.get("family", "")) for r in rows if r.get("family")})),
        }
        for field in fields_raw:
            vals = [float(r.get(field, math.nan)) for r in rows]
            item[field] = mean(vals)
            if field.startswith(("B_", "C_")):
                lo, hi = bootstrap_ci(vals, rng, args.bootstrap)
                item[field + "_ci95"] = f"[{100 * lo:.2f}, {100 * hi:.2f}]"
        b_vals = [float(r["B_full"]) for r in rows]
        c_lex_vals = [float(r["C_lex"]) for r in rows]
        c_sem_vals = [float(r["C_sem"]) for r in rows]
        c_struct_vals = [float(r["C_struct"]) for r in rows]
        denom = mean(b_vals)
        item["C_lex_share"] = item["C_lex"] / denom if math.isfinite(denom) and abs(denom) > 1e-12 else math.nan
        item["C_sem_share"] = item["C_sem"] / denom if math.isfinite(denom) and abs(denom) > 1e-12 else math.nan
        item["C_struct_share"] = item["C_struct"] / denom if math.isfinite(denom) and abs(denom) > 1e-12 else math.nan
        item["positive_C_lex_rate"] = mean([1.0 if v > 0 else 0.0 for v in c_lex_vals])
        item["positive_C_sem_rate"] = mean([1.0 if v > 0 else 0.0 for v in c_sem_vals])
        item["positive_C_struct_rate"] = mean([1.0 if v > 0 else 0.0 for v in c_struct_vals])
        table_rows.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "Method",
        "N",
        "Split",
        "B_full_x100",
        "C_lex_x100",
        "C_sem_x100",
        "C_struct_x100",
        "B_lex_blur_x100",
        "B_sem_blur_x100",
        "C_lex_share_pct",
        "C_sem_share_pct",
        "C_struct_share_pct",
        "lex_replacements",
        "semantic_replacements",
        "full_reasoning_tokens",
        "additive_residual_x100",
    ]
    display_rows: List[Dict[str, Any]] = []
    for row in table_rows:
        display_rows.append(
            {
                "Method": row["Method"],
                "N": row["N"],
                "Split": row["Split"],
                "B_full_x100": 100.0 * row["B_full"],
                "C_lex_x100": 100.0 * row["C_lex"],
                "C_sem_x100": 100.0 * row["C_sem"],
                "C_struct_x100": 100.0 * row["C_struct"],
                "B_lex_blur_x100": 100.0 * row["B_lex_blur"],
                "B_sem_blur_x100": 100.0 * row["B_sem_blur"],
                "C_lex_share_pct": 100.0 * row["C_lex_share"] if math.isfinite(row["C_lex_share"]) else math.nan,
                "C_sem_share_pct": 100.0 * row["C_sem_share"] if math.isfinite(row["C_sem_share"]) else math.nan,
                "C_struct_share_pct": 100.0 * row["C_struct_share"] if math.isfinite(row["C_struct_share"]) else math.nan,
                "lex_replacements": row["lex_replacements"],
                "semantic_replacements": row["semantic_replacements"],
                "full_reasoning_tokens": row["full_reasoning_tokens"],
                "additive_residual_x100": 100.0 * row["additive_residual"],
            }
        )

    write_csv(args.output_dir / "blur_decomposition_summary.csv", display_rows, metric_fields)
    md = markdown_table(metric_fields, [[r.get(f, "") for f in metric_fields] for r in display_rows])
    (args.output_dir / "blur_decomposition_summary.md").write_text(md, encoding="utf-8")

    ci_rows = []
    for row in table_rows:
        ci_rows.append(
            {
                "Method": row["Method"],
                "N": row["N"],
                "B_full_ci95": row.get("B_full_ci95", ""),
                "C_lex_ci95": row.get("C_lex_ci95", ""),
                "C_sem_ci95": row.get("C_sem_ci95", ""),
                "C_struct_ci95": row.get("C_struct_ci95", ""),
            }
        )
    write_csv(args.output_dir / "blur_decomposition_ci.csv", ci_rows, ["Method", "N", "B_full_ci95", "C_lex_ci95", "C_sem_ci95", "C_struct_ci95"])

    method_counts = {method: len(grouped[method]) for method in order if method in grouped}
    summary = {
        "metrics": str(args.metrics),
        "records": len(records),
        "method_counts": method_counts,
        "units": "per-answer-token bits; *_x100 fields multiply by 100",
        "definition": {
            "B_full": "log p(A|Q,R) - log p(A|Q)",
            "C_lex": "B_full - B_lex_blur",
            "C_sem": "B_lex_blur - B_sem_blur",
            "C_struct": "B_sem_blur",
        },
        "max_abs_additive_residual": max(abs(float(r["additive_residual"])) for r in records),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Three-Layer Blur Decomposition",
        "",
        "All bit-gain values are per-answer-token bits multiplied by 100 in the table.",
        "",
        "## Summary",
        md,
        "## Additive Check",
        f"- Max absolute residual: `{summary['max_abs_additive_residual']:.6e}`",
        f"- Records: `{len(records)}`",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.output_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-input")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--sources-manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--embedding-device", default=None)
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--embedding-max-length", type=int, default=256)
    p.add_argument("--semantic-threshold", type=float, default=-1.0)
    p.add_argument("--semantic-top-frac", type=float, default=0.30)
    p.add_argument("--semantic-min-replace", type=int, default=1)
    p.add_argument("--semantic-min-words", type=int, default=4)
    p.add_argument("--limit-per-method", type=int)
    p.add_argument("--fixed-replacements", action="store_true", default=True)
    p.add_argument("--corpus-replacements", dest="fixed_replacements", action="store_false")
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--no-semantic-blur", action="store_true")
    p.set_defaults(func=build_blur_input)

    p = sub.add_parser("score")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scoring-model", type=Path, default=DEFAULT_SCORING_MODEL)
    p.add_argument("--methods", default="")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--debug-errors", action="store_true")
    p.set_defaults(func=score_blurs)

    p = sub.add_parser("aggregate")
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method-order", default="")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=19)
    p.set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
