#!/usr/bin/env python3
"""Generate two paraphrase controls per answer with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROMPT = """You are a careful paraphrasing assistant.
Rewrite the given answer into two deep paraphrases.

Requirements:
- Preserve the original meaning and factual commitments.
- Change wording, sentence order, section headings, bullet wording, and paragraph organization whenever possible.
- Keep each paraphrase within +/-20% of the original length.
- Do not add new claims.
- Do not copy full sentences unless technical names, code, formulas, or exact quoted terms require it.

Return exactly this format and nothing else:
<paraphrase_1>
...
</paraphrase_1>
<paraphrase_2>
...
</paraphrase_2>
"""

WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\W\s]", flags=re.UNICODE)


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


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def apply_chat_template(tokenizer: Any, answer: str) -> str:
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Answer:\n{answer}"},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_completed(path: Path) -> Dict[int, Dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[int(row["row_idx"])] = row
    return out


def parse_output(text: str) -> Dict[str, str]:
    raw = text.strip()
    tag_1 = re.search(r"<paraphrase_1>(.*?)</paraphrase_1>", raw, flags=re.IGNORECASE | re.DOTALL)
    tag_2 = re.search(r"<paraphrase_2>(.*?)</paraphrase_2>", raw, flags=re.IGNORECASE | re.DOTALL)
    if tag_1 and tag_2:
        return {
            "paraphrase_1": tag_1.group(1).strip(),
            "paraphrase_2": tag_2.group(1).strip(),
        }
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {
        "paraphrase_1": str(obj.get("paraphrase_1", "")).strip(),
        "paraphrase_2": str(obj.get("paraphrase_2", "")).strip(),
    }


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


def validation_stats(
    paraphrase: str,
    answer: str,
    length_tolerance: float,
    max_lcs_recall: float,
    require_lcs: bool,
) -> Dict[str, Any]:
    answer_len = len(tokenize_words(answer))
    para_len = len(tokenize_words(paraphrase))
    if answer_len == 0:
        length_ratio = 0.0
        length_ok = False
    else:
        length_ratio = para_len / answer_len
        length_ok = (1.0 - length_tolerance) <= length_ratio <= (1.0 + length_tolerance)
    if require_lcs:
        lcs = lcs_recall(paraphrase, answer)
        lcs_ok = lcs <= max_lcs_recall
    else:
        lcs = None
        lcs_ok = True
    accepted = bool(length_ok and (lcs_ok or not require_lcs))
    return {
        "word_length": para_len,
        "answer_word_length": answer_len,
        "length_ratio": length_ratio,
        "lcs_recall_with_answer": lcs,
        "length_ok": length_ok,
        "lcs_ok": lcs_ok,
        "lcs_required": require_lcs,
        "accepted": accepted,
    }


def build_output(
    input_rows: List[Dict[str, Any]],
    completed: Dict[int, Dict[str, Any]],
    length_tolerance: float,
    max_lcs_recall: float,
    require_lcs: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    rejected = []
    for idx, src in enumerate(input_rows):
        item = completed.get(idx)
        if not item:
            rejected.append({"row_idx": idx, "reason": "missing_raw_output"})
            continue
        parsed = parse_output(str(item.get("raw_output", "")))
        p1, p2 = parsed.get("paraphrase_1", ""), parsed.get("paraphrase_2", "")
        if not p1 or not p2:
            rejected.append({"row_idx": idx, "reason": "parse_failed", "raw_output": item.get("raw_output", "")[:500]})
            continue
        answer = src.get("answers", {}).get("NEU", "")
        stats = [
            validation_stats(p1, answer, length_tolerance, max_lcs_recall, require_lcs),
            validation_stats(p2, answer, length_tolerance, max_lcs_recall, require_lcs),
        ]
        if not all(s["accepted"] for s in stats):
            rejected.append({
                "row_idx": idx,
                "reason": "validation_failed",
                "validation": stats,
                "raw_output": item.get("raw_output", "")[:500],
            })
            continue
        rows.append({
            "id": src.get("id", idx),
            "sample_idx": idx,
            "answer": answer,
            "paraphrases": [p1, p2],
            "validation": stats,
        })
    return rows, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=8192)
    parser.add_argument("--length-tolerance", type=float, default=0.20)
    parser.add_argument("--max-lcs-recall", type=float, default=0.95)
    parser.add_argument("--require-lcs", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    input_rows = read_jsonl(args.input)
    if args.limit is not None:
        input_rows = input_rows[:args.limit]

    completed = load_completed(args.raw_output)
    if not args.build_only:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompts = []
        prompt_rows = []
        for idx, row in enumerate(input_rows):
            if idx in completed:
                continue
            answer = str(row.get("answers", {}).get("NEU", ""))
            if not answer.strip():
                continue
            prompts.append(apply_chat_template(tokenizer, answer))
            prompt_rows.append((idx, row.get("id", idx)))
        print(json.dumps({"requests": len(input_rows), "completed": len(completed), "pending": len(prompts)}, indent=2))

        llm = LLM(
            model=str(args.model),
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
        )
        sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
        for start in range(0, len(prompts), args.chunk_size):
            chunk_prompts = prompts[start:start + args.chunk_size]
            chunk_rows = prompt_rows[start:start + args.chunk_size]
            outputs = llm.generate(chunk_prompts, sampling)
            raw_rows = []
            for (idx, sample_id), out in zip(chunk_rows, outputs):
                raw_rows.append({
                    "row_idx": idx,
                    "id": sample_id,
                    "raw_output": out.outputs[0].text if out.outputs else "",
                    "finish_reason": out.outputs[0].finish_reason if out.outputs else None,
                    "output_tokens": len(out.outputs[0].token_ids) if out.outputs else 0,
                })
            append_jsonl(args.raw_output, raw_rows)
            print(json.dumps({"chunk_start": start, "chunk_size": len(raw_rows), "raw_output": str(args.raw_output)}))

    completed = load_completed(args.raw_output)
    rows, rejected = build_output(input_rows, completed, args.length_tolerance, args.max_lcs_recall, args.require_lcs)
    write_jsonl(args.output, rows)
    write_jsonl(args.rejected_output or args.output.with_suffix(".rejected.jsonl"), rejected)
    print(json.dumps({"rows": len(rows), "rejected": len(rejected), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
