# -*- coding: utf-8 -*-
"""
A1. Forward-draft + Bridge (FDB)
  Phase 1: 只给 Q，正向生成 R_fwd + 草稿答案 Â（零锚定 by construction）
  Phase 2: 给 (Q, A, R_fwd)，生成预算受限的 bridge 段 B
  最终 trace = R_fwd ⊕ B

熵口径：FDB 未修改解码分布，A_ent 用生成时 top-20 熵；两段各自在其真实
生成条件下采集（这正是要测的 generation dynamics），段边界强制断步。

用法：
  python gen_a1_fdb.py --model Qwen/Qwen3-235B-Instruct-2507 --tp 8 \
      --input data/analysis_2k.jsonl --out out/a1_fdb_implicit.jsonl \
      --bridge-mode implicit --bridge-budget 200
之后：
  python metrics_anchoring.py --in out/a1_fdb_implicit.jsonl --out scored.jsonl \
      --scorer-model <目标模型> --segments
"""
import argparse

from transformers import AutoTokenizer

from common import (build_chat, load_jsonl, save_jsonl, split_steps_by_blankline,
                    substring_token_span, vllm_generate)
from prompts_tier1 import (FDB_P1_SYSTEM, FDB_P2_SYSTEM_EXPLICIT,
                           FDB_P2_SYSTEM_IMPLICIT, FDB_P2_USER, USER_Q_ONLY,
                           parse_tagged)


def slice_gen(tok, cand, sub):
    """从一次生成里切出子串 sub 对应的 token_ids / tok_entropy 片段。"""
    span = substring_token_span(tok, cand["token_ids"], sub)
    if span is None:
        return cand["token_ids"], cand.get("tok_entropy"), False
    s, e = span
    ent = cand.get("tok_entropy")
    return cand["token_ids"][s:e], (ent[s:e] if ent else None), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--input", required=True, help="jsonl: qid, question, answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bridge-mode", choices=["implicit", "explicit"],
                    default="implicit")
    ap.add_argument("--bridge-budget", type=int, default=200)
    ap.add_argument("--max-tokens-p1", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=32768)
    args = ap.parse_args()

    from vllm import LLM

    data = load_jsonl(args.input)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.92)

    # ---------------- Phase 1：纯正向，绝对不见 A ----------------
    p1_prompts = [build_chat(tok, FDB_P1_SYSTEM,
                             USER_Q_ONLY.format(question=d["question"]),
                             enable_thinking=False) for d in data]
    p1 = vllm_generate(llm, p1_prompts, temperature=args.temperature,
                       max_tokens=args.max_tokens_p1)

    drafts = []
    for d, cands in zip(data, p1):
        c = cands[0]
        r_fwd = parse_tagged(c["text"], "reasoning")
        a_hat = parse_tagged(c["text"], "answer")
        ids, ent, ok = slice_gen(tok, c, r_fwd)
        drafts.append({"r_fwd": r_fwd, "a_hat": a_hat,
                       "ids": ids, "ent": ent, "align_ok": ok})

    # ---------------- Phase 2：bridge ----------------
    sys_t = (FDB_P2_SYSTEM_IMPLICIT if args.bridge_mode == "implicit"
             else FDB_P2_SYSTEM_EXPLICIT)
    sys_p2 = sys_t.format(bridge_budget=args.bridge_budget)
    p2_prompts = [build_chat(
        tok, sys_p2,
        FDB_P2_USER.format(question=d["question"], answer=d["answer"],
                           draft=dr["r_fwd"]),
        enable_thinking=False) for d, dr in zip(data, drafts)]
    # 硬预算：max(200, 15%·|R_fwd|) 取 per-sample max_tokens 不便，统一用
    # budget+64 slack 的全局上限，超出部分自然截断（标记尾部即可）。
    p2 = vllm_generate(llm, p2_prompts, temperature=args.temperature,
                       max_tokens=args.bridge_budget + 64)

    recs = []
    for d, dr, cands in zip(data, drafts, p2):
        c = cands[0]
        bridge = parse_tagged(c["text"], "bridge")
        b_ids, b_ent, b_ok = slice_gen(tok, c, bridge)

        # 拼接熵流：段边界强制断步（spans 先各算再平移合并）
        ids_all = dr["ids"] + b_ids
        ent_all = ((dr["ent"] or []) + (b_ent or [])) or None
        spans = split_steps_by_blankline(tok, dr["ids"])
        off = len(dr["ids"])
        spans += [(s + off, e + off)
                  for s, e in split_steps_by_blankline(tok, b_ids)]

        recs.append({
            "qid": d.get("qid"), "question": d["question"],
            "answer": d["answer"],
            "method": f"fdb_{args.bridge_mode}",
            "trace": (dr["r_fwd"] + "\n\n" + bridge).strip(),
            "segments": {"r_fwd": dr["r_fwd"], "bridge": bridge},
            "draft_answer": dr["a_hat"],
            "tok_entropy": ent_all, "step_spans": spans,
            "align_ok": dr["align_ok"] and b_ok,
            "n_tokens": {"r_fwd": len(dr["ids"]), "bridge": len(b_ids)},
        })

    save_jsonl(recs, args.out)
    bad = sum(1 for r in recs if not r["align_ok"])
    print(f"done: {len(recs)} traces -> {args.out} | align fallback: {bad}")


if __name__ == "__main__":
    main()
