# -*- coding: utf-8 -*-
"""
A6. Gist-first（语义要点两阶段）—— 归因关键对照
  Phase 1: 给 (Q, A) 生成 3-8 条【内容性】方法论要点（gist），
           变体: invariant（与 SSR 同等内容不变性约束，公平比较）
                 free（去掉约束，更弱对照）
  Phase 2: 按 gist 展开全 trace（A 仍可见，与 SSR phase 2 镜像）

预算控制：gist 总 token 预算默认 300，对齐 SSR 骨架（~15 步 × ≤20 tok）。
gist 文本单独入档：后续可对 gist 本身算 a_lex（vs A），与 SSR 骨架的泄漏对比
（呼应 A9 的 PS-plan 泄漏分析）。

用法：
  python gen_a6_gist.py --model <generator> --tp 8 \
      --input data/analysis_2k.jsonl --out out/a6_gist_inv.jsonl \
      --variant invariant --gist-budget 300
"""
import argparse

from transformers import AutoTokenizer

from common import (build_chat, load_jsonl, save_jsonl, split_steps_by_blankline,
                    substring_token_span, vllm_generate)
from prompts_tier1 import (GIST_P1_SYSTEM_FREE, GIST_P1_SYSTEM_INVARIANT,
                           GIST_P2_SYSTEM, GIST_P2_USER, USER_QA, parse_tagged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["invariant", "free"],
                    default="invariant")
    ap.add_argument("--gist-budget", type=int, default=300)
    ap.add_argument("--max-tokens-p2", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=32768)
    args = ap.parse_args()

    from vllm import LLM

    data = load_jsonl(args.input)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.92)

    # ---------------- Phase 1：gist ----------------
    sys_t = (GIST_P1_SYSTEM_INVARIANT if args.variant == "invariant"
             else GIST_P1_SYSTEM_FREE)
    sys_p1 = sys_t.format(gist_budget=args.gist_budget)
    p1_prompts = [build_chat(
        tok, sys_p1,
        USER_QA.format(question=d["question"], answer=d["answer"]),
        enable_thinking=False) for d in data]
    p1 = vllm_generate(llm, p1_prompts, temperature=args.temperature,
                       max_tokens=args.gist_budget + 64, logprobs=None)
    gists = [parse_tagged(c[0]["text"], "gist") for c in p1]

    # ---------------- Phase 2：按 gist 展开 ----------------
    p2_prompts = [build_chat(
        tok, GIST_P2_SYSTEM,
        GIST_P2_USER.format(question=d["question"], answer=d["answer"],
                            gists=g),
        enable_thinking=False) for d, g in zip(data, gists)]
    p2 = vllm_generate(llm, p2_prompts, temperature=args.temperature,
                       max_tokens=args.max_tokens_p2)

    recs = []
    for d, g, cands in zip(data, gists, p2):
        c = cands[0]
        trace = parse_tagged(c["text"], "reasoning")
        span = substring_token_span(tok, c["token_ids"], trace)
        if span:
            s, e = span
            ids = c["token_ids"][s:e]
            ent = c["tok_entropy"][s:e] if c.get("tok_entropy") else None
            ok = True
        else:
            ids, ent, ok = c["token_ids"], c.get("tok_entropy"), False
        recs.append({
            "qid": d.get("qid"), "question": d["question"],
            "answer": d["answer"],
            "method": f"gist_{args.variant}",
            "trace": trace, "gist": g,
            "tok_entropy": ent,
            "step_spans": split_steps_by_blankline(tok, ids),
            "align_ok": ok,
        })

    save_jsonl(recs, args.out)
    print(f"done: {len(recs)} traces -> {args.out}")
    # gist 泄漏快查（词级粗筛，正式数用 metrics 的 token 级 a_lex 跑 gist 字段）
    n_warn = sum(1 for r in recs
                 if len(set(r["gist"].lower().split())
                        & set(r["answer"].lower().split())) > 30)
    print(f"gist-leak rough warning (>30 shared words): {n_warn}/{len(recs)}")


if __name__ == "__main__":
    main()
