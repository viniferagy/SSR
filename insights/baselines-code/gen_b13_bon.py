# -*- coding: utf-8 -*-
"""
B13. Best-of-N + 锚定拒绝采样 —— 回答"指标本身够不够当选择器"
流程（三个子命令，指标打分复用 metrics_anchoring CLI）：

  1) gen    : NEU 采 N 条/query（vLLM, logprobs=20），候选展平成 jsonl
  2)         python metrics_anchoring.py --in cands.jsonl --out cands_scored.jsonl \
                 --scorer-model <目标模型>          # 未改分布，无需 rescore
  3) judge  : 质量门控 judge（235B），写入 quality_score / consistent
  4) select : 三种规则各出一份选中集
       rule_prob   : min a_prob
       rule_combo  : min [0.5·mm(a_prob) + 0.5·mm(a_ent)]（mm = 组内 minmax）
       rule_gated  : quality_score ≥ 阈值 且 consistent 的子集内跑 combo，
                     全军覆没则回退组内最高 quality（防"低锚定的烂 trace"）

解读锚点：BoN-8 给出"采样分布内"的改善下限；SSR 若显著超过 → SSR 改变了
生成分布而非只挑好尾巴（selection vs distribution shift 的辨析）。
"""
import argparse
from collections import defaultdict

from common import (build_chat, load_jsonl, save_jsonl, split_steps_by_blankline,
                    substring_token_span, vllm_generate, vllm_judge)
from prompts_tier1 import (NEU_SYSTEM, NEU_USER_WITH, QUALITY_GATE_SYSTEM,
                           QUALITY_GATE_USER, parse_neu)


# ------------------------------- gen ---------------------------------------

def cmd_gen(args):
    from transformers import AutoTokenizer
    from vllm import LLM
    data = load_jsonl(args.input)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.92)

    prompts = [build_chat(tok, NEU_SYSTEM,
                          NEU_USER_WITH.format(question=d["question"],
                                               answer=d["answer"]))
               for d in data]
    outs = vllm_generate(llm, prompts, temperature=args.temperature,
                         top_p=0.95, max_tokens=args.max_new_tokens,
                         n=args.n, logprobs=20, seed=args.seed)

    recs = []
    for d, p, cands in zip(data, prompts, outs):
        for k, c in enumerate(cands):
            trace = parse_neu(c["text"])
            span = substring_token_span(tok, c["token_ids"], trace)
            if span:
                s, e = span
                ids = c["token_ids"][s:e]
                ent = c["tok_entropy"][s:e]
            else:
                ids, ent = c["token_ids"], c["tok_entropy"]
            recs.append({"qid": d.get("qid"), "cand_idx": k,
                         "question": d["question"], "answer": d["answer"],
                         "method": f"bon{args.n}",
                         "trace": trace, "tok_entropy": ent,
                         "step_spans": split_steps_by_blankline(tok, ids),
                         "gen_ctx": p})
    save_jsonl(recs, args.out)
    print(f"gen done: {len(data)} queries x {args.n} -> {args.out}")


# ------------------------------- judge -------------------------------------

def cmd_judge(args):
    from transformers import AutoTokenizer
    from vllm import LLM
    recs = load_jsonl(args.input)
    tok = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(model=args.judge_model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.92)

    users = [QUALITY_GATE_USER.format(question=r["question"],
                                      answer=r["answer"], trace=r["trace"])
             for r in recs]
    parsed = vllm_judge(llm, tok, QUALITY_GATE_SYSTEM, users)
    n_fail = 0
    for r, p in zip(recs, parsed):
        if p is None:
            r["quality_score"], r["consistent"] = None, None
            n_fail += 1
        else:
            r["quality_score"] = p.get("score")
            r["consistent"] = bool(p.get("consistent"))
    save_jsonl(recs, args.out)
    print(f"judge done -> {args.out} | unparsable: {n_fail}/{len(recs)}")


# ------------------------------- select ------------------------------------

def _mm(vals):
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in vals]


def cmd_select(args):
    recs = load_jsonl(args.input)
    groups = defaultdict(list)
    for r in recs:
        groups[r["qid"]].append(r)

    out = {"rule_prob": [], "rule_combo": [], "rule_gated": []}
    for qid, cs in groups.items():
        cs = [c for c in cs if c.get("a_prob") is not None]
        if not cs:
            continue
        out["rule_prob"].append(min(cs, key=lambda c: c["a_prob"]))

        ps = _mm([c["a_prob"] for c in cs])
        ent_raw = [c.get("a_ent") for c in cs]
        valid = [v for v in ent_raw
                 if isinstance(v, (int, float)) and v == v]
        if valid:
            it = iter(_mm(valid))
            es = [next(it) if (isinstance(v, (int, float)) and v == v)
                  else 0.5 for v in ent_raw]
        else:
            es = [0.5] * len(cs)
        combo = [0.5 * p + 0.5 * e for p, e in zip(ps, es)]
        for c, v in zip(cs, combo):
            c["combo_score"] = v
        out["rule_combo"].append(min(cs, key=lambda c: c["combo_score"]))

        gated = [c for c in cs
                 if (c.get("quality_score") or 0) >= args.quality_th
                 and c.get("consistent")]
        out["rule_gated"].append(
            min(gated, key=lambda c: c["combo_score"]) if gated
            else max(cs, key=lambda c: c.get("quality_score") or 0))

    for rule, sel in out.items():
        path = f"{args.out_prefix}.{rule}.jsonl"
        for s in sel:
            s["method"] = f"{s['method']}_{rule}"
        save_jsonl(sel, path)
        if sel and all("a_prob" in s for s in sel):
            import numpy as np
            print(f"{rule}: n={len(sel)} | a_prob x100 = "
                  f"{100 * np.mean([s['a_prob'] for s in sel]):.1f} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--input", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=8)
    g.add_argument("--tp", type=int, default=4)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--max-new-tokens", type=int, default=8192)
    g.add_argument("--max-model-len", type=int, default=32768)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(fn=cmd_gen)

    j = sub.add_parser("judge")
    j.add_argument("--input", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--judge-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    j.add_argument("--tp", type=int, default=8)
    j.add_argument("--max-model-len", type=int, default=32768)
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("select")
    s.add_argument("--input", required=True, help="经 metrics + judge 后的候选文件")
    s.add_argument("--out-prefix", required=True)
    s.add_argument("--quality-th", type=int, default=3)
    s.set_defaults(fn=cmd_select)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
