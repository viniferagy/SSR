# -*- coding: utf-8 -*-
"""
三层锚定指标实现 + 统一打分 CLI。与论文公式逐项对齐：

  A_lex  = LCS(R, A) / |A|                       (Eq.1, ROUGE-L recall, token 级)
  A_ent  = sqrt(G_unif * L_nonunif)              (Eq.2-5, τg = 0.1)
  A_prob = (1/|A|) * log2[P(A|Q,R) / P(A|Q)]     (Eq.6, bits/token)

协议要点（与设计文档 Sec.0 一致）：
- scorer 永远是【未修改的目标模型】。B12 改了解码分布、B11 改了部分位置 logits，
  这两个方法的 A_ent 不能用生成时熵，必须 --rescore-entropy（teacher-forcing 全词表精确熵）。
- A_prob 打分上下文模板全局固定（见 SCORING_CTX_*），换模板 = 换指标，所有方法必须同模板。
- token 边界：context 与 A 分别 encode 后在 token 级拼接，避免字符串拼接后
  re-tokenize 的边界漂移。
报告约定：三指标 ×100 进表。

用法：
  python metrics_anchoring.py --in traces.jsonl --out scored.jsonl \
      --scorer-model Qwen/Qwen3-4B-Thinking-2507 [--rescore-entropy] [--segments]
输入 jsonl 每行至少含: question, answer, trace；
可选: tok_entropy + step_spans（vLLM 采集，未 rescore 时用）, segments（FDB 用）。
"""
import argparse
import math

import numpy as np
import torch
from rapidfuzz.distance import LCSseq
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_jsonl, save_jsonl

LN2 = math.log(2.0)
TAU_G = 0.1

# A_prob 打分上下文（全局固定，勿改）
SCORING_CTX_WITH_R = (
    "[QUESTION]\n{question}\n\n[REASONING]\n{reasoning}\n\n"
    "Based on the reasoning above, give the final answer."
)
SCORING_CTX_NO_R = "[QUESTION]\n{question}\n\nGive the final answer."


# ============================ A_lex ========================================

def a_lex(tokenizer, trace: str, answer: str) -> float:
    r = tokenizer.encode(trace, add_special_tokens=False)
    a = tokenizer.encode(answer, add_special_tokens=False)
    if not a:
        return 0.0
    return LCSseq.similarity(r, a) / len(a)


# ============================ A_ent ========================================

def a_ent(tok_entropy, step_spans) -> float:
    """tok_entropy: 每个生成 token 的熵；step_spans: [(s,e),...]。"""
    ids = [float(np.mean(tok_entropy[s:e])) for s, e in step_spans if e > s]
    if len(ids) < 3:
        return float("nan")
    u = np.asarray(ids, dtype=np.float64)
    rng = u.max() - u.min()
    u = (u - u.min()) / rng if rng > 1e-9 else np.zeros_like(u)   # 归一到单位区间
    g_unif = 1.0 / (1.0 + np.var(u) / TAU_G)                       # Eq.3
    d = np.abs(np.diff(u))
    mu = d.mean()
    l_non = 0.0 if mu < 1e-9 else (d.std() / mu) / (1.0 + d.std() / mu)  # Eq.4
    return float(np.sqrt(g_unif * l_non))                          # Eq.5


# ==================== A_prob / 精确熵（HF teacher-forcing） ==================

class AnswerScorer:
    """HF 加载未修改的目标模型，提供：
    - logp_answer: sum log P(A | ctx)，token 级拼接，边界安全
    - exact_entropy: 对给定 (ctx, R) teacher-forcing 出 R 各位置的全词表精确熵
    """

    def __init__(self, model_name, dtype="bfloat16", device_map="auto",
                 max_len=32768):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype),
            device_map=device_map).eval()
        self.max_len = max_len

    def _ids(self, text, special=False):
        return self.tok.encode(text, add_special_tokens=special)

    @torch.no_grad()
    def logp_answer(self, ctx: str, answer: str) -> float:
        ids_ctx = self._ids(ctx, special=True)
        ids_a = self._ids(answer, special=False)
        ids = (ids_ctx + ids_a)[-self.max_len:]
        n_a = min(len(ids_a), len(ids) - 1)
        x = torch.tensor([ids], device=self.model.device)
        logits = self.model(x).logits[0].float()
        lp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = x[0, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return float(tok_lp[-n_a:].sum())

    def a_prob(self, question, answer, reasoning) -> float:
        lp_wr = self.logp_answer(
            SCORING_CTX_WITH_R.format(question=question, reasoning=reasoning),
            answer)
        lp_wo = self.logp_answer(
            SCORING_CTX_NO_R.format(question=question), answer)
        n_a = len(self._ids(answer))
        return (lp_wr - lp_wo) / (max(n_a, 1) * LN2)

    def a_prob_segments(self, question, answer, segments: dict) -> dict:
        """FDB 分段增量 bit gain：Δseg = [lp(A|Q,R≤seg) - lp(A|Q,R<seg)] / |A|ln2。
        segments 按生成顺序，如 {"r_fwd": ..., "bridge": ...}。"""
        n_a = max(len(self._ids(answer)), 1)
        lp_prev = self.logp_answer(
            SCORING_CTX_NO_R.format(question=question), answer)
        out, acc = {}, ""
        for name, seg in segments.items():
            acc = (acc + "\n\n" + seg).strip()
            lp = self.logp_answer(
                SCORING_CTX_WITH_R.format(question=question, reasoning=acc),
                answer)
            out[name] = (lp - lp_prev) / (n_a * LN2)
            lp_prev = lp
        return out

    @torch.no_grad()
    def exact_entropy(self, gen_ctx: str, reasoning: str,
                      chunk=512):
        """B11/B12 专用：在【生成时的真实上下文】gen_ctx 下 teacher-force R，
        返回 (每 token 全词表精确熵 list, R 的 token_ids)。逐 chunk 算 softmax 防爆显存。"""
        ids_ctx = self._ids(gen_ctx, special=True)
        ids_r = self._ids(reasoning, special=False)
        ids = (ids_ctx + ids_r)[-self.max_len:]
        n_r = min(len(ids_r), len(ids) - 1)
        x = torch.tensor([ids], device=self.model.device)
        logits = self.model(x).logits[0]
        # 预测 R 第 j 个 token 的分布在位置 len(ids)-n_r-1+j
        pred = logits[len(ids) - n_r - 1: len(ids) - 1]
        ents = []
        for i in range(0, pred.size(0), chunk):
            lp = torch.log_softmax(pred[i:i + chunk].float(), dim=-1)
            h = -(lp.exp() * lp).sum(-1)
            ents.extend([round(float(v), 4) for v in h])
        return ents, ids_r[-n_r:] if n_r else []


# ============================ CLI ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scorer-model", required=True,
                    help="必须 = 论文目标模型（如 Qwen3-4B-Thinking-2507），全方法统一")
    ap.add_argument("--rescore-entropy", action="store_true",
                    help="B11/B12 必开：忽略生成时 top-k 熵，teacher-forcing 精确熵")
    ap.add_argument("--gen-ctx-field", default="gen_ctx",
                    help="rescore 时记录里存放生成上下文字符串的字段名")
    ap.add_argument("--segments", action="store_true",
                    help="对含 segments 字段的记录（FDB）额外算分段增量 A_prob")
    args = ap.parse_args()

    sc = AnswerScorer(args.scorer_model)
    from common import split_steps_by_blankline

    recs = load_jsonl(args.inp)
    for i, r in enumerate(recs):
        q, a, tr = r["question"], r["answer"], r["trace"]
        r["a_lex"] = a_lex(sc.tok, tr, a)
        r["a_prob"] = sc.a_prob(q, a, tr)

        if args.rescore_entropy:
            ents, ids_r = sc.exact_entropy(r[args.gen_ctx_field], tr)
            spans = split_steps_by_blankline(sc.tok, ids_r)
            r["a_ent"] = a_ent(ents, spans)
        elif "tok_entropy" in r and "step_spans" in r:
            r["a_ent"] = a_ent(r["tok_entropy"], r["step_spans"])
        else:
            r["a_ent"] = float("nan")

        if args.segments and "segments" in r:
            r["a_prob_seg"] = sc.a_prob_segments(q, a, r["segments"])
            r["a_lex_seg"] = {k: a_lex(sc.tok, v, a)
                              for k, v in r["segments"].items()}
        if (i + 1) % 50 == 0:
            print(f"[score] {i + 1}/{len(recs)}")

    save_jsonl(recs, args.out)
    valid = [x for x in recs if not math.isnan(x.get("a_ent", float("nan")))]
    if valid:
        for k in ("a_lex", "a_ent", "a_prob"):
            print(f"{k} x100: {100 * np.mean([v[k] for v in valid]):.1f}")


if __name__ == "__main__":
    main()
