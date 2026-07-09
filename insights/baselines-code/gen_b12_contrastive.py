# -*- coding: utf-8 -*-
"""
B12. 对比 / 负向引导解码（CFG 反向用法）
  l_final = l_wo + γ · (l_w − l_wo)
  l_w  = logit(t | Q, A, R_<t)   l_wo = logit(t | Q, R_<t)
  γ=1 还原 NEU；γ∈{0.3,0.5,0.7} 削弱 A；γ=1.5 可作"超锚定"诊断点。
  --anneal-frac f: 末段 f 比例 token 内 γ 线性退火回 1（终点对齐）。

两个上下文除 REFERENCE ANSWER 块外指令逐字相同（prompts_tier1 保证），
共享生成 token 流在双上下文下均 in-distribution。

实现要点（坑都在这）：
  - 左 padding + 显式 position_ids（HF 在带 cache 时默认按 past_len 排位置，
    左 padding 下是错的，必须自己传每行非 pad 计数）；
  - done 行后续 mask 置 0，不让垃圾 token 进注意力；
  - A_ent 必须 rescore（生成分布是插值分布，不是原模型），rescore 上下文
    用 with 版（与其他方法同在 (Q,A) 条件下测）。

用法：
  python gen_b12_contrastive.py --model Qwen/Qwen3-4B-Thinking-2507 \
      --input data/analysis_2k.jsonl --out out/b12_g0.5.jsonl \
      --gamma 0.5 --anneal-frac 0.15 --batch-size 8
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_chat, load_jsonl, save_jsonl
from prompts_tier1 import (NEU_SYSTEM, NEU_USER_WITH, NEU_USER_WITHOUT,
                           parse_neu)


def sample_top_p(logits, temperature, top_p):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits.float(), dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = sp.cumsum(-1)
    sp[cum - sp > top_p] = 0.0
    sp = sp / sp.sum(-1, keepdim=True)
    return si.gather(-1, torch.multinomial(sp, 1)).squeeze(-1)


def gamma_at(t, gamma0, max_new, anneal_frac):
    start = int(max_new * (1 - anneal_frac))
    if anneal_frac <= 0 or t < start:
        return gamma0
    return gamma0 + (1.0 - gamma0) * (t - start) / max(max_new - start, 1)


def prefill(model, tok, prompts, device):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                position_ids=pos, use_cache=True)
    counts = enc.attention_mask.sum(-1)              # 每行下一个 position id
    return out.logits[:, -1, :], out.past_key_values, enc.attention_mask, counts


@torch.no_grad()
def contrastive_batch(model, tok, batch, args, device):
    pw = [build_chat(tok, NEU_SYSTEM,
                     NEU_USER_WITH.format(question=d["question"],
                                          answer=d["answer"]))
          for d in batch]
    po = [build_chat(tok, NEU_SYSTEM,
                     NEU_USER_WITHOUT.format(question=d["question"]))
          for d in batch]

    lw, cache_w, am_w, cnt_w = prefill(model, tok, pw, device)
    lo, cache_o, am_o, cnt_o = prefill(model, tok, po, device)

    eos_ids = tok.eos_token_id
    eos_ids = set(eos_ids) if isinstance(eos_ids, (list, tuple)) else {eos_ids}
    B = lw.size(0)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    gen = [[] for _ in range(B)]

    for t in range(args.max_new_tokens):
        g = gamma_at(t, args.gamma, args.max_new_tokens, args.anneal_frac)
        logits = lo + g * (lw - lo)
        nxt = sample_top_p(logits, args.temperature, args.top_p)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
        for i in range(B):
            if not done[i]:
                gen[i].append(int(nxt[i]))
        done = done | torch.tensor([int(nxt[i]) in eos_ids for i in range(B)],
                                   device=device)
        if done.all():
            break

        step_mask = (~done).long().unsqueeze(1)
        am_w = torch.cat([am_w, step_mask], dim=1)
        am_o = torch.cat([am_o, step_mask], dim=1)
        ids_step = nxt.unsqueeze(1)
        ow = model(input_ids=ids_step, attention_mask=am_w,
                   position_ids=cnt_w.unsqueeze(1),
                   past_key_values=cache_w, use_cache=True)
        oo = model(input_ids=ids_step, attention_mask=am_o,
                   position_ids=cnt_o.unsqueeze(1),
                   past_key_values=cache_o, use_cache=True)
        cache_w, cache_o = ow.past_key_values, oo.past_key_values
        lw, lo = ow.logits[:, -1, :], oo.logits[:, -1, :]
        cnt_w = cnt_w + step_mask.squeeze(1)
        cnt_o = cnt_o + step_mask.squeeze(1)

    recs = []
    for d, ids, ctx in zip(batch, gen, pw):
        ids = [i for i in ids if i not in eos_ids]
        text = tok.decode(ids, skip_special_tokens=True)
        recs.append({"qid": d.get("qid"), "question": d["question"],
                     "answer": d["answer"],
                     "method": f"contrastive_g{args.gamma}"
                               + (f"_anneal{args.anneal_frac}"
                                  if args.anneal_frac > 0 else ""),
                     "trace": parse_neu(text), "raw_output": text,
                     "gen_ctx": ctx})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--anneal-frac", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    device = model.device

    data = load_jsonl(args.input)
    recs = []
    for b0 in range(0, len(data), args.batch_size):
        recs += contrastive_batch(model, tok, data[b0:b0 + args.batch_size],
                                  args, device)
        print(f"[b12 g={args.gamma}] {len(recs)}/{len(data)}")
    save_jsonl(recs, args.out)
    print(f"done -> {args.out}\n务必 --rescore-entropy 打分；并跑终点一致性 judge"
          "（γ 小时偏离率是关键副指标）")


if __name__ == "__main__":
    main()
