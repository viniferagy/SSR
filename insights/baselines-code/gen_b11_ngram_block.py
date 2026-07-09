# -*- coding: utf-8 -*-
"""
B11. N-gram blocking（机械词面抑制）—— ironic process 的判别实验
核心设定：prompt 用 NEU 原版，模型对抑制【完全不知情】；
抑制只发生在 logits 层（禁掉 A 的 content n-gram）。

判别逻辑：
  机械抑制 A_ent/A_prob ≈ NEU 且 SUP 反弹  -> ironic process 成立且精确到"指令诱发监控"
  机械抑制同样反弹                          -> 反弹是任何抑制的副产物，4.4 节需改写

两种引擎：
  --engine hf    自定义 processor，支持 --unblock-tail-frac（末段解禁，trace 才能收尾）
  --engine vllm  SamplingParams(bad_words=...) per-request，全程禁、可扩到大规模

注意：B11 修改了部分位置的解码分布，A_ent 必须 rescore：
  python metrics_anchoring.py --in out.jsonl --out scored.jsonl \
      --scorer-model <目标模型> --rescore-entropy --gen-ctx-field gen_ctx
"""
import argparse
import string

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList)

from common import build_chat, load_jsonl, save_jsonl
from prompts_tier1 import NEU_SYSTEM, NEU_USER_WITH, parse_neu

STOPWORDS = set("""a an the of to in on at for with by from as and or but if then
than that this these those it its is are was were be been being do does did not no
so such can could may might will would shall should must have has had i you he she
we they them his her their our your my me him us what which who whom how when where
why there here also into over under about above below between during after before
all any both each few more most other some only own same very just""".split())
PUNCT = set(string.punctuation)


# -------------------- 禁用集构建（per-sample） ------------------------------

def is_content_ngram(text: str, max_stop_frac=2 / 3) -> bool:
    words = [w.strip("".join(PUNCT)).lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return False
    n_stop = sum(1 for w in words if w in STOPWORDS or not any(c.isalnum() for c in w))
    return n_stop / len(words) < max_stop_frac


def banned_ngrams_token(tok, answer: str, ns=(3, 4)):
    """token 级禁用 n-gram 集合（含前导空格再分词变体，缓解边界 token 不一致）。"""
    ids = tok.encode(answer, add_special_tokens=False)
    out = set()
    for n in ns:
        for i in range(len(ids) - n + 1):
            win = tuple(ids[i:i + n])
            text = tok.decode(list(win))
            if not is_content_ngram(text):
                continue
            out.add(win)
            for v in (text, " " + text.lstrip()):
                enc = tuple(tok.encode(v, add_special_tokens=False))
                if len(enc) == n:
                    out.add(enc)
    return out


def banned_ngrams_str(tok, answer: str, ns=(3, 4)):
    """vLLM bad_words 用字符串形式（vLLM 内部自行处理空格分词变体）。"""
    ids = tok.encode(answer, add_special_tokens=False)
    out = set()
    for n in ns:
        for i in range(len(ids) - n + 1):
            text = tok.decode(ids[i:i + n]).strip()
            if text and is_content_ngram(text):
                out.add(text)
    return sorted(out)


# -------------------- HF 自定义 processor ----------------------------------

class PerRowNoBadNgrams(LogitsProcessor):
    """每行独立禁用集；前缀 (n-1) token 命中则 ban 后继 token。
    unblock_at: 生成长度达到该值后整行解禁（末段收尾用）。"""

    def __init__(self, ban_maps, init_len: int, unblock_at=None):
        # ban_maps[i]: dict[tuple(prefix_ids) -> set(next_ids)]
        self.maps = ban_maps
        self.init_len = init_len
        self.unblock_at = unblock_at
        self.prefix_lens = sorted({len(k) for m in ban_maps for k in m} | {0})

    def __call__(self, input_ids, scores):
        gen_len = input_ids.shape[1] - self.init_len
        if self.unblock_at is not None and gen_len >= self.unblock_at:
            return scores
        for i in range(input_ids.shape[0]):
            m = self.maps[i]
            if not m:
                continue
            row = input_ids[i].tolist()
            for pl in self.prefix_lens:
                if pl == 0 or pl > len(row):
                    continue
                nxt = m.get(tuple(row[-pl:]))
                if nxt:
                    scores[i, list(nxt)] = float("-inf")
        return scores


def to_ban_map(ngrams):
    m = {}
    for g in ngrams:
        m.setdefault(g[:-1], set()).add(g[-1])
    return m


# -------------------- 主流程 ------------------------------------------------

def run_hf(args, data, tok):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    recs = []
    for b0 in range(0, len(data), args.batch_size):
        batch = data[b0:b0 + args.batch_size]
        prompts = [build_chat(tok, NEU_SYSTEM,
                              NEU_USER_WITH.format(question=d["question"],
                                                   answer=d["answer"]))
                   for d in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        maps = [to_ban_map(banned_ngrams_token(tok, d["answer"]))
                for d in batch]
        unblock_at = (int(args.max_new_tokens * (1 - args.unblock_tail_frac))
                      if args.unblock_tail_frac > 0 else None)
        proc = PerRowNoBadNgrams(maps, enc.input_ids.shape[1], unblock_at)

        out = model.generate(
            **enc, do_sample=True, temperature=args.temperature, top_p=0.95,
            max_new_tokens=args.max_new_tokens,
            logits_processor=LogitsProcessorList([proc]),
            pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        for j, (d, g, p) in enumerate(zip(batch, gen, prompts)):
            text = tok.decode(g, skip_special_tokens=True)
            recs.append(_record(d, text, p, args,
                                n_ban=sum(len(v) for v in maps[j].values())))
        print(f"[hf] {min(b0 + args.batch_size, len(data))}/{len(data)}")
    return recs


def run_vllm(args, data, tok):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.92)
    prompts, sps = [], []
    for d in data:
        prompts.append(build_chat(tok, NEU_SYSTEM,
                                  NEU_USER_WITH.format(question=d["question"],
                                                       answer=d["answer"])))
        bw = banned_ngrams_str(tok, d["answer"])
        sps.append(SamplingParams(temperature=args.temperature, top_p=0.95,
                                  max_tokens=args.max_new_tokens,
                                  bad_words=bw))
    outs = llm.generate(prompts, sps)
    return [_record(d, o.outputs[0].text, p, args, n_ban=len(sp.bad_words))
            for d, o, p, sp in zip(data, outs, prompts, sps)]


def _record(d, text, gen_ctx, args, n_ban):
    return {"qid": d.get("qid"), "question": d["question"],
            "answer": d["answer"],
            "method": f"ngram_block_{args.engine}"
                      + (f"_tail{args.unblock_tail_frac}"
                         if args.unblock_tail_frac > 0 else "_full"),
            "trace": parse_neu(text), "raw_output": text,
            "gen_ctx": gen_ctx, "n_banned_ngrams": n_ban}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unblock-tail-frac", type=float, default=0.0,
                    help="0=全程禁; 0.1=末 10%% token 解禁（仅 hf 引擎支持）")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=32768)
    args = ap.parse_args()
    if args.engine == "vllm" and args.unblock_tail_frac > 0:
        raise SystemExit("末段解禁变体请用 --engine hf")

    data = load_jsonl(args.input)
    tok = AutoTokenizer.from_pretrained(args.model)
    recs = run_hf(args, data, tok) if args.engine == "hf" else run_vllm(args, data, tok)
    save_jsonl(recs, args.out)
    print(f"done -> {args.out}\n之后务必用 --rescore-entropy 打分（解码分布被修改过）")


if __name__ == "__main__":
    main()
