# -*- coding: utf-8 -*-
"""
共享工具：所有 Tier 1 pipeline 复用。
关键约定：
- 生成时如用 vLLM 且分布未被修改（A1/A6/B13），熵直接从 top-20 logprobs 采集
  （截断近似，绝对值偏低但跨方法可比）；
- B11/B12 修改了解码分布，A_ent 必须用 metrics_anchoring.py 的 --rescore-entropy
  以原模型 teacher-forcing 重算（全词表精确熵）。
"""
import json
import math
import os
from typing import Optional


# ----------------------------- IO -----------------------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------- chat ---------------------------------------

def build_chat(tokenizer, system: Optional[str], user: str,
               enable_thinking: Optional[bool] = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    kw = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                msgs, enable_thinking=enable_thinking, **kw)
        except TypeError:
            pass  # 非 Qwen3-thinking 系模板不认这个参数
    return tokenizer.apply_chat_template(msgs, **kw)


# ------------------------ vLLM 生成 + 熵采集 --------------------------------

def vllm_generate(llm, prompts, temperature=0.8, top_p=0.95,
                  max_tokens=8192, n=1, logprobs=20, seed=None):
    """返回 list[list[dict]]：外层对齐 prompts，内层 n 个候选。
    每个候选: {text, token_ids, tok_entropy}（logprobs=None 时无 tok_entropy）。"""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, n=n,
                        logprobs=logprobs, seed=seed)
    outs = llm.generate(prompts, sp)
    results = []
    for o in outs:
        cands = []
        for c in o.outputs:
            rec = {"text": c.text, "token_ids": list(c.token_ids)}
            if logprobs and c.logprobs is not None:
                rec["tok_entropy"] = [
                    round(_entropy_from_top(d), 4) for d in c.logprobs]
            cands.append(rec)
        results.append(cands)
    return results


def _entropy_from_top(logprob_dict):
    """top-k 截断熵（nats）。d: {token_id: Logprob}。"""
    ps = [math.exp(v.logprob) for v in logprob_dict.values()]
    return -sum(p * math.log(max(p, 1e-12)) for p in ps)


# ------------------------ token → step 切分 --------------------------------

def split_steps_by_blankline(tokenizer, token_ids, min_step_tokens=5):
    """按 '\\n\\n' 把生成 token 序列切成推理步，返回 [(start, end), ...)（半开区间）。
    逐 token decode 累积检测边界；过短步与前一步合并。"""
    spans, start, buf = [], 0, ""
    for i, tid in enumerate(token_ids):
        buf += tokenizer.decode([tid], skip_special_tokens=False)
        if "\n\n" in buf:
            spans.append((start, i + 1))
            start, buf = i + 1, ""
    if start < len(token_ids):
        spans.append((start, len(token_ids)))
    merged = []
    for s, e in spans:
        if merged and (e - s) < min_step_tokens:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# -------------------- trace 子串 ↔ token 区间对齐 ---------------------------

def per_token_concat_text(tokenizer, token_ids):
    """逐 token decode 的累积文本与各 token 的字符区间。
    注意：多字节 unicode 跨 token 时与联合 decode 有细微差异（英文数据可忽略），
    子串搜索必须在这份 concat 文本上做而非 vLLM 的 c.text。"""
    pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in token_ids]
    offs, pos = [], 0
    for p in pieces:
        offs.append((pos, pos + len(p)))
        pos += len(p)
    return "".join(pieces), offs


def substring_token_span(tokenizer, token_ids, sub: str):
    """返回 sub 对应的 token 半开区间 (s, e)；找不到返回 None。"""
    full, offs = per_token_concat_text(tokenizer, token_ids)
    c0 = full.find(sub)
    if c0 < 0:
        # 兜底：长 trace 用首尾各 80 字符锚定
        head = full.find(sub[:80])
        tail = full.find(sub[-80:])
        if head < 0 or tail < 0:
            return None
        c0, c1 = head, tail + 80
    else:
        c1 = c0 + len(sub)
    s = next((i for i, (a, b) in enumerate(offs) if b > c0), None)
    e = None
    for i in range(len(offs) - 1, -1, -1):
        if offs[i][0] < c1:
            e = i + 1
            break
    return (s, e) if s is not None and e is not None and e > s else None


# ------------------------ judge 批量调用 ------------------------------------

def vllm_judge(llm, tokenizer, system, users, max_tokens=512):
    """judge 模型批量打分，强制 JSON 解析，失败返回 None 占位。"""
    prompts = [build_chat(tokenizer, system, u, enable_thinking=False)
               for u in users]
    outs = vllm_generate(llm, prompts, temperature=0.0, top_p=1.0,
                         max_tokens=max_tokens, logprobs=None)
    parsed = []
    for cands in outs:
        t = cands[0]["text"]
        try:
            j = t[t.index("{"): t.rindex("}") + 1]
            parsed.append(json.loads(j))
        except (ValueError, json.JSONDecodeError):
            parsed.append(None)
    return parsed
