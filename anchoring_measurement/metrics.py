import os
import json
import math
import argparse
import gc
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

# ===========================================
# [Metrics] Hyperparameters
# ===========================================
TIMING_WEIGHT = 0.1
EFFICIENCY_WEIGHT = 0.1
SHAPE_WEIGHT = 0.1
CERTAINTY_WEIGHT = 0.7
LOG10_TO_BITS = 3.321928

SEQ_SEP = '\n\n'
MAX_STEPS_PER_SAMPLE: Optional[int] = None
TORCH_DTYPE = torch.bfloat16
STEP_MICROBATCH = 8
USE_SHARED_PREFIX_PAST = True
COMPUTE_ONLY_SAMPLED_STEPS = True


def split_reasoning(R: str) -> List[str]:
    parts = [p.strip() for p in R.strip().split(SEQ_SEP)]
    return [p for p in parts if p]


def apply_chat_prefix(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def step_idx_generator(n: int):
    if n <= 0:
        yield 0
        return
    if n > 50:
        for i in range(0, min(50, n) + 1): yield i
        last = None
        for i in range(54, n + 1, 4):
            yield i
            last = i
        if last != n: yield n
    else:
        for i in range(n + 1): yield i


def calculate_token_lcs_recall(candidate_ids: List[int], reference_ids: List[int]) -> float:
    if not reference_ids or not candidate_ids: return 0.0
    m, n = len(reference_ids), len(candidate_ids)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference_ids[i - 1] == candidate_ids[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n] / len(reference_ids)


def compute_probabilistic_anchoring(ys: List[float]) -> Dict[str, float]:
    default_result = {"ProbabilisticAnchoring": 0.0, "GainTiming": 0.0, "GainEfficiency": 0.0, "CurveShape": 0.0, "FinalCertainty": 0.0}
    if len(ys) < 2: return default_result
    ys_arr = np.array(ys, dtype=float)
    
    gains = np.diff(ys_arr)
    abs_gains = np.abs(gains)
    total_movement = np.sum(abs_gains)
    DirectGain = float(ys_arr[-1] - ys_arr[0])
    
    if total_movement > 1e-9:
        w_norm = abs_gains / total_movement
        t_indices = np.arange(len(gains))
        expected_t = np.sum(t_indices * w_norm)
        GainTiming = 1.0 - expected_t / max(len(gains) - 1, 1)
        GainEfficiency = np.clip(DirectGain / total_movement, 0.0, 1.0)
    else:
        GainTiming = 0.5 
        GainEfficiency = 0.0 if DirectGain <= 0 else 1.0
        
    ys_min, ys_max = np.min(ys_arr), np.max(ys_arr)
    amplitude = ys_max - ys_min
    if amplitude > 1e-9:
        ys_normalized = (ys_arr - ys_min) / amplitude
        CurveShape = float(np.mean(ys_normalized))
    else:
        CurveShape = 0.5 
        
    final_log_prob = ys_arr[-1]
    FinalCertainty = float(np.exp(np.clip(final_log_prob, -10, 0)))
    
    ProbabilisticAnchoring = (
        TIMING_WEIGHT * GainTiming + EFFICIENCY_WEIGHT * GainEfficiency +
        SHAPE_WEIGHT * CurveShape + CERTAINTY_WEIGHT * FinalCertainty
    )
    return {"ProbabilisticAnchoring": float(ProbabilisticAnchoring), "GainTiming": float(GainTiming), "GainEfficiency": float(GainEfficiency), "CurveShape": float(CurveShape), "FinalCertainty": float(FinalCertainty)}


def compute_entropy_anchoring(token_entropies: List[float], step_boundaries: Optional[List[int]] = None) -> Dict[str, float]:
    default_result = {"EntropyAnchoring": 0.0}
    if len(token_entropies) < 2: return default_result
    H_arr = np.array(token_entropies, dtype=float)
    
    if step_boundaries is None:
        window_size = 50
        step_boundaries = [0] + list(range(window_size, len(H_arr), window_size))
        if step_boundaries[-1] != len(H_arr): step_boundaries.append(len(H_arr))
        
    step_IDs = []
    for i in range(len(step_boundaries) - 1):
        start, end = step_boundaries[i], step_boundaries[i + 1]
        if end > start: step_IDs.append(np.mean(H_arr[start:end]))
        
    step_IDs = np.array(step_IDs)
    if len(step_IDs) < 2: return default_result
    
    ID_min, ID_max = np.min(step_IDs), np.max(step_IDs)
    if ID_max - ID_min > 1e-9: ID_normalized = (step_IDs - ID_min) / (ID_max - ID_min)
    else: ID_normalized = np.zeros_like(step_IDs)
    
    global_variance = np.var(ID_normalized)
    GlobalNonUniformity = min(global_variance / 0.25, 1.0)
    GlobalUniformity = 1.0 - GlobalNonUniformity
    
    deltas = np.abs(np.diff(ID_normalized))
    if len(deltas) > 0:
        mean_delta, std_delta = np.mean(deltas), np.std(deltas)
        if mean_delta > 1e-9: LocalUniformity = np.exp(-std_delta / mean_delta)
        else: LocalUniformity = 1.0
    else: LocalUniformity = 1.0
    LocalNonUniformity = 1.0 - LocalUniformity
    
    EntropyAnchoring = np.sqrt(LocalNonUniformity * GlobalUniformity)
    return {"EntropyAnchoring": float(EntropyAnchoring)}


def get_rank_info() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", int(os.environ["RANK"])))
    return 0, 1, 0


def tok_1d(tokenizer, text: str, device: Optional[torch.device] = None) -> torch.LongTensor:
    return torch.tensor(tokenizer(text, add_special_tokens=False).input_ids, dtype=torch.long).to(device)


def tokenize_reasoning_to_flat_ids(tokenizer, R_text: str, sep_ids: torch.LongTensor, max_steps: Optional[int] = None, device: Optional[torch.device] = None) -> Tuple[int, torch.LongTensor, List[int]]:
    r_list = split_reasoning(R_text)
    if max_steps is not None: r_list = r_list[:max_steps]
    n = len(r_list)
    if n == 0: return 0, torch.empty((0,), dtype=torch.long, device=device), [0]
    flat_list, end_pos = [], [0]
    for i, part in enumerate(r_list):
        if i > 0 and len(sep_ids) > 0: flat_list.extend(sep_ids.tolist())
        flat_list.extend(tokenizer(part, add_special_tokens=False).input_ids)
        end_pos.append(len(flat_list))
    return n, torch.tensor(flat_list, dtype=torch.long, device=device), end_pos[1:] 


@torch.no_grad()
def prepare_prefix_past(model, prefix_ids_1d: torch.LongTensor) -> Tuple[Optional[Any], int]:
    device = next(model.parameters()).device
    inp = prefix_ids_1d.unsqueeze(0).to(device)
    out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=True)
    return getattr(out, "past_key_values", None), int(prefix_ids_1d.numel())


def _repeat_past(past, bsz: int):
    if past is None: return None
    if hasattr(past, "key_cache") and DynamicCache is not None:
        new_past = DynamicCache()
        for i in range(len(past.key_cache)):
            k, v = past.key_cache[i], past.value_cache[i]
            if k.size(0) == 1:
                new_past.key_cache.append(k.expand(bsz, *k.shape[1:]).contiguous())
                new_past.value_cache.append(v.expand(bsz, *v.shape[1:]).contiguous())
            else:
                new_past.key_cache.append(k); new_past.value_cache.append(v)
        return new_past
    if not isinstance(past, (tuple, list)): return past
    new = []
    for layer in past:
        if isinstance(layer, (tuple, list)) and len(layer) == 2:
            k, v = layer
            if torch.is_tensor(k) and torch.is_tensor(v) and k.size(0) == 1:
                new.append((k.expand(bsz, *k.shape[1:]).contiguous(), v.expand(bsz, *v.shape[1:]).contiguous()))
            else: new.append(layer)
        else: new.append(layer)
    return tuple(new)


@torch.no_grad()
def robust_batch_answer_logprob_with_shared_past(model, past_prefix, past_len: int, cont_ids_list: List[torch.LongTensor], ans_start_list: List[int], ans_len: int, batch_size: int) -> List[float]:
    if batch_size <= 0: batch_size = 1
    device = next(model.parameters()).device
    out_scores = []
    try:
        for i0 in range(0, len(cont_ids_list), batch_size):
            cont_chunk, ans_start_chunk = cont_ids_list[i0:i0 + batch_size], ans_start_list[i0:i0 + batch_size]
            bsz = len(cont_chunk)
            max_len = max(int(x.numel()) for x in cont_chunk)
            input_ids, cur_attn = torch.zeros((bsz, max_len), dtype=torch.long, device=device), torch.zeros((bsz, max_len), dtype=torch.long, device=device)
            for b, x in enumerate(cont_chunk):
                L = int(x.numel())
                input_ids[b, :L], cur_attn[b, :L] = x.to(device), 1
            attn = torch.cat([torch.ones((bsz, past_len), dtype=torch.long, device=device), cur_attn], dim=1)
            past_rep = _repeat_past(past_prefix, bsz)
            
            logits = model(input_ids=input_ids, attention_mask=attn, past_key_values=past_rep, use_cache=False).logits
            logp = torch.log_softmax(logits, dim=-1)
            for b in range(bsz):
                ans_start = int(ans_start_chunk[b])
                pred_pos = torch.arange(ans_start - 1, ans_start + ans_len - 1, device=device)
                target = input_ids[b, ans_start: ans_start + ans_len]
                sel = logp[b, pred_pos, :].gather(1, target.unsqueeze(1)).squeeze(1)
                out_scores.append(float(sel.sum().item()))
            del past_rep, logits, input_ids, attn
        return out_scores
    except torch.OutOfMemoryError:
        if batch_size == 1: raise
        torch.cuda.empty_cache(); gc.collect()
        return robust_batch_answer_logprob_with_shared_past(model, past_prefix, past_len, cont_ids_list, ans_start_list, ans_len, max(1, batch_size // 2))


@torch.no_grad()
def batch_answer_logprob_no_past(model, prompt_ids_list: List[torch.LongTensor], ans_ids: torch.LongTensor, max_ctx: Optional[int], batch_size: int) -> List[float]:
    device = next(model.parameters()).device
    ans_ids = ans_ids.to(device); Lans = int(ans_ids.numel())
    out_scores = []
    for i0 in range(0, len(prompt_ids_list), batch_size):
        chunk = prompt_ids_list[i0:i0 + batch_size]
        seqs, prompt_lens = [], []
        for p in chunk:
            full = torch.cat([p.to(device), ans_ids], dim=0)
            if max_ctx is not None and full.numel() > max_ctx: full = full[-int(max_ctx):]
            seqs.append(full); prompt_lens.append(int(full.numel()) - Lans)
        max_len = max(int(s.numel()) for s in seqs)
        input_ids, attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device), torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for b, s in enumerate(seqs):
            L = int(s.numel()); input_ids[b, :L], attn[b, :L] = s, 1
        logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        logp = torch.log_softmax(logits, dim=-1)
        for b in range(len(seqs)):
            pl = prompt_lens[b]
            pred_pos = torch.arange(pl - 1, pl + Lans - 1, device=device)
            sel = logp[b, pred_pos, :].gather(1, input_ids[b, pl:pl+Lans].unsqueeze(1)).squeeze(1)
            out_scores.append(float(sel.sum().item()))
        del logits, input_ids, attn
    return out_scores


@torch.no_grad()
def compute_reasoning_token_entropies(model, flat_reason_ids: torch.LongTensor, past_prefix, batch_size: int = 512) -> List[float]:
    device = next(model.parameters()).device
    L = int(flat_reason_ids.numel())
    if L == 0: return []
    entropies, current_past = [], past_prefix
    current_seq_len = current_past.key_cache[0].shape[-2] if hasattr(current_past, "key_cache") else current_past[0][0].shape[-2]

    for i in range(0, L, batch_size):
        chunk_ids = flat_reason_ids[i:min(i + batch_size, L)]
        chunk_len = int(chunk_ids.numel())
        input_ids = chunk_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones((1, current_seq_len + chunk_len), dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=_repeat_past(current_past, 1), use_cache=True)
        
        logits = outputs.logits
        probs, log_probs = torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)
        entropies.extend((-torch.sum(probs * log_probs, dim=-1)).squeeze(0).cpu().tolist())
        
        current_past = outputs.past_key_values
        current_seq_len += chunk_len
        del logits, probs, log_probs, outputs, input_ids, attention_mask
    return entropies


def compute_absolute_curve_for_R_sampled(model, tokenizer, prefix_ids, past_prefix, past_len, think_close_ids, ans_ids, max_ctx, flat_reason_ids, end_pos, step_idxs, batch_size):
    if not end_pos: return [], [], [], 0
    if not COMPUTE_ONLY_SAMPLED_STEPS: step_idxs = list(range(0, len(end_pos) + 1))
    close_len, ans_len = int(think_close_ids.numel()), int(ans_ids.numel())
    xs, ys = [], []
    
    if USE_SHARED_PREFIX_PAST and past_prefix is not None:
        cont_list, ans_start_list = [], []
        for s in step_idxs:
            r_end = 0 if s == 0 else int(end_pos[s - 1]) if (s - 1) < len(end_pos) else int(end_pos[-1])
            xs.append(r_end)
            cont_list.append(torch.cat([flat_reason_ids[:r_end], think_close_ids, ans_ids], dim=0))
            ans_start_list.append(r_end + close_len)
        sum_logps = robust_batch_answer_logprob_with_shared_past(model, past_prefix, past_len, cont_list, ans_start_list, ans_len, batch_size)
    else:
        prompt_ids_list = []
        for s in step_idxs:
            r_end = 0 if s == 0 else int(end_pos[s - 1]) if (s - 1) < len(end_pos) else int(end_pos[-1])
            xs.append(r_end)
            prompt_ids_list.append(torch.cat([prefix_ids, flat_reason_ids[:r_end], think_close_ids], dim=0))
        sum_logps = batch_answer_logprob_no_past(model, prompt_ids_list, ans_ids, max_ctx, batch_size)
        
    for lp in sum_logps: ys.append(float(lp / math.log(10)))
    return step_idxs, xs, ys, ans_len


def has_nan_in_metrics(metrics_record: Dict[str, Any]) -> bool:
    """检查指标记录中是否包含NaN值"""
    for key, value in metrics_record.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return True
    return False


def process_single_method(model, tokenizer, device, max_ctx, think_open_ids, think_close_ids, reason_sep_ids, 
                          method: str, obj: Dict, line_idx: int) -> Optional[Dict[str, Any]]:
    """处理单个方法，返回指标记录或None（如果失败或产生NaN）"""
    try:
        qs_dict = obj.get('questions', {})
        Q = qs_dict.get(method)
        A = obj.get('answers', {}).get(method)
        new_R = obj.get('reasonings', {}).get(method, "")
        C = obj.get('contexts', {}).get(method, "")
        
        if not (Q and A and new_R):
            return None
        
        ans_ids = tok_1d(tokenizer, A, device)
        if int(ans_ids.numel()) <= 0:
            return None
        
        chat_prefix_ids_q = tok_1d(tokenizer, apply_chat_prefix(tokenizer, Q), device)
        prefix_ids_q = torch.cat([chat_prefix_ids_q, think_open_ids], dim=0)
        past_prefix_q, past_len_q = prepare_prefix_past(model, prefix_ids_q) if USE_SHARED_PREFIX_PAST else (None, int(prefix_ids_q.numel()))
        
        n_parts_n, flat_n, end_pos_n = tokenize_reasoning_to_flat_ids(tokenizer, new_R, reason_sep_ids, MAX_STEPS_PER_SAMPLE, device)
        if int(flat_n.numel()) <= 0:
            return None
        
        step_idxs_n = list(step_idx_generator(n_parts_n)) if COMPUTE_ONLY_SAMPLED_STEPS else list(range(n_parts_n + 1))
        _, xs_new, ys_total, ans_len = compute_absolute_curve_for_R_sampled(
            model, tokenizer, prefix_ids_q, past_prefix_q, past_len_q, 
            think_close_ids, ans_ids, max_ctx, flat_n, end_pos_n, step_idxs_n, STEP_MICROBATCH
        )
        
        if not xs_new:
            return None
        
        # 1. Lexical & Probabilistic
        lexical_score = calculate_token_lcs_recall(flat_n.cpu().tolist(), ans_ids.cpu().tolist())
        ys_rate = [(y / ans_len) * LOG10_TO_BITS for y in ys_total] if ans_len > 0 else ys_total
        prob_metrics = compute_probabilistic_anchoring(ys_rate)
        
        # 2. Entropy
        entropy_metrics = {"EntropyAnchoring": 0.0}
        prompt_content_c = C if C and C.strip() else Q
        if prompt_content_c:
            chat_prefix_ids_c = tok_1d(tokenizer, apply_chat_prefix(tokenizer, prompt_content_c), device)
            entropy_prefix_ids = torch.cat([chat_prefix_ids_c, think_open_ids], dim=0)
            past_prefix_c, _ = prepare_prefix_past(model, entropy_prefix_ids) if USE_SHARED_PREFIX_PAST else (None, 0)
            
            token_entropies = compute_reasoning_token_entropies(model, flat_n, past_prefix_c, 512)
            if token_entropies:
                entropy_metrics = compute_entropy_anchoring(token_entropies, end_pos_n)
        
        # 构建指标记录
        metrics_record = {
            "method": method,
            "sample_idx": line_idx,
            "lexical_anchoring": lexical_score,
            "entropy_anchoring": entropy_metrics["EntropyAnchoring"],
            **prob_metrics
        }
        
        # 检查是否有NaN
        if has_nan_in_metrics(metrics_record):
            return None
        
        return metrics_record
        
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Calculate Anchoring Metrics (with NaN filtering)")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--scoring-model", type=str, required=True, help="Model path/name")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    rank, world_size, local_rank = get_rank_info()
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, f"metrics_rank{rank}.jsonl")
    valid_data_file = os.path.join(args.output_dir, f"valid_data_rank{rank}.jsonl")
    
    print(f"[rank{rank}] Loading model from: {args.scoring_model}")
    cfg = AutoConfig.from_pretrained(args.scoring_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.scoring_model, config=cfg, trust_remote_code=True, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.scoring_model, config=cfg, trust_remote_code=True, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True)
    
    model.to(device); model.eval()
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    
    think_open_ids = tok_1d(tokenizer, "<think>\n", device)
    think_close_ids = tok_1d(tokenizer, "</think>\n\n", device)
    reason_sep_ids = tok_1d(tokenizer, SEQ_SEP, device)

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % world_size == rank:
                samples.append((i, json.loads(line)))

    # 统计信息
    total_samples = len(samples)
    valid_samples = 0
    nan_samples = 0

    with open(metrics_file, "w", encoding="utf-8") as fw_metrics, \
         open(valid_data_file, "w", encoding="utf-8") as fw_valid:
        
        for line_idx, obj in tqdm(samples, desc=f"Processing (rank{rank})", disable=(rank != 0)):
            qs_dict = obj.get('questions', {})
            methods = list(qs_dict.keys())  # 自动推断方法
            
            if not methods:
                continue
            
            # 收集当前样本所有方法的结果
            sample_metrics = []
            sample_valid = True
            
            for m in methods:
                metrics_record = process_single_method(
                    model, tokenizer, device, max_ctx,
                    think_open_ids, think_close_ids, reason_sep_ids,
                    m, obj, line_idx
                )
                
                if metrics_record is None:
                    # 任意一个方法失败或产生NaN，标记整个样本为无效
                    sample_valid = False
                    break
                
                sample_metrics.append(metrics_record)
            
            # 只有当所有方法都成功且无NaN时才保存
            if sample_valid and len(sample_metrics) == len(methods):
                # 写入所有方法的指标
                for metrics_record in sample_metrics:
                    fw_metrics.write(json.dumps(metrics_record) + "\n")
                fw_metrics.flush()
                
                # 写入有效的原始数据
                fw_valid.write(json.dumps(obj) + "\n")
                fw_valid.flush()
                
                valid_samples += 1
            else:
                nan_samples += 1
            
            gc.collect()
            torch.cuda.empty_cache()
    
    # 打印统计信息
    print(f"\n[rank{rank}] Processing complete:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid samples (all methods non-NaN): {valid_samples}")
    print(f"  Discarded samples (NaN/error in any method): {nan_samples}")
    print(f"  Metrics saved to: {metrics_file}")
    print(f"  Valid data saved to: {valid_data_file}")


if __name__ == "__main__":
    main()
