import argparse
import json
import os
from collections import defaultdict
import numpy as np

# 计算代码实际写入 JSONL 的字段名（camelCase）
PROB_ANCHORING_KEY = "ProbabilisticAnchoring"
ENT_ANCHORING_KEY  = "entropy_anchoring"   # 计算代码此字段已是 snake_case
LEX_ANCHORING_KEY  = "lexical_anchoring"

def main():
    parser = argparse.ArgumentParser(description="Aggregate Anchoring Metrics")
    parser.add_argument("--input", type=str, required=True, help="Input metrics.jsonl file or directory")
    parser.add_argument("--group-by", type=str, default="method", help="Field to group by")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    input_files = [args.input] if os.path.isfile(args.input) else [
        os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.jsonl')
    ]

    data = defaultdict(lambda: defaultdict(list))
    skipped = defaultdict(int)

    for filepath in input_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                group_val = record.get(args.group_by, "unknown")

                numeric_fields = {
                    k: v for k, v in record.items()
                    if isinstance(v, (int, float)) and k != "sample_idx"
                }

                # 任意数值字段为 nan/inf，跳过整条记录
                if any(not np.isfinite(v) for v in numeric_fields.values()):
                    skipped[group_val] += 1
                    continue

                for k, v in numeric_fields.items():
                    data[group_val][k].append(v)

    summary = {}
    print(f"\n{'='*90}\nAnchoring Metrics Summary (Means)\n{'='*90}")
    print(f"{'Method':<20} | {'N':<6} | {'Skipped':<8} | {'A_lex':<10} | {'A_ent':<10} | {'A_prob':<10}")
    print("-" * 90)

    for group_val, metrics in data.items():
        if not metrics:
            continue

        # 使用计算代码实际写入的 key 名
        prob_list = metrics.get(PROB_ANCHORING_KEY, [])
        if not prob_list:
            print(f"[WARN] group '{group_val}' has no '{PROB_ANCHORING_KEY}' field, skipping.")
            continue

        n = len(prob_list)
        means = {k: float(np.mean(v)) for k, v in metrics.items()}

        # 象限计算：同样使用实际 key 名
        x = np.array(metrics.get(ENT_ANCHORING_KEY, []))
        y = np.array(metrics.get(PROB_ANCHORING_KEY, []))
        quadrants = {
            "Reason": float(np.mean((x < 0.5) & (y < 0.5))),
            "Encode": float(np.mean((x < 0.5) & (y >= 0.5))),
            "Cloze":  float(np.mean((x >= 0.5) & (y < 0.5))),
            "Copy":   float(np.mean((x >= 0.5) & (y >= 0.5)))
        } if len(x) == len(y) and len(x) > 0 else {}

        summary[group_val] = {
            "N":           n,
            "skipped_nan": skipped[group_val],
            "means":       means,
            "quadrants":   quadrants
        }

        print(
            f"{group_val:<20} | {n:<6} | {skipped[group_val]:<8} | "
            f"{means.get(LEX_ANCHORING_KEY,  0):.4f}     | "
            f"{means.get(ENT_ANCHORING_KEY,  0):.4f}     | "
            f"{means.get(PROB_ANCHORING_KEY, 0):.4f}"
        )

    if not summary:
        print("\n[ERROR] summary is empty — check that the JSONL keys match the constants at the top of this file.")
        print(f"  Expected probabilistic key : '{PROB_ANCHORING_KEY}'")
        print(f"  Expected entropy key       : '{ENT_ANCHORING_KEY}'")
        print(f"  Expected lexical key       : '{LEX_ANCHORING_KEY}'")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()