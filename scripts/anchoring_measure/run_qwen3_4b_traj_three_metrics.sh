#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-runs/qwen3_4b_traj_three_metrics}"
MODEL="${MODEL:-/home/pengguangyue/workspace/models/Qwen/Qwen3-4B-Thinking-2507}"
PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
NPROC="${NPROC:-4}"
INPUT="${INPUT:-runs/mismatch_dimensions_all_variants/inputs/mismatch_variants_1000.metric.jsonl}"
TRAJ_CSV="${TRAJ_CSV:-runs/mismatch_dimensions_all_variants_qwen3_4b_thinking_2507/n0512/commitment_confidence_results/commitment_means.csv}"

METRICS_DIR="$RUN_DIR/prob_metrics"
RESULTS_DIR="$RUN_DIR/results"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$METRICS_DIR" "$RESULTS_DIR" "$LOG_DIR"

cat > "$RUN_DIR/run_manifest.json" <<JSON
{
  "model": "$MODEL",
  "input": "$INPUT",
  "prob_metrics_dir": "$METRICS_DIR",
  "traj_csv": "$TRAJ_CSV",
  "results_dir": "$RESULTS_DIR",
  "nproc": $NPROC,
  "definition_alignment": {
    "source_plan": "/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/insights/claude.traj.md",
    "A_lex": "Content_IDF, IDF-weighted answer-content recall in reasoning",
    "A_traj": "100 * ConfidenceGap_mean from Qwen3-4B mismatch diagnostics",
    "A_prob": "clipped normalized answer-surprisal reduction scored by Qwen3-4B-Thinking-2507"
  }
}
JSON

echo "[run] scoring A_prob with Qwen3-4B scorer"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  scripts/anchoring_measure/prob_lex_traj_metrics.py score-prob \
  --input "$INPUT" \
  --scoring-model "$MODEL" \
  --output-dir "$METRICS_DIR" \
  2>&1 | tee "$LOG_DIR/score_prob.log"

echo "[run] aggregating A_lex/A_traj/A_prob"
"$PYTHON_BIN" scripts/anchoring_measure/prob_lex_traj_metrics.py aggregate \
  --input "$INPUT" \
  --prob-metrics "$METRICS_DIR" \
  --traj-csv "$TRAJ_CSV" \
  --output-dir "$RESULTS_DIR" \
  2>&1 | tee "$LOG_DIR/aggregate.log"

echo "[run] done: $RESULTS_DIR/qwen3_4b_traj_three_metrics.csv"
