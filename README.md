<h1 align="center">SSR: Measuring and Mitigating Post-hoc Rationalization in Reverse Chain-of-Thought Generation</h1>

---

<p align="center">
<a href="https://arxiv.org/abs/2602.14469">Paper</a> |
<a href="https://huggingface.co/datasets/Nanbeige/SSR-RCoT-16K">Dataset</a>
</p>

This repository contains the code release for **Measuring and Mitigating
Post-hoc Rationalization in Reverse Chain-of-Thought Generation**.

Reverse Chain-of-Thought Generation (RCG) generates reasoning traces from
question-answer pairs. Because the answer is visible during generation, the
trace can become a post-hoc rationalization rather than a forward-usable
reasoning path. This codebase provides:

- RCoT generation prompts for NEU, SUP, AUG-SUP, and SSR.
- Anchoring metrics for lexical, trajectory, and probabilistic answer
  dependence.
- Controlled-reference and behavioral-zone analysis utilities.
- Diagnostic scripts for stress tests, ablations, and result auditing.

Large generated outputs are written under `runs/`, which is intentionally
ignored by git.

## Environment Setup

The project is tested with Python 3.11.

```bash
UV_CACHE_DIR=.uv-cache uv venv venv --python 3.11

UV_CACHE_DIR=.uv-cache uv pip install --python venv/bin/python -e ".[generation]" \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
```

For metric-only usage without vLLM generation, the base package dependencies in
`pyproject.toml` are sufficient. Generation with `scripts/generate_vllm.py`
requires the `generation` extra.

## Data

For a quick code-path check, use the included metric-format example file:

```bash
scripts/examples/examples.jsonl
```

This file already contains generated NEU/SUP/AUG-SUP/SSR reasoning traces, so
you can skip trace generation and run the anchoring pipeline directly.

For larger experiments, download the public SSR-RCoT-16K subset from Hugging
Face:

```bash
venv/bin/python - <<'PY'
from datasets import load_dataset

ds = load_dataset("Nanbeige/SSR-RCoT-16K")
print(ds)
PY
```

If you export dataset rows to JSONL, convert them to the metric input format
with:

```bash
venv/bin/python scripts/prepare_anchoring_input.py convert-ssr-rcot \
  --input path/to/ssr_rcot_16k.jsonl \
  --output runs/ssr_rcot_16k/inputs/ssr.metric.jsonl \
  --method SSR
```

## Quick Start

Run the anchoring pipeline on the bundled examples:

```bash
venv/bin/python scripts/run_anchoring_pipeline.py \
  --input scripts/examples/examples.jsonl \
  --run-dir runs/rcot_example \
  --model /path/to/local/model \
  --python venv/bin/python \
  --torchrun-nproc 4 \
  --force
```

Build a human-readable report:

```bash
venv/bin/python scripts/report_anchoring_results.py \
  --run-dir runs/rcot_example
```

The selected `--run-dir` will contain metric tables, behavioral-zone summaries,
plots, and `report.md`.

## Generate New RCoT Traces

Use `scripts/generate_vllm.py` when you want to regenerate traces from a
metric-format input file.

```bash
venv/bin/python scripts/generate_vllm.py \
  --mode rcot \
  --input path/to/input.metric.jsonl \
  --output runs/rcot_example/inputs/generated.metric.jsonl \
  --raw-output runs/rcot_example/generation/raw_outputs.jsonl \
  --model /path/to/local/model \
  --methods NEU,SUP,AUG-SUP,SSR \
  --tensor-parallel-size 4
```

Then score the generated traces:

```bash
venv/bin/python scripts/run_anchoring_pipeline.py \
  --input runs/rcot_example/inputs/generated.metric.jsonl \
  --run-dir runs/rcot_example \
  --model /path/to/local/model \
  --python venv/bin/python \
  --torchrun-nproc 4 \
  --force
```

## Input Format

Metric-format JSONL files contain one row per example. The pipeline expects a
question, answer, and method-indexed reasoning traces.

```json
{
  "id": "example-id",
  "question": "User question",
  "answer": "Reference answer",
  "reasonings": {
    "NEU": "reasoning trace",
    "SUP": "reasoning trace",
    "AUG-SUP": "reasoning trace",
    "SSR": "reasoning trace"
  }
}
```

`scripts/generate_vllm.py` writes this format when run in `--mode rcot`.

## Metrics

The main pipeline computes three levels of answer anchoring:

- `A_lex`: question-filtered lexical overlap between answer content and the
  reasoning trace.
- `A_traj`: answer-conditioned trajectory dependence, estimated from the
  next-token entropy gap with and without answer visibility.
- `A_prob`: endpoint recoverability, measuring how much the reasoning trace
  reduces answer surprisal under a scoring model.

The behavioral-zone utilities summarize how examples distribute across
Reason, Encode, Monitor, and Copy regions in the trajectory-probabilistic
anchoring plane.

## Useful Scripts

- `scripts/generate_vllm.py`: generate RCoT or answer-blind traces with vLLM.
- `scripts/run_anchoring_pipeline.py`: run metric scoring, controlled
  references, tables, plots, and report metadata.
- `scripts/report_anchoring_results.py`: assemble a markdown report from a run
  directory.
- `scripts/prepare_anchoring_input.py`: validate, convert, or build metric
  input files.
- `scripts/anchoring_measure/prob_lex_traj_metrics.py`: core lexical,
  trajectory, and probabilistic metric computation.
- `scripts/anchoring_measure/plot_paper_style_behavior_zones.py`: behavioral
  zone plotting.
- `scripts/anchoring_measure/build_controlled_traj_anchors.py`: controlled
  anchor construction.
- `scripts/analyze_answer_swap_sensitivity.py`: answer-swap sensitivity
  diagnostic.
- `scripts/diagnose_candidate_quality_slices.py`: quality slice diagnostics
  such as language matching and simple-task overreasoning.

Additional baseline-generation scripts are provided under
`insights/baselines-code/`.

## Repository Layout

```text
scripts/
  anchoring_measure/      Metric, plotting, and diagnostic utilities.
  rcot_generation/        Prompt templates for RCoT generation.
  examples/               Small metric-format example input.
insights/baselines-code/  Baseline generation helpers.
runs/                     Local experiment outputs; ignored by git.
```

## Notes

- Use local model checkpoints for generation and scoring.
- Keep generated outputs under `runs/` or another ignored directory.
- For large runs, use `--torchrun-nproc` to match the number of available
  scoring GPUs.
- The same metric-format JSONL can be passed through custom generation methods
  as long as method names are stored under `reasonings`.

## Citation

```bibtex
@inproceedings{peng2026rcganchoring,
  title={Measuring and Mitigating Post-Hoc Rationalization in Reverse Chain-of-Thought Generation},
  author={Guangyue Peng and Zongchao Chen and Wen Luo and Yuntao Wen and Wei Li and Ruixiang Feng and Ran Le and Chen Yang and Zhenwei An and Yang Song and Tao Zhang and Houfeng Wang},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=cY0pi9WHvL}
}
```
