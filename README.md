# SSR: Reverse Chain-of-Thought Anchoring

Code for **Measuring and Mitigating Post-hoc Rationalization in Reverse
Chain-of-Thought Generation**.

[Paper](https://arxiv.org/abs/2602.14469) | [SSR-RCoT-16K dataset](https://huggingface.co/datasets/Nanbeige/SSR-RCoT-16K)

This release covers RCoT generation, the final three anchoring metrics,
controlled references and behavioral zones, and the diagnostics used to audit
those measurements. Generated traces, metric results, figures, paper sources,
and model checkpoints are intentionally not tracked.

## Installation

Python 3.11 is supported.

```bash
uv venv venv --python 3.11
uv pip install --python venv/bin/python -e .
```

Install optional dependencies only for the workflows that need them:

```bash
# vLLM trace and baseline generation
uv pip install --python venv/bin/python -e ".[generation]"

# Embedding-based and lexical robustness diagnostics
uv pip install --python venv/bin/python -e ".[diagnostics]"
```

Use a PyTorch/vLLM build compatible with the local CUDA driver. All generation
and scoring commands expect local or Hugging Face model identifiers supplied
explicitly on the command line.

## Input Format

The common JSONL format stores one row per example and one value per method in
each section:

```json
{
  "id": "example-id",
  "questions": {"NEU": "...", "SUP": "...", "AUG-SUP": "...", "SSR": "..."},
  "answers": {"NEU": "...", "SUP": "...", "AUG-SUP": "...", "SSR": "..."},
  "contexts": {"NEU": "...", "SUP": "...", "AUG-SUP": "...", "SSR": "..."},
  "reasonings": {"NEU": "...", "SUP": "...", "AUG-SUP": "...", "SSR": "..."}
}
```

`contexts` contains the conversation shown to the reverse-CoT generator,
normally including the final assistant answer. `questions` contains the
question-only scoring context. Every requested method must be present and
nonempty in all four sections.

The bundled [examples.jsonl](scripts/examples/examples.jsonl) contains three
small synthetic rows for format and parser checks. It is not an experimental
result set.

## Format Check

The dry run validates the main methods, validates a genuine question-only
`Blind CoT`, and mechanically constructs the four controlled references. It
does not load a model or compute metrics.

```bash
venv/bin/python scripts/run_anchoring_pipeline.py \
  --input scripts/examples/examples.jsonl \
  --blind-input scripts/examples/examples.jsonl \
  --blind-method "Blind CoT" \
  --run-dir runs/example_dry_run \
  --dry-run \
  --force
```

## Generate Traces

`SSR` maps to the final paper prompt. `SSR-SCHEMA` and `SSR-DENSE` are the
structured-format ablations; dense paragraph numbering is removed during
output parsing.

```bash
venv/bin/python scripts/generate_vllm.py \
  --mode rcot \
  --input path/to/input.metric.jsonl \
  --output runs/main/inputs/methods.metric.jsonl \
  --raw-output runs/main/generation/raw_outputs.jsonl \
  --model /path/to/generator \
  --methods NEU,SUP,AUG-SUP,SSR \
  --tensor-parallel-size 4
```

Generate independent question-only traces for the controlled references:

```bash
venv/bin/python scripts/generate_vllm.py \
  --mode r0 \
  --input path/to/input.metric.jsonl \
  --output runs/main/inputs/blind.metric.jsonl \
  --raw-output runs/main/generation/blind_raw_outputs.jsonl \
  --model /path/to/generator \
  --method-name "Blind CoT" \
  --tensor-parallel-size 4
```

The Blind CoT must be generated from the question alone. NEU is an
answer-visible reverse-CoT baseline and must not be substituted for it.

## Score Metrics

Run all three metrics and the behavioral-zone analysis with one scorer:

```bash
venv/bin/python scripts/run_anchoring_pipeline.py \
  --input runs/main/inputs/methods.metric.jsonl \
  --blind-input runs/main/inputs/blind.metric.jsonl \
  --blind-method "Blind CoT" \
  --model /path/to/scoring-model \
  --run-dir runs/main \
  --python venv/bin/python \
  --torchrun-nproc 4 \
  --force
```

Omit `--blind-input` to score only the requested methods. In that case the
pipeline skips controlled references and behavioral-zone plots. Outputs are
written under the selected run directory:

```text
inputs/                         Validated method and controlled JSONL
metrics/methods/                Per-rank probability and trajectory records
metrics/controlled/             Per-rank controlled-reference records
results/*/anchoring_metrics.*   Per-record and aggregate final metrics
figures/behavioral_zones_*      Reference-normalized zone plots and summaries
manifest.json                   Inputs, scorer, and metric definitions
report.md                       Compact run report
```

## Final Metrics

Aggregate tables report all three metrics on a 0-100 scale:

- `A_lex = 100 * A_lex_QF`: IDF-weighted recall of answer content in the
  reasoning after removing answer terms already present in the question.
- `A_traj = 100 * ConfidenceGap_mean`: mean reduction in next-token entropy
  when the answer is visible, evaluated at sampled prefixes of the observed
  reasoning trace.
- `A_prob`: 100 times the clipped fraction of baseline answer surprisal removed
  after conditioning on the complete reasoning trace.

The per-record CSV keeps lexical and probabilistic values in `[0, 1]`; aggregate
tables multiply them by 100. `A_traj` is written in percentage-point units at
both levels. `B_100` is the unnormalized answer bit gain multiplied by 100 and
is retained as a robustness diagnostic.

The controlled set contains:

- `Blind CoT`: independently generated from the question only.
- `+Prob Anchor`: Blind CoT plus neutral padding and unordered answer terms.
- `+Traj Anchor`: a mechanically rendered answer prefix that preserves
  structure while sparsely copying content words.
- `Response-as-CoT`: the reference answer used directly as the trace.

## Dataset Conversion

Convert downloaded SSR-RCoT rows into the common format:

```bash
venv/bin/python scripts/prepare_anchoring_input.py convert-ssr-rcot \
  --input path/to/ssr_rcot_16k.jsonl \
  --output runs/ssr_rcot_16k/inputs/ssr.metric.jsonl \
  --method SSR
```

## Scripts

- `scripts/rcot_generation/rcot_prompt.py`: public NEU, SUP, AUG-SUP, SSR,
  schema, dense, and suppression-control prompts.
- `scripts/generate_vllm.py`: answer-visible RCoT and question-only Blind CoT
  generation.
- `scripts/run_anchoring_pipeline.py`: validated end-to-end three-metric run.
- `scripts/anchoring_measure/prob_lex_traj_metrics.py`: final `A_lex` and
  `A_prob` scoring plus three-metric aggregation.
- `scripts/anchoring_measure/commitment_kl.py`: per-prefix trajectory
  sensitivity used for final `A_traj`.
- `scripts/anchoring_measure/build_controlled_traj_anchors.py`: controlled
  reference construction.
- `scripts/anchoring_measure/plot_paper_style_behavior_zones.py`: behavioral
  zone transformation and plotting.
- `scripts/baseline_tier1.py`: FDB, Gist, NGramBlock, and Best-of-N baselines,
  selection, merge, and endpoint evaluation.
- `scripts/anchoring_measure/prefix_curve.py`, `path_diversity.py`,
  `length_effect_diagnostics.py`, and `scripts/analyze_answer_swap_sensitivity.py`:
  paper robustness diagnostics.

`insights/baselines-code/` retains the separate reference implementations for
the contrastive and Tier-1 baseline ablations.

## Release Scope

This code-only release reproduces RCoT generation and anchoring analysis from
prepared question-answer data. The SSR-D training pipeline, downstream
benchmark harnesses, external LLM-as-judge service integration, paper sources,
and reported result files are outside this repository. Their absence is
intentional; the repository does not fabricate substitute implementations or
embed unpublished outputs.

Keep local outputs under `runs/`, which is ignored by Git.

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
