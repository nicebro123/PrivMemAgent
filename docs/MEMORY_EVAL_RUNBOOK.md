# Memory Evaluation Runbook


## Repository, Assets, and Runtime Configs

Keep the tracked repo code-only where possible. Small sample JSONL files and
tracked YAML profiles are fine; large model weights, generated runtime configs,
local privacy stores, and experiment outputs should stay outside git or under
ignored paths. This mirrors the EvoCo-RAG workflow: tracked profiles describe
the reusable experiment intent, and generated runtime configs freeze one
server/run.

Recommended local layout:

```text
parent/
├── PrivMemAgent/
└── memprivate_assets/
    ├── models/bge-m3/
    ├── runtime_configs/
    └── results/
```

Generate a runtime config instead of editing tracked profiles for each machine:

```bash
python -m tools.materialize_eval_config \
  --profile evaluation/eval_config.deepseek.yaml \
  --output ../memprivate_assets/runtime_configs/deepseek_cuda0.yaml \
  --embedding-model ../memprivate_assets/models/bge-m3 \
  --embedding-device cuda:0 \
  --output-path ../memprivate_assets/results/deepseek_cuda0 \
  --print-export

export MEMPRIVACY_EVAL_CONFIG="$PWD/../memprivate_assets/runtime_configs/deepseek_cuda0.yaml"
```

`tools.materialize_eval_config` supports repeatable dotted-key overrides such
as `--set answer_llm.model=deepseek-v4-flash`. It refuses literal secret values
for keys such as `api_key`, `token`, and `password`; store credentials in an env
file and keep tracked YAML values as `$OPENAI_API_KEY` style references.

Memory runners resolve `output_path` from the selected config. Prefer absolute
result roots, or let the materializer convert `--output-path` to an absolute
path, so generated configs can move without changing where results are written.

## Environment

Use Python 3.10 or newer and install the complete evaluation dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Keep credentials out of tracked YAML:

```bash
export OPENAI_BASE_URL="https://your-gateway.example/v1"
export OPENAI_API_KEY="..."
```

Validate the runtime before any experiment:

```bash
python -m tools.preflight_memory_eval --system mem0 --system langmem
```

Memobase additionally requires a reachable server and runtime-only
`MEMOBASE_PROJECT_URL` and `MEMOBASE_API_KEY`.


## DeepSeek API profile

DeepSeek can be used for OpenAI-compatible chat/reasoning calls by loading a
runtime-only environment file and selecting the DeepSeek config profile:

```bash
source ~/.config/memprivate/deepseek.env
python -m tools.materialize_eval_config \
  --profile evaluation/eval_config.deepseek.yaml \
  --output evaluation/runtime_configs/eval_config.deepseek.cuda0.yaml \
  --embedding-model /mnt/infini-data/test/quan_space/codespace/memprivate/models/bge-m3 \
  --embedding-device cuda:0 \
  --output-path evaluation/results \
  --print-export
export MEMPRIVACY_EVAL_CONFIG="$PWD/evaluation/runtime_configs/eval_config.deepseek.cuda0.yaml"
python -m tools.preflight_memory_eval --config "$MEMPRIVACY_EVAL_CONFIG" --system langmem --probe-openai
```

The key file must stay outside git and mode `0600`. DeepSeek official chat
models work for `memory_llm`, `answer_llm`, and `judgment_llm`, but the official
API does not provide an OpenAI-compatible embeddings endpoint. The DeepSeek
profile therefore uses a local BGE-M3 embedding checkpoint at
`/mnt/infini-data/test/quan_space/codespace/memprivate/models/bge-m3` with 1024
dimensions. Set `embedding_model.device` to `cuda:0` if you want faster local
embedding; CPU is the default to avoid occupying GPUs during API-bound runs.

## Infrastructure Smoke

Compile the synthetic smoke dataset:

```bash
python -m evaluation.eval_public_memory \
  --input data/memory_eval_smoke.jsonl \
  --output evaluation/results/smoke_public_records.jsonl \
  --metrics-output evaluation/results/smoke_public_metrics.json \
  --state-dir evaluation/results/smoke_public_state \
  --cloud-safe-dataset-output evaluation/results/smoke_public_benchmark.jsonl \
  --minimum-token-reduction -1 \
  --annotation-source oracle
```

Run the legacy typed-alias condition:

```bash
python -m evaluation.eval_mem0 \
  --input data/memory_eval_smoke.jsonl \
  --mask --mask-level PL2 PL3 PL4 --mask-mode type_specific \
  --annotation-source oracle --mcq --user-num 1 --num-workers 1
```

Run the minimal-public condition:

```bash
python -m evaluation.eval_mem0 \
  --input evaluation/results/smoke_public_benchmark.jsonl \
  --no-mask --mcq --user-num 1 --num-workers 1
```

Repeat both commands with `evaluation.eval_langmem`. Run Memobase only after its
preflight passes. The minimal-public `--no-mask` path is guarded: runners reject
unmasked inputs unless they have cloud-safe shape, preventing accidental raw
benchmark uploads to cloud memory systems. Use `--allow-unsafe-no-mask` only for
trusted local debugging or the explicit raw-memory baseline.

## Full Oracle Upper-Bound Matrix

For each dataset and memory system, run:

- raw memory: `--no-mask --allow-unsafe-no-mask`;
- persistent typed pseudonyms: `--mask --mask-mode type_specific`;
- complete masking: `--mask --mask-mode complete`;
- minimal public memory: compile a cloud-safe benchmark and run `--no-mask`.

Datasets:

- `data/memprivacy_bench_testset.jsonl` with short-answer QA;
- `data/personamem_v2_testset.jsonl` with MCQ.

The released partial datasets contain 5,746 messages and 427 questions. At
`turns_per_chunk=5`, one condition needs 582 memory-write chunks. Mem0 and
LangMem additionally require answer calls for every question and judge calls
for the 149 short-answer questions. Run smoke first and record provider cost
before launching the matrix.

## Scientific Matrix

Oracle annotations are an upper bound. The final paper matrix must regenerate
`privacy_info_llm` with the declared detector checkpoint and repeat all
conditions using `--annotation-source model`.

Report:

- accuracy and valid-answer accuracy by question type;
- exact-value recovery, PL4 retention, and attribute/linkage attacks;
- public tokens and memory records per user;
- budget-sweep Pareto summaries from `tools.public_memory_budget_sweep`;
- raw, typed, complete-mask, and minimal-public paired deltas;
- model annotations separately from oracle annotations.

For a deterministic minimality sweep before expensive memory-system runs:

```bash
python -m tools.public_memory_budget_sweep \
  --input data/personamem_v2_testset.jsonl \
  --output-dir evaluation/results/persona_budget_sweep \
  --summary-output evaluation/results/persona_budget_sweep_summary.json \
  --annotation-source oracle \
  --user-limit 5 \
  --minimum-token-reduction -1.0 \
  --budget 16 --budget 64 --budget 128

python -m tools.public_memory_budget_selector \
  --summary evaluation/results/persona_budget_sweep_summary.json \
  --output evaluation/results/persona_budget_sweep_recommendation.json \
  --min-utility 0.75 \
  --min-local-recoverability 0.95
```

Use the recommendation JSON as the deployable selector-calibration record: it
keeps rejected budgets with failure reasons, the smallest safe budget, the
highest-utility safe budget, and the safe Pareto frontier.
