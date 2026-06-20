# Memory Evaluation Runbook

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
preflight passes.

## Full Oracle Upper-Bound Matrix

For each dataset and memory system, run:

- raw memory: `--no-mask`;
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
- raw, typed, complete-mask, and minimal-public paired deltas;
- model annotations separately from oracle annotations.
