# PrivMemAgent

PrivMemAgent is a research implementation of **minimal sufficient public
memory** for privacy-preserving edge-cloud agents.

Instead of sending raw user memory or replacing every sensitive span with an
opaque placeholder, the framework decomposes a memory into:

```text
raw memory -> task-sufficient public memory + local private residue
```

Only the public memory is written to the cloud memory system. Exact sensitive
values remain local or are assigned `no_retention` by policy.

This repository extends the original
[MemPrivacy](https://github.com/MemTensor/MemPrivacy) research code with the
trainable PMA + PUC + AMA pipeline described in `docs/`.

## Core modules

- **PMA — Privacy Memory Abstractor**
  - prompt-based oracle generation
  - local trained-model inference
  - deterministic heuristic mode for CI
  - typed-placeholder and full-redaction fallbacks
- **PUC — Privacy-Utility Critic**
  - measures downstream QA utility
  - measures privacy leakage through AMA
  - selects the lowest-leakage candidate above a utility threshold
- **AMA — Adversarial Memory Auditor**
  - exact private-value reconstruction
  - sensitive-attribute inference
  - strict matching for credentials and identifiers
  - semantic matching support for non-strict privacy types

## Privacy policy

The default policy is in
`src/privacy_abstraction_config.yaml`.

It enforces:

- allowed abstraction levels for PL1–PL4
- type-specific overrides
- full redaction for verification codes, passwords, API keys, and recovery codes
- `local_only`, `session_only`, and `no_retention` residue modes
- rejection of candidates containing annotated raw private values
- required residue and trace alignment

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For real Mem0 evaluation:

```bash
pip install -e ".[mem0]"
```

For SFT and DPO training:

```bash
pip install -e ".[train]"
```

For the retained upstream vLLM extraction script:

```bash
pip install -e ".[legacy]"
```

## Configuration

`evaluation/eval_config.yaml` uses environment references and does not contain
credentials:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."  # optional for the standard OpenAI endpoint
```

The `privacy_llm` drives oracle candidate generation. `answer_llm` and
`judgment_llm` measure utility. `attack_llm` performs privacy attacks.

## End-to-end pipeline

### 1. Generate candidates

```bash
python -m evaluation.build_pma_candidates \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/pma_candidates.jsonl \
  --users 10 \
  --task-family recommendation \
  --backend oracle_prompt \
  --config evaluation/eval_config.yaml
```

Every privacy-bearing turn receives semantic candidates plus mandatory L4 and
L5 fallbacks.

### 2. Score utility and leakage

Standalone QA scoring:

```bash
python -m evaluation.score_pma_candidates \
  --candidates evaluation/results/pma_candidates.jsonl \
  --output evaluation/results/pma_scores.jsonl \
  --memory-system none \
  --attack all \
  --config evaluation/eval_config.yaml
```

Real Mem0 scoring:

```bash
python -m evaluation.score_pma_candidates \
  --candidates evaluation/results/pma_candidates.jsonl \
  --output evaluation/results/pma_scores_mem0.jsonl \
  --memory-system mem0 \
  --attack all \
  --config evaluation/eval_config.yaml
```

`--dry-run` is available for plumbing tests. Dry-run output is explicitly
marked `paper_evidence: false` and uses a deterministic utility proxy.

### 3. Build SFT and preference data

```bash
python -m evaluation.build_pma_train_data \
  --candidates evaluation/results/pma_candidates.jsonl \
  --scores evaluation/results/pma_scores.jsonl \
  --sft-output evaluation/results/pma_sft.jsonl \
  --preference-output evaluation/results/pma_preference.jsonl
```

The builder removes identical and duplicate chosen/rejected outputs before
creating DPO records.

### 4. Train PMA

SFT:

```bash
python -m training.train_pma_sft \
  --train-file evaluation/results/pma_sft.jsonl \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --model-revision "<immutable-commit-hash>" \
  --output-dir checkpoints/pma-sft \
  --bf16 \
  --lora
```

DPO:

```bash
python -m training.train_pma_preference \
  --train-file evaluation/results/pma_preference.jsonl \
  --model-name-or-path checkpoints/pma-sft \
  --model-revision "<immutable-commit-hash>" \
  --output-dir checkpoints/pma-dpo \
  --bf16 \
  --lora
```

Both commands support `--dry-run` for schema validation without loading a
model.

### 5. Evaluate all methods through Mem0

```bash
python -m evaluation.eval_pma_mem0 \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/pma_mem0.json \
  --methods raw complete generic type_specific pma_oracle pma_sft \
  --pma-model-path checkpoints/pma-sft \
  --pma-model-revision "<immutable-commit-hash>" \
  --memory-backend mem0 \
  --auditor-backend llm \
  --is-mcq true \
  --turns-per-chunk 5
```

The evaluation performs the actual loop:

```text
transform turns -> write memory -> retrieve -> answer -> score -> attack
```

User and assistant turns are both transformed. Result JSON includes:

- utility and leakage for each method
- per-question responses and scores
- selected abstractions
- per-item attacks
- privacy metrics grouped by privacy type

### 6. Validate and render paper evidence

```bash
python -m evaluation.compare_pma_results \
  evaluation/results/pma_mem0.json \
  --required-methods raw complete type_specific pma_sft \
  --markdown-output evaluation/results/main_table.md \
  --csv-output evaluation/results/main_table.csv
```

The command fails if:

- a required baseline is missing
- utility or privacy uses a proxy
- methods use different user/question pairs
- per-type metrics or attack records are absent
- the run is not marked as real Mem0 paper evidence

## CI and local smoke tests

No API key or model checkpoint is needed for:

```bash
pytest
ruff check src evaluation training tests

python -m evaluation.eval_pma_mem0 \
  --input data/memprivacy_bench_testset.jsonl \
  --output /tmp/pma_ci.json \
  --users 1 \
  --max-turns 2 \
  --methods raw complete generic type_specific pma_oracle \
  --memory-backend in_memory \
  --ci-mode
```

CI-mode results are deliberately marked as non-paper evidence.

## Repository status

The code paths required by the implementation plan are present and tested.
The repository does **not** claim that the paper's empirical improvement has
already been established:

- the included public dataset excerpt contains 10 users, not the planned 100
- no API credentials are committed
- no trained PMA checkpoint is committed
- proxy and CI runs are blocked from paper-evidence validation

See `docs/implementation_audit.md` for the requirement-by-requirement audit.

## Original MemPrivacy components

The repository retains the original reversible pseudonymization and legacy
memory-system evaluation code:

- `src/privacy_masking.py`
- `evaluation/eval_mem0.py`
- `evaluation/eval_langmem.py`
- `evaluation/eval_memobase.py`
- `evaluation/metric.py`

The PMA experiment uses the new validated pipeline described above.

## Attribution and license

This work is derived from
[MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud
Agents](https://arxiv.org/abs/2605.09530) by Yining Chen et al.

The upstream README identifies the project under
CC BY-NC-ND 4.0. Preserve upstream attribution and review the license before
public redistribution or commercial use. This repository should remain private
unless the owner has confirmed redistribution rights.
