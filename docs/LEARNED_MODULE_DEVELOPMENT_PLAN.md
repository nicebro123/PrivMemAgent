# Learned Module Development Plan

## Goal

PrivMemAgent currently contains a deterministic public-memory compiler. The next
major development step is to train two modules that turn the system into a
learned minimal-sufficient memory compiler:

1. `LearnedAbstractionGenerator`
2. `LearnedUtilityLeakageSelector`

The target end-to-end pipeline is:

```text
raw dialogue
  -> MemPrivacy-4B-RL privacy extraction
  -> learned abstraction candidate generation
  -> learned utility-leakage selection
  -> policy-constrained public memory
  -> cloud-safe memory QA
  -> adversarial audit and attacker feedback
```

## Non-Goals

This plan does not retrain MemPrivacy-4B-RL. The detector remains an upstream
privacy extractor unless a separate detector-training project is started.

This plan also does not replace local privacy stores, provenance, scoped alias
routing, or adversarial audits. The learned modules should plug into the current
compiler and preserve deterministic fallbacks.

## Module 1: Learned Abstraction Generator

### Purpose

Generate candidate public-memory representations that preserve future task
utility while reducing privacy leakage.

The generator replaces or augments the current rule-based `RuleBasedAbstractor`
in `src/public_memory_compiler.py`.

### Proposed Interface

New file:

```text
src/abstraction_generator.py
```

Initial interface:

```python
@dataclass(frozen=True)
class AbstractionInput:
    user_id: str
    message_id: str
    role: str
    message_text: str
    privacy_item: dict
    neighboring_context: list[dict]
    question_hints: list[str]
    policy: dict

@dataclass(frozen=True)
class AbstractionCandidate:
    text: str
    abstraction_level: int
    representation_type: str
    contains_alias: bool = False
    generator_score: float | None = None

class LearnedAbstractionGenerator:
    def generate(self, item: AbstractionInput) -> list[AbstractionCandidate]: ...
```

The output list must include a `drop` or empty-memory option. It may include a
reversible alias option only when the policy allows it.

### Candidate Levels

Use a stable level taxonomy so selector labels are consistent:

| Level | Meaning | Example |
|---|---|---|
| 0 | drop | no public memory |
| 1 | generic abstraction | `private detail` |
| 2 | category abstraction | `health constraint relevant to future assistance` |
| 3 | task-sufficient abstraction | `medical scheduling preference may affect recommendations` |
| 4 | scoped reversible alias | `<MPM_Health_...>` only with consent and exact utility need |

### Training Data Construction

New scripts:

```text
training/build_abstraction_data.py
training/generate_abstraction_candidates.py
training/filter_abstraction_candidates.py
```

Data sources:

- `data/personamem_v2_testset.jsonl`
- `data/memprivacy_bench_testset.jsonl`
- model predictions from `evaluation/results/memprivacy4b_full/*/predictions.jsonl`
- public-memory artifacts from the deterministic compiler
- future memory questions and answer evidence

Weak supervision sources:

1. Existing rule-based abstractions.
2. LLM-generated abstractions from DeepSeek API.
3. Audit-filtered safe candidates.
4. Utility labels from downstream QA and non-private answer-token recall.

Each training example should contain:

```json
{
  "message_text": "...",
  "privacy_item": {
    "original_text": "...",
    "privacy_type": "...",
    "privacy_level": "PL2"
  },
  "question_hints": ["..."],
  "candidates": [
    {
      "text": "private detail",
      "level": 1,
      "leakage_label": 0,
      "utility_label": 0.3,
      "audit_passed": true
    }
  ],
  "preferred_candidate_ids": [2]
}
```

### Model Choices

Start small and reproducible:

1. Prompted DeepSeek candidate generation for data construction only.
2. Train a small local seq2seq or causal LM adapter if enough data is available.
3. If data is limited, keep the generator as retrieval/rule plus LLM-distilled
   templates and focus training on the selector first.

Possible local models:

- Qwen2.5-0.5B/1.5B Instruct LoRA;
- Qwen3-0.6B/1.7B LoRA;
- a compact T5-style model if Chinese-English support is sufficient.

### Losses

SFT objective:

```text
input: message + privacy item + policy + task hint
output: safe abstraction candidate
```

Preference objective:

```text
safe useful abstraction > over-specific abstraction > unsafe exact leak
```

Hard negative examples must include candidates that copy exact private values.

### Safety Filters

Before a generated candidate can be used for training or evaluation, run:

- exact string leak check against `original_text`;
- PL4 and canary leak check;
- sensitive attribute term check;
- adversarial audit on generated public artifacts;
- optional LLM judge for semantic utility only after exact filters pass.

Generated candidates that expose the original sensitive value must be rejected
or labeled as negative examples.

## Module 2: Learned Utility-Leakage Selector

### Purpose

Select which abstraction candidate should enter public memory, or decide that
nothing should be exported.

The selector replaces or augments `src/sufficiency_selector.py`.

### Proposed Interface

New file:

```text
src/utility_leakage_selector.py
```

Initial interface:

```python
@dataclass(frozen=True)
class SelectorFeatures:
    candidate_text: str
    representation_type: str
    abstraction_level: int
    privacy_type: str
    privacy_level: str
    token_count: int
    embedding_relevance: float
    lexical_answer_overlap: float
    exact_leak_flag: bool
    attribute_risk_score: float
    linkability_scope: str | None

@dataclass(frozen=True)
class SelectorDecision:
    action: str
    score: float
    reason: str

class LearnedUtilityLeakageSelector:
    def select(self, candidates: list[SelectorFeatures]) -> SelectorDecision: ...
```

Valid actions:

```text
select
reject
local_only
drop
public_reversible
```

### Feature Set

Minimum features:

- candidate token count;
- abstraction level;
- privacy level and privacy type;
- route proposed by policy;
- BGE-M3 similarity between candidate and future question/evidence;
- lexical overlap with non-private answer tokens;
- exact leak boolean;
- PL4 boolean;
- sensitive attribute term count;
- alias scope;
- membership-marker risk;
- deterministic selector decision;
- deterministic audit result.

### Label Construction

Build labels from candidate-level experiment sweeps:

```text
candidate public memory set
  -> downstream QA utility
  -> adversarial audit leakage
  -> public token count
  -> linkability score
```

Define a scalar training target:

```text
score = utility
        - lambda_exact * exact_leak
        - lambda_attr * attribute_leak
        - lambda_link * linkability
        - beta * normalized_token_count
```

Also compute Pareto labels:

```text
candidate is positive if it is non-dominated under utility, leakage, and size
and satisfies hard PL4/exact-leak constraints.
```

### Model Choices

Start with a lightweight ranker:

- logistic regression or linear ranker for interpretability;
- LightGBM/XGBoost if available;
- small MLP over engineered features plus BGE embeddings;
- later, cross-encoder reranker if more labeled data is available.

The first trainable selector should be cheap to train, easy to debug, and
compatible with CPU inference.

### Training Scripts

New files:

```text
training/build_selector_data.py
training/train_utility_leakage_selector.py
training/evaluate_utility_leakage_selector.py
```

Outputs:

```text
artifacts/selectors/utility_leakage_selector_v1/
  config.yaml
  model.pkl or model.safetensors
  feature_schema.json
  metrics.json
```

Do not commit large model weights. Commit only schema, config examples, and
small test fixtures.

## Data Pipeline

### Stage A: Extract Privacy Spans

Use the existing 4B workflow:

```bash
python -m tools.launch_memprivacy4b_full_experiments --gpus 0,1,2,3
```

This creates:

```text
evaluation/results/memprivacy4b_full/<dataset>/predictions.jsonl
```

### Stage B: Generate Candidate Abstractions

For each privacy-bearing message:

```text
privacy item + surrounding context + question hints
  -> candidate abstractions at levels 0-4
```

Candidate generation should support both:

- deterministic templates;
- DeepSeek-generated candidates for distillation.

### Stage C: Filter Unsafe Candidates

Run exact and semantic safety filters:

```text
candidate -> leak checks -> audit labels
```

Unsafe candidates are not used as positive outputs but should be retained as
negative examples for the selector.

### Stage D: Build Public Memory Variants

For each selector/generator variant, compile public memory and record:

```text
public_records.jsonl
public_benchmark.jsonl
public_metrics.json
adversarial_audit.json
```

### Stage E: Run Downstream Utility

Evaluate memory QA using:

- internal proxy utility;
- Mem0;
- LangMem;
- Memobase, if service credentials are available.

### Stage F: Train Selector

Use utility, leakage, and size labels to train the selector.

### Stage G: End-to-End Evaluation

Compare:

1. MemPrivacy typed masking baseline;
2. deterministic PrivMemAgent compiler;
3. learned abstraction generator only;
4. learned selector only;
5. learned generator plus learned selector.

## Implementation Milestones

### Milestone 1: Dataset Builder

Deliverables:

- `training/build_abstraction_data.py`
- JSONL schema for abstraction examples
- unit tests with tiny fixtures
- documentation for generated fields

Exit criteria:

- Can build examples from smoke data and full predictions.
- No API keys or raw secrets are written outside ignored artifact paths.

### Milestone 2: Candidate Generator Baseline

Deliverables:

- `src/abstraction_generator.py`
- deterministic template generator compatible with current compiler
- optional DeepSeek candidate generation script

Exit criteria:

- Compiler can switch between rule generator and new generator interface.
- Smoke evaluation produces identical or better safety than current rules.

### Milestone 3: Selector Data Builder

Deliverables:

- `training/build_selector_data.py`
- feature extraction with BGE-M3 similarity
- candidate-level utility/leakage/token labels

Exit criteria:

- Selector examples can be generated for PersonaMem-v2 and MemPrivacy-Bench.
- Feature schema is versioned.

### Milestone 4: First Learned Selector

Deliverables:

- `src/utility_leakage_selector.py`
- `training/train_utility_leakage_selector.py`
- serialized lightweight selector artifact

Exit criteria:

- Selector beats deterministic rule selector on at least one Pareto metric
  without increasing exact or PL4 leakage.

### Milestone 5: Learned Abstraction Generator

Deliverables:

- SFT or distillation training script
- model adapter or compact generator artifact
- safety-filtered generation pipeline

Exit criteria:

- Generated abstractions improve utility at fixed leakage or reduce leakage at
  fixed utility versus deterministic templates.

### Milestone 6: Full End-to-End Paper Evaluation

Deliverables:

- full PersonaMem-v2 and MemPrivacy-Bench results;
- Mem0/LangMem/Memobase utility tables;
- privacy, minimality, and Pareto-frontier tables;
- ablation tables;
- failure analysis.

Exit criteria:

- Utility within 1 absolute point of raw-memory or typed-masking baseline where
  appropriate;
- exact sensitive-value recovery at most 1%;
- PL4 cloud retention exactly 0;
- attribute and linkage attack AUC at most 0.55;
- public token count at least 30% lower than typed masking.

## Engineering Rules

- Keep all generated training data, model weights, API outputs, and full
  experiment results out of git unless explicitly curated as small fixtures.
- Never commit API keys, DeepSeek runtime configs, local model paths with
  secrets, or SQLite state containing original values.
- Every learned module must keep a deterministic fallback.
- Every trained artifact must include feature schema, data hash, git commit,
  training command, and evaluation metrics.
- Any cloud LLM used for data construction must receive only masked or
  public-safe text unless an experiment is explicitly marked local-only.

## Open Research Questions

1. How much abstraction can be applied before memory QA utility collapses?
2. Can selector training generalize across memory systems, or does each backend
   need separate calibration?
3. Which leakage signal is most predictive of downstream adversarial failure:
   exact string match, attribute attack score, membership marker, or linkage
   metadata?
4. Should abstraction generation be conditioned on concrete future questions or
   only on question-type distributions?
5. Can a compact local generator match DeepSeek-generated abstraction quality
   after distillation?

## Implemented Code Entry Points

The first implementation pass adds the following concrete interfaces:

- `src/abstraction_generator.py`: deterministic fallback generator, JSON-artifact-backed learned generator, safe candidate filtering, and an adapter for the existing compiler.
- `src/utility_leakage_selector.py`: learned-selector feature schema, linear utility-leakage ranker, JSON weight loading, and fail-closed exact/PL4 guards.
- `training/build_abstraction_data.py`: builds abstraction-generator training examples from oracle or model annotations.
- `training/build_selector_data.py`: converts abstraction examples into candidate-level selector examples with initial utility/leakage labels.
- `src/public_memory_config.yaml`: optional `abstraction_generator` and `utility_leakage_selector` configuration blocks.

The default configuration remains conservative: rule-based abstraction is enabled
and the learned selector is disabled. Learned artifacts can be activated by
setting `public_memory.abstraction_generator.mode` to `learned` or
`public_memory.utility_leakage_selector.mode` to `linear`.

## Immediate Next Step

After the current full 4B extraction/public-memory run completes, build the
first abstraction-training dataset from:

```text
evaluation/results/memprivacy4b_full/personamem_v2/predictions.jsonl
evaluation/results/memprivacy4b_full/memprivacy_bench/predictions.jsonl
```

Then implement `training/build_abstraction_data.py` and a small fixture test
before starting any model training.
