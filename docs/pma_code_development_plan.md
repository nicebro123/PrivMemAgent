# Code Development Plan: PMA + PUC + AMA

## 1. Purpose

This document turns the paper concept into an implementable code plan.

The target system contains three cooperating modules:

```text
PMA: Privacy Memory Abstractor
PUC: Privacy-Utility Critic
AMA: Adversarial Memory Auditor
```

The implementation goal is not to immediately train all three modules. The first goal is to build a reproducible oracle pipeline that can:

1. generate candidate public memories,
2. score their downstream utility,
3. audit their privacy leakage,
4. select training labels,
5. produce SFT and preference data for a trainable PMA,
6. evaluate PMA against existing masking baselines.

This creates a bridge from the paper claim to executable evidence.

## 2. Research-to-Code Mapping

Paper claim:

```text
Cloud memory should store learned public abstractions instead of masked raw conversations.
```

Code object:

```text
public_memory
private_residue
abstraction_trace
```

Paper method:

```text
PMA_theta(x, p, c, pi) -> (z, r, t)
```

Code method:

```python
PrivacyMemoryAbstractor.abstract(
    dialogue_text: str,
    privacy_items: list[dict],
    task_family: str,
    policy: dict,
) -> AbstractionResult
```

Paper evidence:

```text
PMA improves utility over typed placeholders at comparable exact reconstruction leakage.
```

Code evidence:

```text
evaluation/results/pma_mem0_*.json
```

with fields:

```json
{
  "method": "pma",
  "utility": {
    "mcq_accuracy": 0.0
  },
  "privacy": {
    "exact_reconstruction_rate": 0.0
  },
  "baselines": {
    "type_specific": {
      "mcq_accuracy": 0.0,
      "exact_reconstruction_rate": 0.0
    }
  }
}
```

## 3. Repository Integration Strategy

The existing repository already has:

```text
src/privacy_masking.py
evaluation/privacy_masking.py
evaluation/eval_mem0.py
evaluation/eval_langmem.py
evaluation/eval_memobase.py
evaluation/utils.py
evaluation/metric.py
data/*.jsonl
```

The least risky development path is:

1. add PMA/PUC/AMA as new modules,
2. avoid rewriting existing masking logic,
3. integrate the new mode only into Mem0 first,
4. reuse existing datasets and answer prompts,
5. reuse existing `call_llm` utility for oracle generation and judge calls.

Do not modify LangMem or Memobase until the Mem0 path produces a clear signal.

## 4. Proposed File Layout

Add:

```text
src/privacy_abstraction.py
src/privacy_abstraction_config.yaml
src/privacy_critic.py
src/privacy_auditor.py
src/privacy_schema.py

evaluation/build_pma_candidates.py
evaluation/score_pma_candidates.py
evaluation/build_pma_train_data.py
evaluation/eval_pma_mem0.py
evaluation/compare_pma_results.py

evaluation/prompts/pma_generate_candidates.txt
evaluation/prompts/ama_exact_reconstruct.txt
evaluation/prompts/ama_attribute_infer.txt
evaluation/prompts/puc_judge_utility.txt
evaluation/prompts/puc_judge_semantic_leakage.txt

tests/test_privacy_abstraction_schema.py
tests/test_pma_selection.py
tests/test_ama_matching.py
```

Optional later:

```text
training/train_pma_sft.py
training/train_pma_preference.py
training/train_puc_reward.py
training/data_collator.py
training/export_pma.py
```

## 5. Core Data Schemas

Use explicit schemas early. The most common failure mode will be malformed LLM output.

### 5.1 Privacy Item

Existing data already uses:

```json
{
  "original_text": "829417",
  "privacy_type": "Verification Code",
  "privacy_level": "PL4"
}
```

Validation rules:

```text
original_text: non-empty string
privacy_type: non-empty string
privacy_level: one of PL1, PL2, PL3, PL4
```

### 5.2 Abstraction Candidate

```json
{
  "candidate_id": "uuid-or-stable-hash",
  "source_id": "user_uuid:turn_index:candidate_index",
  "level": "L2",
  "public_memory": "The user prefers low-allergen indoor environments.",
  "private_residue": [
    {
      "raw": "pollen allergy",
      "privacy_type": "Medical Health",
      "privacy_level": "PL3",
      "retention": "local_only"
    }
  ],
  "abstraction_trace": [
    {
      "raw": "pollen allergy",
      "public_abstraction": "low-allergen indoor environments",
      "reason": "Preserves planning constraint while hiding condition name."
    }
  ],
  "metadata": {
    "task_family": "recommendation",
    "generator": "oracle_prompt",
    "created_at": "ISO-8601"
  }
}
```

Validation rules:

```text
level: one of L0, L1, L2, L3, L4, L5
public_memory: non-empty string unless level is L5 and policy allows empty memory
private_residue[].raw must come from privacy_items[].original_text
retention: one of local_only, session_only, no_retention
```

### 5.3 Candidate Score

```json
{
  "candidate_id": "...",
  "utility": {
    "mcq_accuracy": 0.0,
    "answer_consistency": 0.0,
    "num_questions": 0,
    "num_valid": 0
  },
  "privacy": {
    "exact_reconstruction_success": false,
    "exact_reconstruction_rate": 0.0,
    "attribute_inference_success": false,
    "semantic_leakage_score": 0.0
  },
  "attacks": [
    {
      "privacy_item": {
        "original_text": "...",
        "privacy_type": "...",
        "privacy_level": "..."
      },
      "attack_type": "exact_reconstruction",
      "guess": "...",
      "success": false,
      "match_score": 0.0
    }
  ]
}
```

### 5.4 Selected Training Example

```json
{
  "example_id": "...",
  "input": {
    "dialogue": "...",
    "privacy_items": [...],
    "task_family": "recommendation",
    "policy": {...}
  },
  "chosen": {
    "public_memory": "...",
    "private_residue": [...],
    "abstraction_trace": [...]
  },
  "rejected": [
    {
      "public_memory": "...",
      "reason": "higher leakage at comparable utility"
    }
  ],
  "scores": {
    "chosen_utility": 0.0,
    "chosen_leakage": 0.0
  }
}
```

## 6. Module 1: PMA

### 6.1 Responsibility

PMA generates public-memory candidates and later produces the final deployed abstraction.

It does not decide whether a candidate is safe enough by itself in the oracle phase. The selector uses PUC and AMA scores.

### 6.2 Interface

```python
from dataclasses import dataclass
from typing import Any, Literal

AbstractionLevel = Literal["L0", "L1", "L2", "L3", "L4", "L5"]
RetentionMode = Literal["local_only", "session_only", "no_retention"]

@dataclass
class PrivateResidue:
    raw: str
    privacy_type: str
    privacy_level: str
    retention: RetentionMode

@dataclass
class AbstractionTrace:
    raw: str
    public_abstraction: str
    reason: str

@dataclass
class AbstractionCandidate:
    candidate_id: str
    level: AbstractionLevel
    public_memory: str
    private_residue: list[PrivateResidue]
    abstraction_trace: list[AbstractionTrace]
    metadata: dict[str, Any]

class PrivacyMemoryAbstractor:
    def generate_candidates(
        self,
        dialogue_text: str,
        privacy_items: list[dict],
        task_family: str,
        policy: dict,
    ) -> list[AbstractionCandidate]:
        ...

    def abstract(
        self,
        dialogue_text: str,
        privacy_items: list[dict],
        task_family: str,
        policy: dict,
    ) -> AbstractionCandidate:
        ...
```

### 6.3 Backends

PMA should support:

```text
oracle_prompt
trained_model
typed_placeholder
redaction
```

Backend behavior:

```text
oracle_prompt:
  call LLM to generate L1-L5 candidates.

trained_model:
  call local fine-tuned model to produce final abstraction.

typed_placeholder:
  reuse evaluation/privacy_masking.py type-specific mode.

redaction:
  reuse complete_mask_dialogue.
```

### 6.4 Candidate Levels

Candidate generation should always include these fallback candidates:

```text
L4: typed placeholder
L5: complete redaction
```

This prevents the selector from failing if LLM-generated abstractions are invalid.

### 6.5 Policy Handling

Policy file:

```yaml
privacy_abstraction:
  default_task_family: "general"
  utility_threshold: 0.85
  max_candidates_per_turn: 6
  allowed_levels:
    PL1: ["L0", "L1", "L2", "L3"]
    PL2: ["L1", "L2", "L3", "L4", "L5"]
    PL3: ["L2", "L3", "L4", "L5"]
    PL4: ["L4", "L5"]
  type_overrides:
    Verification Code:
      allowed_levels: ["L5"]
      retention: "no_retention"
    Password:
      allowed_levels: ["L5"]
      retention: "no_retention"
    API Key:
      allowed_levels: ["L5"]
      retention: "no_retention"
    Detailed Address:
      allowed_levels: ["L2", "L3", "L4", "L5"]
      retention: "local_only"
```

The policy validator must reject a candidate if:

1. it uses a level not allowed for a privacy item,
2. it keeps raw PL4 values in `public_memory`,
3. it omits required private residue for local-only items,
4. it uses `local_only` for items whose override is `no_retention`.

## 7. Module 2: AMA

### 7.1 Responsibility

AMA simulates adversarial attempts to recover private information from cloud-visible memory.

It is used for:

1. evaluating PMA privacy,
2. scoring candidates,
3. building preference data,
4. finding failure cases for future adversarial training.

### 7.2 Interface

```python
@dataclass
class AttackResult:
    attack_type: str
    original_text: str
    privacy_type: str
    privacy_level: str
    guess: str
    success: bool
    match_score: float
    rationale: str

class AdversarialMemoryAuditor:
    def attack_exact_reconstruction(
        self,
        public_memory: str,
        privacy_item: dict,
        auxiliary_context: dict | None = None,
    ) -> AttackResult:
        ...

    def attack_attribute_inference(
        self,
        public_memory: str,
        privacy_item: dict,
        auxiliary_context: dict | None = None,
    ) -> AttackResult:
        ...

    def audit_candidate(
        self,
        candidate: AbstractionCandidate,
        privacy_items: list[dict],
        auxiliary_context: dict | None = None,
    ) -> list[AttackResult]:
        ...
```

### 7.3 Attack Types

Implement in this order:

1. exact reconstruction,
2. attribute inference,
3. semantic narrowing,
4. compositional linkage.

First milestone only needs exact reconstruction.

### 7.4 Exact Reconstruction Matching

Use a tiered matcher:

```text
Tier 1: exact normalized string match
Tier 2: substring containment
Tier 3: token F1 for non-strict privacy types
Tier 4: LLM judge for semantic equivalence
```

Strict types should require exact or near-exact match:

```text
Verification Code
Password
API Key
Recovery Code
Government ID
Financial Account
Email
Phone Number
Detailed Address
```

Non-strict types can use semantic matching:

```text
Medical Health
Relationship Info
Political Views/Stance
Religious Beliefs
Itinerary/Trajectory
Job Intent/Status
```

### 7.5 AMA Prompt Contract

`evaluation/prompts/ama_exact_reconstruct.txt` should require JSON:

```json
{
  "guess": "...",
  "confidence": 0.0,
  "rationale": "..."
}
```

The prompt must explicitly state that the attacker only sees the public memory. It should not leak the original value in the prompt.

## 8. Module 3: PUC

### 8.1 Responsibility

PUC evaluates whether a candidate is good enough.

It combines:

```text
utility evidence + privacy evidence + policy constraints
```

In the first version, PUC is a deterministic scorer and selector. Later it can become a learned critic.

### 8.2 Interface

```python
@dataclass
class UtilityScore:
    mcq_accuracy: float
    answer_consistency: float | None
    num_questions: int
    num_valid: int

@dataclass
class PrivacyScore:
    exact_reconstruction_rate: float
    attribute_inference_rate: float | None
    semantic_leakage_score: float | None

@dataclass
class CandidateScore:
    candidate_id: str
    utility: UtilityScore
    privacy: PrivacyScore
    attacks: list[AttackResult]

class PrivacyUtilityCritic:
    def score_utility(...):
        ...

    def score_privacy(...):
        ...

    def select_candidate(
        self,
        candidates: list[AbstractionCandidate],
        scores: list[CandidateScore],
        utility_threshold: float,
    ) -> AbstractionCandidate:
        ...
```

### 8.3 Selection Rule

Primary rule:

```text
choose candidate with minimum leakage among candidates whose utility >= threshold
```

Tie-breakers:

1. lower privacy level exposure,
2. shorter public memory,
3. lower abstraction level only if leakage tie remains,
4. typed placeholder fallback if no candidate passes.

Pseudo-code:

```python
valid = [
    (candidate, score)
    for candidate, score in zip(candidates, scores)
    if score.utility.mcq_accuracy >= utility_threshold
]

if not valid:
    return typed_placeholder_candidate

return min(
    valid,
    key=lambda x: (
        x[1].privacy.exact_reconstruction_rate,
        x[1].privacy.attribute_inference_rate or 0.0,
        len(x[0].public_memory),
    ),
)[0]
```

### 8.4 Learned PUC Later

Once candidate data exists, train a lightweight critic:

```text
input: raw dialogue + privacy_items + candidate public_memory
output: predicted utility_score and leakage_score
```

The first trained PUC can be a classifier/regressor that predicts:

```json
{
  "utility_pass": true,
  "leakage_bucket": "low | medium | high"
}
```

This is easier and more robust than predicting exact real-valued scores.

## 9. Pipeline 1: Candidate Generation

Command:

```bash
python evaluation/build_pma_candidates.py \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/pma_candidates.jsonl \
  --users 100 \
  --task-family recommendation \
  --config evaluation/eval_config.yaml
```

Input:

```text
data/*.jsonl
```

Output:

```text
evaluation/results/pma_candidates.jsonl
```

Each line:

```json
{
  "source": {
    "user_id": "...",
    "turn_index": 0,
    "role": "user"
  },
  "dialogue": "...",
  "privacy_items": [...],
  "candidates": [...]
}
```

Implementation details:

1. read each JSONL user,
2. iterate dialogues with privacy items,
3. skip messages with no privacy unless `--include-non-private` is set,
4. generate L1-L5 candidates,
5. validate schema,
6. add typed-placeholder and redaction fallbacks,
7. write one source record per message.

## 10. Pipeline 2: Candidate Scoring

Command:

```bash
python evaluation/score_pma_candidates.py \
  --candidates evaluation/results/pma_candidates.jsonl \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/pma_candidate_scores.jsonl \
  --memory-system none \
  --attack exact
```

Two scoring modes:

```text
memory-system none:
  score candidate in isolation with question answering prompts.

memory-system mem0:
  insert candidate public memories into Mem0 and evaluate retrieval QA.
```

Start with `none` for speed, then use `mem0` for final evidence.

Output:

```json
{
  "source_id": "...",
  "candidate_id": "...",
  "utility": {...},
  "privacy": {...},
  "attacks": [...]
}
```

## 11. Pipeline 3: Training Data Construction

Command:

```bash
python evaluation/build_pma_train_data.py \
  --candidates evaluation/results/pma_candidates.jsonl \
  --scores evaluation/results/pma_candidate_scores.jsonl \
  --sft-output evaluation/results/pma_sft_train.jsonl \
  --preference-output evaluation/results/pma_pref_train.jsonl \
  --utility-threshold 0.85
```

Selection:

```text
chosen = lowest leakage candidate with utility >= threshold
rejected = all candidates with worse utility-leakage tradeoff
```

SFT output format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a local privacy memory abstractor..."
    },
    {
      "role": "user",
      "content": "{input_json}"
    },
    {
      "role": "assistant",
      "content": "{target_output_json}"
    }
  ]
}
```

Preference output format:

```json
{
  "prompt": "{input_json}",
  "chosen": "{chosen_output_json}",
  "rejected": "{rejected_output_json}",
  "metadata": {
    "chosen_utility": 0.91,
    "chosen_leakage": 0.0,
    "rejected_utility": 0.92,
    "rejected_leakage": 1.0
  }
}
```

## 12. Pipeline 4: PMA Evaluation with Mem0

Command:

```bash
python evaluation/eval_pma_mem0.py \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/pma_mem0_eval.json \
  --users 100 \
  --is-mcq true \
  --turns-per-chunk 5 \
  --methods raw complete type_specific pma_oracle
```

Evaluation loop:

1. load user data,
2. for each method, transform user and assistant turns,
3. write transformed turns into Mem0,
4. retrieve memories for each question,
5. answer MCQ using existing answer prompt,
6. score accuracy,
7. run AMA against cloud-visible memories,
8. report utility and leakage.

The first version can skip assistant turns if implementation time is limited, but the paper experiment should include both user and assistant turns because sensitive information can be echoed by assistant responses.

## 13. Baseline Implementation

The evaluation script must support:

```text
raw
complete
generic
type_specific
pma_oracle
pma_sft
```

Mapping:

```text
raw:
  no transform

complete:
  complete_mask_dialogue(...)

generic:
  PrivacyStore(mask_mode="generic") + mask_dialogue(...)

type_specific:
  PrivacyStore(mask_mode="type_specific") + mask_dialogue(...)

pma_oracle:
  PMA oracle candidates + PUC selector

pma_sft:
  trained local PMA
```

## 14. Result File Contract

Final eval output:

```json
{
  "run": {
    "dataset": "data/memprivacy_bench_testset.jsonl",
    "memory_system": "mem0",
    "num_users": 100,
    "is_mcq": true,
    "created_at": "ISO-8601"
  },
  "methods": {
    "type_specific": {
      "utility": {
        "mcq_accuracy": 0.0,
        "total_score": 0,
        "total_num": 0
      },
      "privacy": {
        "exact_reconstruction_rate": 0.0,
        "num_attacks": 0
      }
    },
    "pma_oracle": {
      "utility": {
        "mcq_accuracy": 0.0,
        "total_score": 0,
        "total_num": 0
      },
      "privacy": {
        "exact_reconstruction_rate": 0.0,
        "num_attacks": 0
      }
    }
  },
  "records": [
    {
      "user_id": "...",
      "question": "...",
      "answer": "...",
      "method": "pma_oracle",
      "response": "...",
      "score": 1,
      "selected_abstractions": [...],
      "attacks": [...]
    }
  ]
}
```

The paper table should be generated from this JSON, not hand-calculated.

## 15. Tests and Validation

### 15.1 Unit Tests

`tests/test_privacy_abstraction_schema.py`:

1. valid candidate passes,
2. missing `public_memory` fails,
3. invalid level fails,
4. PL4 raw value in public memory fails,
5. private residue must align with privacy item.

`tests/test_pma_selection.py`:

1. chooses lower leakage when utility passes,
2. rejects high utility candidate if leakage is high and safer candidate has sufficient utility,
3. falls back to typed placeholder when no candidate passes threshold,
4. handles ties deterministically.

`tests/test_ama_matching.py`:

1. exact OTP match succeeds,
2. different OTP fails,
3. address substring match succeeds,
4. semantic health match succeeds only for non-strict mode.

### 15.2 Integration Tests

Minimal integration test:

```bash
python evaluation/build_pma_candidates.py \
  --input data/memprivacy_bench_testset.jsonl \
  --output /tmp/pma_candidates.jsonl \
  --users 1 \
  --max-turns 2
```

Expected:

```text
non-empty JSONL
each record has candidates
each candidate validates
fallback candidates exist
```

Scoring smoke test:

```bash
python evaluation/score_pma_candidates.py \
  --candidates /tmp/pma_candidates.jsonl \
  --output /tmp/pma_scores.jsonl \
  --attack exact \
  --dry-run
```

Expected:

```text
one score per candidate
privacy fields present
no raw secret inserted into attack prompt
```

### 15.3 Paper-Evidence Validation

Before using numbers in the paper:

1. confirm result JSON includes all baselines,
2. confirm type-specific baseline uses typed placeholders, not generic,
3. confirm AMA only sees public memory,
4. confirm raw private values are used only for scoring matches,
5. confirm no original private value leaks into candidate generation output except private residue,
6. confirm MCQ scoring uses the existing answer key.

## 16. Milestones

### Milestone 0: Documentation

Deliverables:

```text
docs/paper_concept_pma.md
docs/pma_code_development_plan.md
```

Exit criteria:

```text
paper claims have matching code modules and evaluation artifacts
```

### Milestone 1: Oracle Candidate Pipeline

Deliverables:

```text
src/privacy_abstraction.py
evaluation/build_pma_candidates.py
evaluation/prompts/pma_generate_candidates.txt
```

Exit criteria:

```text
candidate JSONL generated for 10 users
schema validation passes
typed and redaction fallbacks included
```

### Milestone 2: AMA Exact Reconstruction

Deliverables:

```text
src/privacy_auditor.py
evaluation/prompts/ama_exact_reconstruct.txt
tests/test_ama_matching.py
```

Exit criteria:

```text
exact reconstruction score produced for each candidate/privacy item pair
strict matching works for identifiers and credentials
```

### Milestone 3: PUC Selection

Deliverables:

```text
src/privacy_critic.py
evaluation/score_pma_candidates.py
evaluation/build_pma_train_data.py
tests/test_pma_selection.py
```

Exit criteria:

```text
chosen/rejected pairs generated
selector behavior is deterministic and tested
```

### Milestone 4: Mem0 Evaluation

Deliverables:

```text
evaluation/eval_pma_mem0.py
evaluation/compare_pma_results.py
```

Exit criteria:

```text
raw, complete, type_specific, pma_oracle compared on same users/questions
result JSON contains utility and leakage metrics
```

### Milestone 5: PMA-SFT

Deliverables:

```text
training/train_pma_sft.py
trained model checkpoint
evaluation/eval_pma_mem0.py --methods pma_sft
```

Exit criteria:

```text
pma_sft output schema validity >= 95%
pma_sft evaluated against type_specific baseline
```

### Milestone 6: Preference PMA

Deliverables:

```text
training/train_pma_preference.py
preference-tuned checkpoint
```

Exit criteria:

```text
preference PMA improves privacy-utility frontier over SFT PMA
```

## 17. Development Order

Recommended order:

1. schemas and validators,
2. fallback candidate generation,
3. prompt-based candidate generation,
4. AMA exact reconstruction matcher,
5. PUC selector,
6. candidate and score JSONL builders,
7. Mem0 evaluation integration,
8. SFT data exporter,
9. PMA-SFT training,
10. preference data exporter and training.

This order avoids training before proving that the abstraction target is useful.

## 18. Engineering Risks

Risk: LLM candidate JSON is malformed.

Mitigation:

```text
json_repair + schema validation + fallback candidates
```

Risk: LLM abstractions hallucinate unsupported claims.

Mitigation:

```text
add support-check prompt or require abstraction_trace to cite raw spans
```

Risk: AMA sees original value accidentally.

Mitigation:

```text
separate attack prompt construction from scoring function
unit-test that prompt string does not contain original_text
```

Risk: utility scoring is too expensive with Mem0.

Mitigation:

```text
two-stage scoring: cheap no-memory scoring first, Mem0 only for selected candidates
```

Risk: PMA learns unsafe paraphrases.

Mitigation:

```text
include leakage-ranked rejected samples and policy validation at inference
```

Risk: evaluation scripts duplicate logic.

Mitigation:

```text
centralize transform methods in src/privacy_abstraction.py and import them in evaluation scripts
```

## 19. Minimum Viable Implementation

If implementation time is limited, build this subset first:

```text
src/privacy_abstraction.py
src/privacy_auditor.py
src/privacy_critic.py
evaluation/build_pma_candidates.py
evaluation/score_pma_candidates.py
evaluation/build_pma_train_data.py
```

Skip:

```text
LangMem integration
Memobase integration
preference training
learned PUC
compositional linkage attack
```

The minimum useful claim becomes:

```text
oracle-selected abstractions beat typed placeholders on utility at comparable exact reconstruction leakage.
```

This is enough to justify training PMA.

## 20. Definition of Done

The code implementation is complete enough for the first paper if:

1. all baselines run on the same dataset split,
2. PMA oracle generates valid candidates,
3. PUC selects candidates using a documented rule,
4. AMA exact reconstruction attacks all public memories,
5. result JSON reports utility and leakage for each method,
6. PMA training data can be produced from candidate scores,
7. PMA-SFT can be evaluated with the same script as the oracle,
8. every number used in the paper can be traced to a result JSON file.

## 21. Direct Links to Paper Sections

Use this mapping when writing or reviewing the paper:

| Paper Section | Code Evidence |
|---|---|
| Problem formulation | `docs/paper_concept_pma.md`, schemas in `src/privacy_schema.py` |
| PMA method | `src/privacy_abstraction.py` |
| AMA privacy attack | `src/privacy_auditor.py`, `evaluation/prompts/ama_exact_reconstruct.txt` |
| PUC selection | `src/privacy_critic.py` |
| Training data construction | `evaluation/build_pma_train_data.py` |
| Mem0 experiment | `evaluation/eval_pma_mem0.py` |
| Main table | `evaluation/results/pma_mem0_*.json` |
| Ablations | `evaluation/eval_pma_mem0.py --ablation ...` |

## 22. First Implementation Checklist

Use this checklist before writing code:

```text
[ ] Define dataclasses or pydantic models for candidates and scores.
[ ] Add YAML policy with allowed abstraction levels.
[ ] Implement typed-placeholder and redaction fallback candidates.
[ ] Implement oracle prompt candidate generation.
[ ] Validate candidate schema and policy constraints.
[ ] Implement AMA prompt and exact matching.
[ ] Implement PUC selection rule.
[ ] Build candidate JSONL.
[ ] Build score JSONL.
[ ] Build SFT and preference JSONL.
[ ] Run 1-user smoke test.
[ ] Run 10-user oracle test.
[ ] Run 100-user Mem0 evaluation.
```

## 23. Non-Goals for First Version

Do not attempt these in the first implementation:

1. formal differential privacy guarantees,
2. full compositional re-identification modeling,
3. simultaneous support for three memory systems,
4. end-to-end adversarial training,
5. production-grade encrypted local residue store,
6. human subject privacy study.

These are valid future work, but they should not block the first paper loop.

## 24. Final Engineering Principle

Every paper claim must map to an artifact:

```text
claim -> code path -> result JSON -> table/figure
```

If a claim cannot be traced this way, it should not appear as a main result.
