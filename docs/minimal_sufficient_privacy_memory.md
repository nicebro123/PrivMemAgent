# Minimal Sufficient Privacy Memory

## 1. Research Position

This project should move beyond reversible placeholder masking and study a deeper question:

> What is the minimal public memory representation that preserves downstream personalization utility while preventing recovery of sensitive private values?

The current MemPrivacy design protects raw sensitive spans by replacing them with type-aware placeholders, for example:

```text
Paris 1st arrondissement, 15 Rue de Rivoli -> <Detailed_Address_1>
829417 -> <Verification_Code_1>
```

This is effective for value hiding, but it treats privacy protection mostly as span replacement. The proposed direction treats memory as a representation learning problem:

```text
raw memory M -> public abstract memory Z + local private residue R
```

Only `Z` is sent to cloud memory. `R` remains local. The key claim is that many personalized tasks do not need the raw private value; they need a task-sufficient abstraction.

Example:

```text
Raw:
The user lives in a top-floor apartment at 15 Rue de Rivoli, Paris 1st arrondissement.

Placeholder:
The user lives at <Detailed_Address_1>.

Minimal sufficient abstraction:
The user is based in central Paris and strongly prefers low-commute indoor activities.
```

The third representation is often more useful than a placeholder and safer than the raw address.

## 2. Core Hypothesis

Typed placeholders are one endpoint of a privacy-utility spectrum. They preserve privacy values but often discard task-relevant semantics. A learned or policy-guided abstraction can dominate placeholders by:

1. preserving more task-relevant information,
2. leaking less recoverable private information than raw memory,
3. reducing the need for downstream local unmasking.

Formally, for raw memory `M`, public memory `Z`, sensitive value or attribute `S`, and future task target `Y`:

```text
maximize utility:  I(Z; Y)
minimize leakage: I(Z; S)
```

This follows the spirit of information bottleneck and privacy funnel formulations, but specializes them to long-term LLM agent memory.

## 3. Main Contribution

The main contribution should be framed as:

> Minimal Sufficient Privacy Memory: a memory abstraction framework that decomposes user memories into cloud-storable task-sufficient abstractions and local-only private residues.

This is not just another detector. It changes the memory object itself.

Current MemPrivacy:

```text
M_cloud = mask(M_raw)
M_local = placeholder <-> raw value map
```

Proposed framework:

```text
M_cloud = abstract(M_raw, task_family, privacy_policy)
M_local = private_residue(M_raw, M_cloud)
```

The cloud memory becomes an irreversible abstraction by default. Local restoration is reserved only for cases where exact values are truly needed.

## 4. Representation Levels

For each privacy item, generate a lattice of abstraction candidates:

```text
L0 raw value:
I have pollen allergy and took loratadine this morning.

L1 medical category abstraction:
I have seasonal allergy and took an antihistamine this morning.

L2 functional constraint abstraction:
I have an environment-triggered health sensitivity and prefer indoor activities.

L3 task preference abstraction:
I prefer low-allergen indoor environments.

L4 typed placeholder:
I have <Medical_Health_1>.

L5 full redaction:
I have ***.
```

The system chooses the lowest-leakage level that still preserves utility for the expected task family.

## 5. Algorithmic Design

### 5.1 Memory Decomposition

Input:

```text
dialogue turn x
privacy annotations p = [{original_text, privacy_type, privacy_level}]
task family c, such as recommendation, drafting, medical advice avoidance, travel planning
policy pi
```

Output:

```json
{
  "public_memory": "...",
  "private_residue": [
    {
      "raw": "...",
      "type": "...",
      "level": "...",
      "linked_public_claim": "...",
      "retention": "local_only | session_only | no_retention"
    }
  ],
  "decision_trace": [
    {
      "span": "...",
      "chosen_level": "L2",
      "utility_reason": "...",
      "privacy_reason": "..."
    }
  ]
}
```

### 5.2 Candidate Generation

For each privacy-bearing memory, generate abstraction candidates using an LLM or local small model:

```text
generate_candidates(M, privacy_items, task_family) -> [Z_0, Z_1, ..., Z_k]
```

The prompt should force monotonic abstraction:

1. `Z_0`: raw-equivalent paraphrase.
2. `Z_1`: category-level abstraction.
3. `Z_2`: functional implication.
4. `Z_3`: task preference or constraint.
5. `Z_4`: typed placeholder.
6. `Z_5`: full redaction.

### 5.3 Utility Scoring

Use counterfactual answer consistency:

```text
answer_raw = AnswerLLM(question, memories=M_raw)
answer_abs = AnswerLLM(question, memories=Z)
utility_score = Judge(answer_raw, answer_abs, reference_answer)
```

For MCQ, use exact option match. For open QA, use the existing judge prompt.

This avoids needing a perfect symbolic definition of utility. Utility means: does the abstraction preserve the answer behavior that raw memory would have enabled?

### 5.4 Leakage Scoring

Evaluate whether an attacker can recover private information:

```text
attack_output = AttackLLM(Z, privacy_type, auxiliary_context)
leakage_score = match(attack_output, raw_private_value)
```

Three attack levels:

1. direct reconstruction: recover the exact value,
2. attribute inference: recover sensitive category or high-risk trait,
3. linkage attack: combine multiple abstract memories to narrow down identity or location.

For exact values, use string and semantic match. For attributes, use classification accuracy. For linkage, measure candidate set reduction if external priors are available; otherwise use LLM judge as a first approximation.

### 5.5 Candidate Selection

Choose the abstraction with minimum leakage subject to utility constraints:

```text
select Z_i such that:
  utility_score(Z_i) >= tau_u
and
  leakage_score(Z_i) is minimal
```

If no candidate satisfies utility:

1. fall back to type-specific placeholder,
2. keep exact value local,
3. optionally ask the local policy whether exact cloud utility is allowed.

## 6. Experimental Design

### 6.1 Datasets

Start with the existing repository datasets:

1. `data/memprivacy_bench_testset.jsonl`
2. `data/personamem_v2_testset.jsonl`

They already contain:

1. multi-turn dialogues,
2. privacy annotations,
3. user metadata,
4. downstream questions.

This makes them sufficient for the first prototype.

### 6.2 Baselines

Compare against four methods:

1. raw memory, upper-bound utility and worst privacy,
2. complete mask, low utility and strong value hiding,
3. generic placeholder, `<MASK_1>`,
4. type-specific placeholder, current MemPrivacy baseline.

Add the proposed method:

5. minimal sufficient abstraction.

### 6.3 Metrics

Utility:

```text
QA accuracy
MCQ accuracy
counterfactual answer consistency
retrieval hit rate
```

Privacy:

```text
exact value reconstruction rate
privacy type inference rate
sensitive attribute inference rate
linkage risk score
```

Tradeoff:

```text
privacy-utility frontier
area under privacy-utility curve
utility retained at fixed leakage budget
leakage reduced at fixed utility floor
```

### 6.4 Main Research Question

The most important plot is not a single accuracy number. It is a frontier:

```text
x-axis: privacy leakage
y-axis: utility retained
```

The paper is strong if minimal sufficient abstraction shifts the frontier upward-left relative to type-specific placeholders.

## 7. Implementation Plan

### Phase 1: Non-training Prototype

Goal: prove the evaluation signal exists before training any model.

Add:

```text
src/privacy_abstraction.py
evaluation/eval_abstraction.py
evaluation/attack.py
evaluation/abstraction_metrics.py
prompts/generate_abstractions.txt
prompts/attack_reconstruct.txt
prompts/judge_abstraction_utility.txt
```

Implement:

1. read one JSONL user record,
2. generate abstraction candidates for each memory chunk,
3. score utility using existing question-answer pipeline,
4. score leakage using attacker prompts,
5. choose the best candidate under a utility threshold,
6. write results to `evaluation/results/abstraction_*`.

Expected difficulty: medium.

Why medium: no model training, but careful experiment plumbing is required.

### Phase 2: Policy-Guided Abstraction

Goal: make abstraction decisions deterministic enough for ablations.

Add a YAML policy:

```yaml
privacy_abstraction:
  default_utility_threshold: 0.85
  levels:
    PL4:
      allowed_public_levels: ["L4", "L5"]
      retention: "session_only"
    PL3:
      allowed_public_levels: ["L2", "L3", "L4", "L5"]
      retention: "local_only"
    PL2:
      allowed_public_levels: ["L1", "L2", "L3", "L4"]
      retention: "local_only"
  type_overrides:
    Verification Code:
      allowed_public_levels: ["L5"]
      retention: "no_retention"
    Detailed Address:
      allowed_public_levels: ["L2", "L3", "L4"]
```

Expected difficulty: medium-low.

Why medium-low: mostly configuration, validation, and integration.

### Phase 3: Trainable Abstractor

Goal: reduce dependence on large LLM prompting and make the framework edge-friendly.

Data generation:

1. use Phase 1 to produce candidate abstractions,
2. label selected candidates using utility/leakage scoring,
3. train a small instruction model to output the selected abstraction and residue schema.

Training target:

```json
{
  "public_memory": "...",
  "private_residue": [...],
  "decision_trace": [...]
}
```

Expected difficulty: high.

Why high: needs high-quality synthetic supervision, robust JSON output, and careful leakage evaluation.

### Phase 4: Adversarial Training

Goal: make the abstractor robust against reconstruction attacks.

Loop:

```text
abstractor generates Z
utility judge checks task preservation
attacker tries to infer S
abstractor is penalized if attacker succeeds
```

Expected difficulty: very high.

Why very high: LLM-as-attacker and LLM-as-judge are noisy, optimization may become unstable, and automatic leakage labels can be brittle.

## 8. Code Integration Points

Current useful hooks:

1. `evaluation/privacy_masking.py` already implements `type_specific`, `generic`, and `complete`.
2. `evaluation/eval_mem0.py`, `eval_langmem.py`, and `eval_memobase.py` already contain memory-write and QA-evaluation loops.
3. `evaluation/utils.py` already provides OpenAI-compatible LLM calls and JSON parsing.
4. `evaluation/metric.py` already contains privacy extraction matching logic that can be reused for leakage scoring.

Minimal invasive integration:

```python
if mask_mode == "abstraction":
    user_content = abstract_dialogue(
        user_content,
        privacy_items,
        abstraction_store,
        task_family=...,
        policy=...
    )
```

This can be added without rewriting the existing memory-system evaluations.

## 9. Innovation Assessment

### 9.1 Novelty

High, if framed correctly.

MemPrivacy already contributes type-aware placeholder masking for edge-cloud memory. The proposed direction changes the object stored in cloud memory from "masked raw text" to "minimal task-sufficient abstraction." That is a deeper representation-level contribution.

The closest theoretical neighbors are information bottleneck and privacy funnel, but those are general privacy-utility frameworks. The novel specialization is:

1. long-term LLM agent memory,
2. personalized downstream tasks,
3. local private residue,
4. adversarial reconstruction evaluation,
5. multi-granularity natural-language memory abstraction.

### 9.2 Research Risk

Medium-high.

The main risk is that LLM-generated abstractions may look good but leak too much through world knowledge. Another risk is evaluation noise: utility and leakage are both difficult to measure automatically.

Mitigation:

1. start with MCQ tasks where utility is objective,
2. use exact-value reconstruction for first leakage metric,
3. separate "value leakage" from "attribute leakage",
4. report privacy-utility frontiers, not only single numbers.

### 9.3 Paper Strength

Potentially strong if experiments show one of these:

1. same utility as type-specific placeholders with lower leakage,
2. same leakage as placeholders with higher utility,
3. better privacy-utility frontier across several memory systems.

The strongest result would be:

```text
At the same reconstruction leakage budget, abstraction retains more personalization accuracy than type-aware placeholders.
```

### 9.4 Implementation Difficulty

Overall difficulty: medium-high.

Breakdown:

| Component | Difficulty | Reason |
|---|---:|---|
| Candidate abstraction prompting | Medium | Prompt design and JSON validation needed |
| Utility evaluation | Medium-low | Existing QA pipeline can be reused |
| Exact reconstruction attack | Medium | Attacker prompt plus matching logic |
| Attribute/linkage attack | High | Requires careful threat model |
| Policy engine | Medium-low | Mostly deterministic config logic |
| Memory-system integration | Medium | Existing scripts are duplicated and need careful patching |
| Trainable abstractor | High | Requires data generation and model training |
| Adversarial training | Very high | Noisy optimization and hard labels |

## 10. Recommended First Milestone

Do not start with training.

The first milestone should be a 100-user prompt-based study:

```text
Dataset: memprivacy_bench_testset.jsonl
Memory system: start with Mem0 only
Task mode: MCQ first
Baselines: raw, complete, generic, type_specific
Proposed: abstraction
Mask levels: PL2 + PL3 + PL4
Main output: privacy-utility frontier
```

Success criterion:

```text
abstraction_utility >= type_specific_utility
and
abstraction_exact_reconstruction <= type_specific_exact_reconstruction
```

If this fails, inspect by privacy type. The method may still be useful for address, health, itinerary, and preference-like privacy, while not useful for credentials and identifiers.

## 11. Expected Failure Cases

1. `PL4` credentials should not be abstracted; they should be blocked.
2. IDs, phone numbers, emails, account numbers usually have no useful abstraction beyond type.
3. Rare diseases, rare locations, and unique occupations can leak identity even after abstraction.
4. Multiple harmless abstractions can compose into a unique fingerprint.
5. LLM judges may overestimate semantic equivalence.

These should not be hidden. They are part of the research contribution because they define where minimal sufficient abstraction is valid.

## 12. Concrete Next Step

Implement Phase 1 with one new abstraction mode:

```text
mask_mode = "abstraction"
```

Start only in `evaluation/eval_mem0.py` to avoid three-way integration cost. Once the signal is validated, port the mode to LangMem and Memobase.

The first implementation should produce a per-question record:

```json
{
  "user_id": "...",
  "question": "...",
  "answer": "...",
  "baseline_type_specific_response": "...",
  "abstraction_response": "...",
  "utility_score": 1,
  "attacks": {
    "exact_reconstruction": {
      "success": false,
      "guess": "..."
    },
    "attribute_inference": {
      "success": true,
      "guess": "..."
    }
  },
  "selected_abstractions": [...]
}
```

This record format will make analysis possible without rerunning expensive memory experiments.

## 13. References

1. MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents, arXiv:2605.09530.
2. Tishby, Pereira, and Bialek, The Information Bottleneck Method, arXiv:physics/0004057.
3. Makhdoumi, Salamatian, Fawaz, Médard, and Calmon, From the Information Bottleneck to the Privacy Funnel, arXiv:1402.1774.
4. Zarrabian and Sadeghi, An Algorithm for Enhancing Privacy-Utility Tradeoff in the Privacy Funnel and Other Lift-based Measures, arXiv:2408.09659.
