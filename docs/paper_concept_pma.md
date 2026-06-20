# Paper Concept: Trainable Privacy Memory Abstraction for Cloud Agents

## 1. Working Title

**Learning Minimal Sufficient Public Memories for Privacy-Preserving Personalized Agents**

Alternative titles:

1. **PMA: A Trainable Privacy Memory Abstractor for Edge-Cloud Agents**
2. **Beyond Placeholder Masking: Learning Task-Sufficient Privacy Abstractions for Agent Memory**
3. **Private Residue, Public Memory: Learning What Cloud Agents Should Remember**

Recommended title:

> Learning Minimal Sufficient Public Memories for Privacy-Preserving Personalized Agents

This title is stronger than a module-only title because it emphasizes the research object: public memory representation.

## 2. One-Sentence Thesis

Long-term personalized agents should not store masked raw conversations in the cloud; they should store learned public abstractions that preserve task-relevant user semantics while leaving sensitive private residues on device.

## 3. Core Problem

Existing privacy-preserving memory methods mostly protect sensitive spans through masking:

```text
raw sensitive value -> placeholder or redaction
```

This protects raw values, but it creates a structural problem:

```text
typed placeholders hide values but also remove task-relevant semantics.
```

Example:

```text
Raw memory:
The user lives at 15 Rue de Rivoli, Paris, and wants quiet indoor cultural activities nearby.

Typed placeholder memory:
The user lives at <Detailed_Address_1> and wants quiet indoor cultural activities nearby.

Learned public memory:
The user is based in central Paris and prefers low-commute quiet indoor cultural activities.
```

The learned public memory is less private than full redaction but more useful than a placeholder. It is also far less reconstructive than the raw address.

This motivates the central research question:

> Can we train a local module to produce public memories that are sufficient for personalization tasks but insufficient for private value reconstruction?

## 4. Key Insight

Privacy protection for agent memory should not be formulated only as span-level sanitization. It should be formulated as memory representation learning.

The raw memory `M` can be decomposed into:

```text
M = Z_public + R_private
```

where:

- `Z_public` is sent to cloud memory,
- `R_private` remains local,
- `Z_public` should preserve information needed for future tasks,
- `Z_public` should remove information needed to reconstruct private values.

This gives the paper its conceptual contribution:

> The correct cloud memory is not the masked conversation. It is the minimal sufficient public abstraction of the conversation.

## 5. Proposed Method

### 5.1 Module: Privacy Memory Abstractor

Introduce **Privacy Memory Abstractor, PMA**, a trainable local sequence-to-structure model:

```text
PMA_theta(x, p, c, pi) -> (z, r, t)
```

where:

- `x`: raw dialogue or memory chunk,
- `p`: privacy items, each with `original_text`, `privacy_type`, `privacy_level`,
- `c`: task family, such as recommendation, planning, drafting, QA,
- `pi`: privacy policy,
- `z`: public abstract memory,
- `r`: private residue,
- `t`: abstraction trace.

Output schema:

```json
{
  "public_memory": "...",
  "private_residue": [
    {
      "raw": "...",
      "privacy_type": "...",
      "privacy_level": "...",
      "retention": "local_only | session_only | no_retention"
    }
  ],
  "abstraction_trace": [
    {
      "raw": "...",
      "public_abstraction": "...",
      "reason": "..."
    }
  ]
}
```

### 5.2 Abstraction Levels

PMA learns to select among abstraction levels:

```text
L0: raw or near-raw memory
L1: category-level abstraction
L2: functional implication
L3: task preference or constraint
L4: typed placeholder
L5: full redaction
```

Example:

```text
L0: I have pollen allergy and took loratadine.
L1: I have seasonal allergy and took an antihistamine.
L2: I have an environment-triggered sensitivity and prefer indoor plans.
L3: I prefer low-allergen indoor environments.
L4: I have <Medical_Health_1>.
L5: I have ***.
```

The model does not blindly pick the most abstract representation. It picks the least-leaking representation that still preserves task utility.

### 5.3 Training Signal

PMA is trained from candidate abstractions scored by utility and leakage.

For each raw memory:

1. generate multiple abstraction candidates,
2. evaluate downstream utility using existing memory QA tasks,
3. evaluate privacy leakage using reconstruction attacks,
4. select the best candidate under a utility constraint,
5. train PMA using SFT and preference pairs.

Selection rule:

```text
Z* = argmin leakage(Z_i)
subject to utility(Z_i) >= tau
```

Training objective:

```text
L = L_sft + lambda_u * L_utility_rank + lambda_p * L_privacy_rank + lambda_schema * L_schema
```

This makes PMA trainable and evaluable, not just a prompt template.

### 5.4 Three-Module Framework: PMA + PUC + AMA

The full method contains three modules:

```text
PMA: Privacy Memory Abstractor
PUC: Privacy-Utility Critic
AMA: Adversarial Memory Auditor
```

Their interaction is:

```text
raw dialogue + privacy annotations
        |
        v
PMA generates candidate public memories
        |
        v
PUC estimates or measures personalization utility
        |
        v
AMA attacks each public memory for privacy leakage
        |
        v
PUC selects the lowest-leakage candidate that preserves utility
        |
        v
selected candidate becomes supervision for trainable PMA
```

PMA is the generator. It proposes different public memory candidates.

PUC is the selector and critic. In the first version, it uses real downstream QA accuracy and AMA leakage scores. In later versions, it can be trained as a reward model that predicts whether a candidate will preserve utility and avoid leakage.

AMA is the adversary. It tries to reconstruct private values or infer sensitive attributes from cloud-visible public memory. Its failures and successes become the privacy signal for PMA.

This three-module design gives the method a closed training loop:

```text
generate -> evaluate utility -> attack privacy -> select -> train
```

The paper should emphasize that AMA is not only an evaluation tool. It is also a source of hard negative examples for PMA training.

## 6. Threat Model

Cloud memory is honest-but-curious or partially compromised.

The cloud can see:

```text
public memory Z
retrieval queries
cloud-side model outputs
```

The cloud cannot see:

```text
raw private value
local private residue R
local mapping database
```

Attacker goals:

1. reconstruct exact private values,
2. infer sensitive attributes,
3. combine multiple public memories to narrow down identity or location.

The first paper should focus on exact private value reconstruction and attribute inference. Linkage attack can be included as an extended analysis or future work if time is limited.

## 7. Experimental Plan

### 7.1 Datasets

Use the current repository datasets:

```text
data/memprivacy_bench_testset.jsonl
data/personamem_v2_testset.jsonl
```

They already provide:

1. long multi-turn dialogues,
2. privacy annotations,
3. user metadata,
4. downstream questions and answers.

The datasets do not directly contain PMA labels, so labels are derived by candidate generation and scoring.

### 7.2 Baselines

Compare:

1. **Raw**: cloud stores raw memory.
2. **Complete Redaction**: sensitive spans become `***`.
3. **Generic Placeholder**: sensitive spans become `<MASK_1>`.
4. **Typed Placeholder**: sensitive spans become `<Medical_Health_1>`.
5. **PMA**: learned public abstraction plus local private residue.

The strongest baseline is typed placeholder because it is the current MemPrivacy-style method.

### 7.3 Utility Metrics

Primary:

```text
MCQ accuracy
open QA correctness
```

Secondary:

```text
retrieval hit rate
answer consistency against raw-memory oracle
```

Start with MCQ because it is objective and reduces judge noise.

### 7.4 Privacy Metrics

Primary:

```text
exact value reconstruction success rate
```

Secondary:

```text
sensitive attribute inference accuracy
semantic leakage score
privacy-type-specific leakage
```

For exact reconstruction:

```text
Attacker(Z, privacy_type, auxiliary_context) -> guessed raw value
```

A reconstruction is successful if the guessed value exactly or semantically matches `privacy_info.original_text`.

### 7.5 Main Result

The main result should be a privacy-utility frontier:

```text
x-axis: leakage
y-axis: utility
```

The paper is strong if PMA moves the curve upward-left relative to typed placeholders.

Most important claim:

```text
At comparable exact reconstruction leakage, PMA retains higher personalization utility than typed placeholders.
```

## 8. Ablation Studies

Run these ablations:

1. **No task family**: PMA without `task_family`.
2. **No privacy type**: PMA without `privacy_type` labels.
3. **No utility ranking**: SFT-only PMA.
4. **No leakage ranking**: utility-only PMA.
5. **Fixed abstraction level**: always choose L2 or L3.
6. **No private residue**: public memory only, no local residue.
7. **Policy-only abstraction**: hand-written rules without trainable PMA.

Expected result:

```text
PMA with both utility and leakage signals should dominate SFT-only and policy-only versions.
```

## 9. Analysis by Privacy Type

The paper should not claim PMA works equally for all privacy types.

Likely good cases:

```text
Detailed Address -> coarse location / commute preference
Medical Health -> functional constraint
Itinerary -> temporal or travel preference
Relationship Info -> social preference or role
Religious Beliefs -> content constraint
Political Views -> content sensitivity or topic preference
Job Status -> schedule or career constraint
```

Likely bad cases:

```text
OTP
password
API key
recovery code
full account number
government ID
exact email
exact phone number
```

For bad cases, the right output is usually block, typed placeholder, or full redaction.

This distinction strengthens the paper because it shows the method is not overclaiming.

## 10. Expected Contributions

Contribution 1:

> A new formulation of privacy-preserving agent memory as minimal sufficient public memory learning.

Contribution 2:

> A trainable local module, PMA, that decomposes raw memories into public abstractions and local private residues.

Contribution 3:

> A utility-leakage training pipeline that derives PMA supervision from existing privacy-annotated memory QA data.

Contribution 4:

> A privacy-utility evaluation protocol comparing raw memory, redaction, generic placeholders, typed placeholders, and learned abstraction.

Contribution 5:

> Empirical analysis showing which privacy types benefit from abstraction and which should remain blocked or placeholdered.

## 11. Why Reviewers Should Care

This paper addresses a concrete limitation in current privacy-preserving LLM memory:

```text
Masking protects values but weakens personalization.
Raw memory supports personalization but leaks privacy.
```

PMA proposes a third option:

```text
learned public memory that preserves task semantics without preserving raw private values.
```

The idea is scientifically interesting because it connects:

1. privacy-preserving representation learning,
2. long-term LLM agent memory,
3. personalized downstream utility,
4. adversarial reconstruction evaluation.

It is practically important because cloud agents increasingly need long-term memory, but raw memory storage is hard to justify for privacy-sensitive users.

## 12. Implementation Roadmap

### Stage 1: Prompt-Based Oracle

Implement candidate generation and scoring without training PMA.

Files:

```text
evaluation/build_abstraction_candidates.py
evaluation/score_abstraction_utility.py
evaluation/attack_abstraction.py
```

Output:

```text
candidate abstraction dataset with utility and leakage scores
```

Purpose:

```text
verify that useful abstractions exist.
```

### Stage 2: SFT PMA

Train PMA on selected best candidates.

Files:

```text
evaluation/build_pma_sft_data.py
evaluation/train_pma_sft.py
src/privacy_abstraction.py
```

Purpose:

```text
replace expensive prompt-based oracle with a trainable local model.
```

### Stage 3: Preference-Tuned PMA

Use candidate comparisons:

```text
chosen: lower leakage at sufficient utility
rejected: higher leakage or insufficient utility
```

Files:

```text
evaluation/build_pma_preference_data.py
evaluation/train_pma_preference.py
```

Purpose:

```text
teach PMA the privacy-utility tradeoff, not just imitation.
```

### Stage 4: Memory-System Evaluation

Integrate PMA into Mem0 first:

```text
mask_mode = "abstraction"
```

Then port to LangMem and Memobase.

Purpose:

```text
show that PMA improves real memory system performance, not just standalone text quality.
```

## 13. Initial Paper Structure

### Abstract

State the problem: long-term cloud memory creates privacy risk; masking protects privacy but harms utility.

State the method: trainable privacy memory abstractor that learns public abstractions and local private residues.

State the result: higher utility than typed placeholders at comparable reconstruction leakage.

### 1. Introduction

Use a concrete example:

```text
exact address -> central Paris + low-commute preference
medical condition -> low-allergen indoor constraint
OTP -> no retention
```

Emphasize that the paper is about learning what should be remembered, not detecting PII.

### 2. Related Work

Cover:

1. LLM memory systems,
2. privacy-preserving LLM agents,
3. PII detection and masking,
4. information bottleneck and privacy-utility tradeoff,
5. text anonymization and abstraction.

### 3. Problem Formulation

Define:

```text
M: raw memory
Z: public memory
R: private residue
S: sensitive value
Y: downstream task
```

Objective:

```text
maximize utility(Z, Y)
minimize leakage(Z, S)
```

### 4. Method

Describe PMA input/output, abstraction levels, training data construction, objective, and deployment pipeline.

### 5. Experimental Setup

Datasets, memory systems, baselines, utility metrics, privacy attacks.

### 6. Results

Main utility-leakage frontier, baseline comparison, per-privacy-type analysis.

### 7. Ablations and Analysis

Training signal ablations, task-family conditioning, leakage failure cases.

### 8. Limitations

Discuss:

1. abstraction may still leak rare traits,
2. attacker strength affects privacy measurement,
3. exact identifiers usually cannot be meaningfully abstracted,
4. local model quality matters,
5. no formal privacy guarantee in first version.

### 9. Conclusion

Return to the main claim:

```text
privacy-preserving memory should store learned public abstractions, not merely masked raw text.
```

## 14. Strongest Version of the Claim

Avoid overclaiming:

Weak claim:

> PMA is another privacy filter.

Do not use this.

Strong claim:

> PMA learns a new cloud memory representation that improves personalization under private value reconstruction constraints.

This is the valuable version.

## 15. Risks and How to Handle Them

Risk 1: PMA leaks through coarse abstractions.

Mitigation:

```text
report leakage by privacy type and add attacker evaluation.
```

Risk 2: PMA utility gains are small.

Mitigation:

```text
focus on privacy types where abstraction is meaningful, such as address, health, itinerary, and preferences.
```

Risk 3: LLM judge noise.

Mitigation:

```text
start with MCQ tasks and exact reconstruction.
```

Risk 4: PMA simply learns to paraphrase raw sensitive text.

Mitigation:

```text
add leakage ranking loss and strict policy constraints for high-risk privacy types.
```

Risk 5: Reviewers ask why typed placeholders are not enough.

Mitigation:

```text
show retrieval and QA cases where typed placeholders remove necessary semantics.
```

## 16. Minimum Publishable Experiment

If time is limited, run only:

```text
Dataset: MemPrivacy-Bench
Memory system: Mem0
Question type: MCQ
Users: 100
Baselines: raw, complete, typed placeholder
Method: PMA-SFT
Privacy metric: exact reconstruction attack
Utility metric: MCQ accuracy
```

Minimum publishable result:

```text
PMA-SFT improves MCQ accuracy over typed placeholders while keeping exact reconstruction at or below typed-placeholder level.
```

If this result holds, the paper has a clear story.

## 17. Best Next Engineering Step

Implement the oracle first:

```text
raw memory + privacy info -> candidates -> utility/leakage scores -> selected abstraction
```

Do not train PMA until the oracle demonstrates that selected abstractions beat typed placeholders.

The oracle answers the most important feasibility question:

```text
Does a useful and safer abstraction exist for this task?
```

If yes, training PMA becomes justified.

## 18. Paper-to-Code Verification Contract

The paper and implementation should be reviewed together. Each major paper claim must have a matching code artifact and measurable evidence.

| ID | Paper Claim | Required Code Artifact | Required Evidence |
|---|---|---|---|
| C1 | PMA converts raw memories into public abstractions plus private residues. | `src/privacy_abstraction.py` | Candidate JSONL contains `public_memory`, `private_residue`, and `abstraction_trace`. |
| C2 | PMA supports multiple abstraction levels from semantic abstraction to redaction. | `PrivacyMemoryAbstractor.generate_candidates` | Each privacy-bearing turn has candidates for L1-L5 or documented fallback levels. |
| C3 | AMA measures exact private value reconstruction from cloud-visible memory. | `src/privacy_auditor.py` and `evaluation/prompts/ama_exact_reconstruct.txt` | Result JSON includes `exact_reconstruction_rate` and per-item attack records. |
| C4 | PUC selects the lowest-leakage candidate subject to utility preservation. | `src/privacy_critic.py` | Candidate score JSON shows chosen/rejected candidates and utility/leakage scores. |
| C5 | PMA training data is derived from existing privacy-annotated memory QA data. | `evaluation/build_pma_train_data.py` | SFT and preference JSONL examples link back to original dataset user/turn IDs. |
| C6 | PMA improves utility over typed placeholders at comparable leakage. | `evaluation/eval_pma_mem0.py` | Mem0 result JSON compares `type_specific` and `pma` on the same users/questions. |
| C7 | PMA is not claimed to work for all privacy types. | Policy config and per-type reporting | Result JSON reports utility/leakage by `privacy_type`. |

The implementation details for these artifacts are specified in `docs/pma_code_development_plan.md`.

The paper should not include a result unless it can be traced through:

```text
paper claim -> code path -> result JSON -> table or figure
```

## 19. Final Pitch

This paper proposes that privacy-preserving agent memory should be learned as a public abstraction problem. Instead of storing raw conversations with masked sensitive spans, a local trainable abstractor produces cloud-storable memories that preserve the semantics needed for personalization while withholding private residues on device. This directly addresses the utility loss of placeholder masking and provides a measurable privacy-utility tradeoff through downstream QA and reconstruction attacks.
