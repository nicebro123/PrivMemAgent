# Trainable Privacy Memory Abstractor

## 1. Module Name

**Privacy Memory Abstractor, PMA**

PMA is a trainable local module that converts raw dialogue memory into a cloud-storable abstract memory while keeping sensitive private residues local.

It is designed to replace this current pipeline:

```text
raw dialogue -> privacy span detection -> typed placeholder masking -> cloud memory
```

with:

```text
raw dialogue -> privacy-aware memory abstraction -> public memory + private residue
```

The module should run before memory insertion.

## 2. Concrete Problem It Solves

Typed placeholders protect raw values, but they often damage memory utility.

Example:

```text
Raw:
The user lives at 15 Rue de Rivoli in Paris and wants a quiet cultural activity nearby.

Typed placeholder:
The user lives at <Detailed_Address_1> and wants a quiet cultural activity nearby.

Problem:
The memory system loses location semantics. It cannot infer "central Paris", "near museums",
"low commute", or "indoor cultural recommendations".
```

PMA solves this by producing:

```text
Public memory:
The user is based in central Paris and prefers quiet, low-commute indoor cultural activities.

Private residue:
15 Rue de Rivoli, Paris -> Detailed Address, local only
```

This solves a real gap:

> How can a cloud memory system preserve personalization utility without storing recoverable private values?

## 3. Input and Output

### Input

```json
{
  "dialogue": "I live at 15 Rue de Rivoli and my pollen allergy is acting up. I want a quiet indoor exhibition nearby.",
  "privacy_items": [
    {
      "original_text": "15 Rue de Rivoli",
      "privacy_type": "Detailed Address",
      "privacy_level": "PL2"
    },
    {
      "original_text": "pollen allergy",
      "privacy_type": "Medical Health",
      "privacy_level": "PL3"
    }
  ],
  "task_family": "recommendation",
  "policy": {
    "cloud_memory_allowed_levels": ["L2", "L3", "L4"],
    "block_levels": ["PL4"]
  }
}
```

### Output

```json
{
  "public_memory": "The user is based in central Paris, prefers low-commute quiet indoor cultural activities, and benefits from low-allergen environments.",
  "private_residue": [
    {
      "raw": "15 Rue de Rivoli",
      "privacy_type": "Detailed Address",
      "privacy_level": "PL2",
      "retention": "local_only"
    },
    {
      "raw": "pollen allergy",
      "privacy_type": "Medical Health",
      "privacy_level": "PL3",
      "retention": "local_only"
    }
  ],
  "abstraction_trace": [
    {
      "raw": "15 Rue de Rivoli",
      "public_abstraction": "central Paris",
      "reason": "preserves location utility while removing exact address"
    },
    {
      "raw": "pollen allergy",
      "public_abstraction": "low-allergen environments",
      "reason": "preserves recommendation constraint while removing condition name"
    }
  ]
}
```

## 4. Model Formulation

PMA is a conditional sequence-to-structure model:

```text
PMA_theta(x, p, c, pi) -> (z, r, t)
```

where:

- `x`: raw dialogue or memory chunk,
- `p`: privacy annotations,
- `c`: task family,
- `pi`: privacy policy,
- `z`: public abstract memory,
- `r`: private residue,
- `t`: abstraction trace.

The trainable module can be implemented as a small instruction-tuned language model, for example Qwen/Llama-class local models, trained to produce strict JSON.

## 5. Training Objective

PMA should not be trained only with supervised text imitation. It needs a utility-leakage objective.

### 5.1 Supervised Structure Loss

Train on generated target JSON:

```text
L_sft = - log P_theta(target_json | x, p, c, pi)
```

This teaches the model to produce valid schema and reasonable abstractions.

### 5.2 Utility Preservation Loss

Given downstream question `q`, reference answer `a`, and public memory `z`:

```text
answer_z = AnswerModel(q, z)
utility_reward = Judge(answer_z, a)
```

Use this as a ranking signal. If candidate `z_good` preserves utility better than `z_bad`:

```text
L_utility_rank = - log sigmoid(score_theta(z_good) - score_theta(z_bad))
```

In practice, this can be implemented with preference tuning after SFT.

### 5.3 Privacy Leakage Loss

Train an attacker model or evaluator:

```text
attack_z = Attacker(z, privacy_type, auxiliary_context)
leakage = Match(attack_z, raw_private_value)
```

The abstractor should prefer candidates with lower leakage:

```text
L_privacy_rank = - log sigmoid(score_theta(z_low_leak) - score_theta(z_high_leak))
```

### 5.4 Combined Objective

```text
L = L_sft + lambda_u * L_utility_rank + lambda_p * L_privacy_rank + lambda_schema * L_schema
```

The important design is not just the loss formula. The important part is creating paired candidates:

```text
same raw memory, different abstraction levels, measured utility and leakage
```

That gives the module a learnable notion of "minimal sufficient abstraction".

## 6. Training Data Construction

The existing repository already contains enough structure for bootstrapping:

```text
dialogues + privacy_info + questions + answers
```

### Step 1: Generate Candidate Abstractions

For each privacy-bearing memory chunk, generate multiple candidates:

```text
Z0: raw or near-raw
Z1: category abstraction
Z2: functional implication
Z3: task preference or constraint
Z4: typed placeholder
Z5: full redaction
```

### Step 2: Score Utility

For each candidate `Zi`, insert it into memory and evaluate questions:

```text
utility(Zi) = QA accuracy or answer consistency
```

For the first version, start with MCQ because it gives objective labels.

### Step 3: Score Leakage

Run reconstruction attacks:

```text
Attacker(Zi) -> guessed private value
```

Compute:

```text
exact_reconstruction_success
attribute_inference_success
semantic_leakage_score
```

### Step 4: Select Training Target

Choose the best candidate:

```text
Z* = argmin leakage(Zi)
subject to utility(Zi) >= tau
```

If no candidate passes utility threshold, choose typed placeholder and store necessary exact value locally.

### Step 5: Build Preference Pairs

Create pairs:

```text
positive: Z*
negative: candidates with higher leakage at similar utility
negative: candidates with lower utility at similar leakage
```

These pairs are more valuable than single SFT labels.

## 7. Architecture

### 7.1 Minimal Version

One local instruction model:

```text
PMA_theta(input_json) -> output_json
```

Pros:

- easiest to implement,
- directly compatible with current code,
- can be trained with SFT and preference tuning.

Cons:

- hard to guarantee monotonic privacy behavior,
- may hallucinate abstractions not supported by the original memory.

### 7.2 Better Version

Three submodules:

```text
1. Abstraction Generator G_theta
2. Utility Predictor U_phi
3. Leakage Predictor L_psi
```

Flow:

```text
G_theta generates candidates
U_phi estimates downstream utility
L_psi estimates privacy leakage
selector chooses candidate
```

Pros:

- easier to debug,
- supports explicit privacy-utility frontier,
- can improve selection without retraining generator.

Cons:

- more engineering,
- needs labeled utility and leakage data.

Recommended first trainable version:

```text
G_theta + rule-based selector using measured utility/leakage labels
```

Then train `U_phi` and `L_psi` later.

## 8. Why This Is a Real Trainable Module

PMA is trainable because it has:

1. supervised JSON targets,
2. preference pairs from utility-leakage comparisons,
3. measurable downstream rewards,
4. explicit negative examples,
5. a stable deployment interface.

It is not just prompt engineering. Prompting is only used to bootstrap candidate data.

After training, PMA can run locally and produce abstractions without calling a large cloud model.

## 9. Integration with Current Code

Add:

```text
src/privacy_abstraction.py
src/privacy_abstraction_config.yaml
evaluation/eval_abstraction_mem0.py
evaluation/attack.py
evaluation/build_abstraction_dataset.py
evaluation/train_pma.py
```

Core interface:

```python
class PrivacyMemoryAbstractor:
    def abstract(
        self,
        dialogue_text: str,
        privacy_items: list[dict],
        task_family: str,
        policy: dict,
    ) -> dict:
        ...
```

Expected return:

```python
{
    "public_memory": str,
    "private_residue": list[dict],
    "abstraction_trace": list[dict],
}
```

Memory integration:

```python
if mask_mode == "abstraction":
    result = abstractor.abstract(
        dialogue_text=user_content,
        privacy_items=user_msg[privacy_info_key],
        task_family="recommendation",
        policy=abstraction_policy,
    )
    user_content = result["public_memory"]
    abstraction_store.save_private_residue(user_id, result["private_residue"])
```

## 10. First Experiment

Keep the first experiment narrow.

```text
Dataset: memprivacy_bench_testset.jsonl
Memory system: Mem0 only
Question type: MCQ only
Users: 50 to 100
Mask levels: PL2 + PL3 + PL4
Compared methods:
  raw
  complete
  generic
  type_specific
  PMA abstraction
```

Main metric:

```text
utility at fixed exact-reconstruction leakage
```

The experiment should answer:

> Can PMA retain more MCQ accuracy than type-specific placeholders without increasing exact private value reconstruction?

If yes, the module solves a concrete problem.

## 11. Expected Effective Privacy Types

PMA should help most for privacy types where exact values are not needed but functional meaning matters:

```text
Detailed Address
Medical Health
Itinerary/Trajectory
Relationship Info
Political Views/Stance
Religious Beliefs
Job Intent/Status
Accommodation Record
Financial behavior descriptions
```

PMA should not try to semantically abstract these into cloud memory:

```text
Verification Code
Password
API Key
Recovery Code
Full account number
Government ID
Exact phone number
Exact email address
```

For those, the correct action is usually:

```text
block from long-term memory or replace with coarse type only
```

This distinction is important. It makes the module practical instead of pretending all privacy types can be abstracted.

## 12. Feasibility Assessment

### Innovation

High.

The module is not a better PII detector. It is a trainable representation transformer for privacy-preserving agent memory. The central novelty is learning which semantics to keep for future personalization and which private values to remove.

### Implementation Difficulty

Medium-high.

A prompt-generated prototype is medium difficulty. A truly trained PMA with preference tuning is high difficulty, but feasible because the repository already has:

1. privacy annotations,
2. memory QA questions,
3. multiple memory-system evaluation scripts,
4. baseline masking modes.

### Main Risk

The hardest part is leakage evaluation. If the attacker is weak, the model may look safer than it really is.

Mitigation:

1. use exact reconstruction first,
2. then add attribute inference,
3. then add multi-memory linkage attacks,
4. report results by privacy type.

## 13. Minimum Viable Implementation

The minimum useful implementation is:

1. generate abstraction candidates with a strong LLM,
2. evaluate each candidate with Mem0 MCQ accuracy,
3. evaluate exact reconstruction attack,
4. select the best candidate under utility threshold,
5. train PMA on selected outputs,
6. compare trained PMA against type-specific placeholder.

This already gives a complete paper loop:

```text
problem -> module -> training data -> training objective -> evaluation -> baseline comparison
```

## 14. Success Criterion

PMA is worth continuing only if it satisfies:

```text
PMA_utility > type_specific_utility
and
PMA_exact_reconstruction <= type_specific_exact_reconstruction
```

If exact reconstruction is equal but PMA utility is higher, that is already a strong result.

If PMA utility is higher but leakage also higher, report the privacy-utility frontier. The method may still be useful under tunable policies.

## 15. One-Sentence Paper Claim

> We introduce a trainable privacy memory abstractor that learns to transform raw user memories into minimal task-sufficient public abstractions, improving personalization utility over typed placeholders while preserving protection against private value reconstruction.
