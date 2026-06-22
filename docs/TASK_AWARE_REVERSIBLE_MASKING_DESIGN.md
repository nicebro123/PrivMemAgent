# Task-Aware Reversible Masking For Edge-Cloud Memory Collaboration

## Purpose

This document refines the PrivMemAgent direction after re-evaluating the MemPrivacy-style baseline. The central research angle is **edge-cloud collaboration from the memory perspective**: private memory should live on the edge, while the cloud should receive only task-safe memory representations or constraints.

The goal is not to replace MemPrivacy's reversible masking idea. The goal is to preserve its strongest property, namely that exact private values stay local and can be restored when necessary, while adding a task-aware memory layer that makes the cloud-visible representation more useful and less attackable.

The system should be framed as a memory architecture rather than a generic masking wrapper. The local side owns complete user memory and exact reversible mappings. The cloud side owns scalable reasoning and generation, but never needs raw private memory by default.

```text
MemPrivacy baseline:
private value -> typed reversible mask -> cloud memory -> local restoration

PrivMemAgent revised direction:
raw interaction -> edge private memory vault
                -> task-aware memory compiler
                -> utility-leakage selection
                -> cloud-safe public memory view
                -> cloud LLM reasoning
                -> local restoration or privacy-preserving final rendering
```

In short, we keep bidirectional masking, but make the mask interface adaptive to the downstream task and explicit about leakage risk. The paper should emphasize that this is a **cloud-edge memory collaboration problem**: the edge keeps the private memory, compiles it into task-safe public memory, and asks the cloud to reason only over that compiled view.

## Core Edge-Cloud Memory Thesis

PrivMemAgent should be presented around the following thesis:

```text
A personalized agent does not need to expose complete user memory to the cloud.
It needs an edge memory manager that converts private memory into the minimum
cloud-safe memory view required for the current task.
```

This gives the method a clearer identity than ordinary anonymization. The unit of protection is not only a private span inside one message. The protected object is the user's long-term memory state.

The memory state is split into two coordinated stores:

| Store | Location | Content | Access |
|---|---|---|---|
| Private memory vault | Edge / phone / local device | raw PL2/PL3/PL4 facts, reversible masks, provenance, consent, revocation metadata | local small model and local tools |
| Public memory view | Cloud-safe memory layer | task constraints, low-risk abstractions, scoped aliases only when needed | cloud LLM and cloud memory system |

This split should be the main architectural contribution. The cloud is still useful because it receives a task-sufficient memory view. The edge is essential because it owns the complete private memory and decides what the cloud is allowed to know.

## Why Keep Reversible Masking

Pure abstraction is attractive for privacy, but it throws away exact values. In real personal agents, exact values are sometimes necessary:

- filling an address or email field;
- drafting a message with a known recipient;
- checking whether a provided account suffix matches a local record;
- restoring a user-approved detail in the final answer;
- supporting deletion, revocation, and provenance at the original-value level.

Reversible masking gives us a local anchor for these operations. Therefore, the new method should treat reversible masks as a local memory capability, not as a weak baseline to discard.

## Main Limitation Of Plain Typed Masks

Plain typed masks usually expose the privacy category:

```text
"160/110" -> <Health_Info_1>
"recovery code RC-7291" -> <Recovery_Code_1>
"Loire Valley villa" -> <Location_1>
```

This hides the exact value, but it can still leak sensitive attributes, membership signals, and linkable structure. For example, a cloud model that sees many records containing `<Health_Info_1>`, `<Medication_1>`, and `<Hospital_1>` can infer that the user has health-related private memory even without seeing the original text.

The revised design therefore separates two ideas:

1. The local reversible binding from mask to original value.
2. The cloud-visible expression of what the downstream task actually needs.

## Design Principle

The central principle is:

```text
Expose task constraints, not identity anchors.
Restore exact values only locally, only when the task requires them.
```

An original user fact is decomposed into two parts:

| Component | Meaning | Example |
|---|---|---|
| Identity anchor | The value that identifies or strongly reveals the user | exact address, child name, account number, specific diagnosis |
| Task constraint | The reason the fact matters for future assistance | nearby recommendations, child-friendly content, low-risk financial wording, avoiding dust |

The local model may keep both components. The cloud memory should normally see only the task constraint.

## Revised Architecture

```text
Raw dialogue and historical interactions
  -> edge private memory vault
  -> local privacy detector and memory updater
  -> task-aware memory compiler
  -> task-aware mask/abstraction generator
  -> utility-leakage selector
  -> cloud-safe public memory view
  -> cloud memory retrieval and cloud LLM reasoning
  -> local output auditor
  -> optional local restoration or abstract final rendering
```

### Edge Private Memory Vault

The vault stores exact private values, reversible masks, provenance, scope, expiration, and revocation metadata. This is the first-class memory store in the system, not a temporary masking cache. It is the only layer allowed to access raw PL2/PL3/PL4 values by default.

Example local bindings:

```text
<LOCAL_HEALTH_7>  <->  "mild intermittent asthma"
<LOCAL_ADDR_2>    <->  "1824 Belmont Ave Apt 3B, Seattle"
<LOCAL_CHILD_3>   <->  "Nadia"
```

These local masks do not have to be shown to the cloud directly.

### Task-Aware Memory Compiler And Masked Abstraction Generator

This module receives the private item, privacy type, privacy level, local memory context, and optional downstream task hints. It produces several candidate cloud-safe memory representations. Its job is not merely to hide a span; its job is to compile local private memory into a public memory view that the cloud can use.

Example:

```text
Original: "mild intermittent asthma", Medical Health, PL3
Task: decide whether attic cleaning is suitable

Candidates:
0. drop
1. private health detail
2. health constraint relevant to physical-environment recommendations
3. user should avoid dusty or airway-irritating environments
4. <LOCAL_HEALTH_7>  (only if exact local restoration is consented and needed)
```

The important shift is that the best candidate is often not the typed mask itself. It is a task-relevant constraint derived from the private value.

### Utility-Leakage Selector

The selector chooses the safest candidate that still supports personalization. It should reject candidates that are too specific, even if they do not copy the exact private value.

For the private fact:

```text
"Loire Valley villa next month"
```

Possible candidates are:

| Candidate | Utility | Leakage | Decision |
|---|---:|---:|---|
| user will go to Loire Valley next month | high | high | reject |
| user has a travel plan | low | low | maybe too vague |
| user prefers low-stress, low-travel-burden leisure plans | high | low | select |

This turns masking from a fixed transformation into a policy-constrained selection problem.

### Local Restoration And Final Rendering

The final response does not always need to restore the original value.

For example, if the cloud says:

```text
Given the user's health constraint, avoid dusty spaces.
```

the local renderer can return that sentence as-is. It does not need to restore:

```text
Given the user's asthma, avoid dusty spaces.
```

Exact restoration should be reserved for cases where the user explicitly asks for exact-value use, such as filling a form or verifying a known record.

## Adaptive Granularity

The same private value can support multiple cloud-visible granularities.

Address example:

| Level | Representation | Typical use |
|---|---|---|
| 0 | drop | no location need |
| 1 | user needs nearby options | generic recommendation |
| 2 | Seattle-area options | city-level planning |
| 3 | options in the user's neighborhood | local search without exact address |
| 4 | exact address | local-only or consent-gated one-shot task |

Health example:

| Level | Representation | Typical use |
|---|---|---|
| 0 | drop | irrelevant task |
| 1 | private health constraint | weak personalization |
| 2 | avoid physically demanding or irritating environments | recommendation |
| 3 | avoid dusty/airway-irritating environments | task-specific advice |
| 4 | exact diagnosis/medication | local-only or explicit consent |

This is a better framing than simply saying that the data is blurred. The local agent chooses the minimum granularity that is sufficient for the task.

## Scope And Linkability

Masks should not be globally stable by default. The system should support:

- turn-scoped masks for ephemeral reasoning;
- session-scoped masks for short conversations;
- task-scoped masks for multi-step workflows;
- persistent masks only when long-term linkage is necessary and allowed.

Cloud-visible public memory should prefer abstract constraints over persistent typed masks. Persistent masks are useful for exact local restoration, but they increase cross-session linkability if exposed to cloud memory.

## Big-Small Model Collaboration Over Memory

The design assumes an edge-cloud setup where the small model is the private memory manager and the large model is the cloud reasoning engine:

| Component | Access | Responsibility |
|---|---|---|
| Local small model | full private memory | privacy detection, task decomposition, abstraction, restoration, output audit |
| Cloud large model | cloud-safe constraints only | high-quality reasoning, generation, broad knowledge use |

The local model owns private precision and memory governance. The cloud model owns generation strength. This division should be described as memory-centric cloud-edge collaboration, not merely model compression or privacy filtering.

This resolves the question: if the cloud sees only abstracted information, how can it still satisfy precise user needs? The answer is that the local model translates exact private facts into precise task constraints before the cloud is called.

## Training Targets

The revised method still needs two learned modules.

### Learned Task-Aware Masked Abstraction Generator

Input:

```text
message text
privacy item
privacy type and level
task hint or future query hint
policy configuration
```

Output:

```text
a ranked set of cloud-safe candidates:
drop, generic abstraction, category abstraction, task constraint, scoped mask
```

Training supervision can come from:

- deterministic rule outputs;
- DeepSeek-generated candidates filtered by exact-leak checks;
- downstream QA utility labels;
- adversarial audit labels;
- human or LLM preference pairs where safe useful candidates beat unsafe or overly vague candidates.

### Learned Utility-Leakage Selector

Input:

```text
candidate text
candidate level
privacy type and level
task relevance features
exact-leak and fragment-leak flags
semantic leakage score
token cost
scope/linkability features
```

Output:

```text
select candidate or drop
```

The selector should be trained with hard negatives:

- exact copies of private values;
- partial fragments such as email prefixes, account suffixes, street names;
- typed masks that reveal overly sensitive categories;
- semantically specific paraphrases that preserve the private fact;
- candidates that are safe but useless.

## Attack Surface To Audit

The revised design should audit both raw-value leakage and mask-level leakage.

Required checks:

- exact secret recovery;
- partial fragment leakage;
- sensitive attribute inference from mask type or surrounding context;
- cross-session linkability from repeated aliases;
- membership inference from public memory records;
- prompt-injection attempts to reveal local memory;
- final-rendering leakage after local restoration;
- multilingual leakage, especially Chinese phrase-level and short-token leaks.

This is important because a system can pass exact-value masking while still failing semantic privacy.

## Relationship To Current Code

The current repository already contains several matching components:

| Design concept | Current module |
|---|---|
| privacy detection interface | `src/privacy_masking.py` |
| local reversible aliases | `src/alias_router.py`, `PrivacyStore` |
| policy routing | `src/policy.py` |
| public memory compilation | `src/public_memory_compiler.py` |
| deterministic abstraction | `RuleBasedAbstractor`, `src/abstraction_generator.py` |
| deterministic/learned selector interface | `src/sufficiency_selector.py`, `src/utility_leakage_selector.py` |
| provenance and revocation | `src/provenance.py` |
| adversarial audit | `tools/adversarial_audit.py` |

The main missing piece is not the skeleton. The missing piece is to train and evaluate the learned generator/selector so that the system chooses task-aware representations rather than relying on rule templates.

## Suggested Paper Positioning

The final positioning should be memory-centric and respectful to MemPrivacy:

```text
MemPrivacy shows that reversible typed masking can protect exact private values while preserving memory utility. We build on this insight and study the next question: what should the cloud-visible memory representation be when the typed mask itself leaks sensitive attributes or unnecessary structure?
```

One concise contribution statement:

```text
We propose task-aware reversible masking for edge-cloud memory collaboration,
a memory interface that keeps complete private memory in an edge reversible
vault while exposing to the cloud only the minimum task-sufficient public memory
view selected under explicit utility-leakage budgets.
```

## Implementation Plan

1. Preserve the existing typed reversible masking baseline as an ablation.
2. Add candidate levels for task-aware masked abstractions.
3. Build training data from MemPrivacy-Bench, PersonaMem-v2, deterministic compiler outputs, DeepSeek candidate generation, QA utility, and adversarial audit labels.
4. Train or distill the abstraction generator.
5. Train the utility-leakage selector with hard negative leakage examples.
6. Re-run full experiments with adversarial audit as a hard gate.
7. Report a Pareto frontier over utility, leakage, public-token cost, and linkability.

## Expected Innovation Points

This design yields five innovation claims without changing the original task definition. The first and most important claim should be memory-centric edge-cloud collaboration:

1. Memory-centric edge-cloud collaboration where private memory is stored and governed locally, while the cloud receives only a compiled public memory view.
2. Task-aware reversible masking instead of fixed typed masking.
3. Adaptive granularity disclosure instead of one-size-fits-all placeholders.
4. Learned utility-leakage selection over candidate memory representations.
5. Big-small model collaboration where the local model owns private precision and the cloud model only receives safe task constraints.
