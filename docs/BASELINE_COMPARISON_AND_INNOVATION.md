# Innovation Positioning Against MemTensor/MemPrivacy

## One-Sentence Positioning

PrivMemAgent does not claim novelty from training another privacy-span detector.
It builds on MemPrivacy-style local privacy extraction and typed
pseudonymization, then moves the problem boundary from **span-level masking** to
**policy-constrained minimal public memory construction**.

In short:

```text
MemPrivacy: detect sensitive spans and replace them with reversible typed placeholders.
PrivMemAgent: decide what should become cloud memory at all, and export only the
least-specific task-sufficient public representation under explicit privacy,
utility, linkability, and token budgets.
```

## What Belongs To The MemPrivacy Baseline

The following capabilities should be treated as baseline or upstream assets, not
as PrivMemAgent contributions:

- PL1-PL4 privacy taxonomy and privacy-span annotation format;
- MemPrivacy SFT/RL detector checkpoints, including MemPrivacy-4B-RL;
- local extraction of sensitive spans from dialogue turns;
- typed placeholder or pseudonymized replacement of sensitive values;
- local reversible mapping from placeholder to original value;
- evaluation on MemPrivacy-Bench and PersonaMem-v2;
- memory-system benchmarking against Mem0, LangMem, and Memobase.

PrivMemAgent can use MemPrivacy-4B-RL as the privacy extractor, but the paper
claim must not imply that this repository retrains or owns that detector unless
a separate training pipeline and data release are added.

## Core Difference

MemPrivacy mainly answers:

```text
Which spans are private, and how can they be replaced before cloud calls?
```

PrivMemAgent answers:

```text
Which pieces of interaction history are worth retaining as long-term public
memory, at what abstraction level, under which user policy, and with what
measured leakage risk?
```

This difference changes the protected object:

| Dimension | MemPrivacy baseline | PrivMemAgent |
|---|---|---|
| Protected object | private span in a message | long-term memory item exported to cloud memory |
| Main transformation | typed reversible pseudonymization | drop / local-only / public abstraction / scoped reversible alias |
| Cloud artifact | mostly original context with placeholders | minimized public memory records and cloud-safe benchmark |
| Privacy risk addressed | exact value exposure | exact value exposure, sensitive attribute leakage, linkability, membership leakage, canary exposure, prompt-injection strings |
| Utility mechanism | preserve semantic type with placeholder | preserve only task-sufficient memory content |
| Linkability | placeholder may be stable unless separately scoped | alias scope is explicit: turn, session, task, or persistent |
| Optimization target | detector quality and masking correctness | utility-leakage-token Pareto frontier for public memory |

## PrivMemAgent Innovation Modules

### 1. Policy-Aware Memory Router

Code: `src/policy.py`

The router maps each privacy-bearing candidate fact to one of four actions:

```text
drop
local_only
public_abstract
public_reversible
```

The default policy is:

```text
PL2 -> public_abstract
PL3 -> local_only
PL4 -> drop
```

The router also handles task-specific requirements through
`exact_required_types` and `consented_reversible_types`, so exact cloud-visible
aliases are allowed only when exact future utility is needed and user consent is
available.

### 2. Minimal Public Memory Compiler

Code: `src/public_memory_compiler.py`

The compiler converts privacy-aware dialogue content into public memory. The
important conceptual shift is that a sensitive fact is not automatically kept in
typed placeholder form. Instead, the compiler creates public representations at
multiple specificity levels:

```text
exact sensitive fact
-> typed abstraction
-> category abstraction
-> generic abstraction
-> local-only or drop
```

The compiler chooses the least-specific representation that remains useful for
future memory tasks. This is the main method-level contribution.

### 3. Sufficiency Selector

Code: `src/sufficiency_selector.py`

The current selector is deterministic. It selects a candidate only if it passes
hard budgets:

```text
utility_score >= utility_floor
leakage_score <= max_leakage
token_count <= max_public_tokens
```

Among feasible candidates, it chooses the least-specific one. This encodes the
minimal-sufficiency objective, but it is currently a rule baseline. The learned
utility-leakage selector described in the development plan should replace or
augment this module.

### 4. Controlled Linkability Through Scoped Aliases

Code: `src/alias_router.py`

Instead of treating all placeholders as globally stable identities, PrivMemAgent
makes alias scope explicit:

```text
turn
session
task
persistent
```

This allows cross-session linkability to become a controlled budget rather than
an accidental side effect of pseudonymization.

### 5. Context Minimization Before Memory Export

Code: `src/context_minimizer.py`

The context minimizer removes or downranks content that is not memory-worthy,
including ephemeral requests, generic assistant boilerplate, and memory-control
commands. This prevents the system from treating every dialogue turn as useful
long-term memory.

### 6. Adversarial Public-Memory Audit

Code: `tools/adversarial_audit.py`

The audit evaluates exported artifacts for:

- exact secret exposure;
- PL4 exposure;
- user identifier exposure;
- sensitive attribute leakage;
- linkage metadata leakage;
- membership-marker exposure;
- canary exposure;
- prompt-injection strings.

Audit failures are not merely runtime errors. They are experimental signals
that identify where the public-memory construction leaks too much information.

## Required Learned Modules For The Final Method

The current implementation is a deterministic research baseline. To make the
full method claim strong enough, two trainable modules are required.

### Learned Abstraction Generator

Purpose:

```text
Generate multiple safe public-memory candidates at different abstraction levels.
```

Input should include:

- dialogue message or extracted memory fact;
- detected privacy spans from MemPrivacy-4B-RL;
- privacy type and privacy level;
- local dialogue context;
- future question or memory-need category when available;
- policy constraints.

Output should be a candidate set such as:

```text
drop
private detail
health constraint relevant to future assistance
daily medical routine may affect scheduling recommendations
scoped reversible alias, only when allowed
```

The generator must be trained to avoid exact sensitive values while preserving
utility-relevant semantics.

### Learned Utility-Leakage Selector

Purpose:

```text
Select the candidate or candidate set that optimizes utility under leakage,
linkability, and memory-size budgets.
```

Input should include:

- abstraction candidates;
- candidate token count;
- privacy level and privacy type;
- query relevance features;
- BGE embedding similarities;
- audit-risk features;
- downstream QA utility signals;
- memory-system feedback when available.

Output can be:

```text
select / reject / local_only / drop
```

or a scalar score:

```text
utility - lambda * leakage - beta * token_cost - gamma * linkability
```

This selector is the learned replacement for the current rule-based
`SufficiencySelector`.

## Defensible Paper Claim

A defensible final claim is:

> PrivMemAgent learns to compile private interaction histories into minimal
> sufficient public memories. Unlike span-level pseudonymization, it optimizes
> the exported memory representation itself under explicit utility, leakage,
> linkability, and token budgets.

A weaker but accurate current claim is:

> PrivMemAgent implements a deterministic policy-constrained public-memory
> compiler on top of MemPrivacy-4B-RL privacy extraction, providing a testbed for
> minimal-sufficient memory construction and adversarial leakage auditing.

## Claims To Avoid Until The Learned Modules Exist

Do not claim yet that:

- PrivMemAgent trains a new privacy detector;
- the current abstraction mechanism is learned;
- the current selector is learned or attacker-optimized;
- the method estimates mutual information;
- audit-passing is guaranteed across unseen memory systems;
- utility is preserved on full memory-system QA before Mem0, LangMem, and
  Memobase evaluations finish.

## Evaluation Needed To Prove Innovation

The final comparison against MemPrivacy should report:

1. **Utility**: memory QA accuracy on MemPrivacy-Bench and PersonaMem-v2.
2. **Privacy**: exact leakage, PL4 leakage, attribute attack, linkage attack,
   membership leakage, and prompt-injection leakage.
3. **Minimality**: public tokens per user, number of exported memory records,
   and token reduction against typed masking.
4. **Pareto frontier**: utility versus leakage versus public-memory size.
5. **Ablations**:
   - MemPrivacy typed masking baseline;
   - rule-based compiler;
   - learned abstraction only;
   - learned selector only;
   - learned abstraction plus learned selector;
   - no scoped aliases;
   - no adversarial audit feedback.

The strongest result would show that PrivMemAgent retains comparable memory QA
utility while exporting fewer tokens and leaking less than typed
pseudonymization.
