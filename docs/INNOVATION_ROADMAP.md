# Minimal Sufficient Public Memory Roadmap

## Status

The deterministic research baseline is now implemented:

- policy-aware `drop`, `local_only`, `public_abstract`, and
  `public_reversible` routing;
- least-specific feasible representation selection under utility, leakage, and
  token budgets;
- scoped alias rotation and local query-time hydration;
- cloud-safe default export, local-only debug metadata, revocation hooks,
  context minimization, memory-control and ephemeral-request filtering, and
  deterministic adversarial leakage auditing.

This is not yet the final learned method. The deterministic audit now covers
exact recovery, source identifiers, sensitive metadata, linkability markers,
membership markers, canary exposure, and prompt-injection strings, with per-user
scoping for partial artifacts. A lightweight held-out bag-of-words attribute
attacker is available as a cheap learned-attack baseline, but stronger neural or
LLM-based attackers are still required. The compiler also has a conservative
residual scrubber for known same-user sensitive tokens. Downstream memory QA,
utility recovery after conservative scrubbing, stronger held-out learned
attackers, and learned selector training remain required before making the full
paper claim described below.

## Problem Statement

Typed masking protects exact values but still exports every non-masked detail,
reveals privacy categories, preserves long-term linkability, and does not ask
whether a cloud memory item is necessary for future tasks.

The target problem is:

> Given a private interaction `X`, sensitive attributes `S`, a user policy `P`,
> and a distribution of future memory queries `Q`, construct the smallest
> public memory `Z` that preserves task utility while keeping measured leakage
> below a policy-specific budget.

An operational constrained objective is:

```text
minimize    task_loss(Z, Q) + beta * public_memory_cost(Z)
subject to  attack_success(Z -> S) <= epsilon(P)
            policy_violations(Z, P) = 0
```

This formulation should be evaluated with explicit attack models. Mutual
information should not be claimed unless it is estimated and validated.

## Proposed Architecture

### 1. Policy-Aware Memory Router

Route each candidate memory fact to one of four destinations:

- `drop`: never retain PL4 credentials or unnecessary facts;
- `local_only`: retain encrypted on the edge for local tools;
- `public_abstract`: export a task-sufficient abstraction;
- `public_reversible`: export a typed alias when exact local restoration is
  required.

The router must emit a reason, policy rule, confidence, and provenance record.

### 2. Public Memory Compiler

Convert dialogue into atomic candidate facts, then generate progressively less
specific alternatives:

```text
"blood pressure was 160/110"
-> "user has a high-risk blood-pressure condition"
-> "health constraints may affect recommendations"
-> drop
```

The compiler selects the least specific candidate that still supports expected
memory queries. This is the core minimal-sufficiency mechanism, not another
masking format.

### 3. Controlled Linkability

Replace globally persistent aliases with policy-controlled alias scopes:

- per turn for PL4;
- per session for high-risk PL3;
- per task or time window for PL2;
- persistent only when a user explicitly permits long-term linkage.

A local entity graph maintains continuity while the cloud receives rotated
aliases. Linkability becomes a measurable budget instead of an implicit side
effect.

### 4. Leakage Auditor

Train or evaluate an attacker ensemble against public memory:

- exact secret recovery;
- sensitive attribute inference;
- cross-session record linkage;
- membership inference;
- prompt-injection-assisted extraction;
- canary exposure.

The router or compiler should be optimized against the strongest held-out
attacker, not only against the same detector used to construct the memory.

### 5. Revocation And Provenance

Every exported memory item needs:

- source message IDs;
- policy version;
- alias scope and expiration;
- abstraction lineage;
- cloud memory IDs;
- deletion and re-compilation hooks.

This makes "forget this" enforceable across local mappings and cloud stores.

## Defensible Novelty

The closest positioning should be explicit:

| Work | Main mechanism | Missing axis addressed here |
|---|---|---|
| [MemPrivacy](https://arxiv.org/abs/2605.09530) | local extraction plus typed reversible placeholders | no minimality objective or attack-constrained memory selection |
| [GAMA](https://arxiv.org/abs/2509.25671) | generative-agent memory anonymization and local reversal | no policy-budgeted sufficiency or controlled linkability |
| [Agent-Memory Protocol](https://proceedings.mlr.press/v317/wu26a.html) | redact-pack-hydrate protocol and device-side PII | protocol isolation rather than learned minimal public memory |

The proposed contribution is not "we also anonymize agent memory." It is:

> A policy-constrained compiler that learns the minimum public representation
> required for long-term agent utility, with explicit resistance to recovery
> and linkage attacks.

## Planned Code Interfaces

```yaml
public_memory:
  mode: minimal_sufficient
  policy:
    pl2: public_abstract
    pl3: local_only
    pl4: drop
  utility_floor: 0.98
  leakage_budget:
    exact_recovery: 0.01
    attribute_attack_auc: 0.55
    linkage_attack_auc: 0.55
  alias_scope:
    pl2: task
    pl3: session
    pl4: turn
  selector:
    max_public_tokens: 128
    candidate_levels: 4
```

Planned modules:

- `src/policy.py`
- `src/public_memory_compiler.py`
- `src/sufficiency_selector.py`
- `src/alias_router.py`
- `src/leakage_auditor.py`
- `src/provenance.py`

All new modules must preserve the current typed-pseudonymization path as a
baseline and ablation.

The deterministic versions of these modules are implemented. The
typed-pseudonymization path remains available as the legacy baseline.

## Required Experiments

### Utility

- MemPrivacy-Bench and PersonaMem-v2 memory QA. The current proxy evaluator
  resolves MCQ answer labels against `all_options` before computing
  non-private answer-token recall.
- Mem0, LangMem, and Memobase.
- Basic, temporal, dynamic-update, aggregation, implicit-inference, and
  adversarial questions reported separately.

### Privacy

- Exact value recovery from public memory.
- Privacy-type and attribute inference.
- Cross-session linkage under alias rotation.
- Membership inference and canary extraction.
- Prompt injection against stored memory.

### Minimality

- Public tokens and memory records per user.
- Redundant-fact rate.
- Utility as a function of public-memory budget.
- Pareto frontier of utility, leakage, and memory size.

### Generalization

- Chinese and English separately.
- PersonaMem-v2 out-of-distribution evaluation.
- Unseen privacy types and unseen memory systems.

### Ablations

- no policy router;
- no abstraction candidates;
- no leakage auditor;
- persistent versus rotated aliases;
- no provenance/revocation;
- typed masking baseline versus minimal-sufficient compiler.

## Promotion Gates

Do not claim the new method is successful unless it meets all of these on held
out users:

- memory QA accuracy is within 1 absolute point of raw-memory utility;
- exact sensitive-value recovery is at most 1%;
- attribute and linkage attack AUC are at most 0.55;
- exported memory tokens fall by at least 30% versus persistent typed masking;
- PL4 values have zero cloud retention in storage and logs;
- gains hold on both MemPrivacy-Bench and PersonaMem-v2.

If utility is preserved only under oracle annotations, report the result as an
upper bound rather than as an end-to-end system result.

## Immediate Implementation Order

1. Add policy and provenance schemas without changing model behavior.
2. Implement deterministic rule-based routing as a testable baseline.
3. Add rotating aliases and linkage-attack evaluation.
4. Add abstraction candidates and a utility-constrained selector.
5. Add attacker-aware training only after deterministic baselines are stable.

Steps 1-4 are implemented, and the deterministic adversarial audit now covers
most roadmap attack surfaces as rule-based checks. A budget-sweep tool now
produces utility/leakage/minimality summaries across `max_public_tokens` values
for Pareto-frontier analysis. A follow-up selector turns those summaries into a
deployable budget recommendation by applying privacy, cloud-safety,
adversarial, utility, and local-recoverability gates. A lightweight held-out
Naive Bayes attribute attacker is implemented as an initial learned-attack
baseline. The current safety-first residual scrubber can reduce utility on some
partial PersonaMem-v2 runs, so stronger held-out attackers, learned
utility-aware selection, and end-to-end memory-system utility experiments are
pending. See [Preliminary results](PRELIMINARY_RESULTS.md).
