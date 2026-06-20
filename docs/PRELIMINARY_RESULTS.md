# Minimal Public Memory Preliminary Results

## Scope

These results validate privacy invariants and public-memory reduction for the
deterministic compiler. They do not establish downstream memory QA utility.
Both runs use oracle privacy annotations and therefore represent an extraction
upper bound.

## Full Released Datasets

| Dataset | Records | Sensitive items | Exact recovery | Cross-scope linkage | PL4 public retention | Public token reduction |
|---|---:|---:|---:|---:|---:|---:|
| MemPrivacy-Bench | 3,310 | 15,512 | 0.00% | N/A | 0.00% | 30.14% |
| PersonaMem-v2 | 2,436 | 1,528 | 0.00% | N/A | 0.00% | 70.90% |

MemPrivacy-Bench contains 10 invalid span annotations. They are reported
separately and excluded from routing and audit denominators.

Sensitive-item counts are protected occurrences after propagating each user's
valid known values across all of that user's messages. This prevents repeated
values in unannotated messages from escaping protection.

Cross-scope linkage is not applicable in these default-policy runs because no
reversible aliases are exported. It must be evaluated in the consented
`public_reversible` ablation.

Token reduction is corpus-weighted: `1 - sum(public tokens) / sum(source
tokens)`. The corresponding unweighted per-message averages are 29.71% and
45.87%; they are diagnostic only and are not used for the promotion gate.

MemPrivacy-Bench routing:

- `drop`: 2,135
- `local_only`: 6,133
- `public_abstract`: 4,399

PersonaMem-v2 routing:

- `drop`: 12
- `local_only`: 304
- `public_abstract`: 1,084

## Utility Proxy

The proxy checks only whether exact private values referenced verbatim by an
answer or evidence remain recoverable from the encrypted local store, and how
many non-private answer tokens occur in public memory.

MemPrivacy-Bench:

- 149 questions;
- 90 questions contain exact annotated private values in answer/evidence;
- all 270 policy-eligible exact references are recoverable locally;
- non-private answer-token recall: 95.80%;
- PL4 local retention: 0.00%.

PersonaMem-v2:

- 278 questions;
- no annotated private value occurs verbatim in answer/evidence;
- local private recoverability is therefore not applicable, not 100%;
- non-private answer-token recall: 59.81%;
- PL4 local retention: 0.00%.

The PersonaMem-v2 proxy result is a warning: public-memory compression may have
discarded paraphrased or implicit utility. The next gate must run actual
Mem0/LangMem/Memobase QA and report question categories separately.

The same-schema cloud-safe benchmark artifacts also protect questions, options,
answers, evidence, and user identifiers. Across the released data they retain
zero exact sensitive values and zero PL4 values. The 30% token-reduction gate is
not applied to these short evaluation fields; it applies only to long-term
memory content.

## Reproduce

```bash
python -m evaluation.eval_public_memory \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/minimal_public_memprivacy.jsonl \
  --metrics-output evaluation/results/minimal_public_memprivacy_metrics.json \
  --state-dir evaluation/results/minimal_public_memprivacy_state \
  --cloud-safe-dataset-output evaluation/results/minimal_public_memprivacy_benchmark.jsonl \
  --annotation-source oracle

python -m evaluation.eval_public_memory \
  --input data/personamem_v2_testset.jsonl \
  --output evaluation/results/minimal_public_personamem.jsonl \
  --metrics-output evaluation/results/minimal_public_personamem_metrics.json \
  --state-dir evaluation/results/minimal_public_personamem_state \
  --cloud-safe-dataset-output evaluation/results/minimal_public_personamem_benchmark.jsonl \
  --annotation-source oracle
```

For an end-to-end scientific run, generate `privacy_info_llm` with the declared
detector checkpoint and rerun using `--annotation-source model`.
