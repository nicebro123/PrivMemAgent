# Correctness And Reproducibility Notes

## Fixed In This Branch

- Raw privacy detection is restricted to localhost unless remote disclosure is
  explicitly enabled.
- SQLite no longer stores original values as plaintext columns.
- Mapping is namespace-isolated, concurrency-safe, and supports legacy
  migration.
- First-use schema creation is serialized per database path so concurrent
  workers cannot create multiple aliases for the same value during startup.
- Placeholder restoration supports legacy punctuation and new collision-
  resistant aliases.
- Invalid or hallucinated spans fail closed in production.
- Privacy extraction matching uses maximum-weight assignment.
- Predictions with an incorrect span receive zero total score.
- Failed model outputs remain in the evaluation denominator.
- Model predictions and oracle annotations are separate experiment conditions.
- Runner paths, output directories, CLI arguments, and repeated Mem0 runs are
  deterministic and isolated.
- Memory-system evaluation now protects known sensitive values in retrieval
  queries, MCQ options, and judge references before cloud calls.
- Retrieved memories and generated responses remain pseudonymized through the
  cloud answer and judge stages; restoration occurs only for the local result
  artifact.
- Result files include dataset hashes and run configuration.

## Released Data Audit

The two partial JSONL files contain 10 annotations whose `original_text` does
not occur in the corresponding message. Run:

```bash
python tools/validate_dataset.py data/*.jsonl
```

Production masking raises on such records. Evaluation wrappers warn and skip
them so an impossible span is not passed to the cloud as if it were masked.
Minimal-public-memory audits report the skipped count and exclude these records
from privacy denominators.

## Query-Side Evaluation Boundary

The released questions can repeat values that appeared in dialogue privacy
annotations. The corrected runners protect those known values using the same
local mapping before memory search or answer-model calls. This benchmark helper
does not replace a production query detector: unseen values introduced for the
first time in a question still require local detection before any cloud call.

In the released partial data, exact known sensitive values occur in 69 of 149
MemPrivacy-Bench questions and 84 of 278 PersonaMem-v2 questions. Previous
runners sent those queries directly and, on the MCQ path, restored retrieved
memory before calling the answer model. Corrected results must therefore be
rerun; old memory-system tables are not evidence for the repaired boundary.

## Remaining Reproduction Dependencies

The repository does not include the complete training split, GRPO split,
training launcher, released checkpoint weights, cloud API credentials, or a
running Memobase service. Therefore:

- unit and data round-trip tests are reproducible from this checkout;
- complete model training and paper-table reproduction require external
  artifacts;
- displayed README tables must be treated as paper-reported values until rerun
  with the corrected metric.

Run `python -m tools.preflight_memory_eval` under Python 3.10+ before memory
experiments. It validates optional packages, runtime credentials, OpenAI
connectivity, and Memobase reachability without printing secret values.
