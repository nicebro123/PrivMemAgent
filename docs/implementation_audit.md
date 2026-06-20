# Implementation audit

This audit maps the paper concept and code-development plan to executable
artifacts. It distinguishes implementation completeness from empirical proof.

## Paper-to-code contract

| ID | Requirement | Implementation evidence | Verification |
|---|---|---|---|
| C1 | Public memory plus private residue and trace | `src/privacy_schema.py`, `src/privacy_abstraction.py` | Schema and policy tests |
| C2 | L1–L5 abstraction support | Oracle/trained/heuristic PMA plus mandatory L4/L5 fallbacks | Backend and 10-user candidate smoke |
| C3 | AMA exact reconstruction from public memory only | `src/privacy_auditor.py`, attack prompts | Prompt-isolation and matching tests |
| C4 | Lowest leakage subject to utility | `src/privacy_critic.py` | Deterministic selector tests |
| C5 | SFT and preference data with lineage | `evaluation/build_pma_train_data.py` | 10-user data build and trainer validation |
| C6 | Same-script Mem0 baseline comparison | `evaluation/eval_pma_mem0.py` | In-memory integration test; real Mem0 requires credentials |
| C7 | Per-privacy-type reporting | AMA aggregation and final result contract | Integration and result-contract tests |

## Code-development Definition of Done

1. **All baselines run on the same split** — implemented by one evaluation loop
   and validated by `compare_pma_results.py`.
2. **PMA oracle generates valid candidates** — implemented with a real
   OpenAI-compatible completion backend, schema repair, policy validation, and
   safe fallback behavior.
3. **PUC uses the documented selection rule** — implemented and tested.
4. **AMA attacks all cloud-visible memories** — exact reconstruction and
   attribute inference are implemented; raw values enter matching only after
   attack generation.
5. **Result JSON contains utility and leakage** — implemented, including
   per-type metrics and per-question records.
6. **Training data can be produced** — verified on all 10 included users:
   2,703 SFT examples and 7,685 deduplicated preference examples in the local
   smoke run.
7. **PMA-SFT uses the same evaluation script** — implemented through
   `--methods pma_sft --pma-model-path ...`.
8. **Paper numbers trace to JSON** — enforced by the result validator and table
   exporter.

## Bugs found and corrected

- Dataclass validation did not run on direct construction.
- Policy constraints were documented but not enforced.
- PL4 items could receive typed placeholders despite redaction-only overrides.
- Oracle and trained-model backends silently used heuristics.
- Utility was hard-coded from abstraction levels.
- Mem0 evaluation was a per-turn proxy rather than a memory QA experiment.
- Training scripts emitted manifests without training.
- Attack prompt templates contained unescaped format braces.
- DPO data could contain identical chosen and rejected outputs.
- The preference trainer used a removed TRL configuration argument.
- Result files did not enforce same-question comparisons or per-type reporting.

## Verified checks

```text
pytest: 25 passed
ruff: all checks passed
bandit: no findings in the new PMA/evaluation/training paths
compileall: passed
10-user candidate generation: 2,703 source records
10-user dry-run scoring: 13,512 candidate score records
SFT export: 2,703 examples
preference export: 7,685 examples
SFT and DPO data validation: passed
in-memory end-to-end evaluation: passed
```

## Empirical work still required

The implementation is ready for the planned experiment, but the paper claim is
not yet empirically proven.

- The repository includes 10 public users, while the plan requests 100.
- No model API key is available in the current environment.
- No trained PMA checkpoint is available.
- A real Mem0 run and PMA-SFT training have therefore not been executed here.

The repository intentionally prevents CI/proxy output from being accepted as
paper evidence.
