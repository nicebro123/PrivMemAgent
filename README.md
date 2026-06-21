<h1 align="center">
    PrivMemAgent: Policy-Constrained Minimal Public Memory for Edge-Cloud Agents
</h1>

<p align="center">
<a href="https://spdx.org/licenses/CC-BY-NC-ND-4.0.html">
    <img alt="License: CC-BY-NC-ND-4.0" src="https://img.shields.io/badge/License-CC_BY_NC_ND_4.0-brightgreen.svg">
</a>
<a href="https://github.com/MemTensor/MemPrivacy/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/MemTensor/MemPrivacy?color=blueviolet">
</a>
<a href="https://arxiv.org/abs/2605.09530">
     <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white">
</a>
<a href="https://huggingface.co/collections/IAAR-Shanghai/memprivacy">
    <img alt="Huggingface" src="https://img.shields.io/badge/🤗_Huggingface-Model-ff9800.svg">
</a>
<a href="https://modelscope.cn/collections/MemTensor/MemPrivacy">
    <img alt="ModelScope" src="https://img.shields.io/badge/🤖_ModelScope-Model-7B42BC?style=flat-square">
</a>
</p>


PrivMemAgent is a research extension of the **MemPrivacy privacy-preserving
personalized memory framework** for **edge-cloud agents**. This branch adds
policy-constrained minimal public memory while retaining MemPrivacy's reversible
pseudonymization path. The cloud-isolation guarantee applies only when privacy
detection runs on a trusted local endpoint; remote detection requires explicit
opt-in and exposes raw input to that provider.


---

## Reproducibility Layout

Following the same code/assets separation used by
[nicebro123/EvoCo-RAG](https://github.com/nicebro123/EvoCo-RAG), keep this Git
repository focused on code, tracked config profiles, small sample data, tests,
and documentation. Large or machine-specific assets should live outside git.

```text
parent/
├── PrivMemAgent/          # this repo: code, configs, scripts, docs, tests
└── memprivate_assets/     # not committed: model weights, runtime configs, outputs
    ├── models/
    │   └── bge-m3/
    ├── runtime_configs/
    ├── results/
    └── logs/
```

On the current H20 server we use the equivalent shared asset root:

```text
/mnt/infini-data/test/quan_space/codespace/memprivate/
├── PrivMemAgent/
├── models/bge-m3/
└── PrivMemAgent/evaluation/runtime_configs/   # ignored local runtime YAMLs
```

The normal reproduction path is:

```text
clone/update repo -> create env -> configure provider keys -> download local embedding -> materialize runtime config -> smoke test -> user-limited experiment -> audit results
```

---

## Why MemPrivacy?

Cloud agents typically send user messages to remote LLMs and store conversation traces in memory systems (e.g., **Mem0**, **LangMem**, **Memobase**) for long-term personalization. This creates a large privacy attack surface:

- plaintext prompts and logs may contain **PII**, medical/financial data, credentials
- cloud memory stores can leak via retrieval, prompt injection, inversion, or misconfiguration
- naïve mitigation (e.g., `***` masking) **destroys task semantics**, harming retrieval and personalization

**Goal:** reduce privacy leakage **without sacrificing utility**.

---

## Core Idea

<div align="center">
    <table border="0">
        <tr>
            <td width="45%" align="center">
                <img src="assets/framework.jpg" width="100%">
                <br>
                <em><strong>Fig 1.</strong> Overview of MemPrivacy. </em>
            </td>
        </tr>
    </table>
</div>

MemPrivacy implements **local reversible pseudonymization**:

1. **On-device privacy detection (local)**
   Detect privacy spans in user input and classify them by:
   - **privacy level** (PL1–PL4)
   - **privacy type** (e.g., Email, Real Name, Medical Health, Recovery Code)

2. **Typed placeholder replacement (local → cloud)**
   Replace protected spans with **semantically meaningful typed placeholders**, e.g.:
   - `160/110` (blood pressure) → `<Health_Info_1>`
   - `recovery code RC-7291` → `<Recovery_Code_1>`

3. **Local encrypted mapping (persistent across sessions)**
   Store an HMAC index and Fernet-encrypted original value in a namespace-isolated local SQLite DB.

4. **Cloud reasoning and memory operations (cloud)**
   The cloud agent/memory only sees placeholders—preserving semantic roles while hiding raw values.

5. **Downlink restoration (local)**
   Restore placeholders in the cloud response back to the original values for a fluent user experience.

With a local detector, this yields **architecture-level isolation**: cloud components never see/store raw sensitive values.

---

## Key Contributions & Advantages

<div align="center">
    <table border="0">
        <tr>
            <td width="45%" align="center">
                <img src="assets/intro.jpg" width="100%">
                <br>
                <em><strong>Fig 2.</strong> Comparison of privacy protection strategies for local-to-cloud agent interactio. </em>
            </td>
        </tr>
    </table>
</div>

### 1) Privacy–Utility Balance (vs. masking)
- **Irreversible masking** (`***`) protects privacy but loses meaning and breaks memory retrieval.
- **Untyped placeholders** (`<Mask_1>`) keep structure but lose semantic roles.
- **MemPrivacy (typed placeholders)** preserve the semantic role *and* hide raw values, minimizing utility loss.

### 2) Configurable Protection via a 4-Level Privacy Taxonomy
MemPrivacy introduces **PL1–PL4** to support user-configurable policies:

| Level | Meaning | Examples | Typical Default Policy |
|---|---|---|---|
| PL1 | low sensitivity / preferences | “I like sci-fi”, tone, generic habits | can be kept for personalization |
| PL2 | identifiable PII | real name, phone, email, detailed address, account IDs | disallowed by default in long-term memory |
| PL3 | highly sensitive PII | health records, financial records, precise location, religion/ethnicity | not permitted in general memory |
| PL4 | critical secrets (immediately exploitable) | passwords, OTPs, recovery codes, API keys | **zero retention**; must be blocked/redacted |

### 3) Benchmark & Evaluation for Memory Systems
This repo builds **MemPrivacy-Bench** and evaluates privacy protection strategies across real memory systems:
- **MemPrivacy-Bench**: 200 synthetic users, bilingual (Chinese/English), multi-turn dialogues with dense privacy exposure, plus memory QA tasks.
- Evaluations on **MemPrivacy-Bench** (in-distribution) and **PersonaMem-v2** (out-of-distribution, annotated here).

### 4) Lightweight & Practical
The framework is designed for **edge deployment**:
- local detection + placeholder substitution + SQLite lookup are low-latency operations
- works as a drop-in privacy layer for existing cloud agents / memory systems

### 5) Open-Source MemPrivacy Models
We release a family of MemPrivacy models trained via Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) across different parameter sizes. You can access the full model collections on [Hugging Face](https://huggingface.co/collections/IAAR-Shanghai/memprivacy) and [ModelScope](https://modelscope.cn/collections/MemTensor/MemPrivacy).

| Model Name | Parameters | Method | HuggingFace Link | ModelScope Link |
| :--- | :---: | :---: | :--- | :--- |
| **MemPrivacy-4B-RL** | 4B | RL | [🤗 MemPrivacy-4B-RL](https://huggingface.co/IAAR-Shanghai/MemPrivacy-4B-RL) | [🤖 MemPrivacy-4B-RL](https://modelscope.cn/models/MemTensor/MemPrivacy-4B-RL) |
| **MemPrivacy-4B-SFT** | 4B | SFT | [🤗 MemPrivacy-4B-SFT](https://huggingface.co/IAAR-Shanghai/MemPrivacy-4B-SFT) | [🤖 MemPrivacy-4B-SFT](https://modelscope.cn/models/MemTensor/MemPrivacy-4B-SFT) |
| **MemPrivacy-1.7B-RL** | 1.7B | RL | [🤗 MemPrivacy-1.7B-RL](https://huggingface.co/IAAR-Shanghai/MemPrivacy-1.7B-RL) | [🤖 MemPrivacy-1.7B-RL](https://modelscope.cn/models/MemTensor/MemPrivacy-1.7B-RL) |
| **MemPrivacy-1.7B-SFT** | 1.7B | SFT | [🤗 MemPrivacy-1.7B-SFT](https://huggingface.co/IAAR-Shanghai/MemPrivacy-1.7B-SFT) | [🤖 MemPrivacy-1.7B-SFT](https://modelscope.cn/models/MemTensor/MemPrivacy-1.7B-SFT) |

---

## Evaluation Results

> **Reproducibility note:** the tables below are the values reported by the paper. This checkout now uses strict span-gated scoring and maximum-weight matching. Re-run the extractor and memory experiments before treating the displayed values as results produced by the current code.

### 1. Privacy Extraction Performance

<div align="center">
    <em><strong>Table 1.</strong> Performance comparison of different LLMs and MemPrivacy models on MemPrivacy-Bench and PersonaMem-v2.</em>
    <img src="assets/table1.png" width="100%" alt="Performance comparison of different LLMs and MemPrivacy models on MemPrivacy-Bench and PersonaMem-v2.">
    <br>
</div>


**Key Takeaways:**

* **Superior Accuracy:** MemPrivacy consistently outperforms 11 general LLMs and **OpenAI-Privacy-Filter**. The best model (MemPrivacy-4B-RL) achieves F1 scores of **85.97%** and **94.48%**, significantly surpassing the top general models (78.41% and 92.18%). Even our smallest 0.6B model beats most general models.
* **Robustness on Complex Data:** While lightweight filters like OpenAI-Privacy-Filter are fast, they struggle with implicit and linguistically diverse privacy expressions (only 35.50% F1 on MemPrivacy-Bench). MemPrivacy accurately handles fine-grained, heterogeneous conversational scenarios.
* **High Efficiency:** Despite its accuracy, MemPrivacy remains highly efficient. Processing latency per message is consistently **below one second** on PersonaMem-v2, making it well-suited for seamless on-device deployment without noticeable delays.

### 2. Memory System Performance under Different Protection Methods

<div align="center">
    <em><strong>Table 2.</strong> Performance comparison under different privacy protection methods on three memory systems.</em>
    <img src="assets/table2.png" width="100%" alt="Performance comparison under different privacy protection methods on three memory systems.">
    <br>
</div>


**Key Takeaways:**

* **Optimal Privacy-Utility Trade-off:** Compared to traditional masking (`***`) or untyped placeholders (`<Mask_1>`), MemPrivacy preserves the utility of downstream systems (LangMem, Mem0, Memobase) significantly better by retaining critical semantic roles.
* **Minimal Degradation:** When applying stringent protection (PL2–PL4), system accuracy drops by merely **0.71%–1.60%**. If protecting only critical secrets (PL4), the drop is **below 0.89%**.
* **Extractor Dependency:** The effectiveness of the entire framework heavily depends on accurate privacy extraction. Replacing the MemPrivacy model with general LLMs (e.g., DeepSeek-V3.2-Think, GPT-5.2) causes substantial accuracy degradation, validating the necessity of our specialized fine-tuning.

---


## What’s in This Repository?

High-level structure:

```text
MemPrivacy/
├── data/        # partial user data from the MemPrivacy-Bench and PersonaMem-v2 test sets
├── evaluation/  # evaluation on memory systems + metrics
└── src/         # privacy masking/pseudonymization core
```

### Core Components

- **Reversible pseudonymization module** (`src/privacy_masking.py`)
  - `PrivacyStore` (SQLite mapping store)
  - `mask_dialogue()`, `unmask_dialogue()`, `detect_and_mask_dialogue()`
  - masking modes: `type_specific`, `generic`, `complete`
- **Evaluation suite** (`evaluation/`)
  - memory systems: `eval_mem0.py`, `eval_langmem.py`, `eval_memobase.py`
  - metrics: `metric.py` (privacy extraction P/R/F1, level/type matching, etc.)
  - results saved to `evaluation/results/`

---

## How It Works (End-to-End)

### Stage A — Uplink Desensitization (Local)
- detect privacy spans locally (original text, privacy level, privacy type)
- apply a user policy: e.g., mask only **PL3+**, or **PL2–PL4**
- replace spans with typed placeholders
- store mapping locally (persistent across sessions)

### Stage B — Cloud Processing
- send only placeholderized text to the cloud LLM / memory system
- the cloud performs normal agent workflows (reasoning, tool use, memory write/retrieval) **and generates a response**
- cloud memory stores placeholders, not raw secrets

### Stage C — Downlink Restoration (Local)
- restore placeholders in the response using the local mapping DB
- user sees original values; cloud never receives them

---

## Quickstart

### 1) Installation

```bash
git clone https://github.com/nicebro123/PrivMemAgent.git
cd PrivMemAgent

conda create -n memprivate python=3.11 -y
conda activate memprivate
pip install -U pip
pip install -r requirements-dev.txt
```

A plain virtual environment also works if you are not on a shared GPU server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For local detector checkpoint evaluation through vLLM on a supported Linux/CUDA
host, install the additional vLLM requirements after the base environment:

```bash
pip install -r requirements-vllm.txt
```

Verify CUDA before running local embeddings on a GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

### 2) Configuration

To both use the core MemPrivacy framework and run the evaluation benchmarks, you need to configure two YAML files:

**1. `src/privacy_config.yaml` (For using the framework)**
This file controls the core reversible pseudonymization module. Key configurations include:
- **`llm`**: local OpenAI-compatible detector endpoint and model parameters. Non-local endpoints are rejected unless `allow_remote: true` is explicitly set.
- **`privacy`**: The local SQLite database path (`db_path`) for storing mapping rules, and the `mask_levels` (e.g., `PL3`, `PL4`) to define your privacy protection policy.

Set `MEMPRIVACY_STORE_KEY` to a Fernet key in production. If it is omitted, the library creates a mode-`0600` key under `~/.config/memprivacy/keys/`. The database and key must not be copied to the cloud together.

**2. `evaluation/eval_config.yaml` (For evaluating memory systems)**
This file configures the benchmarking suite across different memory systems (Mem0, Memobase, etc.). Key configurations include:
- **Global API Keys**: `openai_base_url` and `openai_api_key`.
- **Role-specific LLMs**: Distinct model settings for memory operations (`memory_llm`), generating answers (`answer_llm`), and automated evaluation (`judgment_llm`, `privacy_llm`).
- **System Configs**: Database paths and connection URLs for specific memory systems (e.g., `mem0_config`, `memobase`).

---


### 3) Experiment Profiles and Runtime Configs

Tracked evaluation profiles live under `evaluation/` and contain only reusable,
non-secret defaults:

- `evaluation/eval_config.yaml`: default OpenAI-compatible profile.
- `evaluation/eval_config.deepseek.yaml`: DeepSeek official API for chat plus
  local BGE-M3 embeddings.

Do not commit API keys, generated runtime configs, model weights, or experiment
outputs. Put provider credentials in a shell env file outside the repo, for
example:

```bash
mkdir -p ~/.config/memprivate
chmod 700 ~/.config/memprivate
cat > ~/.config/memprivate/deepseek.env <<'EOF'
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_API_KEY="replace-with-your-key"
EOF
chmod 600 ~/.config/memprivate/deepseek.env
source ~/.config/memprivate/deepseek.env
```

Then materialize a machine-specific runtime YAML. This is the PrivMemAgent
equivalent of EvoCo-RAG's generated per-run `run_config.yaml`: it freezes the
provider profile, local embedding path, GPU choice, and result root for one
server/run, while staying outside git.

```bash
python -m tools.materialize_eval_config \
  --profile evaluation/eval_config.deepseek.yaml \
  --output evaluation/runtime_configs/eval_config.deepseek.cuda0.yaml \
  --embedding-model /mnt/infini-data/test/quan_space/codespace/memprivate/models/bge-m3 \
  --embedding-device cuda:0 \
  --output-path evaluation/results \
  --print-export

export MEMPRIVACY_EVAL_CONFIG="$PWD/evaluation/runtime_configs/eval_config.deepseek.cuda0.yaml"
```

The memory runners resolve `output_path` from the selected config. The
materializer stores `--output-path` as an absolute path so moving the runtime
YAML does not accidentally redirect outputs under `evaluation/runtime_configs/`.

To switch large-model providers later, copy or add another tracked profile such
as `evaluation/eval_config.openai.yaml` or `evaluation/eval_config.local.yaml`,
keep secrets as environment references, and materialize a new runtime config:

```bash
python -m tools.materialize_eval_config \
  --profile evaluation/eval_config.deepseek.yaml \
  --output ../memprivate_assets/runtime_configs/deepseek_v4_flash_cuda0.yaml \
  --set memory_llm.model=deepseek-v4-flash \
  --set answer_llm.model=deepseek-v4-flash \
  --set judgment_llm.model=deepseek-v4-pro \
  --embedding-device cuda:0 \
  --output-path ../memprivate_assets/results/deepseek_v4_flash
```

Use `--set dotted.key=value` for non-secret per-run overrides. Literal secrets
are rejected by default; use `$ENV_VAR` references instead.

### 4) DeepSeek + Local BGE Smoke Test

DeepSeek supplies the OpenAI-compatible chat/reasoning model roles
(`memory_llm`, `answer_llm`, and `judgment_llm`). BGE-M3 runs locally as the
embedding model, so user memory text does not need to be sent to a remote
embedding endpoint. The BGE checkpoint should be downloaded outside git, for
example with ModelScope or Hugging Face:

```bash
# Example expected path on the H20 server
test -d /mnt/infini-data/test/quan_space/codespace/memprivate/models/bge-m3
```

Run preflight before spending API/GPU time:

```bash
source ~/.config/memprivate/deepseek.env
export MEMPRIVACY_EVAL_CONFIG="$PWD/evaluation/runtime_configs/eval_config.deepseek.cuda0.yaml"
python -m tools.preflight_memory_eval \
  --config "$MEMPRIVACY_EVAL_CONFIG" \
  --system mem0 \
  --system langmem \
  --no-probe-openai
```

Compile a tiny cloud-safe public-memory artifact:

```bash
python -m evaluation.eval_public_memory \
  --input data/memory_eval_smoke.jsonl \
  --output evaluation/results/smoke_deepseek_bge_public_records.jsonl \
  --metrics-output evaluation/results/smoke_deepseek_bge_public_metrics.json \
  --state-dir evaluation/results/smoke_deepseek_bge_public_state \
  --cloud-safe-dataset-output evaluation/results/smoke_deepseek_bge_public_benchmark.jsonl \
  --annotation-source oracle \
  --minimum-token-reduction -1.0
```

Audit the generated cloud-safe artifacts:

```bash
python -m tools.adversarial_audit \
  --source data/memory_eval_smoke.jsonl \
  --artifact evaluation/results/smoke_deepseek_bge_public_records.jsonl \
  --artifact evaluation/results/smoke_deepseek_bge_public_benchmark.jsonl \
  --report evaluation/results/smoke_deepseek_bge_adversarial_audit.json
```

Run one-user Mem0 and LangMem smoke checks on the cloud-safe benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.eval_mem0 \
  --input evaluation/results/smoke_deepseek_bge_public_benchmark.jsonl \
  --no-mask --mcq --user-num 1 --num-workers 1

CUDA_VISIBLE_DEVICES=0 python -m evaluation.eval_langmem \
  --input evaluation/results/smoke_deepseek_bge_public_benchmark.jsonl \
  --no-mask --mcq --user-num 1 --num-workers 1
```

### 5) User-Limited PersonaMem-v2 Run

After smoke passes, run a small reproducibility experiment before launching a
full matrix. The following commands use 5 users and GPU 0 for local embeddings:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.eval_public_memory \
  --input data/personamem_v2_testset.jsonl \
  --output evaluation/results/user5_deepseek_bge/persona_public_records.jsonl \
  --metrics-output evaluation/results/user5_deepseek_bge/persona_public_metrics.json \
  --state-dir evaluation/results/user5_deepseek_bge/persona_public_state \
  --cloud-safe-dataset-output evaluation/results/user5_deepseek_bge/persona_public_benchmark.jsonl \
  --annotation-source oracle \
  --user-limit 5 \
  --minimum-token-reduction -1.0

python -m tools.adversarial_audit \
  --source data/personamem_v2_testset.jsonl \
  --source-user-limit 5 \
  --artifact evaluation/results/user5_deepseek_bge/persona_public_records.jsonl \
  --artifact evaluation/results/user5_deepseek_bge/persona_public_benchmark.jsonl \
  --report evaluation/results/user5_deepseek_bge/persona_public_adversarial_audit.json

CUDA_VISIBLE_DEVICES=0 python -m evaluation.eval_mem0 \
  --input evaluation/results/user5_deepseek_bge/persona_public_benchmark.jsonl \
  --no-mask --mcq --user-num 5 --num-workers 1

CUDA_VISIBLE_DEVICES=0 python -m evaluation.eval_langmem \
  --input evaluation/results/user5_deepseek_bge/persona_public_benchmark.jsonl \
  --no-mask --mcq --user-num 5 --num-workers 1
```

A complete run should leave public-memory metrics, adversarial-audit reports,
and memory-system result JSON files under the selected `output_path`. Treat
`adversarial_audit.passed == true` and zero direct leakage failures as the gate
before sending artifacts to a cloud memory backend.



### 6) Default MemPrivacy-4B-RL Memory Workflow

The default research workflow now uses the released **MemPrivacy-4B-RL** model
as the privacy extractor. Oracle annotations remain useful for upper-bound
analysis, but new experiments should use model annotations unless a paper table
explicitly says otherwise.

Expected local assets on the H20 server:

```text
/mnt/infini-data/test/quan_space/codespace/memprivate/models/
├── MemPrivacy-4B-RL-hf/   # downloaded from Hugging Face via HF mirror
└── bge-m3/                # local embedding model
```

Run a tiny end-to-end smoke on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.run_memprivacy4b_memory_workflow   --mode smoke
```

Run the 5-user PersonaMem-v2 pilot and include Mem0/LangMem baselines:

```bash
source ~/.config/memprivate/deepseek.env
CUDA_VISIBLE_DEVICES=0 python -m tools.run_memprivacy4b_memory_workflow   --mode persona5   --run-memory-systems   --memory-system mem0   --memory-system langmem
```

Run the full released PersonaMem-v2 model-annotation workflow:

```bash
source ~/.config/memprivate/deepseek.env
CUDA_VISIBLE_DEVICES=0 python -m tools.run_memprivacy4b_memory_workflow   --mode full
```

Each mode performs:

```text
MemPrivacy-4B-RL privacy extraction
  -> writes privacy_info_llm
  -> compiles model-annotated public memory with --annotation-source model
  -> writes a cloud-safe benchmark JSONL
  -> runs adversarial leakage audit
  -> optionally runs Mem0/LangMem/Memobase on the cloud-safe benchmark
```

Use `--dry-run` to print the exact commands before launching a long experiment.
All generated runtime configs, model predictions, public-memory artifacts,
states, audits, and baseline result files stay under ignored runtime/results
paths and are not committed to git.

## Evaluate The Privacy Extractor

```bash
python -m evaluation.eval \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/memprivacy_with_predictions.jsonl \
  --metrics-output evaluation/results/extractor_metrics.json \
  --backend vllm \
  --model /absolute/path/to/checkpoint
```

When `--model` is a Hugging Face repository ID, `--revision` must be its
immutable 40-character commit SHA. Type scoring is exact by default; configure
the optional embedding endpoint arguments only when semantic type matching is
part of the declared evaluation protocol.

## Evaluate Memory Systems (Mem0 / LangMem / Memobase)

Example commands:

```bash
python -m evaluation.eval_mem0 \
  --input data/personamem_v2_testset.jsonl --mcq \
  --annotation-source oracle

python -m evaluation.eval_langmem \
  --input data/memprivacy_bench_testset.jsonl --no-mcq \
  --annotation-source oracle

python -m evaluation.eval_memobase \
  --input data/personamem_v2_testset.jsonl --mcq \
  --annotation-source oracle
```

Each command accepts `--mask/--no-mask`, `--mask-level`, `--mask-mode`,
`--user-num`, and `--num-workers`. Result and database directories are created
automatically. Mem0 storage is reset before each complete add-memory + QA run,
so stale memories cannot contaminate a repeated experiment.

`--annotation-source model` is the default scientific setting and requires
`privacy_info_llm` generated by `evaluation.eval`. The examples above explicitly
select `oracle` because the released partial JSONL files contain only reference
annotations. Model and oracle runs use separate storage and output names.

Validate annotations before running:

```bash
python tools/validate_dataset.py data/*.jsonl --report evaluation/results/data_audit.json
```

Validate the complete runtime before a costly memory experiment:

```bash
export OPENAI_BASE_URL="https://your-gateway.example/v1"
export OPENAI_API_KEY="..."
python -m tools.preflight_memory_eval --system mem0 --system langmem
```

Add `--system memobase` only when a Memobase server is reachable. Configuration
files may reference secrets as `$OPENAI_BASE_URL`, `$OPENAI_API_KEY`,
`$MEMOBASE_PROJECT_URL`, and `$MEMOBASE_API_KEY`; unresolved references fail
preflight instead of being treated as literal credentials.



The currently released partial files contain 10 annotations whose
`original_text` is absent from the corresponding message. Production APIs fail
closed on these records; evaluation wrappers log and skip them.

## Compile Minimal Public Memory

The deterministic minimal-sufficient baseline separates cloud-safe public
memory from encrypted local-only facts:

```bash
python -m evaluation.eval_public_memory \
  --input data/memprivacy_bench_testset.jsonl \
  --output evaluation/results/minimal_public.jsonl \
  --metrics-output evaluation/results/minimal_public_metrics.json \
  --state-dir evaluation/results/minimal_public_state \
  --cloud-safe-dataset-output evaluation/results/minimal_public_benchmark.jsonl \
  --annotation-source oracle
```

For the tiny smoke file, the default long-term reduction gate can be stricter
than the sample is meant to satisfy. Use an explicit smoke-only override when
checking plumbing rather than compression quality:

```bash
python -m evaluation.eval_public_memory \
  --input data/memory_eval_smoke.jsonl \
  --output evaluation/results/minimal_public_smoke.jsonl \
  --metrics-output evaluation/results/minimal_public_smoke_metrics.json \
  --state-dir evaluation/results/minimal_public_smoke_state \
  --cloud-safe-dataset-output evaluation/results/minimal_public_smoke_benchmark.jsonl \
  --annotation-source oracle \
  --minimum-token-reduction -1.0
```

The JSONL output is cloud-safe by default: it contains the public text, stable
anonymized user/message aliases, policy version, and coarse token statistics.
Source user/message identifiers are anonymized by default; use `--raw-output-ids`
only for trusted local debugging. Internal fingerprints, privacy categories,
route decisions, alias scopes, and provenance IDs are also withheld by default;
use `--debug-output-metadata` only for local audits. Exact local values remain
in the encrypted state directory; PL4 values are not retained and their entire
source sentence is removed from public text. Use `src/public_memory_config.yaml`
to configure policy, alias scopes, selection budgets, context minimization, and
leakage gates.

The optional cloud-safe dataset preserves the benchmark schema while protecting
dialogues, questions, options, answers, evidence, and user identifiers. Memory
control requests such as `Please forget ...` and one-off task requests such as
`Could you explain ...` are treated as local control flow, not public memory
facts, so they are removed from generated public/cloud-safe artifacts. The
utility proxy resolves multiple-choice answer labels such as
`(b)` through `all_options` before measuring non-private answer-token recall, so
PersonaMem style MCQ labels are not treated as semantic answer content. It can
be passed to the existing memory runners without a second masking layer:

```bash
python -m evaluation.eval_mem0 \
  --input evaluation/results/minimal_public_benchmark.jsonl \
  --no-mask --mcq
```

The memory-system runners refuse `--no-mask` unless the input looks like a
cloud-safe public-memory dataset with anonymized `User-*` identifiers and no raw
privacy/debug metadata. Use `--allow-unsafe-no-mask` only for trusted local
debugging of deliberately raw inputs. Keep the generated state directory on the
trusted edge. Do not upload it with the cloud-safe JSONL.

Run the deterministic adversarial leakage audit before sending generated
artifacts to a cloud memory system:

```bash
python -m tools.adversarial_audit \
  --source data/memory_eval_smoke.jsonl \
  --artifact evaluation/results/minimal_public_smoke.jsonl \
  --artifact evaluation/results/minimal_public_smoke_benchmark.jsonl \
  --report evaluation/results/adversarial_audit_smoke.json
```

When auditing artifacts generated with `eval_public_memory --user-limit N`, pass
`--source-user-limit N` to keep the attack source scope aligned with the
generated artifact. Otherwise secrets from users outside the generated subset can
collide with ordinary public text and create misleading cross-user findings.

This deterministic audit checks exact secret recovery, PL4/canary exposure,
source user identifiers, sensitive privacy-type metadata, linkability metadata,
verbatim membership markers, and prompt-injection strings in public/cloud-safe
JSONL. It scopes checks by artifact user order so partial `--user-limit` runs do
not confuse different users' secrets. Alias reuse, non-sensitive membership
markers, and sensitive category words are reported as warnings; direct leakage
and debug metadata are failures. The compiler also applies a conservative
user-level residual scrubber to remove known sensitive tokens that survive span
replacement in dialogues or QA fields. This is still a deterministic baseline,
not a replacement for held-out learned attribute, linkage, membership, or
prompt-injection attackers.

For minimality and Pareto-frontier analysis, sweep public-memory token budgets
and collect utility/leakage metrics in one JSON summary:

```bash
python -m tools.public_memory_budget_sweep \
  --input data/memory_eval_smoke.jsonl \
  --output-dir evaluation/results/budget_sweep_smoke \
  --summary-output evaluation/results/budget_sweep_smoke_summary.json \
  --annotation-source oracle \
  --minimum-token-reduction -1.0 \
  --budget 16 \
  --budget 64 \
  --budget 128
```

Each budget run writes public memory, optional cloud-safe benchmark data,
`metrics.json`, and `adversarial_audit.json`. The summary includes token
reduction, utility proxy recall, exact/PL4 leakage, adversarial failures, and
warning counts for plotting the utility/privacy/minimality trade-off.

After a sweep, choose a deployable budget with policy gates:

```bash
python -m tools.public_memory_budget_selector \
  --summary evaluation/results/budget_sweep_smoke_summary.json \
  --output evaluation/results/budget_sweep_smoke_recommendation.json \
  --min-utility 0.75 \
  --min-local-recoverability 0.95
```

The selector reports the smallest safe budget, the highest-utility safe budget,
and the Pareto frontier after builtin, cloud-safe, adversarial, exact-recovery,
PL4-retention, utility, and local-recoverability gates.

For a lightweight held-out learned attacker baseline, train a bag-of-words
Naive Bayes attribute attacker on source-user labels and public artifacts:

```bash
python -m tools.learned_attribute_attack \
  --source data/personamem_v2_testset.jsonl \
  --artifact evaluation/results/persona_public.jsonl \
  --target medical \
  --target-term medical \
  --target-term health \
  --source-user-limit 20 \
  --report evaluation/results/persona_medical_attack.json
```

The report includes held-out accuracy, ROC AUC, majority baseline accuracy, and
top positive tokens. Treat it as a cheap attacker sanity check before training
stronger neural or LLM-based attackers.

Evaluation logic:
1. feed dialogues turn-by-turn into the memory system (optionally with MemPrivacy masking)
2. protect known sensitive values in questions and options before cloud retrieval
3. keep retrieved memories and generated responses protected through cloud judging
4. restore only the final local result artifact
5. compute privacy leakage / extraction metrics

---

## Use MemPrivacy in Your Own Agent (Minimal Example)

The reversible pseudonymization APIs live in:

- `src/privacy_masking.py` (core)
- `evaluation/privacy_masking.py` contains tolerant wrappers for auditing imperfect benchmark annotations

Conceptual usage:

```python
from src.privacy_masking import PrivacyStore, mask_dialogue, unmask_dialogue

store = PrivacyStore(
    db_path="local_privacy_store.sqlite",
    namespace=user_id,
)

masked_text = mask_dialogue(
    dialogue_text=user_text,
    privacy_items=detected_privacy_items,  # produced locally by MemPrivacy model
    store=store,
    mask_levels=["PL2", "PL3", "PL4"],
)

# send masked_text to cloud...

restored = unmask_dialogue(cloud_response_text, store=store)
```

### Masking Modes
- `type_specific`: configure `PrivacyStore(mask_mode="type_specific")`
- `generic`: configure `PrivacyStore(mask_mode="generic")`
- `complete`: remove sensitive spans entirely (max privacy, lowest utility)

Generated reversible placeholders include a random collision-resistant suffix,
for example `<MPM_Email_1_a1b2c3d4e5f6>`.

### Policy Control (Privacy Levels)
You can enforce a masking threshold such as:
- protect `PL4` only (credentials)
- protect `PL3+` (highly sensitive + secrets)
- protect `PL2–PL4` (most conservative)

### Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check src evaluation tests tools
```

Development notes:

- [Correctness and reproducibility](docs/CORRECTNESS_NOTES.md)
- [Minimal sufficient public memory roadmap](docs/INNOVATION_ROADMAP.md)
- [Minimal public memory preliminary results](docs/PRELIMINARY_RESULTS.md)
- [Memory evaluation runbook](docs/MEMORY_EVAL_RUNBOOK.md)

---

## Citation

If you use MemPrivacy-Bench, the taxonomy, or the framework, please cite:

```bibtex
@misc{chen2026memprivacyprivacypreservingpersonalizedmemory,
      title={MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents},
      author={Yining Chen and Jihao Zhao and Bo Tang and Haofen Wang and Yue Zhang and Fei Huang and Feiyu Xiong and Zhiyu Li},
      year={2026},
      eprint={2605.09530},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2605.09530},
}
```

---

## Disclaimer

This project is intended for **privacy research and evaluation**.
Do **not** use it to process real user secrets without proper security controls, threat modeling, and compliance review. Always follow local laws and organizational policies.
