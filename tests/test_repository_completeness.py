from pathlib import Path

import yaml

REQUIRED_RESEARCH_FILES = (
    "src/privacy_abstraction.py",
    "src/privacy_critic.py",
    "src/privacy_auditor.py",
    "src/privacy_schema.py",
    "evaluation/build_pma_candidates.py",
    "evaluation/score_pma_candidates.py",
    "evaluation/build_pma_train_data.py",
    "evaluation/eval_pma_mem0.py",
    "evaluation/compare_pma_results.py",
    "evaluation/prompts/ama_exact_reconstruct.txt",
    "evaluation/prompts/ama_attribute_infer.txt",
    "training/train_pma_sft.py",
    "training/train_pma_preference.py",
    "docs/pma_code_development_plan.md",
    "docs/trainable_privacy_memory_abstractor.md",
)


def test_trainable_research_pipeline_is_present():
    missing = [path for path in REQUIRED_RESEARCH_FILES if not Path(path).is_file()]
    assert missing == []


def test_packaging_keeps_research_extras():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    for extra in ("mem0", "memory", "train", "legacy", "dev"):
        assert f"{extra} = [" in pyproject
    assert 'packages = ["src", "evaluation", "training", "tools"]' in pyproject


def test_evaluation_config_keeps_privacy_and_attack_models():
    config = yaml.safe_load(Path("evaluation/eval_config.yaml").read_text())
    assert config["privacy_llm"]["model"]
    assert config["attack_llm"]["model"]
    assert config["openai_api_key"] == "$OPENAI_API_KEY"
