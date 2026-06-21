from __future__ import annotations

import pytest
import yaml

from tools.materialize_eval_config import materialize_config


def test_materialize_config_applies_runtime_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "openai_base_url: $OPENAI_BASE_URL\n"
        "openai_api_key: $OPENAI_API_KEY\n"
        "embedding_model:\n"
        "  provider: huggingface\n"
        "  model: /models/old\n"
        "  dimensions: 1024\n"
        "  device: cpu\n"
        "answer_llm:\n"
        "  model: old-model\n"
        "output_path: results\n",
        encoding="utf-8",
    )

    config = materialize_config(
        profile,
        output_path="outputs/run1",
        embedding_device="cuda:0",
        embedding_model="models/bge-m3",
        overrides=["answer_llm.model=deepseek-chat", "answer_llm.retry_times=7"],
    )

    assert config["openai_api_key"] == "$OPENAI_API_KEY"
    assert config["embedding_model"]["device"] == "cuda:0"
    assert config["embedding_model"]["model"] == str((tmp_path / "models/bge-m3").resolve())
    assert config["output_path"] == str((tmp_path / "outputs/run1").resolve())
    assert config["answer_llm"]["model"] == "deepseek-chat"
    assert config["answer_llm"]["retry_times"] == 7
    yaml.safe_dump(config)


def test_materialize_config_rejects_literal_secret_override(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text("openai_api_key: $OPENAI_API_KEY\n", encoding="utf-8")

    with pytest.raises(ValueError, match="literal secret"):
        materialize_config(profile, overrides=["openai_api_key=sk-secret"])


def test_materialize_config_allows_environment_secret_reference(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text("openai_api_key: $OPENAI_API_KEY\n", encoding="utf-8")

    config = materialize_config(profile, overrides=["openai_api_key=$OTHER_API_KEY"])

    assert config["openai_api_key"] == "$OTHER_API_KEY"
