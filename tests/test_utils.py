from evaluation import utils


def test_config_cache_is_keyed_by_path(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("value: 1\n", encoding="utf-8")
    second.write_text("value: 2\n", encoding="utf-8")

    assert utils._get_config(str(first))["value"] == 1
    assert utils._get_config(str(second))["value"] == 2


def test_verify_mcq_answer_returns_validity():
    assert utils.verify_mcq_answer("(A)", "a") == (True, True)
    assert utils.verify_mcq_answer("answer A", "a") == (False, False)


def test_environment_overrides_do_not_require_secret_files(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "openai_base_url: ''\n"
        "openai_api_key: ''\n"
        "memobase:\n"
        "  project_url: http://localhost:8019\n"
        "  api_key: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "runtime-secret")
    monkeypatch.setenv("MEMOBASE_PROJECT_URL", "https://memory.example")
    monkeypatch.setenv("MEMOBASE_API_KEY", "runtime-memory-secret")

    config = utils._load_config(str(config_path))

    assert config["openai_base_url"] == "https://gateway.example/v1"
    assert config["openai_api_key"] == "runtime-secret"
    assert config["memobase"]["project_url"] == "https://memory.example"
    assert config["memobase"]["api_key"] == "runtime-memory-secret"


def test_config_expands_environment_references(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "openai_base_url: $TEST_OPENAI_URL\n"
        "openai_api_key: ${TEST_OPENAI_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_OPENAI_URL", "https://gateway.example/v1")
    monkeypatch.setenv("TEST_OPENAI_KEY", "runtime-secret")

    config = utils._load_config(str(config_path))

    assert config["openai_base_url"] == "https://gateway.example/v1"
    assert config["openai_api_key"] == "runtime-secret"


def test_question_type_summary_reports_full_and_valid_accuracy():
    summary = utils.summarize_scores_by_question_type(
        [
            {
                "question_type": "Basic Memory",
                "score": 1,
                "is_valid": True,
            },
            {
                "question_type": "Basic Memory",
                "score": 0,
                "is_valid": False,
            },
            {
                "question_type": "Temporal Reasoning",
                "score": 0.5,
                "is_valid": True,
            },
        ]
    )

    assert summary["Basic Memory"]["accuracy"] == 0.5
    assert summary["Basic Memory"]["accuracy_valid"] == 1.0
    assert summary["Temporal Reasoning"]["accuracy"] == 0.5
