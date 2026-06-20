from tools.preflight_memory_eval import _check_memobase, _valid_http_url


def test_preflight_url_validation():
    assert _valid_http_url("https://gateway.example/v1") is True
    assert _valid_http_url("$OPENAI_BASE_URL") is False
    assert _valid_http_url("") is False


def test_memobase_preflight_rejects_invalid_url_without_network():
    assert _check_memobase("$MEMOBASE_PROJECT_URL", timeout=0.01) == {
        "ok": False,
        "error": "invalid project_url",
    }
