import json

from semblo.run_metadata import collect_run_metadata, redact_sensitive_fields, runtime_environment


def test_runtime_credentials_are_not_written(tmp_path, monkeypatch):
    from semblo.lm_eval_runner import _write_run_environment
    from scripts.eval_openai_mrcr import _run_environment

    monkeypatch.setenv("VLLM_API_KEY", "dummy-server-key")
    monkeypatch.setenv("SEMBLO_AUTH_TOKEN", "dummy-token")
    monkeypatch.setenv("SEMBLO_TARGET_TOKENS", "4096")
    monkeypatch.setenv("SEMBLO_TOKEN_OUTER_FIT_TOKENS", "65536")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    _write_run_environment(tmp_path, "model")
    records = [json.loads((tmp_path / "run_environment.json").read_text()),
               _run_environment(), collect_run_metadata()]
    for record in records:
        text = json.dumps(record)
        assert "dummy-server-key" not in text and "dummy-token" not in text
        assert "hostname" not in record
        assert record["environment"]["SEMBLO_TARGET_TOKENS"] == "4096"
        assert record["environment"]["SEMBLO_TOKEN_OUTER_FIT_TOKENS"] == "65536"
        assert record["environment"]["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert "VLLM_API_KEY" not in runtime_environment()


def test_nested_credentials_are_redacted_without_changing_token_budgets():
    settings = {"env": {"HF_TOKEN": "dummy"},
                "engine": {"api_key": "dummy", "max_tokens": 1024},
                "items": [{"password": "dummy"}], "target_tokens": 4096}
    safe = redact_sensitive_fields(settings)
    assert "dummy" not in json.dumps(safe)
    assert safe["engine"]["max_tokens"] == 1024 and safe["target_tokens"] == 4096
    assert settings["env"]["HF_TOKEN"] == "dummy"
