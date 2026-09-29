import json
from dataclasses import replace

import pytest

from semblo.backends.base import GenerationOutput
from semblo.config import GenerationConfig, ModelSpec
from semblo.prompting import make_prompt_render
from semblo.tasks.base import Example
from semblo.writer import RunWriter


def test_writer_records_prompt_chat_template_kwargs(tmp_path):
    prompt = make_prompt_render(
        messages=[{"role": "user", "content": "hello"}],
        rendered_prompt="<user>hello<assistant>",
        prompt_tokens_estimate=3,
        renderer="unit",
        chat_template_kwargs={"enable_thinking": True},
        system_prompt_applied=False,
        token_ids=[1, 2, 3],
    )
    writer = RunWriter(
        tmp_path,
        run_config={},
        model_spec=ModelSpec(key="m", hf_id="dummy", chat_template_kwargs={"enable_thinking": True}),
        generation=GenerationConfig(max_new_tokens=8),
    )
    writer.write(
        Example(id="ex1", task_name="unit", messages=[{"role": "user", "content": "hello"}], answer="A"),
        GenerationOutput(text="answer", prompt_render=prompt, sampling_params={"max_tokens": 8}),
        {"score": 1.0},
        model_key="m",
        run_seed=0,
        base_seed=0,
        seed_policy="fixed",
    )

    record = json.loads((tmp_path / "examples.jsonl").read_text(encoding="utf-8"))
    assert record["prompt_chat_template_kwargs"] == {"enable_thinking": True}


def _writer(path, *, model=None, generation=None, config=None, resume=False):
    return RunWriter(path, config or {}, model or ModelSpec(key="m", hf_id="dummy"),
                     generation or GenerationConfig(), resume=resume)


def test_reusing_output_requires_resume_and_preserves_files(tmp_path):
    _writer(tmp_path)
    manifest = (tmp_path / "run_config.json").read_bytes()
    examples = tmp_path / "examples.jsonl"
    examples.write_text('{"example_id":"existing"}\n')
    with pytest.raises(FileExistsError, match="--resume"):
        _writer(tmp_path)
    assert (tmp_path / "run_config.json").read_bytes() == manifest
    _writer(tmp_path, resume=True)
    assert (tmp_path / "run_config.json").read_bytes() == manifest
    assert examples.read_text() == '{"example_id":"existing"}\n'


@pytest.mark.parametrize("change", ["model", "generation", "benchmark", "runtime"])
def test_resume_rejects_changed_experiment(tmp_path, change):
    model = ModelSpec(key="m", hf_id="dummy")
    generation = GenerationConfig()
    config = {"benchmark_config": {"task": "one"}, "runtime_environment": {"SEMBLO_TARGET_TOKENS": "4096"}}
    _writer(tmp_path, model=model, generation=generation, config=config)
    manifest = (tmp_path / "run_config.json").read_bytes()
    if change == "model":
        model = replace(model, hf_id="another-model")
    elif change == "generation":
        generation = replace(generation, temperature=0.5)
    elif change == "benchmark":
        config = {**config, "benchmark_config": {"task": "two"}}
    else:
        config = {**config, "runtime_environment": {"SEMBLO_TARGET_TOKENS": "8192"}}
    with pytest.raises(ValueError, match="settings differ"):
        _writer(tmp_path, model=model, generation=generation, config=config, resume=True)
    assert (tmp_path / "run_config.json").read_bytes() == manifest


def test_resume_without_manifest_is_rejected(tmp_path):
    (tmp_path / "examples.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="missing"):
        _writer(tmp_path, resume=True)


def test_writer_redacts_model_credentials(tmp_path):
    _writer(tmp_path, model=ModelSpec(key="m", hf_id="dummy", env={"HF_TOKEN": "dummy-secret"}))
    assert "dummy-secret" not in (tmp_path / "run_config.json").read_text()
