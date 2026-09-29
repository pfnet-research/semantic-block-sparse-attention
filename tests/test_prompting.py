from collections import UserDict

from semblo.config import ModelSpec
from semblo.prompting import apply_system_prompt


def test_prepend_if_absent_adds_system_once():
    spec = ModelSpec(key="x", hf_id="dummy", system_prompt="Be concise.", system_prompt_policy="prepend_if_absent")
    messages = [{"role": "user", "content": "hello"}]
    prepared, applied = apply_system_prompt(messages, spec)
    assert applied is True
    assert prepared[0] == {"role": "system", "content": "Be concise."}
    prepared2, applied2 = apply_system_prompt(prepared, spec)
    assert applied2 is False
    assert prepared2 == prepared


def test_replace_or_prepend_replaces_first_system():
    spec = ModelSpec(key="x", hf_id="dummy", system_prompt="New.", system_prompt_policy="replace_or_prepend")
    messages = [{"role": "system", "content": "Old."}, {"role": "user", "content": "hello"}]
    prepared, applied = apply_system_prompt(messages, spec)
    assert applied is True
    assert prepared == [{"role": "system", "content": "New."}, {"role": "user", "content": "hello"}]


def test_none_does_not_insert_system_prompt():
    spec = ModelSpec(key="x", hf_id="dummy", system_prompt="Ignored.", system_prompt_policy="none")
    messages = [{"role": "user", "content": "hello"}]
    prepared, applied = apply_system_prompt(messages, spec)
    assert applied is False
    assert prepared == messages

class FakeChatTokenizer:
    chat_template = "fake-template"
    name_or_path = "fake"
    vocab_size = 10
    model_max_length = 100
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.calls.append({"tokenize": tokenize, "add_generation_prompt": add_generation_prompt, "kwargs": kwargs})
        if tokenize:
            return [101, 102, 103, 104]
        return "<user>hello<assistant>"


def test_render_prompt_uses_chat_template_token_ids_for_count():
    from semblo.prompting import render_prompt

    tokenizer = FakeChatTokenizer()
    spec = ModelSpec(
        key="x",
        hf_id="dummy",
        system_prompt_policy="none",
        chat_template_kwargs={"enable_thinking": False},
    )
    rendered = render_prompt(tokenizer, [{"role": "user", "content": "hello"}], spec)
    assert rendered.rendered_prompt == "<user>hello<assistant>"
    assert rendered.token_ids == [101, 102, 103, 104]
    assert rendered.token_count == 4
    assert tokenizer.calls == [
        {"tokenize": False, "add_generation_prompt": True, "kwargs": {"enable_thinking": False}},
        {"tokenize": True, "add_generation_prompt": True, "kwargs": {"enable_thinking": False}},
    ]


def test_render_prompt_accepts_mapping_token_ids():
    from semblo.prompting import render_prompt

    class MappingTokenizer(FakeChatTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            if tokenize:
                return UserDict({"input_ids": [201, 202, 203]})
            return "<user>hello<assistant>"

    rendered = render_prompt(
        MappingTokenizer(),
        [{"role": "user", "content": "hello"}],
        ModelSpec(key="x", hf_id="dummy", system_prompt_policy="none"),
    )
    assert rendered.token_ids == [201, 202, 203]
    assert rendered.token_count == 3


def test_render_prompt_supports_raw_text_without_chat_template():
    from semblo.prompting import render_prompt

    class RawTokenizer:
        chat_template = None
        name_or_path = "raw"
        vocab_size = 10
        model_max_length = 100
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def __init__(self):
            self.calls = []

        def __call__(self, text, add_special_tokens=True):
            self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
            return {"input_ids": [1, 42, 2]}

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("raw_text prompt_format must not call apply_chat_template")

    tokenizer = RawTokenizer()
    rendered = render_prompt(
        tokenizer,
        [{"role": "user", "content": "Answer A."}],
        ModelSpec(key="x", hf_id="dummy", prompt_format="raw_text", system_prompt_policy="none"),
    )
    assert rendered.rendered_prompt == "Answer A."
    assert rendered.token_ids == [1, 42, 2]
    assert rendered.token_count == 3
    assert rendered.renderer == "tokenizer.raw_text"
    assert tokenizer.calls == [{"text": "Answer A.", "add_special_tokens": True}]


def test_truncate_prompt_render_middle_keeps_head_and_tail():
    from semblo.prompting import make_prompt_render, truncate_prompt_render

    class DecodeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            return " ".join(str(x) for x in token_ids)

    rendered = make_prompt_render(
        messages=[{"role": "user", "content": "x"}],
        rendered_prompt="original",
        prompt_tokens_estimate=10,
        renderer="unit",
        chat_template_kwargs={},
        system_prompt_applied=False,
        token_ids=list(range(10)),
    )
    truncated = truncate_prompt_render(rendered, tokenizer=DecodeTokenizer(), max_prompt_tokens=6)
    assert truncated.token_ids == [0, 1, 2, 7, 8, 9]
    assert truncated.token_count == 6
    assert truncated.rendered_prompt == "0 1 2 7 8 9"
    assert truncated.extra["prompt_truncation"]["original_prompt_tokens"] == 10
    assert truncated.extra["prompt_truncation"]["removed_tokens"] == 4
