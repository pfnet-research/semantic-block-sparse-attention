from __future__ import annotations

import sys
import types

from semblo.hf_loading import load_auto_processor, load_auto_tokenizer


def test_load_auto_tokenizer_forwards_explicit_arguments(monkeypatch):
    module = types.ModuleType("transformers")
    calls = []

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "tokenizer"

    module.AutoTokenizer = AutoTokenizer
    module.AutoProcessor = object
    monkeypatch.setitem(sys.modules, "transformers", module)

    result = load_auto_tokenizer(
        "org/model",
        revision="rev1",
        trust_remote_code=False,
        local_files_only=True,
    )

    assert result == "tokenizer"
    assert calls == [
        (
            ("org/model",),
            {
                "revision": "rev1",
                "trust_remote_code": False,
                "local_files_only": True,
            },
        )
    ]


def test_load_auto_processor_forwards_explicit_arguments(monkeypatch):
    module = types.ModuleType("transformers")
    calls = []

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "processor"

    module.AutoTokenizer = object
    module.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", module)

    result = load_auto_processor(
        "org/model",
        revision=None,
        trust_remote_code=True,
        use_fast=True,
    )

    assert result == "processor"
    assert calls == [
        (
            ("org/model",),
            {
                "revision": None,
                "trust_remote_code": True,
                "use_fast": True,
            },
        )
    ]
