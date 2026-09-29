from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from semblo.config import ModelSpec
from semblo.io import sha256_json, sha256_text

Message = dict[str, str]


@dataclass(frozen=True)
class PromptRender:
    messages: list[Message]
    rendered_prompt: str | None
    rendered_prompt_sha256: str | None
    rendered_prompt_chars: int | None
    request_sha256: str
    message_roles: list[str]
    prompt_tokens_estimate: int | None
    renderer: str
    chat_template_kwargs: dict[str, Any]
    system_prompt_applied: bool
    token_ids: list[int] | None = None
    prompt_preview: str | None = None
    extra: dict[str, Any] | None = None

    @property
    def text(self) -> str | None:
        return self.rendered_prompt

    @property
    def token_count(self) -> int | None:
        return self.prompt_tokens_estimate

    def metadata(self) -> dict[str, Any]:
        return self.to_record(include_text=False, include_token_ids=False)

    def to_record(self, include_text: bool = False, include_token_ids: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_text:
            data.pop("rendered_prompt", None)
        if not include_token_ids:
            data.pop("token_ids", None)
        return data

    def audit_record(self, include_preview: bool = True, include_text: bool = False) -> dict[str, Any]:
        data = self.to_record(include_text=include_text, include_token_ids=False)
        if not include_preview:
            data.pop("prompt_preview", None)
        return data


def preview_text(text: str | None, limit: int = 2000) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...<truncated {len(text) - limit} chars>"


prompt_preview = preview_text


def truncate_prompt_render(
    prompt_render: PromptRender,
    *,
    tokenizer: Any,
    max_prompt_tokens: int,
    policy: str = "middle",
) -> PromptRender:
    """Shorten a rendered prompt to at most ``max_prompt_tokens`` tokens.

    The ``middle`` policy follows LongBench: it keeps the first and last halves of
    the rendered prompt and removes the middle, so the instruction and the final
    question are kept. The truncation is recorded in ``extra``.
    """
    if max_prompt_tokens < 1:
        raise ValueError(f"max_prompt_tokens must be >= 1, got {max_prompt_tokens}")
    if policy != "middle":
        raise ValueError(f"Unknown prompt truncation policy: {policy}")
    token_ids = prompt_render.token_ids
    if token_ids is None:
        raise ValueError("Prompt truncation requires token_ids from the tokenizer")
    original_tokens = len(token_ids)
    if original_tokens <= max_prompt_tokens:
        return prompt_render

    head = max_prompt_tokens // 2
    tail = max_prompt_tokens - head
    truncated_ids = [int(x) for x in token_ids[:head]] + [int(x) for x in token_ids[-tail:]]
    rendered_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    extra = dict(prompt_render.extra or {})
    extra["prompt_truncation"] = {
        "policy": policy,
        "original_prompt_tokens": original_tokens,
        "max_prompt_tokens": int(max_prompt_tokens),
        "truncated_prompt_tokens": len(truncated_ids),
        "removed_tokens": original_tokens - len(truncated_ids),
        "head_tokens": head,
        "tail_tokens": tail,
    }
    return make_prompt_render(
        messages=prompt_render.messages,
        rendered_prompt=rendered_prompt,
        prompt_tokens_estimate=len(truncated_ids),
        renderer=f"{prompt_render.renderer}+truncate_{policy}",
        chat_template_kwargs=prompt_render.chat_template_kwargs,
        system_prompt_applied=prompt_render.system_prompt_applied,
        token_ids=truncated_ids,
        request_payload={
            "messages": prompt_render.messages,
            "truncation": extra["prompt_truncation"],
            "request_sha256_before_truncation": prompt_render.request_sha256,
        },
        extra=extra,
    )


def append_prompt_prefill(
    prompt_render: PromptRender,
    *,
    tokenizer: Any,
    prefill: str,
) -> PromptRender:
    """Append text to a rendered prompt as the start of the assistant turn.

    If scorers should see the prefill as part of the response, the caller must also
    prepend it to the saved completion.
    """
    if not prefill:
        return prompt_render
    token_ids = prompt_render.token_ids
    if token_ids is None:
        raise ValueError("Prompt prefill requires token_ids from the tokenizer")
    prefill_ids = tokenizer.encode(prefill, add_special_tokens=False)
    updated_ids = [int(x) for x in token_ids] + [int(x) for x in prefill_ids]
    rendered_prompt = (prompt_render.rendered_prompt or "") + prefill
    extra = dict(prompt_render.extra or {})
    extra["assistant_prefill"] = {
        "text": prefill,
        "tokens": len(prefill_ids),
    }
    return make_prompt_render(
        messages=prompt_render.messages,
        rendered_prompt=rendered_prompt,
        prompt_tokens_estimate=len(updated_ids),
        renderer=f"{prompt_render.renderer}+assistant_prefill",
        chat_template_kwargs=prompt_render.chat_template_kwargs,
        system_prompt_applied=prompt_render.system_prompt_applied,
        token_ids=updated_ids,
        request_payload={
            "messages": prompt_render.messages,
            "assistant_prefill": prefill,
            "request_sha256_before_prefill": prompt_render.request_sha256,
        },
        extra=extra,
    )


def _load_system_prompt(model_spec: ModelSpec) -> str | None:
    if model_spec.system_prompt is not None:
        return model_spec.system_prompt
    path = model_spec.system_prompt_path or getattr(model_spec, "system_prompt_file", None)
    if path:
        return Path(path).read_text(encoding="utf-8")
    return None


def apply_system_prompt(
    messages: list[Message],
    system_prompt: str | ModelSpec | None,
    policy: str = "prepend_if_missing",
    role: str = "system",
) -> tuple[list[Message], bool]:
    if isinstance(system_prompt, ModelSpec):
        model_spec = system_prompt
        system_prompt = _load_system_prompt(model_spec)
        policy = model_spec.system_prompt_policy
        role = model_spec.system_prompt_role
    prepared = [dict(m) for m in messages]
    if not system_prompt or policy in {"none", "never", "off", "false"}:
        return prepared, False

    system_message = {"role": role, "content": system_prompt}
    system_indices = [i for i, m in enumerate(prepared) if m.get("role") == role]

    if policy in {"prepend_if_missing", "if_missing", "prepend_if_absent"}:
        if system_indices:
            return prepared, False
        return [system_message, *prepared], True
    if policy == "prepend":
        return [system_message, *prepared], True
    if policy in {"replace", "replace_or_prepend"}:
        without_system = [m for m in prepared if m.get("role") != role]
        return [system_message, *without_system], True
    if policy in {"append_to_existing", "append_to_first"}:
        if not system_indices:
            return [system_message, *prepared], True
        first_idx = system_indices[0]
        updated = list(prepared)
        old = updated[first_idx].get("content", "")
        updated[first_idx] = {"role": role, "content": old.rstrip() + "\n\n" + system_prompt}
        return updated, True

    raise ValueError(
        "system_prompt_policy must be one of: never, none, prepend_if_missing, prepend_if_absent, if_missing, prepend, replace, replace_or_prepend, append_to_existing, append_to_first"
    )


def _coerce_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Expected one rendered prompt, got a batch of {len(value)}")
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Tokenized chat template returned {type(value)!r}; expected list[int]")
    return [int(x) for x in value]


def _tokenize_chat_template(tokenizer: Any, messages: list[Message], model_spec: ModelSpec) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=model_spec.add_generation_prompt,
        **model_spec.chat_template_kwargs,
    )
    return _coerce_token_ids(tokenized)


def _messages_to_raw_text(messages: list[Message]) -> str:
    if len(messages) == 1:
        return str(messages[0].get("content", ""))
    return "\n\n".join(str(message.get("content", "")) for message in messages)


def _tokenize_raw_text(tokenizer: Any, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=True)
    return _coerce_token_ids(tokenized)


def make_prompt_render(
    *,
    messages: list[Message],
    rendered_prompt: str | None,
    prompt_tokens_estimate: int | None,
    renderer: str,
    chat_template_kwargs: dict[str, Any],
    system_prompt_applied: bool,
    token_ids: list[int] | None = None,
    request_payload: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> PromptRender:
    request_obj = request_payload if request_payload is not None else messages
    return PromptRender(
        messages=messages,
        rendered_prompt=rendered_prompt,
        rendered_prompt_sha256=sha256_text(rendered_prompt) if rendered_prompt is not None else None,
        rendered_prompt_chars=len(rendered_prompt) if rendered_prompt is not None else None,
        request_sha256=sha256_json(request_obj),
        message_roles=[str(m.get("role")) for m in messages],
        prompt_tokens_estimate=prompt_tokens_estimate,
        renderer=renderer,
        chat_template_kwargs=dict(chat_template_kwargs),
        system_prompt_applied=system_prompt_applied,
        token_ids=token_ids,
        prompt_preview=preview_text(rendered_prompt),
        extra=extra or {},
    )


def render_prompt(tokenizer: Any, messages: list[Message], model_spec: ModelSpec) -> PromptRender:
    system_prompt = _load_system_prompt(model_spec)
    prepared, applied = apply_system_prompt(
        messages,
        system_prompt,
        model_spec.system_prompt_policy,
        role=model_spec.system_prompt_role,
    )
    if model_spec.prompt_format in {"raw", "raw_text", "plain_text"}:
        prompt = _messages_to_raw_text(prepared)
        token_ids = _tokenize_raw_text(tokenizer, prompt)
        return make_prompt_render(
            messages=prepared,
            rendered_prompt=prompt,
            prompt_tokens_estimate=len(token_ids),
            renderer="tokenizer.raw_text",
            chat_template_kwargs=model_spec.chat_template_kwargs,
            system_prompt_applied=applied,
            token_ids=token_ids,
            request_payload={"messages": prepared, "prompt_format": model_spec.prompt_format},
            extra={"prompt_format": model_spec.prompt_format, "add_special_tokens": True},
        )

    prompt = tokenizer.apply_chat_template(
        prepared,
        tokenize=False,
        add_generation_prompt=model_spec.add_generation_prompt,
        **model_spec.chat_template_kwargs,
    )
    if not isinstance(prompt, str):
        raise TypeError(f"apply_chat_template(tokenize=False) returned {type(prompt)!r}; expected str")
    token_ids = _tokenize_chat_template(tokenizer, prepared, model_spec)
    return make_prompt_render(
        messages=prepared,
        rendered_prompt=prompt,
        prompt_tokens_estimate=len(token_ids) if token_ids is not None else None,
        renderer="tokenizer.apply_chat_template",
        chat_template_kwargs=model_spec.chat_template_kwargs,
        system_prompt_applied=applied,
        token_ids=token_ids,
        request_payload=prepared,
        extra={"add_generation_prompt": model_spec.add_generation_prompt},
    )


def render_chat_prompt(tokenizer: Any, messages: list[Message], model_spec: ModelSpec) -> tuple[str, list[Message], bool]:
    bundle = render_prompt(tokenizer, messages, model_spec)
    if bundle.rendered_prompt is None:
        raise ValueError("render_chat_prompt unexpectedly produced no prompt text")
    return bundle.rendered_prompt, bundle.messages, bundle.system_prompt_applied



def tokenizer_prompt_metadata(tokenizer: Any, model_spec: ModelSpec) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template_sha256": sha256_text(chat_template) if isinstance(chat_template, str) else None,
        "chat_template_preview": preview_text(chat_template, 2000) if isinstance(chat_template, str) else None,
        "chat_template_kwargs": dict(model_spec.chat_template_kwargs),
        "system_prompt_sha256": sha256_text(_load_system_prompt(model_spec)),
        "system_prompt_policy": model_spec.system_prompt_policy,
        "system_prompt_role": model_spec.system_prompt_role,
        "prompt_format": model_spec.prompt_format,
        "add_generation_prompt": model_spec.add_generation_prompt,
    }

# Used by the CLI. Maps the policy aliases to their canonical names.
def prepare_messages(messages: list[Message], model_spec: ModelSpec) -> list[Message]:
    system_prompt = _load_system_prompt(model_spec)
    policy = model_spec.system_prompt_policy
    if policy == "prepend_if_absent":
        policy = "prepend_if_missing"
    elif policy == "replace_or_prepend":
        policy = "replace"
    elif policy == "append_to_first":
        policy = "append_to_existing"
    prepared, _ = apply_system_prompt(messages, system_prompt, policy, role=model_spec.system_prompt_role)
    return prepared
