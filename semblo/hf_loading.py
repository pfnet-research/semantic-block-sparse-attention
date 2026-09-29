from __future__ import annotations

from typing import Any


def load_auto_tokenizer(
    model_id: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a tokenizer with ``AutoTokenizer``.

    Network and cache behaviour follow the standard Hugging Face arguments and
    environment variables.
    """

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def load_auto_processor(
    model_id: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
