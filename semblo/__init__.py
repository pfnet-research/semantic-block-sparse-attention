__all__ = ["__version__"]
__version__ = "0.1.0"

# vLLM's YaRN on multimodal RoPE is checked against Transformers the first time it is used, and corrected if it is
# off (semblo/mrope_yarn_fix.py).
from semblo import mrope_yarn_fix as _mrope_yarn_fix  # noqa: E402

_mrope_yarn_fix.install()
