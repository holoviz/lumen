"""Optional evaluation tools for Lumen agents."""

try:
    import pydantic_evals  # noqa: F401
except ImportError as exc:
    raise ImportError(
        'lumen.ai.evals requires pydantic-evals; install it with `pip install "lumen[evals]"`.'
    ) from exc
