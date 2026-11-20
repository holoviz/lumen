try:
    import pydantic_evals  # noqa

    import lumen.ai  # noqa
except ModuleNotFoundError:
    # Without the AI dependencies (the test-core environment) there is nothing to evaluate.
    collect_ignore_glob = ["*.py"]
