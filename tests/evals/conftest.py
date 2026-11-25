try:
    import pydantic_evals  # noqa

    import lumen.ai  # noqa
except ModuleNotFoundError:
    # Without the AI dependencies (the test-core environment) there is nothing to evaluate.
    collect_ignore_glob = ["*.py"]
else:
    from importlib.util import find_spec

    if find_spec("panel_flowdash") is None:
        # The FlowDash subject needs panel-flowdash, which needs Python 3.12.
        collect_ignore = ["test_flowdash_subject.py", "test_followup_cases.py", "test_scenarios.py"]
