import subprocess
import sys

import pytest

BLOCK = "import sys; sys.modules[{module!r}] = None; "


@pytest.mark.parametrize("package, missing, message", [
    ("lumen.flowdash", "panel_flowdash", "lumen.flowdash needs panel-flowdash"),
    ("lumen.agent", "panel_agents", "lumen.agent needs panel-flowdash with its agent extra"),
])
def test_import_fails_clearly_without_optional_dependency(package, missing, message):
    code = BLOCK.format(module=missing) + f"import {package}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert message in result.stderr


def test_lumen_does_not_import_flowdash_or_agent():
    code = (
        "import sys, lumen, lumen.ai, lumen.ai.evals.harness; "
        "print(sorted(m for m in sys.modules if m.split('.')[:2] in (['lumen', 'flowdash'], ['lumen', 'agent']) "
        "or m.startswith('panel_flowdash')))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
