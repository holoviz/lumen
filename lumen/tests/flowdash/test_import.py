import importlib.metadata
import subprocess
import sys

import pytest

from packaging.requirements import Requirement


def test_panel_flowdash_with_its_agent_extra_is_a_core_dependency():
    requirements = [Requirement(r) for r in importlib.metadata.requires("lumen") or []]
    flowdash = [r for r in requirements if r.name == "panel-flowdash"]

    assert len(flowdash) == 1
    assert flowdash[0].marker is None
    assert flowdash[0].extras == {"agent"}


def test_lumen_does_not_import_flowdash_or_agent():
    pytest.importorskip("lumen.ai")
    pytest.importorskip("pydantic_evals")
    code = (
        "import sys, lumen, lumen.ai, lumen.ai.evals.harness; "
        "print(sorted(m for m in sys.modules if m.split('.')[:2] in (['lumen', 'flowdash'], ['lumen', 'agent']) "
        "or m.startswith('panel_flowdash')))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
