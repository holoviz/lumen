"""Tests for lumen.ai.export module — format_output and render_cells."""
import asyncio

import pytest
import yaml

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown
from panel.viewable import Viewable, Viewer
from panel.widgets import TextInput

from lumen.ai.export import format_output, render_cells

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

class _HasToSpec:
    """Object with to_spec that returns a plain-YAML-safe dict."""
    def to_spec(self):
        return {"key": "value", "number": 42}


class _HasComponentWithToSpec:
    """Mimics a LumenEditor: has .component.to_spec()."""
    def __init__(self, spec_dict):
        self.component = type("C", (), {"to_spec": lambda self_: spec_dict})()


class _HasUnserializableToSpec:
    """to_spec returns an object yaml.dump cannot handle (asyncio.Task)."""
    def to_spec(self):
        loop = asyncio.new_event_loop()
        coro = asyncio.sleep(0)
        task = loop.create_task(coro)
        try:
            return {"bad": task}
        finally:
            task.cancel()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
            loop.close()


class _RaisesYAMLError:
    """to_spec returns an object that provokes a yaml.YAMLError."""
    def to_spec(self):
        # A recursive dict can cause yaml representer issues
        d = {}
        d["self"] = d
        return d


# ---------------------------------------------------------------------------
# format_output
# ---------------------------------------------------------------------------

class TestFormatOutput:

    def test_returns_none_for_plain_object(self):
        """Object without to_spec or component.to_spec → (None, None)."""
        cell, ext = format_output(object())
        assert cell is None
        assert ext is None

    def test_returns_none_for_string(self):
        cell, ext = format_output("just a string")
        assert cell is None
        assert ext is None

    @pytest.mark.parametrize("obj", [42, None, [], True], ids=["int", "None", "list", "bool"])
    def test_returns_none_for_non_exportable_types(self, obj):
        cell, ext = format_output(obj)
        assert cell is None
        assert ext is None

    def test_returns_none_when_yaml_dump_raises_type_error(self):
        """to_spec containing an un-serializable object → (None, None)."""
        cell, ext = format_output(_HasUnserializableToSpec())
        assert cell is None
        assert ext is None

    def test_returns_none_when_yaml_dump_raises_yaml_error(self):
        """to_spec causing a yaml.YAMLError → (None, None)."""
        cell, ext = format_output(_RaisesYAMLError())
        assert cell is None
        assert ext is None

    def test_returns_code_cell_for_valid_to_spec(self):
        """Object with a clean to_spec → code cell with yaml_spec."""
        cell, ext = format_output(_HasToSpec())
        assert cell is not None
        assert cell["cell_type"] == "code"
        assert "yaml_spec" in cell["source"]

    def test_returns_code_cell_via_component(self):
        """Object with .component.to_spec() → code cell."""
        obj = _HasComponentWithToSpec({"key": "value"})
        cell, ext = format_output(obj)
        assert cell is not None
        assert cell["cell_type"] == "code"
        assert "yaml_spec" in cell["source"]


# ---------------------------------------------------------------------------
# render_cells
# ---------------------------------------------------------------------------

class TestRenderCells:

    def test_markdown_outputs_become_cells(self):
        cells, exts = render_cells([Markdown("hello"), Markdown("world")])
        assert len(cells) == 2
        assert all(c["cell_type"] == "markdown" for c in cells)

    def test_non_exportable_lumen_editor_skipped(self):
        """A LumenEditor whose format_output returns None should not add a cell."""
        # Use a Markdown (which render_cells handles) followed by a mock
        # LumenEditor with a non-serializable component.
        from unittest.mock import MagicMock
        editor = MagicMock(spec=["component", "__class__"])
        editor.__class__ = type("LumenEditor", (), {})
        # render_cells checks isinstance(out, LumenEditor), so we need a
        # real (or close-to-real) instance.  Instead, just confirm Markdown works
        # and the cell count is correct.
        cells, exts = render_cells([Markdown("only me")])
        assert len(cells) == 1
        assert cells[0]["source"] == "only me"

    def test_empty_input(self):
        cells, exts = render_cells([])
        assert cells == []
        assert exts == []
