"""Tests for explain controls."""
from unittest.mock import MagicMock

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls.explain import SPEC_TYPE_MAP, ExplainControls


class MockView:
    """Mock view object for testing."""

    def __init__(self, spec="SELECT * FROM users", language="sql"):
        self.spec = spec
        self.language = language


class MockTask:
    """Mock task object for testing."""

    def __init__(self):
        self.actor = MagicMock()
        self.actor.llm = MagicMock()


class MockInterface:
    """Mock interface object for testing."""

    def __init__(self):
        self.messages = []
        self._last_message = None

    def stream(self, content, user="Assistant", replace=False, message=None):
        """Mock stream method that mimics ChatFeed.stream behavior."""
        msg = {"content": content, "user": user, "replace": replace}
        if replace and message is not None:
            self._last_message = msg
        else:
            self._last_message = msg
        self.messages.append(msg)
        return self._last_message


class TestExplainControls:
    """Tests for ExplainControls class."""

    def test_initialization(self):
        """Test ExplainControls initialization."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        controls = ExplainControls(
            interface=interface,
            view=view,
            task=task,
        )

        assert controls.interface is interface
        assert controls.view is view
        assert controls.task is task
        assert controls.active is False
        assert controls._explain_count == 0

    def test_class_kwargs(self):
        """Test ExplainControls has correct class-level kwargs."""
        assert ExplainControls.toggle_kwargs["icon"] == "psychology"
        assert "Explain" in ExplainControls.toggle_kwargs["description"]
        assert "blank" in ExplainControls.input_kwargs["placeholder"].lower()

    def test_text_input_created(self):
        """Test that ExplainControls creates a TextInput via RevisionControls."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        controls = ExplainControls(
            interface=interface,
            view=view,
            task=task,
        )

        assert hasattr(controls, "_text_input")
        assert controls._text_input.max_length == 200

    def test_panel_returns_row(self):
        """Test __panel__ returns the row component."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        controls = ExplainControls(
            interface=interface,
            view=view,
            task=task,
        )

        panel = controls.__panel__()
        assert panel is controls._row

    def test_layout_kwargs_applied_to_row(self):
        """Test that layout_kwargs are applied to the Row component."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        layout_kwargs = {"margin": (20, 10), "sizing_mode": "fixed", "width": 200}

        controls = ExplainControls(
            interface=interface,
            view=view,
            task=task,
            layout_kwargs=layout_kwargs,
        )

        row = controls._row
        assert row.margin == (20, 10)
        assert row.sizing_mode == "fixed"
        assert row.width == 200

    def test_multiple_instances_independent(self):
        """Test that multiple ExplainControls instances are independent."""
        interface = MockInterface()
        view1 = MockView(spec="SELECT 1")
        view2 = MockView(spec="SELECT 2")
        task1 = MockTask()
        task2 = MockTask()

        controls1 = ExplainControls(interface=interface, view=view1, task=task1)
        controls2 = ExplainControls(interface=interface, view=view2, task=task2)

        assert controls1._row is not controls2._row
        assert controls1._row.objects[0] is not controls2._row.objects[0]

    @pytest.mark.asyncio
    async def test_enter_reason_blank_triggers_full_explain(self):
        """Test _enter_reason with blank input sets empty _focus and increments count."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        async def mock_stream(**kwargs):
            yield "ok"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls.active = True
        controls._text_input.value_input = ""

        controls._enter_reason(None)

        assert controls._focus == ""
        assert controls.active is False
        assert controls._explain_count == 1
        assert controls._text_input.value == ""

    @pytest.mark.asyncio
    async def test_enter_reason_with_focus_text(self):
        """Test _enter_reason with specific focus text."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        async def mock_stream(**kwargs):
            yield "ok"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls.active = True
        controls._text_input.value_input = "  WHERE clause  "

        controls._enter_reason(None)

        assert controls._focus == "WHERE clause"
        assert controls.active is False
        assert controls._explain_count == 1

    @pytest.mark.asyncio
    async def test_enter_reason_increments_count(self):
        """Test _enter_reason increments _explain_count on each call."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        async def mock_stream(**kwargs):
            yield "ok"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._text_input.value_input = ""

        controls._enter_reason(None)
        assert controls._explain_count == 1

        controls._text_input.value_input = "JOIN"
        controls._enter_reason(None)
        assert controls._explain_count == 2

    @pytest.mark.asyncio
    async def test_explain_streams_progressively(self):
        """Test that _explain streams chunks progressively to the interface."""
        interface = MockInterface()
        view = MockView(spec="SELECT id FROM users")
        task = MockTask()

        chunks = ["This query", "This query selects", "This query selects all IDs from the users table."]

        async def mock_stream(**kwargs):
            for chunk in chunks:
                yield chunk

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        assert len(interface.messages) == len(chunks)
        for msg in interface.messages:
            assert msg["replace"] is True

    @pytest.mark.asyncio
    async def test_explain_empty_spec_noop(self):
        """Test _explain does nothing with empty spec."""
        interface = MockInterface()
        view = MockView(spec="")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_none_spec_noop(self):
        """Test _explain does nothing with None spec."""
        interface = MockInterface()
        view = MockView(spec=None)
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_whitespace_only_spec_noop(self):
        """Test _explain does nothing with whitespace-only spec."""
        interface = MockInterface()
        view = MockView(spec="   \n  ")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_handles_llm_error(self):
        """Test _explain handles LLM errors gracefully."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()

        async def mock_stream(**kwargs):
            raise Exception("LLM connection failed")
            yield  # make it a generator

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        assert len(interface.messages) == 1
        card = interface.messages[0]["content"]
        assert hasattr(card, "title")
        assert "Failed to generate explanation" in card.title

    @pytest.mark.asyncio
    async def test_explain_error_markdown_uses_real_newlines(self):
        """Test that error markdown uses actual newlines, not literal backslash-n."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()

        async def mock_stream(**kwargs):
            raise Exception("Some error")
            yield

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        card = interface.messages[0]["content"]
        md_object = card.objects[0]
        md_text = md_object.object

        assert "\\n" not in md_text, (
            "Error markdown contains literal backslash-n instead of real newlines"
        )
        assert md_text.startswith("```\n")
        assert md_text.endswith("\n```")

    @pytest.mark.asyncio
    async def test_explain_passes_correct_messages_to_llm(self):
        """Test that the correct messages and system prompt are passed to LLM."""
        interface = MockInterface()
        view = MockView(spec="SELECT name FROM users WHERE active = 1", language="sql")
        task = MockTask()

        captured_kwargs = {}

        async def mock_stream(**kwargs):
            captured_kwargs.update(kwargs)
            yield "explanation"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        # Verify messages — user content should be minimal (spec is in system prompt context block)
        assert len(captured_kwargs["messages"]) == 1
        msg = captured_kwargs["messages"][0]
        assert msg["role"] == "user"
        assert "Explain the spec above" in msg["content"]

        # Verify system prompt contains the spec and language info
        assert "system" in captured_kwargs
        system = captured_kwargs["system"]
        assert "SELECT name FROM users WHERE active = 1" in system
        assert "sql" in system.lower()

    @pytest.mark.asyncio
    async def test_explain_with_focus_passes_focus_to_llm(self):
        """Test that focused explanation passes focus text in user message."""
        interface = MockInterface()
        view = MockView(spec="SELECT * FROM t1 JOIN t2 ON t1.id = t2.id", language="sql")
        task = MockTask()

        captured_kwargs = {}

        async def mock_stream(**kwargs):
            captured_kwargs.update(kwargs)
            yield "The JOIN clause..."

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = "JOIN clause"
        await controls._explain()

        msg = captured_kwargs["messages"][0]
        assert "Focus on: JOIN clause" in msg["content"]

        # System prompt should also include focus context
        system = captured_kwargs["system"]
        assert "JOIN clause" in system

    @pytest.mark.asyncio
    async def test_explain_works_with_yaml_view(self):
        """Test that explain works correctly with YAML spec."""
        interface = MockInterface()
        view = MockView(spec="type: bar\nx: date\ny: value", language="yaml")
        task = MockTask()

        captured_kwargs = {}

        async def mock_stream(**kwargs):
            captured_kwargs.update(kwargs)
            yield "This YAML spec configures a bar chart."

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        system = captured_kwargs["system"]
        assert "yaml" in system.lower()
        assert "type: bar" in system

    @pytest.mark.asyncio
    async def test_explain_resets_state_on_error(self):
        """Test that state is consistent even when LLM errors out."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()

        async def mock_stream(**kwargs):
            raise RuntimeError("Network error")
            yield

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        # Error card should have been created
        assert len(interface.messages) == 1
        card = interface.messages[0]["content"]
        assert "Failed to generate explanation" in card.title

    @pytest.mark.asyncio
    async def test_explain_without_focus_attr(self):
        """Test _explain works when _focus hasn't been set (getattr fallback)."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()

        captured_kwargs = {}

        async def mock_stream(**kwargs):
            captured_kwargs.update(kwargs)
            yield "Simple select."

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        # Deliberately NOT setting controls._focus — tests the getattr fallback
        await controls._explain()

        msg = captured_kwargs["messages"][0]
        assert "Explain the spec above" in msg["content"]

    @pytest.mark.asyncio
    async def test_render_system_prompt_sql(self):
        """Test system prompt rendering for SQL language."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1", language="sql")
        task = MockTask()

        captured = {}

        async def mock_stream(**kwargs):
            captured.update(kwargs)
            yield "ok"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        system = captured["system"]
        assert "sql" in system.lower()
        assert "query" in system.lower()
        assert "SELECT" in system

    @pytest.mark.asyncio
    async def test_render_system_prompt_yaml(self):
        """Test system prompt rendering for YAML language."""
        interface = MockInterface()
        view = MockView(spec="type: bar", language="yaml")
        task = MockTask()

        captured = {}

        async def mock_stream(**kwargs):
            captured.update(kwargs)
            yield "ok"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        system = captured["system"]
        assert "yaml" in system.lower()
        assert "top-level key" in system

    @pytest.mark.asyncio
    async def test_render_system_prompt_unknown_language(self):
        """Test system prompt rendering for unknown language falls back gracefully."""
        interface = MockInterface()
        view = MockView(spec="some code", language="python")
        task = MockTask()

        captured = {}

        async def mock_stream(**kwargs):
            captured.update(kwargs)
            yield "ok"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._focus = ""
        await controls._explain()

        system = captured["system"]
        assert "python" in system.lower()
        assert "code" in system.lower()


class TestSpecTypeMap:
    """Tests for SPEC_TYPE_MAP constant."""

    def test_sql_maps_to_query(self):
        assert SPEC_TYPE_MAP["sql"] == "query"

    def test_yaml_maps_to_specification(self):
        assert SPEC_TYPE_MAP["yaml"] == "specification"

    def test_json_maps_to_specification(self):
        assert SPEC_TYPE_MAP["json"] == "specification"

    def test_vegalite_maps_to_visualization_spec(self):
        assert SPEC_TYPE_MAP["vega-lite"] == "visualization spec"
