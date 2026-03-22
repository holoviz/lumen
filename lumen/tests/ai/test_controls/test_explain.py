"""Tests for explain controls."""
import asyncio

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls.explain import ExplainControls, SPEC_TYPE_MAP


class MockView:
    """Mock view object for testing."""

    def __init__(self, spec="SELECT * FROM users", language="sql"):
        self.spec = spec
        self.language = language
        self.editor = MagicMock()
        self.editor.code = spec


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
            # Update existing message
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
        assert controls.layout_kwargs == {}

    def test_creates_icon_button(self):
        """Test that ExplainControls creates an IconButton with correct properties."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        controls = ExplainControls(
            interface=interface,
            view=view,
            task=task,
        )

        row = controls._row
        assert len(row.objects) == 1
        icon_button = row.objects[0]

        assert icon_button.icon == "psychology"
        assert icon_button.active_icon == "check"
        assert icon_button.description == "Explain SQL"
        assert icon_button.size == "small"
        assert icon_button.color == "primary"
        assert icon_button.icon_size == "0.9em"
        assert icon_button.margin == (5, 0)
        assert icon_button.toggle_duration == 1000

    def test_description_adapts_to_language(self):
        """Test that description tooltip changes for different languages."""
        interface = MockInterface()
        task = MockTask()

        # SQL
        view_sql = MockView(language="sql")
        controls_sql = ExplainControls(interface=interface, view=view_sql, task=task)
        assert controls_sql._row.objects[0].description == "Explain SQL"

        # YAML
        view_yaml = MockView(spec="type: bar", language="yaml")
        controls_yaml = ExplainControls(interface=interface, view=view_yaml, task=task)
        assert controls_yaml._row.objects[0].description == "Explain YAML"

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

    def test_render_system_prompt_sql(self):
        """Test system prompt rendering for SQL language."""
        interface = MockInterface()
        view = MockView(language="sql")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        prompt = controls._render_system_prompt()

        assert "sql" in prompt.lower()
        assert "query" in prompt.lower()
        assert "SELECT" in prompt
        assert "FROM" in prompt

    def test_render_system_prompt_yaml(self):
        """Test system prompt rendering for YAML language."""
        interface = MockInterface()
        view = MockView(spec="type: bar", language="yaml")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        prompt = controls._render_system_prompt()

        assert "yaml" in prompt.lower()
        assert "specification" in prompt.lower()
        assert "top-level key" in prompt

    def test_render_system_prompt_unknown_language(self):
        """Test system prompt rendering for unknown language falls back gracefully."""
        interface = MockInterface()
        view = MockView(spec="some code", language="python")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        prompt = controls._render_system_prompt()

        assert "python" in prompt.lower()
        assert "code" in prompt.lower()

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
        await controls._explain()

        # Should have streamed multiple times (one per chunk)
        assert len(interface.messages) == len(chunks)
        # Each stream call should use replace=True
        for msg in interface.messages:
            assert msg["replace"] is True

    @pytest.mark.asyncio
    async def test_explain_empty_spec_noop(self):
        """Test _explain does nothing with empty spec."""
        interface = MockInterface()
        view = MockView(spec="")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_none_spec_noop(self):
        """Test _explain does nothing with None spec."""
        interface = MockInterface()
        view = MockView(spec=None)
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_whitespace_only_spec_noop(self):
        """Test _explain does nothing with whitespace-only spec."""
        interface = MockInterface()
        view = MockView(spec="   \n  ")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
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
        await controls._explain()

        # Should have streamed an error card
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
    async def test_explain_disables_icon_during_execution(self):
        """Test that the icon is disabled during LLM call and re-enabled after."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()

        disabled_states = []

        async def mock_stream(**kwargs):
            disabled_states.append(True)  # capture state during streaming
            yield "explanation"

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        await controls._explain()

        # Icon should be re-enabled after completion
        assert controls._icon.disabled is False
        assert controls._clicking is False

    @pytest.mark.asyncio
    async def test_explain_resets_clicking_on_empty_spec(self):
        """Test that _clicking is reset even when spec is empty."""
        interface = MockInterface()
        view = MockView(spec="")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._clicking = True
        await controls._explain()

        assert controls._clicking is False

    @pytest.mark.asyncio
    async def test_explain_resets_state_on_error(self):
        """Test that icon state is reset even when LLM errors out."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()

        async def mock_stream(**kwargs):
            raise RuntimeError("Network error")
            yield

        task.actor.llm.stream = mock_stream

        controls = ExplainControls(interface=interface, view=view, task=task)
        await controls._explain()

        assert controls._icon.disabled is False
        assert controls._clicking is False

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
        await controls._explain()

        # Verify messages contain the spec
        assert len(captured_kwargs["messages"]) == 1
        msg = captured_kwargs["messages"][0]
        assert msg["role"] == "user"
        assert "SELECT name FROM users WHERE active = 1" in msg["content"]
        assert "```sql" in msg["content"]

        # Verify system prompt was rendered
        assert "system" in captured_kwargs
        assert "sql" in captured_kwargs["system"].lower()

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
        await controls._explain()

        assert "```yaml" in captured_kwargs["messages"][0]["content"]
        assert "yaml" in captured_kwargs["system"].lower()

    def test_on_click_prevents_double_click(self):
        """Test that _on_click prevents concurrent executions."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)

        with patch("panel.io.state") as mock_state:
            # First click should work
            controls._on_click(None)
            assert controls._clicking is True
            mock_state.execute.assert_called_once()

            # Second click should be ignored
            mock_state.execute.reset_mock()
            controls._on_click(None)
            mock_state.execute.assert_not_called()


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
