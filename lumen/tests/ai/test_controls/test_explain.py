"""Tests for explain controls."""
from unittest.mock import MagicMock

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls.explain import ExplainControls


class MockView:
    """Mock view object for testing."""

    def __init__(self, spec="SELECT * FROM users", language="sql"):
        self.spec = spec
        self.language = language


class MockTask:
    """Mock task object for testing."""

    def __init__(self):
        self.actor = MagicMock()
        self.history = []
        self.out_context = {}

    def set_explain_chunks(self, chunks):
        """Configure the mock actor.explain to yield given chunks."""
        async def mock_explain(instruction, messages, context, view):
            for chunk in chunks:
                yield chunk
        self.actor.explain = mock_explain

    def set_explain_error(self, error):
        """Configure the mock actor.explain to raise an error."""
        async def mock_explain(instruction, messages, context, view):
            raise error
            yield  # make it a generator
        self.actor.explain = mock_explain


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
        assert "blank" in ExplainControls.input_kwargs["placeholder"].lower() or "overview" in ExplainControls.input_kwargs["placeholder"].lower()

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
        """Test _enter_reason with blank input sets empty _user_question and increments count."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        task.set_explain_chunks(["ok"])

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls.active = True
        controls._text_input.value_input = ""

        controls._enter_reason(None)

        assert controls._user_question == ""
        assert controls.active is False
        assert controls._explain_count == 1
        assert controls._text_input.value == ""

    @pytest.mark.asyncio
    async def test_enter_reason_with_user_question(self):
        """Test _enter_reason with specific user question."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        task.set_explain_chunks(["ok"])

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls.active = True
        controls._text_input.value_input = "  WHERE clause  "

        controls._enter_reason(None)

        assert controls._user_question == "WHERE clause"
        assert controls.active is False
        assert controls._explain_count == 1

    @pytest.mark.asyncio
    async def test_enter_reason_increments_count(self):
        """Test _enter_reason increments _explain_count on each call."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        task.set_explain_chunks(["ok"])

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._text_input.value_input = ""

        controls._enter_reason(None)
        assert controls._explain_count == 1

        controls._text_input.value_input = "JOIN"
        controls._enter_reason(None)
        assert controls._explain_count == 2

    @pytest.mark.asyncio
    async def test_explain_delegates_to_actor(self):
        """Test that _explain delegates to self.task.actor.explain."""
        interface = MockInterface()
        view = MockView(spec="SELECT id FROM users")
        task = MockTask()

        chunks = ["This query", "This query selects", "This query selects all IDs from the users table."]
        task.set_explain_chunks(chunks)

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = ""
        await controls._explain()

        assert len(interface.messages) == len(chunks)
        for msg in interface.messages:
            assert msg["replace"] is True

    @pytest.mark.asyncio
    async def test_explain_passes_args_to_actor(self):
        """Test that _explain passes user_question, messages, context, and view to actor."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()
        task.history = [{"role": "user", "content": "hello"}]
        task.out_context = {"key": "value"}

        captured = {}

        async def mock_explain(instruction, messages, context, view_arg):
            captured["instruction"] = instruction
            captured["messages"] = messages
            captured["context"] = context
            captured["view"] = view_arg
            yield "ok"

        task.actor.explain = mock_explain

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = "JOIN clause"
        await controls._explain()

        assert captured["instruction"] == "JOIN clause"
        assert captured["messages"] == [{"role": "user", "content": "hello"}]
        assert captured["context"] == {"key": "value"}
        assert captured["view"] is view

    @pytest.mark.asyncio
    async def test_explain_empty_spec_noop(self):
        """Test _explain does nothing with empty spec."""
        interface = MockInterface()
        view = MockView(spec="")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = ""
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_none_spec_noop(self):
        """Test _explain does nothing with None spec."""
        interface = MockInterface()
        view = MockView(spec=None)
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = ""
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_whitespace_only_spec_noop(self):
        """Test _explain does nothing with whitespace-only spec."""
        interface = MockInterface()
        view = MockView(spec="   \n  ")
        task = MockTask()

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = ""
        await controls._explain()

        assert len(interface.messages) == 0

    @pytest.mark.asyncio
    async def test_explain_handles_error(self):
        """Test _explain handles errors gracefully."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()
        task.set_explain_error(Exception("LLM connection failed"))

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = ""
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
        task.set_explain_error(Exception("Some error"))

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = ""
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
    async def test_explain_resets_state_on_error(self):
        """Test that state is consistent even when actor.explain errors out."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()
        task.set_explain_error(RuntimeError("Network error"))

        controls = ExplainControls(interface=interface, view=view, task=task)
        controls._user_question = ""
        await controls._explain()

        # Error card should have been created
        assert len(interface.messages) == 1
        card = interface.messages[0]["content"]
        assert "Failed to generate explanation" in card.title

    @pytest.mark.asyncio
    async def test_explain_default_user_question(self):
        """Test _explain works with default empty _user_question from __init__."""
        interface = MockInterface()
        view = MockView(spec="SELECT 1")
        task = MockTask()

        captured = {}

        async def mock_explain(instruction, messages, context, view_arg):
            captured["instruction"] = instruction
            yield "Simple select."

        task.actor.explain = mock_explain

        controls = ExplainControls(interface=interface, view=view, task=task)
        assert controls._user_question == ""
        await controls._explain()

        assert captured["instruction"] == ""
