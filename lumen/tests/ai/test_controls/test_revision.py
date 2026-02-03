"""Tests for revision control classes."""
import asyncio

from unittest.mock import (
    AsyncMock, MagicMock, Mock, patch,
)

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls.revision import (
    AnnotationControls, RetryControls, RevisionControls,
)


class MockView:
    """Mock view object for testing."""
    
    def __init__(self, spec="type: bar", language="yaml"):
        self._spec = spec
        self.language = language
        self.editor = MagicMock()
        self.editor.loading = False
        self.editor.param = MagicMock()
        self.editor.param.update = MagicMock()
        # Make param.update work as context manager
        self.editor.param.update.return_value.__enter__ = Mock(return_value=None)
        self.editor.param.update.return_value.__exit__ = Mock(return_value=None)
        
    @property
    def spec(self):
        return self._spec
        
    @spec.setter
    def spec(self, value):
        self._spec = value


class MockTask:
    """Mock task object for testing."""
    
    def __init__(self):
        self.actor = MagicMock()
        self.actor.param = MagicMock()
        self.actor.param.update = MagicMock()
        # Make param.update work as context manager
        self.actor.param.update.return_value.__enter__ = Mock(return_value=None)
        self.actor.param.update.return_value.__exit__ = Mock(return_value=None)
        
        self.history = []
        self.out_context = {}
        self.parent = None
        self.param = MagicMock()
        self.param.update = MagicMock()
        # Make param.update work as context manager
        self.param.update.return_value.__enter__ = Mock(return_value=None)
        self.param.update.return_value.__exit__ = Mock(return_value=None)
        
    def invalidate(self, keys, start=1):
        """Mock invalidate method."""
        pass
    
    async def execute(self):
        """Mock execute method."""
        pass


class MockInterface:
    """Mock interface object for testing."""
    
    def __init__(self):
        self.messages = []
        
    def stream(self, content, user="Assistant"):
        """Mock stream method."""
        self.messages.append({"content": content, "user": user})


class TestRevisionControls:
    """Tests for base RevisionControls class."""
    
    def test_initialization(self):
        """Test RevisionControls initialization."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        assert controls.interface is interface
        assert controls.view is view
        assert controls.task is task
        assert controls.active is False
        assert controls.instruction == ""
        
    def test_initialization_uses_class_kwargs(self):
        """Test RevisionControls uses class-level input_kwargs and toggle_kwargs."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        # Create a subclass with custom kwargs
        class CustomRevisionControls(RevisionControls):
            input_kwargs = {"placeholder": "Custom placeholder", "max_length": 100}
            toggle_kwargs = {"icon": "custom_icon"}
        
        controls = CustomRevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Verify the text input uses the custom kwargs
        assert controls._text_input.max_length == 100
        
    def test_active_state_change(self):
        """Test that active state can be changed."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Initially False
        assert controls.active is False
        
        # Set to True
        controls.active = True
        assert controls.active is True
        
        # Set back to False
        controls.active = False
        assert controls.active is False
            
    def test_close_updates_active_state(self):
        """Test _close method updates active state."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Simulate closing with event
        event = Mock()
        event.new = False
        controls._close(event)
        
        assert controls.active is False
        
    def test_enter_reason_updates_instruction(self):
        """Test _enter_reason captures instruction and resets input."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Set input value
        controls._text_input.value_input = "Make the chart bigger"
        
        # Trigger enter
        controls._enter_reason(None)
        
        # Verify instruction was captured
        assert controls.instruction == "Make the chart bigger"
        assert controls.active is False
        assert controls._text_input.value == ""
        
    def test_report_status_streams_message(self):
        """Test _report_status streams formatted content to interface."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Report a status
        diff_content = "```diff\n- old line\n+ new line\n```"
        controls._report_status(diff_content)
        
        # Verify message was streamed
        assert len(interface.messages) == 1
        assert interface.messages[0]["user"] == "Assistant"
        
    def test_report_status_empty_output(self):
        """Test _report_status with empty output does nothing."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Report empty status
        controls._report_status("")
        
        # Verify no message was streamed
        assert len(interface.messages) == 0
        
    def test_report_status_custom_title(self):
        """Test _report_status with custom title."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Report status with custom title
        controls._report_status("content", title="Custom Title")
        
        # Verify message was streamed (title is in Card which is part of content)
        assert len(interface.messages) == 1
        # Check that the Card has the custom title
        card = interface.messages[0]["content"]
        assert hasattr(card, 'title')
        assert card.title == "Custom Title"
        
    def test_panel_returns_row(self):
        """Test __panel__ returns the row component."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RevisionControls(
            interface=interface,
            view=view,
            task=task
        )
        
        panel = controls.__panel__()
        assert panel is controls._row


class TestRetryControls:
    """Tests for RetryControls class."""
    
    def test_initialization(self):
        """Test RetryControls initialization with correct defaults."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = RetryControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Verify retry-specific class attributes
        assert "Enter feedback" in RetryControls.input_kwargs["placeholder"]
        assert RetryControls.toggle_kwargs["icon"] == "auto_awesome"
        assert "Ask AI to make changes" in RetryControls.toggle_kwargs["description"]
        
    @pytest.mark.asyncio
    async def test_revise_with_revise_method(self):
        """Test _revise calls actor.revise when available."""
        interface = MockInterface()
        old_spec = "type: bar\ndata: old"
        new_spec = "type: bar\ndata: new"
        view = MockView(spec=old_spec)
        task = MockTask()
        
        # Mock the revise method
        task.actor.revise = AsyncMock(return_value=new_spec)
        task.history = [{"role": "user", "content": "Create a bar chart"}]
        task.out_context = {"data": "test"}
        
        controls = RetryControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Mock generate_diff to avoid actual diff computation
        with patch('lumen.ai.controls.revision.generate_diff', return_value="diff output"):
            # Trigger revision
            controls.instruction = "Make it blue"
            await controls._revise()
            
            # Verify revise was called
            task.actor.revise.assert_called_once()
            call_args = task.actor.revise.call_args
            assert call_args[0][0] == "Make it blue"  # instruction
            assert call_args[0][1] == task.history  # messages
            assert call_args[0][2] == task.out_context  # out_context
            assert call_args[0][3] == view  # view
            
            # Verify spec was updated
            assert view.spec == new_spec
            
    @pytest.mark.asyncio
    async def test_revise_without_revise_method(self):
        """Test _revise invalidates and re-executes when actor has no revise method."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        # Remove revise method from actor
        delattr(task.actor, 'revise')
        
        # Mock output_schema
        task.actor.output_schema = MagicMock()
        task.actor.output_schema.__annotations__ = {"field1": str, "field2": int}
        
        # Mock invalidate and execute
        task.invalidate = Mock()
        task.execute = AsyncMock()
        
        controls = RetryControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Trigger revision
        controls.instruction = "Try again"
        await controls._revise()
        
        # Verify invalidate was called with output keys
        task.invalidate.assert_called_once()
        invalidation_keys = task.invalidate.call_args[0][0]
        assert "field1" in invalidation_keys
        assert "field2" in invalidation_keys
        
        # Verify execute was called
        task.execute.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_revise_empty_instruction(self):
        """Test _revise does nothing with empty instruction."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        task.actor.revise = AsyncMock()
        
        controls = RetryControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Set empty instruction
        controls.instruction = ""
        await controls._revise()
        
        # Verify revise was not called
        task.actor.revise.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_revise_handles_exception(self):
        """Test _revise handles exceptions during revision."""
        interface = MockInterface()
        view = MockView(spec="original")
        task = MockTask()
        
        # Mock revise to raise exception
        task.actor.revise = AsyncMock(side_effect=Exception("LLM Error"))
        task.history = []
        task.out_context = {}
        
        controls = RetryControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Trigger revision and expect exception
        controls.instruction = "Fix this"
        with pytest.raises(Exception, match="LLM Error"):
            await controls._revise()
            
        # Verify error was reported (check that a Card was created with correct title)
        assert len(interface.messages) == 1
        card = interface.messages[0]["content"]
        assert hasattr(card, 'title')
        assert "Failed to generate revisions" in card.title
        
    @pytest.mark.asyncio
    async def test_revise_spec_application_failure(self):
        """Test _revise handles spec application failures."""
        interface = MockInterface()
        old_spec = "type: bar"
        new_spec = "invalid: spec: syntax"
        view = MockView(spec=old_spec)
        task = MockTask()
        
        task.actor.revise = AsyncMock(return_value=new_spec)
        task.history = []
        task.out_context = {}
        
        controls = RetryControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Create a custom property that raises on new_spec
        original_setter = type(view).spec.fset
        
        def failing_setter(self, value):
            if value == new_spec:
                raise ValueError("Invalid spec")
            original_setter(self, value)
                
        type(view).spec = property(type(view).spec.fget, failing_setter)
        
        with patch('lumen.ai.controls.revision.generate_diff', return_value="diff"):
            controls.instruction = "Change it"
            
            # Expect exception to be raised
            with pytest.raises(ValueError, match="Invalid spec"):
                await controls._revise()
                
            # Verify spec was reverted
            assert view.spec == old_spec
            
            # Verify error was reported with correct title
            assert len(interface.messages) == 1
            card = interface.messages[0]["content"]
            assert hasattr(card, 'title')
            assert "Failed to apply revisions" in card.title
            
    @pytest.mark.asyncio
    async def test_revise_with_parent_task(self):
        """Test _revise navigates to root task when needed."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        parent_task = MockTask()
        
        # Set up task hierarchy
        task.parent = parent_task
        parent_task.parent = None
        parent_task.execute = AsyncMock()
        
        # Remove revise method
        delattr(task.actor, 'revise')
        task.actor.output_schema = MagicMock()
        task.actor.output_schema.__annotations__ = {"field": str}
        
        controls = RetryControls(
            interface=interface,
            view=view,
            task=task
        )
        
        controls.instruction = "Retry"
        await controls._revise()
        
        # Verify root execute was called
        parent_task.execute.assert_called_once()


class TestAnnotationControls:
    """Tests for AnnotationControls class."""
    
    def test_initialization(self):
        """Test AnnotationControls initialization with correct defaults."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = AnnotationControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Verify annotation-specific class attributes
        assert "Describe what to annotate" in AnnotationControls.input_kwargs["placeholder"]
        assert AnnotationControls.toggle_kwargs["icon"] == "chat-bubble"
        assert "Add annotations" in AnnotationControls.toggle_kwargs["description"]
        
    @pytest.mark.asyncio
    async def test_annotate_success(self):
        """Test _annotate successfully adds annotations."""
        interface = MockInterface()
        old_spec = "type: bar"
        new_spec = "type: bar\nannotations:\n  - text: Peak"
        view = MockView(spec=old_spec, language="yaml")
        task = MockTask()
        
        # Mock the annotate method
        task.actor.annotate = AsyncMock(return_value=new_spec)
        task.history = [{"role": "user", "content": "Show data"}]
        task.out_context = {"data": "test"}
        
        controls = AnnotationControls(
            interface=interface,
            view=view,
            task=task
        )
        
        with patch('lumen.ai.controls.revision.generate_diff', return_value="diff output"):
            with patch('lumen.ai.controls.revision.load_yaml', return_value={"type": "bar"}):
                # Trigger annotation
                controls.instruction = "Highlight peak values"
                await controls._annotate()
                
                # Verify annotate was called
                task.actor.annotate.assert_called_once()
                call_args = task.actor.annotate.call_args
                assert call_args[0][0] == "Highlight peak values"  # instruction
                assert call_args[0][1] == task.history  # messages
                assert call_args[0][2] == task.out_context  # out_context
                
                # Verify spec was updated
                assert view.spec == new_spec
                
    @pytest.mark.asyncio
    async def test_annotate_empty_instruction(self):
        """Test _annotate does nothing with empty instruction."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        task.actor.annotate = AsyncMock()
        
        controls = AnnotationControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Set empty instruction
        controls.instruction = ""
        await controls._annotate()
        
        # Verify annotate was not called
        task.actor.annotate.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_annotate_handles_exception(self):
        """Test _annotate handles exceptions during annotation."""
        interface = MockInterface()
        view = MockView(spec="original")
        task = MockTask()
        
        # Mock annotate to raise exception
        task.actor.annotate = AsyncMock(side_effect=Exception("Annotation failed"))
        task.history = []
        task.out_context = {}
        
        controls = AnnotationControls(
            interface=interface,
            view=view,
            task=task
        )
        
        with patch('lumen.ai.controls.revision.load_yaml', return_value={}):
            controls.instruction = "Add labels"
            
            with pytest.raises(Exception, match="Annotation failed"):
                await controls._annotate()
                
            # Verify error was reported with correct title
            assert len(interface.messages) == 1
            card = interface.messages[0]["content"]
            assert hasattr(card, 'title')
            assert "Failed to generate annotations" in card.title
            
    @pytest.mark.asyncio
    async def test_annotate_spec_application_failure(self):
        """Test _annotate handles spec application failures."""
        interface = MockInterface()
        old_spec = "type: bar"
        new_spec = "invalid: annotation"
        view = MockView(spec=old_spec)
        task = MockTask()
        
        task.actor.annotate = AsyncMock(return_value=new_spec)
        task.history = []
        task.out_context = {}
        
        controls = AnnotationControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Create a custom property that raises on new_spec
        original_setter = type(view).spec.fset
        
        def failing_setter(self, value):
            if value == new_spec:
                raise ValueError("Invalid annotation spec")
            original_setter(self, value)
                
        type(view).spec = property(type(view).spec.fget, failing_setter)
        
        with patch('lumen.ai.controls.revision.generate_diff', return_value="diff"):
            with patch('lumen.ai.controls.revision.load_yaml', return_value={}):
                controls.instruction = "Annotate"
                
                with pytest.raises(ValueError, match="Invalid annotation spec"):
                    await controls._annotate()
                    
                # Verify spec was reverted
                assert view.spec == old_spec
                
                # Verify error was reported with correct title
                assert len(interface.messages) == 1
                card = interface.messages[0]["content"]
                assert hasattr(card, 'title')
                assert "Failed to apply annotations" in card.title
                
    @pytest.mark.asyncio
    async def test_annotate_uses_load_yaml(self):
        """Test _annotate properly loads and passes spec to annotate method."""
        interface = MockInterface()
        view = MockView(spec="type: bar\ndata: test")
        task = MockTask()
        
        task.actor.annotate = AsyncMock(return_value="updated spec")
        task.history = []
        task.out_context = {}
        
        controls = AnnotationControls(
            interface=interface,
            view=view,
            task=task
        )
        
        mock_loaded_spec = {"type": "bar", "data": "test"}
        
        with patch('lumen.ai.controls.revision.load_yaml', return_value=mock_loaded_spec) as mock_load:
            with patch('lumen.ai.controls.revision.generate_diff', return_value=""):
                controls.instruction = "Add notes"
                await controls._annotate()
                
                # Verify load_yaml was called with the ORIGINAL view spec (before update)
                # The view spec gets updated to "updated spec" after annotate succeeds
                mock_load.assert_called_once_with("type: bar\ndata: test")
                
                # Verify loaded spec was passed to annotate
                call_args = task.actor.annotate.call_args[0]
                assert call_args[3] == {"spec": mock_loaded_spec}
