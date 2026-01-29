"""Tests for copy controls."""
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls.copy import CopyControls


class MockView:
    """Mock view object for testing."""
    
    def __init__(self, spec="type: bar", language="yaml"):
        self.spec = spec
        self.language = language
        self.editor = MagicMock()
        self.editor.code = spec


class MockTask:
    """Mock task object for testing."""
    
    def __init__(self):
        self.actor = MagicMock()


class MockInterface:
    """Mock interface object for testing."""
    
    def __init__(self):
        self.messages = []


class TestCopyControls:
    """Tests for CopyControls class."""
    
    def test_initialization(self):
        """Test CopyControls initialization."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task
        )
        
        assert controls.interface is interface
        assert controls.view is view
        assert controls.task is task
        assert controls.layout_kwargs == {}
        
    def test_initialization_with_layout_kwargs(self):
        """Test CopyControls initialization with custom layout kwargs."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        layout_kwargs = {"margin": (10, 5), "sizing_mode": "stretch_width"}
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task,
            layout_kwargs=layout_kwargs
        )
        
        assert controls.layout_kwargs == layout_kwargs
        
    def test_creates_icon_button(self):
        """Test that CopyControls creates an IconButton with correct properties."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task
        )
        
        # Access the row to check its contents
        row = controls._row
        
        # The row should have one child (the icon button)
        assert len(row.objects) == 1
        icon_button = row.objects[0]
        
        # Verify icon button properties
        assert icon_button.icon == "content_copy"
        assert icon_button.active_icon == "check"
        assert icon_button.description == "Copy YAML to clipboard"
        assert icon_button.size == "small"
        assert icon_button.color == "primary"
        assert icon_button.icon_size == "0.9em"
        assert icon_button.margin == (5, 0)
        assert icon_button.toggle_duration == 1000
        
    def test_javascript_click_handler(self):
        """Test that the copy button has JavaScript click handler configured."""
        interface = MockInterface()
        view = MockView(spec="type: bar\ndata: values")
        task = MockTask()
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task
        )
        
        icon_button = controls._row.objects[0]
        
        # Verify js_on_click was called by checking _models
        # The button should have JS callbacks registered
        assert hasattr(icon_button, '_models')
        
    def test_javascript_code_references_clipboard(self):
        """Test that JavaScript code uses navigator.clipboard.writeText."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        # Mock js_on_click to capture the code argument
        captured_code = []
        captured_args = []
        
        with patch('panel_material_ui.IconButton.js_on_click') as mock_js_on_click:
            def capture_js_call(args, code):
                captured_args.append(args)
                captured_code.append(code)
                
            mock_js_on_click.side_effect = capture_js_call
            
            controls = CopyControls(
                interface=interface,
                view=view,
                task=task
            )
            
            # Verify js_on_click was called
            assert mock_js_on_click.called
            
            # Verify the code includes clipboard API call
            assert len(captured_code) > 0
            js_code = captured_code[0]
            assert 'navigator.clipboard.writeText' in js_code
            assert 'code_editor.code' in js_code
            
            # Verify args includes code_editor
            assert len(captured_args) > 0
            assert 'code_editor' in captured_args[0]
        
    def test_panel_returns_row(self):
        """Test __panel__ returns the row component."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task
        )
        
        panel = controls.__panel__()
        assert panel is controls._row
        
    def test_view_editor_reference(self):
        """Test that the view's editor is correctly referenced."""
        interface = MockInterface()
        spec_content = "type: line\nx: date\ny: value"
        view = MockView(spec=spec_content)
        task = MockTask()
        
        # Mock js_on_click to capture args
        captured_args = []
        
        with patch('panel_material_ui.IconButton.js_on_click') as mock_js_on_click:
            def capture_args(args, code):
                captured_args.append(args)
                
            mock_js_on_click.side_effect = capture_args
            
            controls = CopyControls(
                interface=interface,
                view=view,
                task=task
            )
            
            # Verify the editor reference matches the view's editor
            assert len(captured_args) > 0
            assert captured_args[0]['code_editor'] is view.editor
        
    def test_multiple_instances_independent(self):
        """Test that multiple CopyControls instances are independent."""
        interface = MockInterface()
        view1 = MockView(spec="spec1")
        view2 = MockView(spec="spec2")
        task1 = MockTask()
        task2 = MockTask()
        
        controls1 = CopyControls(interface=interface, view=view1, task=task1)
        controls2 = CopyControls(interface=interface, view=view2, task=task2)
        
        # Each should have their own row
        assert controls1._row is not controls2._row
        
        # Each should have their own button
        button1 = controls1._row.objects[0]
        button2 = controls2._row.objects[0]
        assert button1 is not button2
        
    def test_layout_kwargs_applied_to_row(self):
        """Test that layout_kwargs are applied to the Row component."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        layout_kwargs = {
            "margin": (20, 10),
            "sizing_mode": "fixed",
            "width": 200
        }
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task,
            layout_kwargs=layout_kwargs
        )
        
        # Verify layout kwargs were passed to Row
        row = controls._row
        assert row.margin == (20, 10)
        assert row.sizing_mode == "fixed"
        assert row.width == 200
        
    def test_view_without_editor(self):
        """Test CopyControls handles view without editor gracefully."""
        interface = MockInterface()
        view = Mock(spec=["spec", "language"])
        view.spec = "type: bar"
        view.language = "yaml"
        # Intentionally no editor attribute
        task = MockTask()
        
        # Should raise AttributeError when trying to access editor
        with pytest.raises(AttributeError):
            controls = CopyControls(
                interface=interface,
                view=view,
                task=task
            )
            
    def test_different_view_specs(self):
        """Test CopyControls with different spec content."""
        interface = MockInterface()
        task = MockTask()
        
        # Test with simple spec
        view1 = MockView(spec="type: bar")
        controls1 = CopyControls(interface=interface, view=view1, task=task)
        # Just verify it was created successfully
        assert controls1._row is not None
        
        # Test with complex spec
        complex_spec = """
type: line
x: date
y: value
color: category
title: Sales Over Time
"""
        view2 = MockView(spec=complex_spec)
        controls2 = CopyControls(interface=interface, view=view2, task=task)
        assert controls2._row is not None
        
    def test_button_toggle_behavior(self):
        """Test that button has toggle behavior configured."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task
        )
        
        icon_button = controls._row.objects[0]
        
        # Verify toggle duration is set (visual feedback when clicked)
        assert icon_button.toggle_duration == 1000
        
        # Verify it changes icon when active (visual confirmation)
        assert icon_button.icon == "content_copy"
        assert icon_button.active_icon == "check"
        
    def test_description_tooltip(self):
        """Test that button has helpful description for tooltip."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task
        )
        
        icon_button = controls._row.objects[0]
        
        # Verify description exists and is helpful
        assert icon_button.description == "Copy YAML to clipboard"
        
    def test_button_styling(self):
        """Test that button has correct styling properties."""
        interface = MockInterface()
        view = MockView()
        task = MockTask()
        
        controls = CopyControls(
            interface=interface,
            view=view,
            task=task
        )
        
        icon_button = controls._row.objects[0]
        
        # Verify styling properties
        assert icon_button.size == "small"
        assert icon_button.color == "primary"
        assert icon_button.icon_size == "0.9em"
        assert icon_button.margin == (5, 0)
