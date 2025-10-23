from __future__ import annotations

from pathlib import Path
from typing import Any

import param

from panel.custom import Child, JSComponent

CSS = """
/* Base styles for the split container */
.split {
    display: flex;
    flex-direction: row;
    height: 100%;
    width: 100%;
    background-color: var(--panel-surface-color);
    overflow: clip; /* Clip overflow to prevent scrollbars; inner content will have their own */
}

/* Style for initial load to prevent FOUC */
.split.loading {
    visibility: hidden;
}

/* Max width for comfortable reading */
.left-panel-content {
    max-width: clamp(450px, 95vw, 1200px);
    margin: 0px auto;
}

/* Split panel styles */
.split > div {
    position: relative;
}

.split > div:nth-child(2) {
    overflow: visible;
    position: relative;
    width: 100%;
}

/* Content wrapper styles */
.content-wrapper {
    width: 100%;
    height: 100%;
    display: contents;
    background-color: var(--panel-background-color);
    border-radius: 10px; /* Slight rounded corners */
    box-sizing: border-box; /* Include padding in size calculations */
    padding-inline: 0.5rem; /* Increased padding for better visual spacing */
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08); /* Subtle shadow for depth */
    border: 1px solid rgba(0, 0, 0, 0.08); /* Subtle border for definition */
}

.left-content-wrapper {
    width: 100%;
    height: 100%;
}

/* Toggle button basic styles */
.toggle-button-left, .toggle-button-right {
    position: absolute;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 100; /* High z-index to ensure always on top */
    opacity: 0.5;
    top: 50%;
    transform: translateY(-50%);
    transition: opacity 0.2s;
    border-radius: 4px;
    padding: 2px;
}

/* Left button (<) - positioned on left side of divider */
.toggle-button-left {
    left: -34px; /* 24px width + 10px spacing from divider */
}

/* Right button (>) - positioned on right side of divider */
.toggle-button-right {
    left: 2px;
}

/* Ensure buttons stay visible even when panels are collapsed */
.split > div:first-child {
    min-width: 0 !important; /* Override any minimum width */
}

.split > div:nth-child(2) {
    overflow: visible !important; /* Ensure buttons remain visible */
    position: relative !important;
}

.toggle-button-left:hover, .toggle-button-right:hover {
    opacity: 1;
    background-color: var(--panel-border-color);
}

/* SVG icon styling */
.toggle-button-left svg, .toggle-button-right svg {
    width: 20px;
    height: 20px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2px;
    stroke-linecap: round;
    stroke-linejoin: round;
}

/* Collapsed state */
.collapsed-content {
    display: none;
}

/* Gutter styles */
.gutter {
    background-color: var(--panel-surface-color);
    background-repeat: no-repeat;
    background-position: 50%;
    cursor: col-resize;
    transition: background-color 0.2s;
}

.gutter:hover {
    background-color: var(--panel-border-color);
}

.gutter.gutter-horizontal {
    background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAeCAYAAADkftS9AAAAIklEQVQoU2M4c+bMfxAGAgYYmwGrIIiDjrELjpo5aiZeMwF+yNnOs5KSvgAAAABJRU5ErkJggg==');
    z-index: 1;
    width: 8px;
}

.split > div:nth-child(2) > div:not(.toggle-button-left):not(.toggle-button-right) {
    width: 100%;
    height: 100%;
    overflow: auto;
    padding-top: 36px; /* Space for the toggle buttons */
}

/* Animation for toggle icon */
@keyframes jumpLeftRight {
    0%, 100% { transform: translateY(-50%); }
    25% { transform: translate(-4px, -50%); }
    50% { transform: translateY(-50%); }
    75% { transform: translate(4px, -50%); }
}

.toggle-button-left.animated, .toggle-button-right.animated {
    animation-name: jumpLeftRight;
    animation-duration: 0.5s;
    animation-timing-function: ease;
    animation-iteration-count: 3;
}

/* List navigation styles */
ul.nav.flex-column {
    padding-inline-start: 0;
    margin: 0;
}
"""


class SplitJS(JSComponent):
    """
    SplitJS is a component for creating a responsive split panel layout with a collapsible sidebar.

    This component uses split.js to create a draggable split layout with two panels.
    Key features include:
    - Collapsible panels with toggle button
    - Minimum size constraints for each panel
    - Invertible layout to support different UI configurations
    - Responsive sizing with automatic adjustments
    - Animation for better user experience

    The component is ideal for creating application layouts with a main content area
    and a secondary panel that can be toggled (like a chat interface with output display).
    """

    left = Child(doc="""
        The component to place in the left panel.
        When invert=True, this will appear on the right side.""")

    right = Child(doc="""
        The component to place in the right panel.
        When invert=True, this will appear on the left side.""")

    sizes = param.NumericTuple(default=(100, 0), length=2, doc="""
        The initial sizes of the two panels (as percentages).
        Default is (100, 0) which means the left panel takes up all the space
        and the right panel is not visible.""")

    expanded_sizes = param.NumericTuple(default=(40, 60), length=2, doc="""
        The sizes of the two panels when expanded (as percentages).
        Default is (35, 65) which means the left panel takes up 35% of the space
        and the right panel takes up 65% when expanded.
        When invert=True, these percentages are automatically swapped.""")

    min_sizes = param.NumericTuple(default=(0, 0), length=2, doc="""
        The minimum sizes of the two panels (in pixels).
        Default is (0, 0) which allows both panels to fully collapse.
        Set to (300, 0) or similar values if you want to enforce minimum widths during dragging.
        When invert=True, these values are automatically swapped.""")

    collapsed = param.Boolean(default=True, doc="""
        Whether the secondary panel is collapsed.
        When True, only one panel is visible (determined by invert).
        When False, both panels are visible according to expanded_sizes.""")

    invert = param.Boolean(default=False, constant=True, doc="""
        Whether to invert the layout, changing the toggle button side and panel styles.""")

    _esm = Path(__file__).parent / "models" / "split.js"

    _stylesheets = [CSS]

    def __init__(self, **params):
        super().__init__(**params)
        if self.invert:
            # Swap min_sizes when inverted
            left_min, right_min = self.min_sizes
            self.min_sizes = (right_min, left_min)

            # Swap expanded_sizes when inverted
            left_exp, right_exp = self.expanded_sizes
            self.expanded_sizes = (right_exp, left_exp)

    @param.depends("collapsed", watch=True)
    def _send_collapsed_update(self):
        """Send message to JS when collapsed state changes in Python"""
        self._send_msg({"type": "update_collapsed", "collapsed": self.collapsed})

    def _handle_msg(self, msg):
        """Handle messages from JS"""
        if 'collapsed' in msg:
            collapsed = msg['collapsed']
            with param.discard_events(self):
                # Important to discard so when user drags the panel, it doesn't
                # expand to the expanded sizes
                self.collapsed = collapsed


class Details(JSComponent):
    """
    A component wrapper for the HTML details element with proper content rendering.

    This component provides an expandable details/summary element that can be
    used to hide/show content. It supports any Panel component as content,
    including Markdown panes for rendering markdown content.
    """

    title = param.String(default="Details", doc="""
        The text to display in the summary element (the clickable header).""")

    object = Child(doc="""
        The content to display within the details element. Can be any Panel component,
        including a Markdown pane for rendering markdown content.""")

    collapsed = param.Boolean(default=True, doc="""
        Whether the details element is collapsed by default.""")

    _esm = Path(__file__).parent / "models" / "details.js"

    def __init__(self, object: Any, **params):
        super().__init__(object=object, **params)

    @param.depends("collapsed", watch=True)
    def _send_collapsed_update(self):
        """Send message to JS when collapsed state changes in Python"""
        self._send_msg({"type": "update_collapsed", "collapsed": self.collapsed})

    def _handle_msg(self, msg):
        """Handle messages from JS"""
        if 'collapsed' in msg:
            collapsed = msg['collapsed']
            with param.discard_events(self):
                self.collapsed = collapsed
