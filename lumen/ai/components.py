from pathlib import Path

import param

from panel.custom import Child, JSComponent

CSS = """
/* Max width for comfortable reading */
.left-panel-content {
    max-width: clamp(450px, 95vw, 1200px);
    margin: 0px auto;  /* Center the content */
    padding: 0px;
}

@keyframes jumpLeftRight {
    0%, 100% { transform: translateY(-50%); }
    25% { transform: translate(-4px, -50%); }
    50% { transform: translateY(-50%); }
    75% { transform: translate(4px, -50%); }
}

.split {
    display: flex;
    flex-direction: row;
    height: 100%;
    width: 100%;
}

.gutter {
    background-color: var(--panel-surface-color);
    background-repeat: no-repeat;
    background-position: 50%;
    cursor: col-resize;
}

.gutter.gutter-horizontal {
    background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAeCAYAAADkftS9AAAAIklEQVQoU2M4c+bMfxAGAgYYmwGrIIiDjrELjpo5aiZeMwF+yNnOs5KSvgAAAABJRU5ErkJggg==');
    z-index: 1;
}

ul.nav.flex-column {
    padding-inline-start: 0;
    margin: 0;
}

.toggle-icon {
    position: absolute;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10;
    opacity: 0.65;
    transition: opacity 0.2s, left 0.3s ease;
    left: -30px; /* Default position - will be updated by JS */
    top: 50%;
    transform: translateY(-50%);
}

.toggle-icon-inverted {
    position: absolute;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10;
    opacity: 0.65;
    transition: opacity 0.2s, right 0.3s ease;
    right: -30px; /* Default position - will be updated by JS */
    top: 50%;
    transform: translateY(-50%);
}

.toggle-icon:hover, .toggle-icon-inverted:hover {
    opacity: 1;
}

.toggle-icon svg, .toggle-icon-inverted svg {
    width: 50px;
    height: 50px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2px;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.collapsed-content {
    display: none !important;  /* required to expand / collapse properly */
}

/* Ensure the right panel and its contents are visible when expanded */
.split > div:nth-child(2) {
    overflow: visible;
    position: relative;
}

.split > div:nth-child(2) > div:not(.toggle-icon) {
    width: 100%;
    height: 100%;
    overflow: auto;
    padding-top: 36px; /* Space for the toggle icon */
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

    expanded_sizes = param.NumericTuple(default=(35, 65), length=2, doc="""
        The sizes of the two panels when expanded (as percentages).
        Default is (35, 65) which means the left panel takes up 35% of the space
        and the right panel takes up 65% when expanded.
        When invert=True, these percentages are automatically swapped.""")

    min_sizes = param.NumericTuple(default=(300, 0), length=2, doc="""
        The minimum sizes of the two panels (in pixels).
        Default is (300, 0) which means the left panel has a minimum width of 300px
        and the right panel has no minimum width.
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
