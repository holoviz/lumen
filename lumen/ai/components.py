from pathlib import Path
from typing import Any

import param

from panel.custom import Child, JSComponent
from panel.layout import Column, HSpacer, Row
from panel.pane import Markdown
from panel.viewable import Viewer
from panel_material_ui import (
    Card, CheckBoxGroup, IconButton, Switch,
)

from .config import SOURCE_TABLE_SEPARATOR
from .memory import memory

CSS = """
/* Base styles for the split container */
.split {
    display: flex;
    flex-direction: row;
    height: 100%;
    width: 100%;
    background-color: var(--panel-surface-color);
    overflow-y: clip; /* Clip overflow to prevent scrollbars; inner content will have their own */
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


class TableSourceCard(Viewer):
    """
    A component that displays a single data source as a card with table selection controls.

    The card includes:
    - A header with the source name and a checkbox to toggle all tables
    - A delete button (if multiple sources exist)
    - Individual checkboxes for each table in the source
    """

    all_selected = param.Boolean(default=True, doc="""
        Whether all tables should be selected by default.""")

    collapsed = param.Boolean(default=False, doc="""
        Whether the card should start collapsed.""")

    deletable = param.Boolean(default=True, doc="""
        Whether to show the delete button.""")

    delete = param.Event(doc="""Action to delete this source from memory.""")

    selected = param.List(default=None, doc="""
        List of currently selected table names.""")

    source = param.Parameter(doc="""
        The data source to display in this card.""")

    def __init__(self, **params):
        super().__init__(**params)
        self.all_tables = self.source.get_tables()

        # Determine which tables are currently visible
        if self.selected is None:
            visible_tables = []
            for table in self.all_tables:
                visible_tables.append(table)
            self.selected = visible_tables

        # Create widgets once in init
        self.source_toggle = Switch.from_param(
            self.param.all_selected,
            name=f"{self.source.name}",
            margin=(5, -5, 0, 3),
            sizing_mode='fixed',
        )

        self.delete_button = IconButton.from_param(
            self.param.delete,
            icon='delete',
            icon_size='1em',
            color="danger",
            margin=(5, 0, 0, 0),
            sizing_mode='fixed',
            width=40,
            height=40,
            visible=self.param.deletable
        )

        self.table_checkbox = CheckBoxGroup.from_param(
            self.param.selected,
            options=self.all_tables,
            sizing_mode='stretch_width',
            margin=(0, 10),
            name="",
        )

    @param.depends('all_selected', watch=True)
    def _on_source_toggle(self):
        """Handle source checkbox toggle (all tables on/off)."""
        if not self.all_selected and len(self.selected) == len(self.all_tables):
            # Important to check to see if all tables are selected for intuitive behavior
            self.selected = []
        elif self.all_selected:
            self.selected = self.all_tables

    @param.depends('selected', watch=True)
    def _update_visible_slugs(self):
        """Update visible_slugs in memory based on selected tables."""
        self.all_selected = len(self.selected) == len(self.all_tables)
        for table in self.all_tables:
            table_slug = f"{self.source.name}{SOURCE_TABLE_SEPARATOR}{table}"
            if table in self.selected:
                memory['visible_slugs'].add(table_slug)
            else:
                memory['visible_slugs'].discard(table_slug)
        memory.trigger('visible_slugs')

    @param.depends('delete', watch=True)
    def _delete_source(self):
        """Handle source deletion via param.Action."""
        if self.source in memory.get("sources", []):
            # Remove all tables from this source from visible_slugs
            for table in self.all_tables:
                table_slug = f"{self.source.name}{SOURCE_TABLE_SEPARATOR}{table}"
                memory['visible_slugs'].discard(table_slug)

            memory["sources"] = [
                source for source in memory.get("sources", [])
                if source is not self.source
            ]

    def __panel__(self):
        card_header = Row(
            self.source_toggle,
            HSpacer(),
            self.delete_button,
            sizing_mode='stretch_width',
            align='start',
            height=35,
            margin=0
        )

        # Create the card
        return Card(
            self.table_checkbox,
            header=card_header,
            collapsible=True,
            collapsed=self.param.collapsed,
            sizing_mode='stretch_width',
            margin=0,
        )


class SourceCatalog(Viewer):
    """
    A component that displays all data sources with table selection controls.

    This component shows each source as a collapsible card with:
    - A header checkbox to toggle all tables in the source
    - Individual checkboxes for each table
    - A delete button to remove the source (if multiple sources exist)

    Tables can be selectively shown/hidden using the checkboxes, which updates
    the 'visible_slugs' set in memory.
    """

    sources = param.List(default=[], doc="""
        List of data sources to display in the catalog.""")

    def __init__(self, **params):
        self._title = Markdown(margin=0)
        self._cards_column = Column(
            margin=0,
        )
        self._layout = Column(
            self._title,
            self._cards_column,
            margin=0,
            sizing_mode='stretch_width'
        )
        super().__init__(**params)

    @param.depends("sources", watch=True, on_init=True)
    def _refresh(self, sources=None):
        """
        Trigger the catalog with new sources.

        Args:
            sources: Optional list of sources. If None, uses sources from memory.
        """
        sources = self.sources or memory.get('sources', [])

        # Create a lookup of existing cards by source
        existing_cards = {
            card.source: card for card in self._cards_column.objects
            if isinstance(card, TableSourceCard) and card.source in sources
        }

        # Build the new cards list
        source_cards = []
        multiple_sources = len(sources) > 1
        for source in sources:
            if source in existing_cards:
                # Reuse existing card and update its deletable property
                card = existing_cards[source]
                card.deletable = multiple_sources
                source_cards.append(card)
            else:
                # Create new card for new source
                source_card = TableSourceCard(
                    source=source,
                    deletable=multiple_sources,
                    collapsed=multiple_sources,
                )
                source_cards.append(source_card)

        self._cards_column.objects = source_cards

        if len(self.sources) == 0:
            self._title.object = "No sources available. Add a source to get started."
        else:
            self._title.object = "Select the table and document sources you want visible to the LLM."

    def __panel__(self):
        """
        Create the source catalog UI.

        Returns a Column containing all source cards or a message if no sources exist.
        """
        return self._layout
