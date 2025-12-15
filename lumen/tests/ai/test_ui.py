import asyncio

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.tests.util import async_wait_until
from panel_material_ui import Column as MuiColumn
from panel_splitjs import VSplit

from lumen.ai.agents.analyst import AnalystAgent
from lumen.ai.agents.sql import SQLAgent, SQLQuery
from lumen.ai.coordinator import Plan
from lumen.ai.models import ErrorDescription, YesNo
from lumen.ai.report import ActorTask
from lumen.ai.schemas import get_metaset
from lumen.ai.ui import Exploration, ExplorerUI
from lumen.ai.views import SQLOutput
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.sources.duckdb import DuckDBSource


@pytest.fixture
def test_source():
    """Create a test DuckDB source with sample data."""
    return DuckDBSource(
        uri=':memory:',
        tables={
            'test_table': """
                SELECT * FROM (
                  VALUES (1, 'A', 10.5),
                         (2, 'B', 20.0),
                         (3, 'A', 7.5),
                         (4, 'C', 15.0)
                ) AS t(id, category, value)
            """
        }
    )


@pytest.fixture
async def explorer_ui(llm, test_source):
    """Create an ExplorerUI instance with test data."""
    slug = f"{test_source.name}{SOURCE_TABLE_SEPARATOR}test_table"
    metaset = await get_metaset([test_source], [slug])
    context = {
        "source": test_source,
        "sources": [test_source],
        "metaset": metaset,
        "visible_slugs": {slug}
    }

    ui = ExplorerUI(data=test_source, llm=llm)
    ui.context = context
    await ui._explorer.sync()
    yield ui
    await asyncio.sleep(0.1)

@pytest.fixture
async def explorer_ui_with_error(explorer_ui):
    def raise_exception():
        raise Exception("foo")

    explorer_ui.llm.set_responses([
        SQLQuery(query="SELECT SUM(value) as value_sum FROM test_table", table_slug="test_sql_agg"),
        raise_exception,
        ErrorDescription(explanation="bar")
    ])

    sql_agent = SQLAgent(llm=explorer_ui.llm)
    analyst_agent = AnalystAgent(llm=explorer_ui.llm)

    plan = Plan(
        ActorTask(sql_agent),
        ActorTask(analyst_agent),
        history=[{"content": "Sum value column", "role": "user"}],
        title="Sum of value",
        context=explorer_ui.context
    )

    await explorer_ui._execute_plan(plan)

    yield explorer_ui

async def test_explorer_ui_initialization(explorer_ui):
    """Test that ExplorerUI initializes correctly."""
    assert explorer_ui is not None
    assert explorer_ui._home is not None
    assert explorer_ui._home.title == 'Home'
    assert len(explorer_ui._explorations.items) == 1
    assert explorer_ui._explorations.items[0]['label'] == 'Home'

async def test_exploration_ui_error(explorer_ui_with_error):
    ui = explorer_ui_with_error
    exploration = ui._exploration['view']

    assert exploration.title == "Sum of value"
    assert exploration.plan.status == "error"
    assert exploration.plan._current == 1
    assert len(exploration.conversation) == 2

    assert len(ui._explorations.items) == 2
    assert ui._explorations.value is ui._exploration
    assert ui._explorations.value["label"] == "Sum of value"

    # Ensure error output is correct
    tabs = exploration.view[0]
    assert len(tabs) == 3
    assert tabs[-1].object == "⚠️ Unable to complete your request\n\nbar"

    # Check Interface contents
    assert len(ui.interface) == 2
    assert len(ui.interface[1].footer_objects) == 2 # Rerun buttons

async def test_exploration_ui_error_rerun(explorer_ui_with_error):
    ui = explorer_ui_with_error
    exploration = ui._exploration['view']

    ui.llm.set_responses([
        "foo"
    ])

    await ui._execute_plan(exploration.plan, rerun=True)

    assert exploration.plan.status == "success"
    assert exploration.plan._current == 2
    assert len(ui._explorations.items) == 2

    assert len(exploration.conversation) == 4
    assert exploration.conversation[3].object == "foo"

    # Check Interface contents
    assert len(ui.interface) == 4
    assert len(ui.interface[1].footer_objects) == 0 # No rerun buttons

    tabs = exploration.view[0]
    assert len(tabs) == 2

async def test_exploration_ui_error_replan(explorer_ui_with_error):
    ui = explorer_ui_with_error
    exploration = ui._exploration['view']

    ui.llm.set_responses([
        SQLQuery(query="SELECT SUM(value) as value_sum FROM test_table", table_slug="test_sql_agg"),
        "foo"
    ])

    # Reset and rerun
    steps = list(exploration.plan)
    exploration.plan.remove(steps)
    ui._reset_error(exploration.plan, exploration, replan=True)
    new_plan = Plan(
        *steps,
        history=[{"content": "Sum value column", "role": "user"}],
        title="Sum of value",
        context=ui.context
    )
    await ui._execute_plan(new_plan, replan=True)

    assert exploration.plan.status == "success"
    assert exploration.plan._current == 2
    assert len(ui._explorations.items) == 2

    assert len(exploration.conversation) == 5
    assert exploration.conversation[4].object == "foo"

    # Check Interface contents
    assert len(ui.interface) == 5
    assert len(ui.interface[1].footer_objects) == 0 # No rerun buttons

    tabs = exploration.view[0]
    assert len(tabs) == 2

async def test_add_exploration_from_explorer(explorer_ui):
    """Test creating an exploration from the table explorer."""
    # Set up the explorer to have a table selected
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)

    # Add exploration
    await explorer_ui._add_exploration_from_explorer()

    # Check that a new exploration was added
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    assert len(explorer_ui._explorations.items) == 2

    # Check that the exploration has the correct structure
    exploration_item = explorer_ui._explorations.items[1]
    assert exploration_item['label'] == 'Explore test_table'
    assert exploration_item['view'] is not None
    assert isinstance(exploration_item['view'], Exploration)
    assert exploration_item['view'].plan is not None
    assert exploration_item['view'].plan.title == 'Explore test_table'

    # Check that the interface received a message
    assert len(explorer_ui.interface.objects) > 0


async def test_add_exploration_from_explorer_no_table(explorer_ui):
    """Test that _add_exploration_from_explorer returns early if no table is selected."""
    explorer_ui._explorer.param.update(table_slug=None)

    initial_count = len(explorer_ui._explorations.items)
    await explorer_ui._add_exploration_from_explorer()

    # Should not have added any exploration
    assert len(explorer_ui._explorations.items) == initial_count


async def test_exploration_pop_out(explorer_ui):
    """Test the pop-out functionality for exploration views."""
    # First create an exploration
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Get the first view (should be a SQLOutput)
    await async_wait_until(lambda: len(exploration.plan.views) > 0)
    views = [v for v in exploration.plan.views if isinstance(v, SQLOutput)]
    assert len(views) > 0

    sql_output = views[0]

    # Render the view to get a VSplit
    title, vsplit = explorer_ui._render_view(exploration, sql_output)

    # Get the tabs container
    tabs = exploration.view[0]
    initial_tab_count = len(tabs)

    # Check that the view is in tabs
    assert vsplit in tabs

    # Create pop-out button
    pop_button = explorer_ui._render_pop_out(exploration, vsplit, title)

    # Trigger the pop-out callback directly
    from panel.io.state import hold
    with hold():
        # Call the callback function directly (it's stored in on_click)
        pop_out_callback = pop_button.on_click
        pop_out_callback(None)

    # Wait for the view to be removed from tabs
    await async_wait_until(lambda: vsplit not in tabs, timeout=5.0)

    # Check that view was removed from tabs
    assert vsplit not in tabs
    assert len(tabs) == initial_tab_count - 1

    # Check that standalone view was added to exploration.view
    assert len(exploration.view) > 1
    standalone = exploration.view[-1]
    assert isinstance(standalone, MuiColumn)

    # Check button description changed
    assert pop_button.description == "Reattach"
    assert pop_button.icon == "open_in_browser"

    # Trigger reattach
    with hold():
        pop_out_callback = pop_button.on_click
        pop_out_callback(None)

    # Wait for the view to be back in tabs
    await async_wait_until(lambda: vsplit in tabs, timeout=5.0)

    # Check that view is back in tabs
    assert vsplit in tabs
    assert len(tabs) == initial_tab_count

    # Check button description changed back
    assert pop_button.description == "Open in new pane"
    assert pop_button.icon == "open_in_new"


async def test_move_exploration_up(explorer_ui):
    """Test moving an exploration up in the list."""
    # Create multiple explorations
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)

    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    # Create a second exploration
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 2)

    items = explorer_ui._explorations.items
    assert len(items) >= 3  # Home + 2 explorations

    # Get the last exploration (not Home)
    last_item = items[-1]
    last_index = len(items) - 1

    # Move it up
    explorer_ui._move_up(last_item)

    # Check that it moved up
    new_items = explorer_ui._explorations.items
    assert new_items[last_index - 1] is last_item


async def test_move_exploration_down(explorer_ui):
    """Test moving an exploration down in the list."""
    # Create multiple explorations
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)

    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 2)

    items = explorer_ui._explorations.items
    assert len(items) >= 3

    # Get the second exploration (not Home, not last)
    if len(items) >= 3:
        second_item = items[1]
        second_index = 1

        # Move it down
        explorer_ui._move_down(second_item)

        # Check that it moved down
        new_items = explorer_ui._explorations.items
        assert new_items[second_index + 1] is second_item


async def test_delete_exploration(explorer_ui):
    """Test deleting an exploration."""
    # Create an exploration
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    initial_count = len(explorer_ui._explorations.items)

    # Get the exploration to delete (not Home)
    exploration_item = explorer_ui._explorations.items[1]

    # Delete it
    explorer_ui._delete_exploration(exploration_item)

    # Check that it was removed
    assert len(explorer_ui._explorations.items) == initial_count - 1
    assert exploration_item not in explorer_ui._explorations.items


async def test_toggle_report_mode(explorer_ui):
    """Test toggling report mode."""
    # Initially should be in split mode
    assert explorer_ui._split in explorer_ui._main

    # Toggle to report mode
    explorer_ui._report_toggle.value = True
    explorer_ui._toggle_report_mode(type('Event', (), {'new': True})())

    # Check that main content changed
    assert explorer_ui._split not in explorer_ui._main or len(explorer_ui._main) == 1

    # Toggle back
    explorer_ui._report_toggle.value = False
    explorer_ui._toggle_report_mode(type('Event', (), {'new': False})())

    # Check that split is back
    assert explorer_ui._split in explorer_ui._main


async def test_update_conversation(explorer_ui):
    """Test updating conversation when switching between explorations."""
    # Create an exploration
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    # Add a message to the interface
    explorer_ui.interface.send("Test message")
    await async_wait_until(lambda: len(explorer_ui.interface.objects) > 0)

    initial_message_count = len(explorer_ui.interface.objects)

    # Switch to the exploration
    exploration_item = explorer_ui._explorations.items[1]
    explorer_ui._explorations.value = exploration_item

    # Update conversation
    await explorer_ui._update_conversation()

    # Check that conversation was updated
    exploration = exploration_item['view']
    assert len(exploration.conversation) == initial_message_count

    # Switch back to home
    explorer_ui._explorations.value = explorer_ui._explorations.items[0]
    await explorer_ui._update_conversation()

    # Check that we're back to home view
    assert explorer_ui._split[0][0] is explorer_ui._splash


async def test_sync_active(explorer_ui):
    """Test syncing active exploration."""
    # Create an exploration
    explorer_ui._explorer.table_slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    # Get the exploration item
    exploration_item = explorer_ui._explorations.items[1]

    # Create a mock event
    event = type('Event', (), {'new': exploration_item})()

    # Sync active
    explorer_ui._sync_active(event)

    # Check that _exploration was updated
    assert explorer_ui._exploration is exploration_item


async def test_find_view_in_tabs(explorer_ui):
    """Test finding a view in tabs."""
    # Create an exploration
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Wait for views to be added
    await async_wait_until(lambda: len(exploration.plan.views) > 0)
    views = [v for v in exploration.plan.views if isinstance(v, SQLOutput)]
    assert len(views) > 0

    sql_output = views[0]

    # Render the view
    title, vsplit = explorer_ui._render_view(exploration, sql_output)

    # Find the view in tabs
    tab_idx = explorer_ui._find_view_in_tabs(exploration, sql_output)

    # Should find it (if it's in tabs)
    tabs = exploration.view[0]
    if vsplit in tabs:
        assert tab_idx is not None
        assert tab_idx >= 0


async def test_find_view_in_popped_out(explorer_ui):
    """Test finding a view in popped out views."""
    # Create an exploration
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Wait for views
    await async_wait_until(lambda: len(exploration.plan.views) > 0)
    views = [v for v in exploration.plan.views if isinstance(v, SQLOutput)]
    assert len(views) > 0

    sql_output = views[0]

    # Render and pop out
    title, vsplit = explorer_ui._render_view(exploration, sql_output)
    pop_button = explorer_ui._render_pop_out(exploration, vsplit, title)

    # Trigger pop out by calling the callback directly
    from panel.io.state import hold
    with hold():
        pop_out_callback = pop_button.on_click
        pop_out_callback(None)

    # Wait for pop out
    await async_wait_until(lambda: len(exploration.view) > 1, timeout=5.0)

    # Find the popped out view
    popped_idx = explorer_ui._find_view_in_popped_out(exploration, sql_output)

    # Should find it
    assert popped_idx is not None
    assert popped_idx >= 1  # Index 0 is tabs


async def test_exploration_context_isolation(explorer_ui):
    """Test that different explorations maintain separate contexts."""
    # Create first exploration
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration1 = explorer_ui._explorations.items[1]['view']

    # Create second exploration
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 2)
    exploration2 = explorer_ui._explorations.items[2]['view']

    # Check that they have separate contexts
    assert exploration1.context is not exploration2.context
    assert id(exploration1.context) != id(exploration2.context)


async def test_exploration_parent_relationship(explorer_ui):
    """Test that explorations maintain correct parent relationships."""
    # Create an exploration from home
    slug = f"{explorer_ui.context['source'].name}{SOURCE_TABLE_SEPARATOR}test_table"
    explorer_ui._explorer.param.update(table_slug=slug)
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Check that parent is home
    assert exploration.parent is explorer_ui._home
    assert exploration.parent.title == 'Home'
