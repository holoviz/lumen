import asyncio

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.layout import Column, Row
from panel.tests.util import async_wait_until
from panel.util import edit_readonly
from panel_material_ui import Container
from panel_splitjs import VSplit

from lumen.ai.agents.chat import ChatAgent
from lumen.ai.agents.sql import SQLAgent, make_sql_model
from lumen.ai.coordinator import Plan
from lumen.ai.models import ErrorDescription
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
    await asyncio.sleep(0.1)
    yield ui
    await asyncio.sleep(0.1)

@pytest.fixture
async def explorer_ui_with_error(explorer_ui):
    def raise_exception():
        raise Exception("foo")

    # Create the proper SQL model with tables field for single source
    test_source = explorer_ui.context["source"]
    SQLQueryWithTables = make_sql_model([(test_source.name, "test_table")])
    explorer_ui.llm.set_responses([
        SQLQueryWithTables(
            query="SELECT SUM(value) as value_sum FROM test_table",
            table_slug="test_sql_agg",
            tables=["test_table"]
        ),
        raise_exception,
        ErrorDescription(explanation="bar")
    ])

    sql_agent = SQLAgent(llm=explorer_ui.llm)
    chat_agent = ChatAgent(llm=explorer_ui.llm)

    plan = Plan(
        ActorTask(sql_agent),
        ActorTask(chat_agent),
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
    assert tabs[-1].object == "❌ **Unable to complete your request**\n\nbar"

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

    # Create the proper SQL model with tables field for single source
    test_source = ui.context["source"]
    SQLQueryWithTables = make_sql_model([(test_source.name, "test_table")])
    ui.llm.set_responses([
        SQLQueryWithTables(
            query="SELECT SUM(value) as value_sum FROM test_table",
            table_slug="test_sql_agg",
            tables=["test_table"]
        ),
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
    explorer_ui._explorer.param.update(table_slug="test_table")

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
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Get the first view (should be a SQLOutput)
    await async_wait_until(lambda: len(exploration.plan.views) > 0)
    views = [v for v in exploration.plan.views if isinstance(v, SQLOutput)]
    assert len(views) > 0

    sql_output = views[0]

    # Get the tabs container
    tabs = exploration.view[0]
    initial_tab_count = len(tabs)

    assert initial_tab_count == 2

    # Check that the view is in tabs
    view = tabs[1]
    title = tabs._names[1]
    assert isinstance(view, Column)
    assert len(view) == 2
    controls, vsplit = view
    assert isinstance(controls, Row)
    assert isinstance(vsplit, VSplit)

    # Create pop-out button
    pop_button = explorer_ui._render_pop_out(exploration, view, title)

    # Trigger the pop-out callback directly
    pop_button.param.trigger("clicks")

    # Wait for the view to be removed from tabs
    await async_wait_until(lambda: view not in tabs, timeout=5.0)

    # Check that view was removed from tabs
    assert view not in tabs
    assert len(tabs) == initial_tab_count - 1

    # Check that standalone view was added to exploration.view
    assert len(exploration.view) > 1
    standalone = exploration.view[-1]
    assert isinstance(standalone, Column)

    # Check button description changed
    assert pop_button.description == "Reattach"
    assert pop_button.icon == "open_in_browser"
    pop_button.param.trigger("clicks")

    # Wait for the view to be back in tabs
    await async_wait_until(lambda: view in tabs, timeout=5.0)

    # Check that view is back in tabs
    assert view in tabs
    assert len(tabs) == initial_tab_count

    # Check button description changed back
    assert pop_button.description == "Open in new pane"
    assert pop_button.icon == "open_in_new"


async def test_sync_active(explorer_ui):
    """Test syncing active exploration."""
    # Create an exploration
    explorer_ui._explorer.table_slug = "test_table"
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
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Wait for views to be added
    await async_wait_until(lambda: len(exploration.plan.views) > 0)
    views = [v for v in exploration.plan.views if isinstance(v, SQLOutput)]
    assert len(views) > 0

    sql_output = views[0]

    # Render the view
    title, view = explorer_ui._render_view(exploration, sql_output)

    # Find the view in tabs
    tab_idx = explorer_ui._find_view_in_tabs(exploration, sql_output)

    # Should find it (if it's in tabs)
    tabs = exploration.view[0]
    if view in tabs:
        assert tab_idx is not None
        assert tab_idx >= 0


async def test_find_view_in_popped_out(explorer_ui):
    """Test finding a view in popped out views."""
    # Create an exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Wait for views
    await async_wait_until(lambda: len(exploration.plan.views) > 0)
    views = [v for v in exploration.plan.views if isinstance(v, SQLOutput)]
    assert len(views) > 0

    sql_output = views[0]

    # Render and pop out
    tabs = exploration.view[0]
    title = tabs._names[1]
    view = tabs[1]
    pop_button = explorer_ui._render_pop_out(exploration, view, title)

    # Trigger pop out by calling the callback directly
    pop_button.param.trigger("clicks")

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
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration1 = explorer_ui._explorations.items[1]['view']

    # Create second exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 2)
    exploration2 = explorer_ui._explorations.items[2]['view']

    # Check that they have separate contexts
    assert exploration1.context is not exploration2.context
    assert id(exploration1.context) != id(exploration2.context)


async def test_exploration_parent_relationship(explorer_ui):
    """Test that explorations maintain correct parent relationships."""
    # Create an exploration from home
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Check that parent is home
    assert exploration.parent is explorer_ui._home
    assert exploration.parent.title == 'Home'


async def test_chat_upload_flow_opens_dialog_and_starts_exploration(explorer_ui, monkeypatch):
    ui = explorer_ui
    old_sources = list(ui.context.get("sources", []))

    execute_plan = AsyncMock()
    respond = AsyncMock(return_value=object())
    monkeypatch.setattr(ui, "_execute_plan", execute_plan)
    monkeypatch.setattr(ui._coordinator, "respond", respond)

    ui._chat_input.value_input = "Show me the uploaded data"

    with patch("panel.state.add_periodic_callback", lambda *a, **kw: None):
        with edit_readonly(ui._chat_input):
            ui._chat_input.pending_uploads = 1
            ui._chat_input.value_uploaded = {}
            ui._on_submit()

    assert ui._pending_query == "Show me the uploaded data"

    with edit_readonly(ui._chat_input):
        ui._chat_input.pending_uploads = 0
        ui._chat_input.value_uploaded = {
            "uploaded.csv": {"value": b"id,value\n1,2\n"}
        }
    ui._on_submit(wait=True)
    assert ui._sources_dialog_content.open is True
    assert ui._source_content.active == ui._source_controls.index(ui._upload_controls)

    ui._upload_controls.param.trigger("add")

    await async_wait_until(lambda: ui._sources_dialog_content.open is False)
    await async_wait_until(lambda: len(ui.context.get("sources", [])) > len(old_sources))
    await async_wait_until(lambda: execute_plan.await_count > 0)

    new_sources = [src for src in ui.context["sources"] if src not in old_sources]
    assert len(new_sources) == 1
    assert "uploaded" in new_sources[0].get_tables()

    assert ui._main[0] is ui.interface
    assert ui._main[0][0].object == "Show me the uploaded data"


# Tests for view transition states

async def test_initial_state_shows_splash(explorer_ui):
    """Test 1: On start the user sees just the self._splash in main."""
    # Initially, _main should contain only _splash
    assert len(explorer_ui._main) == 1
    assert explorer_ui._main[0] is explorer_ui._splash

    # Should be on home
    assert explorer_ui._explorations.value['view'] is explorer_ui._home
    assert len(explorer_ui._explorations.items) == 1  # Only home


async def test_exploration_launch_transitions_to_split(explorer_ui):
    """Test 2: When an exploration is launched we switch from splash to split view."""
    # Initially showing splash
    assert explorer_ui._main[0] is explorer_ui._splash

    # Create an exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    # Wait for exploration to be created
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    # After exploration is created, should transition to split view
    # The view should update when views are added (which happens in _add_exploration_from_explorer)
    await async_wait_until(lambda: explorer_ui._split in explorer_ui._main or explorer_ui._navigation in explorer_ui._main)

    # Should now show split view (with navigation if there are explorations)
    if explorer_ui._should_show_navigation():
        assert len(explorer_ui._main) == 2
        assert explorer_ui._main[0] is explorer_ui._navigation
        assert explorer_ui._main[1] is explorer_ui._split
    else:
        assert len(explorer_ui._main) == 1
        assert explorer_ui._main[0] is explorer_ui._split

    # Should not be showing splash anymore
    assert explorer_ui._splash not in explorer_ui._main


async def test_exploration_with_pipeline_data_shows_split(explorer_ui):
    """Test 2b: When plan produces data, should show split view with navigation."""
    # Create an exploration with pipeline data
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration = explorer_ui._explorations.items[1]['view']

    # Wait for views to be added (pipeline data)
    await async_wait_until(lambda: len(exploration.plan.views) > 0, timeout=5.0)

    # Should show split view with navigation (since we have explorations)
    assert explorer_ui._should_show_navigation()
    assert len(explorer_ui._main) == 2
    assert explorer_ui._main[0] is explorer_ui._navigation
    assert explorer_ui._main[1] is explorer_ui._split

    # Output should contain the exploration
    assert len(explorer_ui._output) > 1
    assert explorer_ui._output[1] is exploration.view


async def test_delete_exploration_switches_to_home(explorer_ui):
    """Test 3: When deleting an exploration, switch to home if no parent exploration."""
    # Create an exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()

    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    exploration_item = explorer_ui._explorations.items[1]

    # Delete the exploration
    await explorer_ui._delete_exploration(exploration_item)
    await asyncio.sleep(0.1)

    # Should switch back to home (splash view)
    assert len(explorer_ui._explorations.items) == 1
    assert explorer_ui._explorations.value['view'] is explorer_ui._home
    assert explorer_ui._main[0] is explorer_ui._splash


async def test_delete_exploration_switches_to_parent(explorer_ui):
    """Test 3b: When deleting a child exploration, switch to parent if available."""
    # Create first exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)
    parent_item = explorer_ui._explorations.items[1]
    parent_exploration = parent_item['view']

    sql_agent = SQLAgent(llm=explorer_ui.llm)
    test_source = explorer_ui.context["source"]
    SQLQueryWithTables = make_sql_model([(test_source.name, "test_table")])
    explorer_ui.llm.set_responses([
        SQLQueryWithTables(
            query="SELECT * FROM test_table LIMIT 5",
            table_slug="test_limit",
            tables=["test_table"]
        )
    ])

    child_plan = Plan(
        ActorTask(sql_agent),
        history=[{"content": "Show first 5 rows", "role": "user"}],
        title="First 5 rows",
        context=explorer_ui._explorations.value["view"].context,
        is_followup=True
    )

    await explorer_ui._add_exploration(child_plan, parent_exploration)

    await asyncio.sleep(0.1)

    # Verify we're currently on the child
    child_item = explorer_ui._explorations.items[1]['items'][0]  # First child of first exploration
    assert explorer_ui._explorations.value is child_item

    # Delete the child exploration
    await explorer_ui._delete_exploration(child_item)

    # Should switch back to parent exploration
    assert explorer_ui._explorations.value is explorer_ui._explorations.items[1]
    assert explorer_ui._explorations.value['view'] is parent_exploration


async def test_switch_to_report_mode_with_no_explorations(explorer_ui):
    """Test 4: When switching to report mode with no explorations, show placeholder."""
    # Initially no explorations (only home)
    assert len(explorer_ui._explorations.items) == 1

    # Switch to report mode
    explorer_ui._toggle_report_mode(True)

    # Should show placeholder message
    assert len(explorer_ui._main) == 1
    main_content = explorer_ui._main[0]
    assert isinstance(main_content, Column)
    # Should contain the "No Explorations Yet" message
    assert len(main_content) >= 1

    # Navigation should show "Report"
    assert explorer_ui._navigation_title.object == "Report"


async def test_switch_to_report_mode_with_explorations(explorer_ui):
    """Test 4b: When switching to report mode with explorations, show Report."""
    # Create an exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    # Wait for views to be added
    exploration = explorer_ui._explorations.items[1]['view']
    await async_wait_until(lambda: len(exploration.plan.views) > 0, timeout=5.0)

    # Switch to report mode
    explorer_ui._handle_sidebar_event(explorer_ui._sidebar_menu.items[1])

    # Should show Report component with navigation
    assert explorer_ui._should_show_navigation()
    assert len(explorer_ui._main) == 2
    assert explorer_ui._main[0] is explorer_ui._navigation

    # Main content should be a Report instance
    assert isinstance(explorer_ui._main[1], Container)
    assert len(explorer_ui._main[1]) == 2

    # Navigation should show "Report"
    assert explorer_ui._navigation_title.object == "Report"


async def test_switch_back_from_report_to_exploration(explorer_ui):
    """Test 5: When switching back from report to exploration, show selected exploration."""
    # Create an exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    exploration_item = explorer_ui._explorations.items[1]
    exploration = exploration_item['view']

    # Wait for views to be added
    await async_wait_until(lambda: len(exploration.plan.views) > 0, timeout=5.0)

    # Switch to report mode
    explorer_ui._handle_sidebar_event(explorer_ui._sidebar_menu.items[1])
    assert isinstance(explorer_ui._main[1], Container)

    # Switch back to exploration mode
    explorer_ui._handle_sidebar_event(explorer_ui._sidebar_menu.items[0])

    # Should show split view with the selected exploration
    assert len(explorer_ui._main) == 2
    assert explorer_ui._main[0] is explorer_ui._navigation
    assert explorer_ui._main[1] is explorer_ui._split

    # Output should contain the exploration
    assert explorer_ui._output[1] is exploration.view

    # Navigation should show "Exploration"
    assert explorer_ui._navigation_title.object == "Exploration"


async def test_navigation_visibility_with_explorations(explorer_ui):
    """Test that navigation pane is shown/hidden correctly based on explorations."""
    # Initially no explorations (only home) - no navigation
    assert not explorer_ui._should_show_navigation()
    assert len(explorer_ui._main) == 1
    assert explorer_ui._navigation not in explorer_ui._main

    # Create an exploration
    explorer_ui._explorer.param.update(table_slug="test_table")
    await explorer_ui._add_exploration_from_explorer()
    await async_wait_until(lambda: len(explorer_ui._explorations.items) > 1)

    # Now should show navigation
    assert explorer_ui._should_show_navigation()
    # View should update to include navigation
    explorer_ui._update_main_view()
    assert len(explorer_ui._main) == 2
    assert explorer_ui._main[0] is explorer_ui._navigation

    # Delete exploration
    exploration_item = explorer_ui._explorations.items[1]
    await explorer_ui._delete_exploration(exploration_item)

    # Navigation should be hidden again
    assert not explorer_ui._should_show_navigation()
    assert len(explorer_ui._main) == 1
    assert explorer_ui._navigation not in explorer_ui._main
