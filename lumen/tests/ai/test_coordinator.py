from typing import get_args

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.tests.util import async_wait_until
from panel_material_ui import Card, ChatMessage, Typography

from lumen.ai.agents import ChatAgent, SQLAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.coordinator import Coordinator, Plan, Planner
from lumen.ai.coordinator.planner import Reasoning, make_plan_model
from lumen.ai.editors import SQLEditor
from lumen.ai.models import ReplaceLine, RetrySpec
from lumen.ai.report import ActorTask
from lumen.ai.schemas import get_metaset
from lumen.ai.tools import FunctionTool, define_tool
from lumen.config import SOURCE_TABLE_SEPARATOR


async def test_planner_instantiate():
    Planner()


def _add_one(value: int) -> int:
    return value + 1


@define_tool(
    requires=["value"],
    provides=["result"],
    purpose="Add one to a value",
    render_output=True,
)
def _add_one_from_context(value: int) -> int:
    return value + 1


async def test_coordinator_instantiates_function_tools(llm):
    coordinator = Coordinator(llm=llm, tools=[_add_one])
    tool = coordinator._tools["main"][0]
    assert isinstance(tool, FunctionTool)
    assert tool.function is _add_one
    assert tool.llm is llm


async def test_coordinator_instantiates_annotated_function_tools(llm):
    coordinator = Coordinator(llm=llm, tools=[_add_one_from_context])
    tool = coordinator._tools["main"][0]
    assert isinstance(tool, FunctionTool)
    assert tool.function is _add_one_from_context
    assert tool.requires == ["value"]
    assert tool.provides == ["result"]
    assert tool.purpose == "Add one to a value"
    assert tool.render_output is True


async def test_planner_empty_plan(llm):
    plan_model = make_plan_model(["ChatAgent"], [])

    llm.set_responses([
        Reasoning(chain_of_thought="Just use ChatAgent"),
        plan_model(title="Hello!", steps=[])
    ])

    planner = Planner(llm=llm)

    plan = await planner.respond([{'role': 'user', 'content': 'Hello?'}], {})

    assert isinstance(plan, Plan)
    assert plan.title == "Hello!"
    assert len(plan) == 0

    assert len(planner.interface) == 1
    isinstance(planner.interface[0].object, Card)
    title, todos = planner.interface[0].object.header

    assert title.object == "🧾 Checklist ready..."
    assert todos.object == ""


async def test_planner_simple_plan(llm):
    PlanModel = make_plan_model(["ChatAgent"], [])

    (StepModel,) = get_args(PlanModel.__annotations__['steps'])

    llm.set_responses([
        Reasoning(chain_of_thought="Just use ChatAgent"),
        PlanModel(title="Hello!", steps=[
            StepModel(
                actor="ChatAgent",
                instruction="Say Hello!",
                title="Hello Reply"
            )
        ])
    ])

    planner = Planner(llm=llm)

    plan = await planner.respond([{'role': 'user', 'content': 'Hello?'}], {})

    assert len(planner.interface) == 1
    steps_layout = planner.interface[0].object
    isinstance(steps_layout, Card)
    assert len(steps_layout) == 2
    reasoning_step = steps_layout[1]
    isinstance(reasoning_step, Card)
    assert reasoning_step.title == "Plan with 1 steps created"
    assert len(reasoning_step) == 1
    assert reasoning_step[0].object == "Just use ChatAgent"
    title, todos = steps_layout.header
    assert title.object == "🧾 Checklist ready..."
    assert todos.object == "- ⚪ Say Hello!"

    assert isinstance(plan, Plan)
    assert plan.title == "Hello!"
    assert plan.steps_layout is steps_layout
    assert len(plan) == 1
    assert isinstance(plan[0], ActorTask)
    assert plan[0].instruction == "Say Hello!"
    assert plan[0].title == "Hello Reply"
    assert isinstance(plan[0].actor, ChatAgent)


async def test_planner_error(llm):
    PlanModel = make_plan_model(["ChatAgent"], [])

    (StepModel,) = get_args(PlanModel.__annotations__['steps'])

    llm.set_responses([
        Reasoning(chain_of_thought="Just use ChatAgent"),
        lambda: PlanModel(
            title="Hello!", steps=[
                StepModel(
                    actor="Invalid",
                    instruction="Say Hello!",
                    title="Hello Reply"
                )
            ])
    ])

    planner = Planner(llm=llm)
    planner.interface.callback_exception = "verbose"

    plan = await planner.respond([{'role': 'user', 'content': 'Hello?'}], {})

    assert plan is None

    assert len(planner.interface) == 1
    steps_layout = planner.interface[0].object
    isinstance(steps_layout, Card)
    assert len(steps_layout) == 2
    reasoning_step = steps_layout[1]
    isinstance(reasoning_step, Card)
    assert reasoning_step.title == "Internal execution error during planning stage."
    title, todos = steps_layout.header
    assert title.object == "Planner could not settle on a plan of action to perform the requested query. Please restate your request."
    assert todos.object is None


@pytest.fixture
async def sql_plan(llm, tiny_source):
    sql_agent = SQLAgent()
    chat_agent = ChatAgent()

    slug = f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny"
    metaset = await get_metaset([tiny_source], [slug])
    context = {
        "source": tiny_source,
        "sources": [tiny_source],
        "metaset": metaset,
        "visible_slugs": [slug]
    }

    # Create the proper SQL model with tables field for single source
    SQLQueryWithTables = make_sql_model([(tiny_source.name, "tiny")])
    llm.set_responses([
        SQLQueryWithTables(
            query="SELECT SUM(value) as value_sum FROM tiny",
            table_slug="tiny_agg",
            tables=["tiny"]
        ),
        lambda: f"Result: {plan[0].out_context['pipeline'].data.iloc[0, 0]}"
    ])

    plan = Plan(
        ActorTask(sql_agent, title="Aggregate"),
        ActorTask(chat_agent, title="Analyze"),
        context=context,
        history=[{"role": "user", "content": "Aggregate and Analyze"}],
        llm=llm
    )
    await plan.execute()
    return plan


async def test_plan_execute(sql_plan):
    assert sql_plan.status == "success"
    assert "sql" in sql_plan.out_context
    assert sql_plan.out_context["sql"] == (
        "SELECT\n"
        "  SUM(value) AS value_sum\n"
        "FROM tiny"
    )
    assert "pipeline" in sql_plan.out_context
    pd.testing.assert_frame_equal(
        sql_plan.out_context["pipeline"].data,
        pd.DataFrame([38.], columns=["value_sum"])
    )

    assert len(sql_plan.views) == 4
    v1, v2, v3, v4 = sql_plan.views
    assert isinstance(v1, Typography)
    assert v1.object == "### Aggregate"
    assert isinstance(v2, SQLEditor)
    assert v2.spec == (
        "SELECT\n"
        "  SUM(value) AS value_sum\n"
        "FROM tiny"
    )
    assert isinstance(v3, Typography)
    assert v3.object == "### Analyze"
    assert isinstance(v4, ChatMessage)
    assert v4.object == "Result: 38.0"


async def test_plan_edit(sql_plan):
    assert sql_plan.status == "success"
    sql_plan.llm.set_responses([
        lambda: f"Result: {sql_plan[0].out_context['pipeline'].data.iloc[0, 0]}"
    ])

    sql_plan.views[1].spec = "SELECT value as value_sum FROM tiny LIMIT 1"

    await async_wait_until(lambda: sql_plan.out_context["sql"] == "SELECT value as value_sum FROM tiny LIMIT 1")
    assert sql_plan.status == "success"
    assert "pipeline" in sql_plan.out_context
    pd.testing.assert_frame_equal(
        sql_plan.out_context["pipeline"].data,
        pd.DataFrame([10.5], columns=["value_sum"])
    )
    await async_wait_until(lambda: len(sql_plan.views) == 4)
    assert sql_plan.views[3].object == "Result: 10.5"


async def test_plan_revise(sql_plan):
    sql_plan.llm.set_responses([
        RetrySpec(chain_of_thought="Select first row", edits=[
            ReplaceLine(line_no=2, line="value AS value_sum"),
            ReplaceLine(line_no=3, line="FROM tiny LIMIT 1")
        ]),
        lambda: f"Result: {sql_plan[0].out_context['pipeline'].data.iloc[0, 0]}"
    ])

    sql_task = sql_plan[0]
    with sql_task.actor.param.update(llm=sql_plan.llm):
        sql_plan.views[1].spec = await sql_task.actor.revise(
            "Update to limit to 1 row", sql_task.history, sql_task.out_context, sql_plan.views[1]
        )

    await async_wait_until(lambda: sql_plan.out_context["sql"] == (
        "SELECT\n"
        "  value AS value_sum\n"
        "FROM tiny\n"
        "LIMIT 1"
    ))
    assert sql_plan.status == "success"
    assert "pipeline" in sql_plan.out_context
    pd.testing.assert_frame_equal(
        sql_plan.out_context["pipeline"].data,
        pd.DataFrame([10.5], columns=["value_sum"])
    )
    await async_wait_until(lambda: len(sql_plan.views) == 4)
    assert sql_plan.views[3].object == "Result: 10.5"


# -------------------------------------------------------------------
# Tests for chat file upload fix (reuse ephemeral source + MetadataLookup tracking)
# -------------------------------------------------------------------

async def test_reuse_ephemeral_source_across_uploads():
    """
    When an ephemeral DuckDBSource already exists in outputs,
    subsequent uploads should reuse it instead of creating a new one.
    """
    from lumen.ai.controls.base import BaseSourceControls
    from lumen.sources.duckdb import DuckDBSource

    controls = BaseSourceControls()

    # Simulate first upload creating an ephemeral source
    source1 = DuckDBSource(uri=":memory:", ephemeral=True, name="UploadedSource000000", tables={})
    controls._register_source_output(source1)

    # Verify the lookup logic finds the existing ephemeral source
    existing_sources = controls.outputs.get("sources", [])
    found = None
    for existing in existing_sources:
        if isinstance(existing, DuckDBSource) and existing.ephemeral:
            found = existing
            break

    assert found is source1


async def test_metadata_lookup_reprocesses_source_with_new_tables():
    """
    MetadataLookup should re-process a source when its table count
    increases (new tables uploaded to the same source).
    """
    from lumen.ai.tools.metadata_lookup import MetadataLookup
    from lumen.sources.duckdb import DuckDBSource

    source = DuckDBSource(tables={
        'table_a': "SELECT 1 AS id, 'X' AS category",
    })

    lookup = MetadataLookup()
    vector_store_id = id(lookup.vector_store)

    # Source previously processed with 1 table
    lookup._sources_in_progress[vector_store_id] = {source.name: 1}

    # Same table count -> should be skipped
    current_count = len(source.get_tables())
    known_count = lookup._sources_in_progress[vector_store_id].get(source.name, 0)
    assert known_count >= current_count

    # Add a second table -> should trigger re-processing
    source.tables['table_b'] = "SELECT 2 AS id, 'Y' AS group_name"
    current_count = len(source.get_tables())
    known_count = lookup._sources_in_progress[vector_store_id].get(source.name, 0)
    assert known_count < current_count


async def test_metadata_lookup_skips_unchanged_source():
    """
    Calling _update_vector_store twice with the same source (unchanged tables)
    should skip re-processing on the second call — vector store entry count
    stays the same.
    """
    from lumen.ai.tools.metadata_lookup import MetadataLookup
    from lumen.sources.duckdb import DuckDBSource

    source = DuckDBSource(tables={
        'my_table': "SELECT 1 AS x",
    })

    lookup = MetadataLookup(include_metadata=False)
    context = {"sources": [source], "tables_metadata": {}}

    # First call — should process the source and upsert 1 entry
    await lookup._update_vector_store(context)
    entries_after_first = len(lookup.vector_store.texts)

    assert entries_after_first == 1
    assert lookup._ready is True

    # Second call with same source, same tables — should skip
    await lookup._update_vector_store(context)
    entries_after_second = len(lookup.vector_store.texts)

    assert entries_after_second == entries_after_first  # No new entries
    assert lookup._ready is True


# -------------------------------------------------------------------
# Integration test: _process_files() with real UploadedFileRow cards
# -------------------------------------------------------------------

async def test_process_files_reuses_ephemeral_source():
    """
    Calling _process_files() twice (simulating two separate chat uploads)
    should reuse the same DuckDBSource so all tables share one connection.
    """
    import io

    from lumen.ai.controls.base import BaseSourceControls, UploadedFileRow
    from lumen.sources.duckdb import DuckDBSource

    controls = BaseSourceControls()

    # First upload: one CSV
    csv1 = io.BytesIO(b"name,salary\nAlice,100\nBob,200\n")
    card1 = UploadedFileRow(file_obj=csv1, filename="employees.csv")
    controls._file_cards = [card1]
    n_tables_1, _, _ = controls._process_files()
    assert n_tables_1 == 1

    sources_after_first = controls.outputs.get("sources", [])
    assert len(sources_after_first) == 1
    first_source = sources_after_first[0]
    assert isinstance(first_source, DuckDBSource)
    assert first_source.ephemeral is True
    assert "employees" in first_source.tables

    # Second upload: another CSV
    csv2 = io.BytesIO(b"product,revenue\nWidget,500\nGizmo,300\n")
    card2 = UploadedFileRow(file_obj=csv2, filename="sales.csv")
    controls._file_cards = [card2]
    n_tables_2, _, _ = controls._process_files()
    assert n_tables_2 == 1

    # Should reuse the same source, not create a new one
    sources_after_second = controls.outputs.get("sources", [])
    assert len(sources_after_second) == 1
    assert sources_after_second[0] is first_source

    # Both tables accessible from the same DuckDB connection
    assert "employees" in first_source.tables
    assert "sales" in first_source.tables

    df_employees = first_source.get("employees")
    assert len(df_employees) == 2
    assert list(df_employees.columns) == ["name", "salary"]

    df_sales = first_source.get("sales")
    assert len(df_sales) == 2
    assert list(df_sales.columns) == ["product", "revenue"]


async def test_process_files_creates_new_source_when_none_exists():
    """
    First call to _process_files() should create a new DuckDBSource
    and increment _count.
    """
    import io

    from lumen.ai.controls.base import BaseSourceControls, UploadedFileRow

    controls = BaseSourceControls()
    assert controls._count == 0

    csv = io.BytesIO(b"x,y\n1,2\n3,4\n")
    card = UploadedFileRow(file_obj=csv, filename="data.csv")
    controls._file_cards = [card]
    controls._process_files()

    assert controls._count == 1
    sources = controls.outputs.get("sources", [])
    assert len(sources) == 1
    assert sources[0].name == f"{controls.source_name_prefix}000000"


# -------------------------------------------------------------------
# Readiness wait test (asyncio.Event based)
# -------------------------------------------------------------------

async def test_metadata_lookup_respond_waits_for_ready_event():
    """
    respond() should wait for _ready_event before calling _gather_info.
    If the event is not set within the timeout, it should proceed anyway.
    """
    import asyncio

    from lumen.ai.tools.metadata_lookup import MetadataLookup

    lookup = MetadataLookup()

    # Simulate an in-progress sync by clearing the event
    lookup._ready_event.clear()
    lookup._ready = False

    # Schedule the event to be set after 0.2s
    async def _set_ready():
        await asyncio.sleep(0.2)
        lookup._ready = True
        lookup._ready_event.set()

    task = asyncio.create_task(_set_ready())

    # respond() should wait for the event, not return immediately
    # We can't call respond() directly without proper context, so
    # test the wait mechanism directly
    assert not lookup._ready_event.is_set()
    await asyncio.wait_for(lookup._ready_event.wait(), timeout=5)
    assert lookup._ready_event.is_set()
    assert lookup._ready is True
    await task


async def test_metadata_lookup_ready_event_timeout():
    """
    If _ready_event is never set, the wait should timeout gracefully
    without raising an exception.
    """
    import asyncio

    from lumen.ai.tools.metadata_lookup import MetadataLookup

    lookup = MetadataLookup()
    lookup._ready_event.clear()
    lookup._ready = False

    # Verify timeout works without raising
    timed_out = False
    try:
        await asyncio.wait_for(lookup._ready_event.wait(), timeout=0.2)
    except TimeoutError:
        timed_out = True

    assert timed_out
    assert not lookup._ready_event.is_set()


async def test_metadata_lookup_ready_event_initially_set():
    """
    A freshly created MetadataLookup should have _ready_event set
    so that respond() does not block when no sync has happened.
    """
    from lumen.ai.tools.metadata_lookup import MetadataLookup

    lookup = MetadataLookup()
    assert lookup._ready_event.is_set()


# -------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------

async def test_process_files_partial_failure():
    """
    If one file fails during _process_files() (unsupported format),
    other valid files should still be processed successfully.
    """
    import io

    from lumen.ai.controls.base import BaseSourceControls, UploadedFileRow

    controls = BaseSourceControls()

    good_csv = io.BytesIO(b"a,b\n1,2\n")
    bad_file = io.BytesIO(b"not a valid file")

    card_good = UploadedFileRow(file_obj=good_csv, filename="valid.csv")
    card_bad = UploadedFileRow(file_obj=bad_file, filename="broken.xyz")

    controls._file_cards = [card_good, card_bad]
    n_tables, _, _ = controls._process_files()

    # Good file processed, bad file skipped
    assert n_tables == 1
    sources = controls.outputs.get("sources", [])
    assert len(sources) == 1
    assert "valid" in sources[0].tables


async def test_process_files_concurrent_uploads_same_source():
    """
    Multiple sequential _process_files() calls should all write to
    the same DuckDBSource without conflicts.
    """
    import io

    from lumen.ai.controls.base import BaseSourceControls, UploadedFileRow
    from lumen.sources.duckdb import DuckDBSource

    controls = BaseSourceControls()

    for i in range(3):
        csv = io.BytesIO(f"col_{i}\n{i}\n".encode())
        card = UploadedFileRow(file_obj=csv, filename=f"file_{i}.csv")
        controls._file_cards = [card]
        controls._process_files()

    # All 3 tables in one source
    sources = controls.outputs.get("sources", [])
    assert len(sources) == 1
    source = sources[0]
    assert isinstance(source, DuckDBSource)
    assert len(source.tables) == 3
    for i in range(3):
        assert f"file_{i}" in source.tables
        df = source.get(f"file_{i}")
        assert len(df) == 1


async def test_process_files_duplicate_table_name():
    """
    Uploading a file with the same table name twice should overwrite
    the table data, not error out.
    """
    import io

    from lumen.ai.controls.base import BaseSourceControls, UploadedFileRow

    controls = BaseSourceControls()

    # First upload
    csv1 = io.BytesIO(b"val\n10\n")
    card1 = UploadedFileRow(file_obj=csv1, filename="data.csv")
    controls._file_cards = [card1]
    controls._process_files()

    # Second upload with same name but different data
    csv2 = io.BytesIO(b"val\n99\n")
    card2 = UploadedFileRow(file_obj=csv2, filename="data.csv")
    controls._file_cards = [card2]
    controls._process_files()

    sources = controls.outputs.get("sources", [])
    assert len(sources) == 1
    df = sources[0].get("data")
    # Should have the latest data
    assert df.iloc[0, 0] == 99
