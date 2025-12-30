from typing import get_args

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.tests.util import async_wait_until
from panel_material_ui import Card, ChatMessage, Typography

from lumen.ai.agents import AnalystAgent, ChatAgent, SQLAgent
from lumen.ai.agents.sql import SQLQuery
from lumen.ai.coordinator import Plan, Planner
from lumen.ai.coordinator.planner import Reasoning, make_plan_model
from lumen.ai.models import ReplaceLine, RetrySpec
from lumen.ai.report import ActorTask
from lumen.ai.schemas import get_metaset
from lumen.ai.tools import MetadataLookup
from lumen.ai.vector_store import NumpyVectorStore
from lumen.ai.views import SQLOutput
from lumen.config import SOURCE_TABLE_SEPARATOR


async def test_planner_instantiate():
    Planner()


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
    assert todos.object == "- [ ] Say Hello!"

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
    analyst_agent = AnalystAgent()

    slug = f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny"
    metaset = await get_metaset([tiny_source], [slug])
    context = {
        "source": tiny_source,
        "sources": [tiny_source],
        "metaset": metaset,
        "visible_slugs": [slug]
    }

    llm.set_responses([
        SQLQuery(query="SELECT SUM(value) as value_sum FROM tiny", table_slug="tiny_agg"),
        lambda: f"Result: {plan[0].out_context['pipeline'].data.iloc[0, 0]}"
    ])

    plan = Plan(
        ActorTask(sql_agent, title="Aggregate"),
        ActorTask(analyst_agent, title="Analyze"),
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
    assert isinstance(v2, SQLOutput)
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
