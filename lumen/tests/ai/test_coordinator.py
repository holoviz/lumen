from typing import get_args

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel_material_ui import Card

from lumen.ai.agents import ChatAgent
from lumen.ai.coordinator import Plan, Planner
from lumen.ai.models import Reasoning, make_plan_model
from lumen.ai.report import ActorTask
from lumen.ai.tools import IterativeTableLookup, TableLookup
from lumen.ai.vector_store import NumpyVectorStore


async def test_planner_instantiate():
    Planner()


async def test_planner_instantiate_tools_shared_vector_store():
    planner = Planner()

    assert len(planner._tools['main']) == 2
    tool1, tool2 = planner._tools['main']
    assert isinstance(tool1, TableLookup)
    assert isinstance(tool2, IterativeTableLookup)

    assert tool1.vector_store is tool2.vector_store


async def test_planner_instantiate_tools_provided_vector_store():
    vector_store = NumpyVectorStore()
    planner = Planner(vector_store=vector_store)

    assert len(planner._tools['main']) == 2
    tool1, tool2 = planner._tools['main']
    assert isinstance(tool1, TableLookup)
    assert isinstance(tool2, IterativeTableLookup)

    assert tool1.vector_store is vector_store
    assert tool2.vector_store is vector_store


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
    assert len(steps_layout) == 1
    reasoning_step = steps_layout[0]
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
    assert len(steps_layout) == 1
    reasoning_step = steps_layout[0]
    isinstance(reasoning_step, Card)
    assert reasoning_step.title == "Internal execution error during planning stage."
    title, todos = steps_layout.header
    assert title.object == "Planner could not settle on a plan of action to perform the requested query. Please restate your request."
    assert todos.object is None
