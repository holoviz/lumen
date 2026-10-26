import datetime

from io import BytesIO
from unittest.mock import MagicMock

import jinja2
import pytest

from PIL import Image

try:
    from lumen.ai.agents.chat import ChatAgent
    from lumen.ai.agents.sql import SQLAgent
    from lumen.ai.agents.validation import ValidationAgent
    from lumen.ai.config import PROMPTS_DIR
    from lumen.ai.coordinator.base import Plan
    from lumen.ai.editors import LumenEditor
    from lumen.ai.report import ActorTask
    from lumen.views import View
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)


@pytest.fixture
def jinja_env():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters['json_to_yaml'] = lambda x: str(x)
    return env


def test_validation_agent_previous_keys_gating(jinja_env):
    template = jinja_env.get_template("ValidationAgent/main.jinja2")
    
    # Test 1: Keys present and NOT in previous_keys (should render gating instructions)
    context = {
        "current_datetime": datetime.datetime.now(),
        "memory": {"sql": "SELECT 1", "view": "...", "chat": "hello", "listing": "list"},
        "previous_keys": set()
    }
    rendered = template.render(**context)
    
    # Check instructions are present
    assert "SQL: check SELECT, WHERE, GROUP BY" in rendered
    assert "View: spec already rendered" in rendered
    assert "Listing: VALID if items match" in rendered
    assert "Assistant response: VALID if consistent" in rendered
    
    # Check examples are present
    assert "SQL: SELECT region, SUM(sales)" in rendered
    assert "User: \"sales by region\" / table shown" in rendered
    assert "User: \"bar chart of sales by region\"" in rendered
    
    # Test 2: Keys present but IN previous_keys (should NOT render gating instructions)
    context = {
        "current_datetime": datetime.datetime.now(),
        "memory": {"sql": "SELECT 1", "view": "...", "chat": "hello", "listing": "list"},
        "previous_keys": {"sql", "view", "chat", "listing"}
    }
    rendered = template.render(**context)
    
    # Check instructions are absent
    assert "SQL: check SELECT, WHERE, GROUP BY" not in rendered
    assert "View: spec already rendered" not in rendered
    assert "Listing: VALID if items match" not in rendered
    assert "Assistant response: VALID if consistent" not in rendered
    
    # Check examples are absent
    assert "SQL: SELECT region, SUM(sales)" not in rendered
    assert "User: \"sales by region\" / table shown" not in rendered
    assert "User: \"bar chart of sales by region\"" not in rendered


@pytest.mark.parametrize(
    ("agent_type", "out_context", "expected_previous_keys"),
    [
        pytest.param(ChatAgent, {}, {"sql"}, id="stale_sql"),
        pytest.param(SQLAgent, {}, set(), id="reused_sql"),
        pytest.param(ChatAgent, {"sql": "SELECT current"}, set(), id="current_sql"),
    ],
)
@pytest.mark.asyncio
async def test_validation_agent_uses_prior_plan_dependencies(
    agent_type, out_context, expected_previous_keys,
):
    validation_agent = ValidationAgent()
    plan = Plan(
        ActorTask(agent_type(), out_context=out_context),
        ActorTask(validation_agent),
    )

    prompt_context = await validation_agent._gather_prompt_context(
        "main", [], {"plan": plan, "sql": "SELECT current"},
    )

    assert prompt_context["previous_keys"] == expected_previous_keys


def test_validation_agent_marks_previous_plan_data(jinja_env):
    template = jinja_env.get_template("ValidationAgent/main.jinja2")
    rendered = template.render(
        current_datetime=datetime.datetime.now(),
        memory={"sql": "SELECT 1", "data": "rows: 1", "chat": "hello"},
        previous_keys={"sql", "data"},
    )
    assert "(From previous plan) Data overview" in rendered
    # The table is from a previous plan, so the response must not be excused for it
    assert "need not restate every value" not in rendered


def test_validation_agent_chat_instructions_need_chat(jinja_env):
    template = jinja_env.get_template("ValidationAgent/main.jinja2")
    rendered = template.render(
        current_datetime=datetime.datetime.now(),
        memory={"sql": "SELECT 1"},
        previous_keys=set(),
    )
    assert "Assistant response: VALID if consistent" not in rendered
    assert "(From previous plan)" not in rendered


@pytest.mark.parametrize(
    ("satisfied", "expected"),
    [pytest.param([], True, id="no_requirements"),
     pytest.param([True, True], True, id="all_met"),
     pytest.param([True, False], False, id="one_unmet")],
)
@pytest.mark.asyncio
async def test_validation_agent_verdict_comes_from_checks(satisfied, expected):
    from lumen.ai.agents.validation import (
        QueryCompletionValidation, RequirementCheck,
    )

    agent = ValidationAgent()
    checks = [RequirementCheck(requirement=f"r{i}", evidence="e", satisfied=s) for i, s in enumerate(satisfied)]

    async def invoke(*args, **kwargs):
        return QueryCompletionValidation(checks=checks, chain_of_thought="x")

    agent._invoke_prompt = invoke
    _, outputs = await agent.respond([{"role": "user", "content": "sales by region"}], {})
    result = outputs["validation_result"]
    assert result.correct is expected
    assert result.missing_elements == [f"r{i}: e" for i, s in enumerate(satisfied) if not s]


def test_validation_agent_compacts_view():
    view = {
        "type": "vegalite",
        "sizing_mode": "stretch_both",
        "pipeline": {"source": {"tables": {"t": "SELECT 1"}}, "sql_transforms": [], "table": "t"},
        "spec": {"config": {"range": {"category": ["#000"]}}, "$schema": "v5", "mark": "bar",
                 "encoding": {"x": {"field": "region"}}},
    }
    compact = ValidationAgent._compact_view(view)
    assert compact == {
        "type": "vegalite",
        "pipeline": {"table": "t"},
        "spec": {"mark": "bar", "encoding": {"x": {"field": "region"}}},
    }
    assert "config" in view["spec"]  # the context's view is not mutated


def test_validation_agent_drops_roadmap():
    messages = [
        {"role": "user", "content": "horizontal bar chart"},
        {"role": "system", "content": "<roadmap>\n  - 🟢 Create a horizontal bar chart\n</roadmap>"},
        {"role": "system", "content": "other context"},
    ]
    assert ValidationAgent._without_roadmap(messages) == [messages[0], messages[2]]


def _plan_with_view(validation_agent, to_image):
    view = MagicMock(spec=View)
    view.to_image = to_image
    editor = MagicMock(spec=LumenEditor)
    editor.component = view
    producer = ActorTask(ChatAgent())
    producer.views = [editor]
    return Plan(producer, ActorTask(validation_agent))


@pytest.mark.asyncio
async def test_validation_agent_attaches_rendered_view(llm):
    """The rendered view reaches the model, and the prompt is told to judge the chart from it."""
    from lumen.ai.agents.validation import QueryCompletionValidation

    agent = ValidationAgent(llm=llm)
    seen = {}

    async def invoke(prompt_name, messages, context, **kwargs):
        seen["messages"], seen["kwargs"] = messages, kwargs
        return QueryCompletionValidation(chain_of_thought="x")

    agent._invoke_prompt = invoke
    png = BytesIO()
    Image.new("RGB", (4, 4)).save(png, format="PNG")
    plan = _plan_with_view(agent, png.getvalue)
    await agent.respond([{"role": "user", "content": "bar chart"}], {"plan": plan})

    assert seen["kwargs"]["view_image"] is True
    assert seen["messages"][-1]["content"][0] == "Rendered view:"


@pytest.mark.parametrize("vision", [True, False])
@pytest.mark.asyncio
async def test_validation_agent_validates_without_image(llm, vision):
    """No image when the model can't read one or the view fails to render; validation still runs."""
    agent = ValidationAgent(llm=llm)
    llm._supports_vision = vision

    def broken():
        raise RuntimeError("Neither Selenium nor Playwright is installed")

    assert await agent._render_view({"plan": _plan_with_view(agent, broken)}) is None
