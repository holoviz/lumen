import datetime

import jinja2
import pytest

try:
    from lumen.ai.config import PROMPTS_DIR
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
    assert "Analyze if SQL query matches user's request" in rendered
    assert "Analyze if generated view and responses match" in rendered
    assert "Listing of items displayed to user." in rendered
    assert "Analyze if assistant's response addresses user's request" in rendered
    
    # Check examples are present
    assert "SQL: SELECT region, SUM(sales)" in rendered
    assert "User: \"Summarize survey results\"" in rendered
    assert "User: \"bar chart of sales by region\"" in rendered
    
    # Test 2: Keys present but IN previous_keys (should NOT render gating instructions)
    context = {
        "current_datetime": datetime.datetime.now(),
        "memory": {"sql": "SELECT 1", "view": "...", "chat": "hello", "listing": "list"},
        "previous_keys": {"sql", "view", "chat", "listing"}
    }
    rendered = template.render(**context)
    
    # Check instructions are absent
    assert "Analyze if SQL query matches user's request" not in rendered
    assert "Analyze if generated view and responses match" not in rendered
    assert "Listing of items displayed to user." not in rendered
    assert "Analyze if assistant's response addresses user's request" not in rendered
    
    # Check examples are absent
    assert "SQL: SELECT region, SUM(sales)" not in rendered
    assert "User: \"Summarize survey results\"" not in rendered
    assert "User: \"bar chart of sales by region\"" not in rendered


# ---------------------------------------------------------------------------
# Regression tests for plan-provenance based stale context tagging
# ---------------------------------------------------------------------------

from typing import Any, NotRequired, TypedDict

from lumen.ai.agents.validation import get_plan_required_keys


class _SQLInputs(TypedDict):
    """Mock input schema that declares 'sql' as a dependency."""
    sql: NotRequired[str]
    data: NotRequired[Any]
    source: str


class _ViewInputs(TypedDict):
    """Mock input schema that declares 'view'-related inputs (no sql/chat/listing)."""
    data: Any
    pipeline: Any
    table: str


class _ChatOutputSchema(TypedDict):
    """Mock input schema with 'chat' dependency."""
    chat: NotRequired[str]


class _EmptyInputs(TypedDict):
    """Mock input schema with no context key dependencies."""
    pass


class _MockTask:
    """Lightweight stand-in for ActorTask used in tests."""

    def __init__(self, input_schema, out_context=None):
        self.input_schema = input_schema
        self.out_context = out_context or {}


# -- Tests for get_plan_required_keys ----------------------------------------

@pytest.mark.parametrize(
    "plan_tasks, expected",
    [
        pytest.param(
            [_MockTask(_SQLInputs, {"sql": "SELECT 1", "data": "..."})],
            {"sql": True, "view": False, "chat": False, "listing": False},
            id="sql_agent_only-sql_required",
        ),
        pytest.param(
            [_MockTask(_ViewInputs, {"view": "..."})],
            {"sql": False, "view": False, "chat": False, "listing": False},
            id="view_agent_only-no_context_keys_required",
        ),
        pytest.param(
            [
                _MockTask(_SQLInputs, {"sql": "SELECT 1"}),
                _MockTask(_ViewInputs, {"view": "..."}),
            ],
            {"sql": True, "view": False, "chat": False, "listing": False},
            id="sql_then_view-sql_required",
        ),
        pytest.param(
            [_MockTask(_ChatOutputSchema, {"chat": "hello"})],
            {"sql": False, "view": False, "chat": True, "listing": False},
            id="chat_agent_only-chat_required",
        ),
        pytest.param(
            [_MockTask(_EmptyInputs)],
            {"sql": False, "view": False, "chat": False, "listing": False},
            id="empty_inputs-nothing_required",
        ),
    ],
)
def test_get_plan_required_keys(plan_tasks, expected):
    """get_plan_required_keys correctly identifies which context keys are
    declared as inputs by the tasks in a plan."""
    result = get_plan_required_keys(plan_tasks)
    assert result == expected


# -- Regression: stale context should be tagged as previous_keys -------------

@pytest.mark.parametrize(
    "plan_tasks, context_keys_present, expected_previous_keys",
    [
        pytest.param(
            # Plan has only ViewAgent (no sql input) but context has leftover sql
            [_MockTask(_ViewInputs, {"view": "..."})],
            {"sql": "SELECT old", "view": "old_chart"},
            {"sql"},
            id="stale_sql-view_only_plan",
        ),
        pytest.param(
            # Plan has SQLAgent (needs sql input), so sql should NOT be stale
            [_MockTask(_SQLInputs, {"data": "..."})],
            {"sql": "SELECT old"},
            set(),
            id="reused_sql-sql_plan_keeps_it",
        ),
        pytest.param(
            # Plan has empty inputs, all leftover keys should be stale
            [_MockTask(_EmptyInputs)],
            {"sql": "SELECT old", "chat": "hello", "view": "chart"},
            {"sql", "chat", "view"},
            id="all_stale-empty_plan",
        ),
        pytest.param(
            # Plan has ChatAgent, chat should stay, sql/view should be stale
            [_MockTask(_ChatOutputSchema, {"chat": "hi"})],
            {"sql": "SELECT old", "chat": "old_chat", "view": "old_view"},
            {"sql", "view"},
            id="mixed-chat_kept_others_stale",
        ),
    ],
)
def test_previous_keys_provenance(plan_tasks, context_keys_present, expected_previous_keys):
    """Regression test: context keys not required by any task in the plan
    should be tagged as previous_keys (stale). Keys that ARE required
    should be kept active.

    On main (before the fix), all non-produced keys were blanket-tagged
    as previous_keys regardless of plan provenance, causing this test to fail.
    """
    plan = plan_tasks
    context = dict(context_keys_present, plan=plan)

    # Replicate the logic from _gather_prompt_context
    required_keys = get_plan_required_keys(plan)
    produced = {k for task in plan for k in task.out_context}
    previous_keys = set()
    for key in ("chat", "sql", "view", "listing"):
        if key in context and key not in produced:
            if required_keys.get(key, False):
                continue
            previous_keys.add(key)

    assert previous_keys == expected_previous_keys
