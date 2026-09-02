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
