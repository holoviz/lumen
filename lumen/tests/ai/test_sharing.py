from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.agents import ChatAgent, SQLAgent
from lumen.ai.agents.sql import execute_distinct_values, make_sql_model
from lumen.ai.agents.vega_lite import VegaLiteAgent
from lumen.ai.answer import aask, configure
from lumen.ai.data import read
from lumen.ai.sharing import MASK, looks_like_pii, sharing
from lumen.ai.table_stats import (
    EXACT, ColumnStats, TableStats, render_column,
)
from lumen.ai.utils import describe_data_sync, get_schema
from lumen.tests.ai.test_answer import FollowUpLLM

# Values no prompt may contain unless sharing allows it
SECRET_CITY, SECRET_AMOUNT = "Zanzibar", 98765.4321


@pytest.fixture(autouse=True)
def default_sharing():
    yield
    sharing.param.update(share="sample", mask_pii=False)


@pytest.fixture
def frame():
    return pd.DataFrame({
        "city": [SECRET_CITY, "Oslo", "Lima"] * 4,
        "amount": [SECRET_AMOUNT, 1.0, 2.0] * 4,
        "email": ["ann@example.com", "bo@example.org", "cy@example.net"] * 4,
    })


@pytest.mark.parametrize("column, values, expected", [
    ("customer_email", (), True),
    ("BillingAddress", (), True),
    ("date_of_birth", (), True),
    ("contact", ["a@b.com", "c@d.org", "x"], True),
    ("species", (), False),
    ("product_name", (), False),
    ("birth_rate", (), False),
    ("note", ["hello", "world"], False),
])
def test_looks_like_pii(column, values, expected):
    assert looks_like_pii(column, values) is expected


@pytest.mark.parametrize("share", ["schema", "none"])
def test_describe_data_keeps_values_out(frame, share):
    configure(share=share)
    summary = describe_data_sync(frame)
    assert SECRET_CITY not in summary and "98765" not in summary
    assert "city" in summary
    assert ("float64" in summary) is (share == "schema")


def test_describe_data_masks_personal_data(frame):
    configure(mask_pii=True)
    summary = describe_data_sync(frame)
    assert "ann@example.com" not in summary and MASK in summary
    assert SECRET_CITY in summary


async def test_get_schema_drops_values_and_then_types(frame):
    source = read({"t": frame})
    full = await get_schema(source, "t")
    assert "enum" in full["city"]
    configure(share="schema")
    schema = await get_schema(source, "t")
    assert schema["city"] == {"type": "str"} and "min" not in schema["amount"]
    configure(share="none")
    assert "type" not in (await get_schema(source, "t"))["amount"]


def test_table_stats_lines_follow_the_sharing_level():
    stats = TableStats(table="t", method=EXACT, rows=12, rows_exact=True)
    city = ColumnStats(name="city", type="VARCHAR", kind="string", nulls=0.0, distinct=3,
                       values=[[SECRET_CITY, 4], ["Oslo", 4], ["Lima", 4]], values_complete=True)
    email = ColumnStats(name="email", type="VARCHAR", kind="string", nulls=0.0, distinct=3,
                        examples=["ann@example.com"])
    assert SECRET_CITY in render_column(city, stats)
    configure(mask_pii=True)
    assert "ann@example.com" not in render_column(email, stats)
    assert SECRET_CITY in render_column(city, stats)
    configure(share="schema")
    assert render_column(city, stats) == "city VARCHAR 3 distinct"
    configure(share="none")
    assert render_column(city, stats) == "city"


def test_sql_tools_that_return_rows_are_left_out():
    agent = SQLAgent()
    assert {"distinct_values", "run_exploration_sql"} <= {t.name for t in agent._sql_tools({}, None)}
    configure(share="schema")
    assert not {"distinct_values", "run_exploration_sql"} & {t.name for t in agent._sql_tools({}, None)}


async def test_distinct_values_of_personal_data_are_withheld(frame):
    source = read({"t": frame})
    configure(mask_pii=True)
    sources = {(source.name, "t"): source}
    assert "withheld" in await execute_distinct_values("t", "email", sources=sources)
    assert SECRET_CITY in await execute_distinct_values("t", "city", sources=sources)


def test_chart_image_is_not_sent_without_values():
    agent = VegaLiteAgent()
    agent.llm = FollowUpLLM([])
    agent.llm._supports_vision = True
    configure(share="schema")
    with patch.object(VegaLiteAgent, "_export_plot_image", side_effect=AssertionError("rendered")):
        messages = agent._prepare_vision_messages([], object(), "Polish it")
    assert messages == [{"role": "user", "content": "Polish it"}]


class RecordingLLM(FollowUpLLM):
    """Replies like FollowUpLLM and keeps every prompt it was sent."""

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self.prompts = []

    async def invoke(self, messages, system="", response_model=None, **kwargs):
        self.prompts.append(f"{system}\n{messages}")
        return await super().invoke(messages, system=system, response_model=response_model, **kwargs)


async def _prompts_for_a_question(frame) -> str:
    source = read({"sales": frame})
    SQLModel = make_sql_model([(source.name, "sales")])
    llm = RecordingLLM([SQLModel(query="SELECT city, SUM(amount) AS total FROM sales GROUP BY city", table_slug="totals", tables=["sales"])])
    with patch.object(SQLAgent, "_profile_source_rows", new=AsyncMock(return_value=[])):
        answer = await aask(source, "Total per city?", llm=llm, agents=[ChatAgent(), SQLAgent()])
    assert answer.status == "success"
    assert SECRET_CITY in answer.pipeline.data["city"].tolist()
    return "\n".join(llm.prompts)


async def test_prompts_carry_values_at_sample(frame):
    # Shows the check below can see the values when they are sent
    assert SECRET_CITY in await _prompts_for_a_question(frame)


@pytest.mark.parametrize("share", ["schema", "none"])
async def test_prompts_carry_no_values_otherwise(frame, share):
    configure(share=share)
    prompts = await _prompts_for_a_question(frame)
    assert SECRET_CITY not in prompts and "98765" not in prompts and "example.com" not in prompts
    assert "city" in prompts


async def test_prompts_mask_personal_data(frame):
    configure(mask_pii=True)
    prompts = await _prompts_for_a_question(frame)
    assert "ann@example.com" not in prompts
    assert SECRET_CITY in prompts
