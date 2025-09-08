import asyncio
import json
import os
import uuid

import numpy as np
import pandas as pd
import pytest

from panel.tests.util import async_wait_until

from lumen.ai.coordinator import Planner
from lumen.ai.llm import OpenAI
from lumen.ai.logfire import read_trace
from lumen.ai.memory import memory
from lumen.ai.ui import ExplorerUI
from lumen.sources.duckdb import DuckDBSource

pytestmark = [pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("LOGFIRE_READ_TOKEN"), 
    reason="OPENAI_API_KEY and LOGFIRE_READ_TOKEN must be set"
), 
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
]


async def assert_expected_results(df: pd.DataFrame, duration: float, input_tokens: int, output_tokens: int, num_spans: int, agents: list[str]):
    assert df.loc[0, "duration"] < duration
    assert df.loc[0, "input_tokens"] < input_tokens
    assert df.loc[0, "output_tokens"] < output_tokens
    if num_spans > 0:
        assert df.loc[0, "num_spans"] <= num_spans

    for agent in agents:
        assert agent in df["name"].unique()


async def async_wait_until_trace(llm):
    await asyncio.sleep(1)
    while True:
        async_wait_until(lambda: read_trace(tags=llm.logfire_tags))
        try:
            df = await read_trace(tags=llm.logfire_tags)
            return df
        except Exception:
            await asyncio.sleep(1)


class TestOneShot:
    @pytest.fixture
    def tag(self):
        return f"test_one_shot_{uuid.uuid4().hex[:6]}"

    @pytest.fixture
    def llm(self, tag):
        llm = OpenAI(logfire_tags=[tag])
        return llm

    @pytest.fixture
    def source(self, tmp_path):
        return DuckDBSource(
            uri=f"{tmp_path}/test.db",
            tables={"gpt-4o-mini": "assets/gpt-4o-mini.csv", "gpt-41-mini": "assets/gpt-41-mini.csv"}
        )

    @pytest.fixture
    def coordinator(self, llm, source):
        planner = Planner(llm=llm, agents=ExplorerUI.param["default_agents"].default)
        memory["source"] = source
        return planner

    async def test_table_list_agent(self, tag, llm, coordinator):
        llm.logfire_tags = [tag, "table_list_agent"]
        await coordinator.respond([{"role": "user", "content": "Show me a list of the tables"}])
        df = await async_wait_until_trace(llm)

        # Using generous assertions
        await assert_expected_results(
            df,
            duration=10,
            input_tokens=3500,
            output_tokens=300,
            num_spans=4,
            agents=["Planner", "TableListAgent"],
        )

    async def test_chat_agent(self, tag, llm, coordinator):
        llm.logfire_tags = [tag, "chat_agent"]
        await coordinator.respond([{"role": "user", "content": "Tell me about the data."}])
        df = await async_wait_until_trace(llm)

        # Using generous assertions
        await assert_expected_results(
            df,
            duration=15,
            input_tokens=3500,
            output_tokens=500,
            num_spans=5,
            agents=["Planner", "ChatAgent"],
        )

    async def test_sql_agent(self, tag, llm, coordinator, source):
        llm.logfire_tags = [tag, "sql_agent"]
        await coordinator.respond([{"role": "user", "content": "What is the total output tokens from gpt-4o-mini?"}])
        df = await async_wait_until_trace(llm)

        # Using generous assertions
        await assert_expected_results(
            df,
            duration=60,
            input_tokens=15000,
            output_tokens=1500,
            num_spans=0,
            agents=["Planner", "SQLAgent"],
        )
        for _, row in df[::-1].iterrows():
            # Extract the SQL query from the response data
            if not row["response_data"]:
                continue

            try:
                arguments = json.loads(row["response_data"]["message"]["tool_calls"][0]["function"]["arguments"])
            except Exception:
                continue

            if "queries" not in arguments:
                continue

            query = arguments["queries"][0]["query"]
            result = source.execute(query).iloc[0, 0]
            assert np.isclose(result, 3988.0)
            break
