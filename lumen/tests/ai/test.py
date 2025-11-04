#!/usr/bin/env python3
"""
Standalone script to test SQL agent functionality.
Extracted from the pytest test suite.
"""

# Method 1: Using threading.current_thread().ident
import asyncio
import json
import os
import tempfile
import threading
import uuid

import numpy as np
import pandas as pd

from lumen.ai.coordinator import Planner
from lumen.ai.llm import OpenAI
from lumen.ai.logfire import read_trace
from lumen.ai.memory import memory
from lumen.ai.ui import ExplorerUI
from lumen.sources.duckdb import DuckDBSource


async def async_wait_until_trace(llm):
    """Wait until trace data is available."""
    await asyncio.sleep(1)
    while True:
        try:
            df = await read_trace(tags=llm.logfire_tags)
            if not df.empty:
                return df
        except Exception:
            pass
        await asyncio.sleep(1)


async def assert_expected_results(df: pd.DataFrame, duration: float, input_tokens: int, output_tokens: int, num_spans: int, agents: list[str]):
    """Assert that the results meet expected criteria."""
    assert df.loc[0, "duration"] < duration
    assert df.loc[0, "input_tokens"] < input_tokens
    assert df.loc[0, "output_tokens"] < output_tokens
    if num_spans > 0:
        assert df.loc[0, "num_spans"] <= num_spans

    for agent in agents:
        assert agent in df["name"].unique()


async def test_sql_agent_standalone():
    """Standalone version of the SQL agent test."""
    
    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("LOGFIRE_READ_TOKEN"):
        raise RuntimeError("OPENAI_API_KEY and LOGFIRE_READ_TOKEN must be set")
    
    # Generate unique tag for this test run
    tag = f"test_sql_agent_{uuid.uuid4().hex[:6]}"
    print(f"Running test with tag: {tag}")
    
    # Create temporary directory for database
    with tempfile.TemporaryDirectory() as tmp_path:
        # Setup LLM
        llm = OpenAI(logfire_tags=[tag, "sql_agent"])
        print("Created OpenAI LLM instance")
        
        # Setup source
        source = DuckDBSource(
            uri=f"{tmp_path}/test.db",
            tables={
                "gpt-4o-mini": "assets/gpt-4o-mini.csv", 
                "gpt-41-mini": "assets/gpt-41-mini.csv"
            }
        )
        print("Created DuckDB source")
        
        # Setup coordinator
        planner = Planner(llm=llm, agents=ExplorerUI.param["default_agents"].default)
        memory["source"] = source
        coordinator = planner
        print("Created coordinator with planner")
        
        # Run the test query
        print("Sending query to coordinator...")
        await coordinator.respond([{
            "role": "user", 
            "content": "What is the total output tokens from gpt-4o-mini?"
        }])
        
        print("Waiting for trace data...")
        df = await async_wait_until_trace(llm)
        
        print("Validating results...")
        # Using generous assertions
        await assert_expected_results(
            df,
            duration=60,
            input_tokens=15000,
            output_tokens=1500,
            num_spans=0,
            agents=["Planner", "SQLAgent"],
        )
        
        # Extract and validate SQL query results
        print("Checking SQL query execution...")
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
            print(f"Executing SQL query: {query}")
            
            result = source.execute(query).iloc[0, 0]
            print(f"Query result: {result}")
            
            # Validate result is approximately 3988.0
            assert np.isclose(result, 3988.0), f"Expected ~3988.0, got {result}"
            print("✓ SQL query result validation passed!")
            break
        else:
            raise AssertionError("No valid SQL query found in trace data")
        
        print("✓ All tests passed!")


async def main():
    """Main entry point."""
    try:
        await test_sql_agent_standalone()
        print("\n🎉 Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())
