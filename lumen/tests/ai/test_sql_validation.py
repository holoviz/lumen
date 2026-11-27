from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip(
        "lumen.ai could not be imported, skipping tests.",
        allow_module_level=True,
    )

from lumen.ai.agents.sql import SQLAgent


@pytest.mark.asyncio
async def test_validate_sql_surfaces_table_names_on_not_found():
    """On a 'table not found' error the retry feedback should list the
    source's real table names so the model stops using the source name
    (the failure mode for multi-table sources like XArraySQLSource)."""
    agent = SQLAgent()

    source = MagicMock()
    source.dialect = "duckdb"
    source.get_tables.return_value = ["CMI_C01", "DQF_C01"]

    attempts = {"n": 0}

    def execute(sql, *a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # First attempt: model referenced the source name as a table.
            raise ValueError("Error during planning: table 'MySource' not found")
        return None  # Corrected query validates.

    source.execute.side_effect = execute

    captured = {}

    async def fake_revise(feedback, *a, **k):
        captured["feedback"] = feedback
        return "SELECT * FROM CMI_C01"

    agent.revise = fake_revise

    result = await agent._validate_sql(
        context={},
        sql_query='SELECT * FROM "MySource"',
        expr_slug="x",
        source=source,
        messages=[],
        step=MagicMock(),
        max_retries=2,
    )

    # Recovered using the corrected SQL.
    assert "CMI_C01" in result
    # The retry feedback surfaced the real table names + the hint.
    assert "CMI_C01" in captured["feedback"]
    assert "DQF_C01" in captured["feedback"]
    assert "not the source name" in captured["feedback"]


@pytest.mark.asyncio
async def test_validate_sql_valid_query_no_retry():
    """A valid query validates on the first try with no retry/feedback."""
    agent = SQLAgent()

    source = MagicMock()
    source.dialect = "duckdb"
    source.execute.return_value = None  # valid

    agent.revise = AsyncMock(
        side_effect=AssertionError("revise should not be called")
    )

    result = await agent._validate_sql(
        context={},
        sql_query="SELECT * FROM CMI_C01",
        expr_slug="x",
        source=source,
        messages=[],
        step=MagicMock(),
        max_retries=2,
    )
    assert "CMI_C01" in result
    source.get_tables.assert_not_called()
