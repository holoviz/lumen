"""Tests for the SQLAgent tool surface, submission flow and retry context."""

import asyncio
import json
import re

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

try:
    from lumen.ai.agents.sql import (
        PROMPT_OTHER_TABLES, RequestBudget, SQLAgent, _coerce_filter_value,
        build_distinct_values_sql, catalog_exceeds_prompt,
        execute_distinct_values, execute_exploration_sql,
        make_browse_data_catalog_tool, make_load_table_schemas_tool,
        make_run_exploration_sql_tool, make_sql_model, numeric_cast_edit,
        referenced_columns, sql_is_scalar_aggregate, summarize_tool_calls,
    )
    from lumen.ai.config import RequestBudgetExceededError
    from lumen.ai.llm import OpenAI
    from lumen.ai.schemas import (
        Column, Metaset, TableCatalogEntry, get_metaset, resolve_table_slug,
    )
    from lumen.ai.tool_trace import ModelCall, ToolCall
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

import sqlglot

from lumen.config import SOURCE_TABLE_SEPARATOR as SEP
from lumen.sources.duckdb import DuckDBSource

SLUGS = [f"DuckDBSource00001{SEP}orders", f"DuckDBSource00001{SEP}Customers", f"other{SEP}data.csv"]


@pytest.mark.parametrize("reference, expected", [
    (f"DuckDBSource00001{SEP}orders", SLUGS[0]),
    ("orders", SLUGS[0]),
    ("ORDERS", SLUGS[0]),
    ("customers", SLUGS[1]),
    ("duckdbsource00001/customers", SLUGS[1]),
    ("DuckDBSource00001.orders", SLUGS[0]),
    ("source/orders", SLUGS[0]),
    ("db.main.orders", SLUGS[0]),
    ('"orders"', SLUGS[0]),
    ("data.csv", SLUGS[2]),
    ("other/data.csv", SLUGS[2]),
])
def test_resolve_table_slug(reference, expected):
    assert resolve_table_slug(reference, SLUGS) == expected


def test_resolve_table_slug_lists_closest_matches():
    with pytest.raises(ValueError, match=re.escape("Unknown table 'order'. Closest matches: orders")) as error:
        resolve_table_slug("order", SLUGS)
    assert "Available tables: Customers, data.csv, orders" in str(error.value)


def test_resolve_table_slug_rejects_ambiguous_names():
    slugs = [f"a{SEP}orders", f"b{SEP}orders"]
    with pytest.raises(ValueError, match="Ambiguous table 'source/orders'"):
        resolve_table_slug("source/orders", slugs)
    assert resolve_table_slug("b/orders", slugs) == slugs[1]


@pytest.fixture
def values_source():
    return DuckDBSource(tables={
        "people": """
            SELECT * FROM (VALUES (1, 'Female', 10), (2, 'female ', 20), (3, 'Male', 30),
                                  (4, 'Female', 40), (5, NULL, 50)) AS t(id, "Gender", score)
        """,
    })


@pytest.fixture
def values_sources(values_source):
    return {(values_source.name, "people"): values_source}


async def test_distinct_values_counts_and_formats_values(values_source, values_sources):
    metaset = await get_metaset([values_source], ["people"])
    result = await execute_distinct_values("people", "gender", sources=values_sources, metaset=metaset)
    assert result.splitlines() == [
        "Values of 'Gender' in 'people', most frequent first (value: rows):",
        "- 'Female': 2",
        "- 'Male': 1",
        "- 'female ': 1",
        "- NULL: 1",
    ]


async def test_distinct_values_filters_case_insensitively_and_reports_truncation(values_sources):
    result = await execute_distinct_values("PEOPLE", "Gender", like="fem", sources=values_sources, limit=1)
    assert "- 'Female': 2" in result
    assert "'female '" not in result
    assert "More than 1 values match" in result


async def test_distinct_values_unknown_column_lists_closest(values_source, values_sources):
    metaset = await get_metaset([values_source], ["people"])
    result = await execute_distinct_values("people", "gendr", sources=values_sources, metaset=metaset)
    assert result.startswith("Unknown column 'gendr'. Closest matches: Gender.")


def test_distinct_values_sql_transpiles_ilike():
    sql = build_distinct_values_sql('SELECT * FROM "t"', "name", "sqlite", like="ab")
    assert "ILIKE" not in sql.upper()
    assert "LIKE" in sql.upper()
    sqlglot.parse_one(sql, read="sqlite")


async def test_exploration_sql_needs_no_source_with_one_source(values_sources):
    tool = make_run_exploration_sql_tool(values_sources)
    assert set(tool._model.model_fields) == {"sql_query"}
    result = await tool.function(sql_query="SELECT COUNT(*) AS n FROM people")
    assert "1 rows x 1 columns" in result


async def test_exploration_sql_resolves_source_references(values_source, values_sources):
    other = DuckDBSource(tables={"other": "SELECT 1 AS x"})
    sources = {**values_sources, (other.name, "other"): other}
    tool = make_run_exploration_sql_tool(sources)
    assert tool._model.model_fields["source"].default is None
    for reference in (values_source.name.lower(), "people", "source/people"):
        result = await execute_exploration_sql(reference, "SELECT COUNT(*) AS n FROM people", sources=sources)
        assert "1 rows x 1 columns" in result
    result = await execute_exploration_sql(None, "SELECT 1", sources=sources)
    assert "pass `source`" in result


async def test_load_table_schemas_filters_columns():
    slug = f"src{SEP}orders"
    metaset = Metaset(
        query=None,
        catalog={slug: TableCatalogEntry(slug, 1, [Column("order_id", description="Primary key"), Column("status")])},
        schemas={slug: {"__len__": 3, "order_id": {"type": "integer"}, "status": {"type": "enum", "enum": ["paid"]}}},
    )

    result = await make_load_table_schemas_tool(metaset).function(["Orders"], columns=["ORDER_ID", "missing"])

    assert "order_id INTEGER -- Primary key" in result
    assert "status" not in result
    assert result.endswith("Columns not found in the requested tables: missing.")


def _catalog(n):
    slugs = [f"src{SEP}table_{i:03d}" for i in range(n)]
    return Metaset(
        query=None,
        catalog={slug: TableCatalogEntry(slug, 1 - i / 1000, []) for i, slug in enumerate(slugs)},
    )


def test_browse_catalog_is_only_offered_for_large_catalogs():
    shown = SQLAgent.schema_tables_shown
    assert not catalog_exceeds_prompt(_catalog(shown + PROMPT_OTHER_TABLES), shown)
    assert catalog_exceeds_prompt(_catalog(shown + PROMPT_OTHER_TABLES + 1), shown)
    agent = SQLAgent()
    names = [tool.name for tool in agent._sql_tools({("src", "table_000"): None}, _catalog(3))]
    assert names == ["load_table_schemas", "distinct_values", "run_exploration_sql"]


def test_browse_catalog_pages_by_relevance():
    tool = make_browse_data_catalog_tool(_catalog(50))
    page = tool.function(offset=40, limit=5)
    assert page.startswith("Tables 41-45 of 50")
    assert "table_040" in page and "table_044" in page
    assert "table_039" not in page and "table_045" not in page
    assert "past the end" in tool.function(offset=60)


def test_other_tables_continue_after_the_primary_page():
    listing = _catalog(10).table_list(n=2, offset=4, n_others=2)
    assert "table_004" in listing and "table_005" in listing
    others = listing.split("Others available:")[1]
    assert "table_006" in others and "table_007" in others
    assert "table_000" not in others


@pytest.mark.parametrize("schema, value", [
    ({"type": "integer"}, "many"),
    ({"type": "number"}, [1, "x"]),
    ({"type": "number"}, [5, 1]),
    ({"type": "string", "format": "datetime"}, "not a date"),
    ({"type": "boolean"}, "maybe"),
    ({"type": "string"}, []),
])
def test_filter_values_are_validated(schema, value):
    with pytest.raises(ValueError):
        _coerce_filter_value(schema, value)


def test_filter_values_are_coerced():
    assert _coerce_filter_value({"type": "integer"}, "3") == 3
    assert _coerce_filter_value({"type": "number"}, ["1.5", 2]) == (1.5, 2)
    assert _coerce_filter_value({"type": "boolean"}, "True") is True


@pytest.mark.parametrize("sql, expected", [
    ("SELECT COUNT(*) FROM t", True),
    ("WITH x AS (SELECT * FROM t) SELECT MAX(a) - MIN(a) AS spread FROM x", True),
    ("SELECT a, COUNT(*) FROM t GROUP BY a", False),
    ("SELECT a, SUM(b) OVER () FROM t", False),
    ("SELECT a FROM t", False),
    ("SELECT COUNT(*) FROM a UNION SELECT COUNT(*) FROM b", False),
])
def test_sql_is_scalar_aggregate(sql, expected):
    assert sql_is_scalar_aggregate(sql, "duckdb") is expected


def test_referenced_columns():
    assert referenced_columns('SELECT "Name", SUM(t.value) FROM t WHERE Region = 1 GROUP BY 1', "duckdb") == {
        "name", "value", "region",
    }
    assert referenced_columns("SELECT COUNT(*) FROM t", "duckdb") == set()
    assert referenced_columns("SELECT * FROM t", "duckdb") is None
    assert referenced_columns("SELECT t.* FROM t", "duckdb") is None


async def test_profile_source_rows_only_lints_referenced_columns():
    source = DuckDBSource(tables={
        "t": "SELECT * FROM (VALUES ('a', 1, -9999), ('b', 2, -9999), ('c', 3, 5)) AS t(name, value, unused)"
    })
    assert await SQLAgent._profile_source_rows(source, ["t"], "SELECT name, SUM(value) FROM t GROUP BY name") == []
    findings = await SQLAgent._profile_source_rows(source, ["t"], "SELECT name, SUM(unused) FROM t GROUP BY name")
    assert any("-9999" in finding for finding in findings)


def test_numeric_cast_edit_is_dialect_aware():
    assert "TRY_CAST" in numeric_cast_edit("duckdb")
    assert "TRY_CAST" not in numeric_cast_edit("sqlite")
    assert "SAFE_CAST" in numeric_cast_edit("bigquery")


def test_summarize_tool_calls():
    events = [
        ModelCall("m", [], None, 0.1),
        ToolCall("distinct_values", {"table": "t", "column": "c"}, "Values of 'c':\n- 'A': 3"),
    ]
    assert summarize_tool_calls(events) == "- distinct_values(table='t', column='c') -> Values of 'c': - 'A': 3"
    assert summarize_tool_calls([ModelCall("m", [], None, 0.1)]) is None


def test_request_budget():
    events = [ModelCall("m", [], None, 0.1)] * 3
    attempts = [{"error": "ValueError: boom"}]
    RequestBudget(events, 4, attempts).check()
    with pytest.raises(RequestBudgetExceededError, match=re.escape("used 3 LLM calls, the limit is 3. Last error: ValueError: boom")):
        RequestBudget(events, 3, attempts).check()


@pytest.fixture
def test_messages():
    return [{"role": "user", "content": "Show the scores"}]


@pytest.fixture
def sql_context(values_source):
    async def make():
        return {
            "source": values_source,
            "sources": [values_source],
            "metaset": await get_metaset([values_source], ["people"]),
        }
    return make


def _submit(query, call_id="call_1"):
    arguments = json.dumps({"query": query, "table_slug": "people_rows", "tables": ["people"]})
    call = {"id": call_id, "type": "function", "function": {"name": "submit_sql", "arguments": arguments}}
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_sql_agent_fixes_a_failed_submission_inside_the_loop(sql_context, test_messages, monkeypatch):
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-test"}})
    requests = []
    responses = [_submit('SELECT "nope" FROM people'), _submit('SELECT "id", "score" FROM people')]

    async def run_client(model_spec, messages, **kwargs):
        requests.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(llm, "run_client", run_client)
    agent = SQLAgent(llm=llm, clean_data=False)

    with patch.object(SQLAgent, "_validate_sql", new=AsyncMock()) as validate_sql:
        out, _ = await agent.respond(test_messages, await sql_context())

    validate_sql.assert_not_awaited()
    assert len(requests) == 2
    rejection = requests[1][-1]
    assert rejection["role"] == "tool"
    assert "submit_sql rejected" in rejection["content"] and "nope" in rejection["content"]
    assert '"score"' in out[0].spec


async def test_sql_agent_asks_once_before_accepting_an_empty_result(sql_context, test_messages, monkeypatch):
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-test"}})
    empty = "SELECT \"id\" FROM people WHERE \"Gender\" = 'Other'"
    responses = [_submit(empty), _submit(empty)]
    requests = []

    async def run_client(model_spec, messages, **kwargs):
        requests.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(llm, "run_client", run_client)
    out, context = await SQLAgent(llm=llm, clean_data=False).respond(test_messages, await sql_context())

    assert "returned no rows" in requests[1][-1]["content"]
    assert len(context["pipeline"].data) == 0
    assert "Other" in out[0].spec


async def test_sql_agent_retry_receives_the_previous_attempt(sql_context, test_messages, llm):
    agent = SQLAgent(llm=llm)
    seen = []

    async def attempt_query(self, *args, previous_attempt=None, **kwargs):
        seen.append(previous_attempt)
        attempt = args[5]
        attempt["sql"] = "SELECT broken"
        attempt["tools"] = "- distinct_values(...) -> ..."
        if len(seen) == 1:
            raise ValueError("boom")
        return SimpleNamespace(render_context=AsyncMock(return_value={}))

    with patch.object(SQLAgent, "_attempt_query", new=attempt_query), patch("lumen.ai.utils.asyncio.sleep", new=AsyncMock()):
        await agent.respond(test_messages, await sql_context())

    assert seen[0] is None
    assert seen[1] == {"sql": "SELECT broken", "tools": "- distinct_values(...) -> ...", "error": "ValueError: boom"}


async def test_sql_agent_stops_at_the_request_timeout(sql_context, test_messages, llm):
    agent = SQLAgent(llm=llm, request_timeout=0.05)

    async def attempt_query(self, *args, **kwargs):
        await asyncio.sleep(5)

    with patch.object(SQLAgent, "_attempt_query", new=attempt_query):
        with pytest.raises(RequestBudgetExceededError, match=re.escape("0.05s time limit")):
            await agent.respond(test_messages, await sql_context())


async def test_sql_prompt_renders_previous_attempt_and_tools(sql_context, test_messages, llm):
    agent = SQLAgent(llm=llm)
    context = await sql_context()
    prompt = await agent._render_prompt(
        "main", test_messages, context, dialect="duckdb", is_final_step=True,
        sql_plan_context=None, active_filters=None,
        tool_names=["load_table_schemas", "distinct_values", "run_exploration_sql"], tool_rounds=6,
        previous_attempt={"sql": "SELECT broken", "tools": "- distinct_values(...) -> 'A'"},
        errors=["ValueError: boom"],
    )
    assert "browse_data_catalog" not in prompt
    assert "**distinct_values**" in prompt
    assert "You have 6 tool rounds" in prompt
    assert "SELECT broken" in prompt
    assert "- distinct_values(...) -> 'A'" in prompt


def test_sql_model_resolves_table_references():
    single = make_sql_model([("src", "orders"), ("src", "Customers")])
    output = single(query="SELECT 1", table_slug="x", tables=["ORDERS", "source/customers"])
    assert output.tables == ["orders", "Customers"]
    with pytest.raises(ValueError):
        single(query="SELECT 1", table_slug="x", tables=["missing"])

    multi = make_sql_model([("a", "orders"), ("b", "items")])
    output = multi(query="SELECT 1", table_slug="x", tables=[{"source": "source", "table": "Items"}, "A/orders"])
    assert [(item.source, item.table) for item in output.tables] == [("b", "items"), ("a", "orders")]


async def test_sql_agent_rejects_submissions_that_read_no_table(sql_context, test_messages, monkeypatch):
    llm = OpenAI(model_kwargs={"default": {"model": "gpt-test"}})
    responses = [_submit("SELECT 3 AS n"), _submit('SELECT COUNT(*) AS n FROM people')]
    requests = []

    async def run_client(model_spec, messages, **kwargs):
        requests.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(llm, "run_client", run_client)
    out, _ = await SQLAgent(llm=llm, clean_data=False).respond(test_messages, await sql_context())

    assert "The query reads no table" in requests[1][-1]["content"]
    assert "FROM people" in out[0].spec
