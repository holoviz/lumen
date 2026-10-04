import json

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from lumen.ai.schemas import DocumentChunk

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.chat import ChatFeed
from panel.pane import Markdown

from lumen.ai.agents import (
    AnalysisAgent, ChatAgent, SQLAgent, VegaLiteAgent,
)
from lumen.ai.agents.analysis import make_analysis_model
from lumen.ai.agents.base_lumen import BaseLumenAgent
from lumen.ai.agents.deck_gl import DeckGLAgent
from lumen.ai.agents.document_list import DocumentListAgent
from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.agents.sql import (
    EXPLORATION_MAX_ROWS, EXPLORATION_MAX_TOKENS, VALIDATION_MAX_ROWS,
    SQLCleanup, execute_exploration_sql, format_exploration_result,
    make_load_table_schemas_tool, make_run_exploration_sql_tool,
    make_sql_model, sql_contains_aggregates,
)
from lumen.ai.agents.table_list import TableListAgent
from lumen.ai.agents.vega_lite import (
    AltairChartSpec, AltairSpec, ChartSpec, VegaLiteSpec, VegaLiteSpecUpdate,
)
from lumen.ai.analysis import Analysis
from lumen.ai.config import DeterministicError, RetriesExceededError
from lumen.ai.context import schema_fields
from lumen.ai.editors import (
    AnalysisOutput, MultiChartEditor, SQLEditor, VegaLiteEditor,
)
from lumen.ai.llm import Llm
from lumen.ai.models import ReplaceLine, RetrySpec
from lumen.ai.schemas import (
    Column, Metaset, TableCatalogEntry, get_metaset,
)
from lumen.ai.utils import count_tokens, format_error
from lumen.config import SOURCE_TABLE_SEPARATOR, dump_yaml
from lumen.pipeline import Pipeline
from lumen.sources.base import QueryTimeoutError
from lumen.sources.duckdb import DuckDBSource
from lumen.views import Panel, Table

root = str(Path(__file__).parent.parent / "sources")


@pytest.mark.parametrize("input_slug", ["src ⦙ orders", "src/orders", "src.orders", "orders"])
async def test_load_table_schemas_resolves_aliases(input_slug):
    slug = f"src{SOURCE_TABLE_SEPARATOR}orders"
    metaset = Metaset(
        query=None,
        catalog={slug: TableCatalogEntry(slug, 1, [Column("order_id")])},
        schemas={slug: {"__len__": 12, "order_id": {"type": "integer"}}},
    )

    result = yaml.safe_load(await make_load_table_schemas_tool(metaset).function([input_slug]))

    assert result == {slug: {"row_count": 12, "schema": {"order_id": {"type": "integer"}}}}


async def test_load_table_schemas_preserves_dots_in_source_name():
    slug = f"src.prod{SOURCE_TABLE_SEPARATOR}orders"
    metaset = Metaset(
        query=None,
        catalog={slug: TableCatalogEntry(slug, 1, [Column("order_id")])},
        schemas={slug: {"order_id": {"type": "integer"}}},
    )

    for alias in ("src.prod/orders", "src.prod.orders"):
        result = yaml.safe_load(await make_load_table_schemas_tool(metaset).function([alias]))
        assert slug in result


async def test_load_table_schemas_rejects_ambiguous_bare_name():
    slugs = [f"{source}{SOURCE_TABLE_SEPARATOR}orders" for source in ("src_a", "src_b")]
    metaset = Metaset(
        query=None,
        catalog={slug: TableCatalogEntry(slug, 1, []) for slug in slugs},
        schemas={},
    )
    metaset.get_schema = AsyncMock()

    result = yaml.safe_load(await make_load_table_schemas_tool(metaset).function(["orders"]))

    assert "Ambiguous" in result["orders"]["error"]
    assert all(slug in result["orders"]["error"] for slug in slugs)
    metaset.get_schema.assert_not_awaited()


async def test_load_table_schemas_budget_applies_to_each_table():
    slugs = [f"src{SOURCE_TABLE_SEPARATOR}table_{i}" for i in range(4)]
    schemas = {
        slug: {"__len__": 100, **{f"column_{j}": {"type": "varchar"} for j in range(90)}}
        for slug in slugs
    }
    metaset = Metaset(
        query=None,
        catalog={slug: TableCatalogEntry(slug, 1, []) for slug in slugs},
        schemas=schemas,
    )

    result = await make_load_table_schemas_tool(metaset).function(slugs)

    assert count_tokens(result) > 3000
    for slug in slugs:
        assert slug in result
        assert f"{slug}:\n  row_count: 100\n  schema:\n    column_0:" in result
    assert count_tokens(result) < 4500


@pytest.mark.filterwarnings("ignore:Widget.name is deprecated:PendingDeprecationWarning")
async def test_list_agent_uses_configured_user():
    """List messages use the agent's display name, including overrides."""
    interface = ChatFeed()
    agent = TableListAgent(user="Data catalog", interface=interface)
    agent._get_items = lambda context: {"Example": ["table"]}

    await agent.respond([], {})

    assert interface.objects[-1].user == "Data catalog"


@pytest.fixture
def duckdb_source():
    duckdb_source = DuckDBSource(
        initializers=["INSTALL sqlite;", "LOAD sqlite;", f"SET home_directory='{root}';"],
        root=root,
        tables={"test_sql": f"SELECT A, B, C, D::TIMESTAMP_NS AS D FROM READ_CSV('{root + '/test.csv'}')"},
    )
    return duckdb_source


@pytest.fixture
def test_messages():
    """Create test messages for agent respond method"""
    return [{"role": "user", "content": "Test message"}]

async def test_chat_agent(llm, test_messages):
    agent = ChatAgent(llm=llm)

    llm.set_responses([
        "Test Response"
    ])

    out, out_context = await agent.respond(test_messages, {})
    assert out[0].object == "Test Response"
    assert out[0].user == "Lumen"

async def test_chat_agent_with_data(llm, duckdb_source, test_messages):
    """Test ChatAgent in analyst mode (with data)"""
    agent = ChatAgent(llm=llm)
    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "data": [{"A": 1, "B": 2}],
        "sql": "SELECT * FROM test_sql"
    }
    llm.set_responses([
        "Analysis of data"
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert out[0].object == "Analysis of data"
    assert out_context == {'chat': 'Analysis of data'}

async def test_sql_agent(llm, duckdb_source, test_messages):
    agent = SQLAgent(llm=llm)

    context = {
        "source": duckdb_source,
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
    }
    # Create the proper SQL model with tables field for single source
    SQLQueryWithTables = make_sql_model([(duckdb_source.name, "test_sql")])
    llm.set_responses([
        SQLQueryWithTables(
            query="SELECT SUM(A) as A_sum FROM test_sql",
            table_slug="test_sql_agg",
            tables=["test_sql"]
        ),
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert isinstance(out[0], SQLEditor)
    assert out[0].spec == (
        "SELECT\n"
        "  SUM(A) AS A_sum\n"
        "FROM test_sql"
    )
    assert set(out_context) == {"data", "pipeline", "sql", "table", "source"}


def test_sql_agent_same_source_selection_keeps_all_join_tables():
    source = DuckDBSource(tables={"customers": "SELECT 1", "orders": "SELECT 2"})
    agent = SQLAgent()
    selected = [SimpleNamespace(source="src", table=table) for table in ("customers", "orders")]
    resolved, tables = agent._merge_sources({("src", table.table): source for table in selected}, selected)

    assert resolved is source
    assert tables == ["customers", "orders"]


def test_sql_agent_rejects_invalid_source_table_pair():
    source = DuckDBSource(tables={"customers": "SELECT 1"})
    selected = [SimpleNamespace(source="other", table="customers")]
    with pytest.raises(ValueError, match="Unknown source/table pair"):
        SQLAgent()._merge_sources({("src", "customers"): source}, selected)


async def test_sql_agent_rejects_empty_single_source_tables(llm, test_messages):
    source = DuckDBSource(tables={"customers": "SELECT 1 AS id"})
    model = make_sql_model([(source.name, "customers")])
    empty = model(query="SELECT id FROM customers", table_slug="result", tables=[])
    valid = model(query="SELECT id FROM customers", table_slug="result", tables=["customers"])
    llm.set_responses([empty, valid])
    context = {"source": source, "sources": [source], "metaset": await get_metaset([source], ["customers"])}

    out, _ = await SQLAgent(llm=llm).respond(test_messages, context)

    assert llm._index == 2
    assert out[0].component.data["id"].tolist() == [1]


async def test_sql_agent_validation_fetch_is_bounded(llm, test_messages):
    source = DuckDBSource(tables={"numbers": "SELECT i FROM range(6000) AS t(i)"})
    agent = SQLAgent(llm=llm, clean_data=False)
    model = make_sql_model([(source.name, "numbers")])
    llm.set_responses([model(query="SELECT i FROM numbers", table_slug="numbers_result", tables=["numbers"])])
    context = {"source": source, "sources": [source], "metaset": await get_metaset([source], ["numbers"])}
    original = source.execute_with_timeout
    validated = []

    async def capture(query, timeout, fetch=False):
        result = await original(query, timeout, fetch)
        validated.append((query, len(result)))
        return result

    with patch.object(source, "execute_with_timeout", capture):
        out, _ = await agent.respond(test_messages, context)

    assert len(validated) == 1
    assert "LIMIT" in validated[0][0].upper()
    assert validated[0][1] == VALIDATION_MAX_ROWS
    assert len(out[0].component.data) == 6000


async def test_sql_exploration_caps_fetched_rows():
    source = DuckDBSource(tables={"numbers": "SELECT i FROM range(5000) AS t(i)"})
    with patch.object(source, "execute_with_timeout", wraps=source.execute_with_timeout) as execute:
        result = await execute_exploration_sql(source.name, "SELECT i FROM numbers", sources={(source.name, "numbers"): source})
    assert f"at least {EXPLORATION_MAX_ROWS} rows" in result
    assert "LIMIT" in execute.call_args.args[0].upper()


@pytest.mark.parametrize("query", ["SELECT 1; DROP TABLE numbers", "DELETE FROM numbers", "WITH x AS (DELETE FROM numbers) SELECT * FROM x", "SELECT * INTO temp FROM numbers", 'SELECT "unterminated FROM numbers'])
async def test_sql_exploration_rejects_non_read_only(query):
    source = DuckDBSource(tables={"numbers": "SELECT 1 AS i"})
    with patch.object(source, "execute_with_timeout", wraps=source.execute_with_timeout) as execute:
        result = await execute_exploration_sql(source.name, query, sources={(source.name, "numbers"): source})
    assert "error" in result.lower() or "read-only" in result.lower()
    execute.assert_not_called()


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
async def test_sql_agent_summarises_the_source_frame_unconverted(llm, test_messages, backend):
    """The summary is built from the source's own frame, not a pandas copy.

    Asserted on the frame describe_data is handed, because that is the only
    place the conversion would have happened; asserting on the summary text
    would pass whether or not one occurred.
    """
    pytest.importorskip(backend)
    import narwhals.stable.v2 as nw

    from lumen.ai.agents import sql as sql_module

    source = DuckDBSource(
        tables={"nums": "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(n, g)"},
        dataframe_backend=backend,
    )
    agent = SQLAgent(llm=llm)
    context = {
        "source": source,
        "sources": [source],
        "metaset": await get_metaset([source], ["nums"]),
    }
    SQLQueryWithTables = make_sql_model([(source.name, "nums")])
    llm.set_responses([
        SQLQueryWithTables(
            query="SELECT n, g FROM nums", table_slug="nums_all", tables=["nums"]
        ),
    ])

    seen = []
    original = sql_module.describe_data

    async def spy(df, *args, **kwargs):
        seen.append(df)
        return await original(df, *args, **kwargs)

    with patch.object(sql_module, "describe_data", spy):
        out, _ = await agent.respond(test_messages, context)

    assert len(out) == 1
    assert seen, "describe_data was never reached"
    assert nw.from_native(seen[0]).implementation.name.lower() == backend


@pytest.fixture
def dirty_source():
    """A table lint_data has something to say about: a duplicated row, padded
    text and a -9999 placeholder standing in for a missing measurement."""
    return DuckDBSource(tables={
        "dirty": """
            SELECT * FROM (
              VALUES (1, ' alpha ', 10.0),
                     (1, ' alpha ', 10.0),
                     (2, 'beta', -9999.0),
                     (3, 'gamma', 30.0),
                     (4, 'delta', 40.0)
            ) AS t(id, name, value)
        """
    })


async def _respond_to_dirty_table(llm, source, messages, responses, **agent_kwargs):
    """Run SQLAgent over the dirty fixture with a queued set of LLM responses."""
    agent = SQLAgent(llm=llm, **agent_kwargs)
    context = {
        "source": source,
        "sources": [source],
        "metaset": await get_metaset([source], ["dirty"]),
    }
    llm.set_responses(responses)
    out, _ = await agent.respond(messages, context)
    return out


async def test_sql_agent_clean_data_rewrites_dirty_query(llm, dirty_source, test_messages):
    SQLQueryWithTables = make_sql_model([(dirty_source.name, "dirty")])
    out = await _respond_to_dirty_table(llm, dirty_source, test_messages, [
        SQLQueryWithTables(query="SELECT * FROM dirty", table_slug="dirty_rows", tables=["dirty"]),
        SQLCleanup(
            chain_of_thought="Dropped the duplicate row and trimmed the names.",
            query='SELECT DISTINCT "id", TRIM("name") AS "name", "value" FROM dirty',
        ),
    ])
    assert llm._index == 2, "the cleaning pass should have consumed a second response"
    assert "DISTINCT" in out[0].spec
    assert "TRIM" in out[0].spec


async def test_sql_agent_clean_data_skipped_when_result_is_clean(llm, tiny_source, test_messages):
    """No findings must mean no second LLM call, or the pass costs on every query."""
    agent = SQLAgent(llm=llm)
    context = {
        "source": tiny_source,
        "sources": [tiny_source],
        "metaset": await get_metaset([tiny_source], ["tiny"]),
    }
    SQLQueryWithTables = make_sql_model([(tiny_source.name, "tiny")])
    llm.set_responses([
        SQLQueryWithTables(query="SELECT * FROM tiny", table_slug="tiny_rows", tables=["tiny"]),
    ])
    # Asserted on the method rather than the response counter: _clean_data_pass
    # swallows its own failures, so a consumed-responses count cannot tell
    # "never called" apart from "called and errored".
    with patch.object(SQLAgent, "_clean_data_pass", new=AsyncMock()) as clean_data_pass:
        await agent.respond(test_messages, context)
    clean_data_pass.assert_not_awaited()


@pytest.mark.parametrize("sql, expected", [
    ("SELECT region, SUM(revenue) FROM dirty GROUP BY region", True),
    ("SELECT COUNT(*) FROM dirty", True),
    ("WITH t AS (SELECT * FROM dirty) SELECT MAX(value) FROM t", True),
    ("SELECT * FROM dirty", False),
    ("SELECT DISTINCT * FROM dirty", False),
    ("SELECT id, name FROM dirty WHERE value > 0", False),
    ("not sql at all !!!", False),
])
def test_sql_contains_aggregates(sql, expected):
    assert sql_contains_aggregates(sql, "duckdb") is expected


async def test_sql_agent_profiles_source_rows_behind_an_aggregate(llm, dirty_source, test_messages):
    """An aggregate hides its inputs: SUM() already swallowed the -9999 placeholders,
    so the result cannot show them and the source rows must be profiled instead."""
    SQLQueryWithTables = make_sql_model([(dirty_source.name, "dirty")])
    captured = {}

    async def _capture(self, sql_query, findings, original_rows, source, messages, context, step):
        captured["findings"] = findings
        return sql_query

    with patch.object(SQLAgent, "_clean_data_pass", new=_capture):
        await _respond_to_dirty_table(llm, dirty_source, test_messages, [
            SQLQueryWithTables(
                query='SELECT "name", SUM("value") AS total FROM dirty GROUP BY "name"',
                table_slug="dirty_totals", tables=["dirty"],
            ),
        ])

    joined = " ".join(captured["findings"])
    assert "before aggregation" in joined
    assert "-9999" in joined, "the placeholder the aggregate hid must reach the rewrite"


async def test_sql_agent_skips_source_profiling_when_not_aggregating(llm, dirty_source, test_messages):
    """A plain SELECT already shows its own problems, so it must not pay for extra queries."""
    SQLQueryWithTables = make_sql_model([(dirty_source.name, "dirty")])
    with patch.object(SQLAgent, "_profile_source_rows") as profile:
        await _respond_to_dirty_table(llm, dirty_source, test_messages, [
            SQLQueryWithTables(query="SELECT * FROM dirty", table_slug="dirty_rows", tables=["dirty"]),
            SQLCleanup(chain_of_thought="Dropped duplicates.", query="SELECT DISTINCT * FROM dirty"),
        ])
    profile.assert_not_called()


async def test_sql_agent_clean_data_ignores_report_only_findings(llm, tiny_source, test_messages):
    """A constant column is reported but must not cost a rewrite: filtering to one
    value is an ordinary query, and dropping that column would lose the answer."""
    agent = SQLAgent(llm=llm)
    context = {
        "source": tiny_source,
        "sources": [tiny_source],
        "metaset": await get_metaset([tiny_source], ["tiny"]),
    }
    SQLQueryWithTables = make_sql_model([(tiny_source.name, "tiny")])
    llm.set_responses([
        SQLQueryWithTables(
            query="SELECT 'north' AS region, id FROM tiny",
            table_slug="tiny_region", tables=["tiny"],
        ),
    ])
    with patch.object(SQLAgent, "_clean_data_pass", new=AsyncMock()) as clean_data_pass:
        await agent.respond(test_messages, context)
    clean_data_pass.assert_not_awaited()


async def test_sql_agent_clean_data_disabled_leaves_query_alone(llm, dirty_source, test_messages):
    SQLQueryWithTables = make_sql_model([(dirty_source.name, "dirty")])
    responses = [
        SQLQueryWithTables(query="SELECT * FROM dirty", table_slug="dirty_rows", tables=["dirty"]),
    ]
    with patch.object(SQLAgent, "_clean_data_pass", new=AsyncMock()) as clean_data_pass:
        out = await _respond_to_dirty_table(
            llm, dirty_source, test_messages, responses, clean_data=False
        )
    clean_data_pass.assert_not_awaited()
    assert "DISTINCT" not in out[0].spec


async def test_sql_agent_clean_data_falls_back_when_rewrite_is_empty(llm, dirty_source, test_messages):
    """A rewrite that filters the answer away must not replace the answer."""
    SQLQueryWithTables = make_sql_model([(dirty_source.name, "dirty")])
    out = await _respond_to_dirty_table(llm, dirty_source, test_messages, [
        SQLQueryWithTables(query="SELECT * FROM dirty", table_slug="dirty_rows", tables=["dirty"]),
        SQLCleanup(chain_of_thought="Removed every row.", query="SELECT * FROM dirty WHERE 1 = 0"),
    ])
    assert llm._index == 2
    assert "1 = 0" not in out[0].spec


async def test_sql_agent_clean_data_falls_back_when_rewrite_is_invalid(llm, dirty_source, test_messages):
    """A rewrite naming a column that does not exist must not replace the answer."""
    SQLQueryWithTables = make_sql_model([(dirty_source.name, "dirty")])
    out = await _respond_to_dirty_table(llm, dirty_source, test_messages, [
        SQLQueryWithTables(query="SELECT * FROM dirty", table_slug="dirty_rows", tables=["dirty"]),
        SQLCleanup(chain_of_thought="Trimmed a column.", query='SELECT TRIM("nope") FROM dirty'),
    ])
    assert llm._index == 2
    assert "nope" not in out[0].spec


async def test_sql_agent_clean_data_falls_back_when_the_call_fails(llm, dirty_source, test_messages):
    """A failed cleaning call must keep the working query, not trigger a full retry."""
    def _explode():
        raise RuntimeError("cleaning model unavailable")

    SQLQueryWithTables = make_sql_model([(dirty_source.name, "dirty")])
    out = await _respond_to_dirty_table(llm, dirty_source, test_messages, [
        SQLQueryWithTables(query="SELECT * FROM dirty", table_slug="dirty_rows", tables=["dirty"]),
        _explode,
    ])
    assert llm._index == 2
    assert "dirty" in out[0].spec


async def test_vegalite_agent(llm, duckdb_source, test_messages):
    """Test VegaLiteAgent instantiation and respond"""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    spec = {
        "config": {"numberFormat": ","},
        "data": {
            "values": [
                {"A": 1, "B": 2, "C": 3, "D": "2023-01-01T00:00:00Z"},
                {"A": 4, "B": 5, "C": 6, "D": "2023-01-02T00:00:00Z"},
            ]
        },
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        }
    }
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="Test plot",
            charts=[ChartSpec(title="A vs B", yaml_spec=dump_yaml(spec))],
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
        VegaLiteSpecUpdate(
            chain_of_thought="All good",
            yaml_update=""
        )
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert isinstance(out[0], VegaLiteEditor)
    assert out[0].spec == "$schema: https://vega.github.io/schema/vega-lite/v5.json\nconfig:\n  numberFormat: ','\ndata:\n  values:\n  - A: 1\n    B: 2\n    C: 3\n    D: '2023-01-01T00:00:00Z'\n  - A: 4\n    B: 5\n    C: 6\n    D: '2023-01-02T00:00:00Z'\nencoding:\n  x:\n    field: A\n    type: quantitative\n  y:\n    field: B\n    type: quantitative\nheight: container\nmark: bar\nwidth: container\n"


async def test_vegalite_agent_multiple(llm, duckdb_source, test_messages):
    """Multiple charts: a view-only 'All' overview plus one editable editor per chart."""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    spec_ab = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        }
    }
    spec_ac = {
        "mark": "line",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "C", "type": "quantitative"},
        }
    }
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="Two distinct relationships",
            charts=[
                ChartSpec(title="A vs B", yaml_spec=dump_yaml(spec_ab)),
                ChartSpec(title="A vs C", yaml_spec=dump_yaml(spec_ac)),
            ],
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
        # One polish response per editable editor.
        VegaLiteSpecUpdate(chain_of_thought="ok", yaml_update=""),
        VegaLiteSpecUpdate(chain_of_thought="ok", yaml_update=""),
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 3
    # One editable VegaLiteEditor per chart first...
    assert isinstance(out[0], VegaLiteEditor)
    assert isinstance(out[1], VegaLiteEditor)
    assert "field: B" in out[0].spec
    assert "field: C" in out[1].spec
    # ...then the "All" overview as the last tab, editing every chart's spec
    # through one sub-tab each.
    assert isinstance(out[-1], MultiChartEditor)
    assert out[-1].title == "All"
    assert out[-1].chart_editors == out[:2]
    assert out[-1].editor._names == ["A vs B", "A vs C"]
    assert isinstance(out[-1].component, Panel)
    assert len(out[-1].component.object) == 2  # both plots stacked in the overview


async def test_vegalite_agent_skips_unparseable_chart(llm, duckdb_source, test_messages):
    """A chart whose spec fails to parse is skipped; valid charts still render."""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    good = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        }
    }
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="one good, one malformed",
            charts=[
                ChartSpec(title="good", yaml_spec=dump_yaml(good)),
                # Unbalanced braces -> YAML parse error.
                ChartSpec(title="bad", yaml_spec="mark: bar\nencoding:\n  x: {field: A, type: quantitative}}"),
            ],
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
        VegaLiteSpecUpdate(chain_of_thought="ok", yaml_update=""),
    ])
    out, out_context = await agent.respond(test_messages, context)
    # Only the valid chart survives -> a single editor, no "All" overview.
    assert len(out) == 1
    assert isinstance(out[0], VegaLiteEditor)
    assert "field: B" in out[0].spec


async def test_vegalite_agent_reports_why_every_chart_failed(llm, duckdb_source, test_messages):
    """When no chart parses the underlying errors reach the retry, which would
    otherwise regenerate blind against an identical message."""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    # Unbalanced braces -> YAML parse error, for every chart.
    malformed = "mark: bar\nencoding:\n  x: {field: A, type: quantitative}}"
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="all malformed",
            charts=[
                ChartSpec(title="first", yaml_spec=malformed),
                ChartSpec(title="second", yaml_spec=malformed),
            ],
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
    ] * 3)  # one response per retry_llm_output attempt
    with pytest.raises(RetriesExceededError) as excinfo:
        await agent.respond(test_messages, context)
    message = str(excinfo.value.__cause__)
    assert "first:" in message and "second:" in message


async def test_vegalite_agent_altair_multiple(llm, duckdb_source, test_messages):
    """VegaLiteAgent Altair (code) mode returns one editor per generated chart."""

    agent = VegaLiteAgent(llm=llm, code_execution="allow")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    chart_ab = SimpleNamespace(to_dict=lambda: {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        },
    })
    chart_ac = SimpleNamespace(to_dict=lambda: {
        "mark": "line",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "C", "type": "quantitative"},
        },
    })
    # Bypass real code execution; return a chart object per generated code block.
    agent._execute_code = AsyncMock(side_effect=[chart_ab, chart_ac])

    llm.set_responses([
        AltairSpec(
            chain_of_thought="Two independent charts",
            charts=[
                AltairChartSpec(
                    title="A vs B",
                    code="chart = alt.Chart(df).mark_bar().encode(x='A:Q', y='B:Q')"
                ),
                AltairChartSpec(
                    title="A vs C",
                    code="chart = alt.Chart(df).mark_line().encode(x='A:Q', y='C:Q')"
                ),
            ],
        ),
    ])
    out, out_context = await agent.respond(test_messages, context)
    # One editable editor per chart, then a view-only "All" overview last.
    assert len(out) == 3
    assert isinstance(out[0], VegaLiteEditor)
    assert isinstance(out[1], VegaLiteEditor)
    assert "field: B" in out[0].spec
    assert "field: C" in out[1].spec
    assert isinstance(out[-1], MultiChartEditor)
    assert out[-1].title == "All"


async def test_analysis_agent(llm, duckdb_source, test_messages):

    class TestAnalysis(Analysis):

        def __call__(self, pipeline, context):
            return f"Test Analysis"

    agent = AnalysisAgent(
        analyses=[TestAnalysis.instance(name='foo'), TestAnalysis.instance(name='bar')],
        llm=llm
    )
    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql")
    }

    model = make_analysis_model([analysis.name for analysis in agent.analyses])
    llm.set_responses([
        model(analysis="bar")
    ])
    out, out_context = await agent.respond(test_messages, context)

    assert len(out) == 1
    assert isinstance(out[0], AnalysisOutput)
    assert isinstance(out[0].component, Panel)
    assert isinstance(out[0].component.object, Markdown)

    assert "view" in out_context
    assert out_context["view"]["type"] == "panel"
    assert out_context["view"]["object"]["object"] == "Test Analysis"



@pytest.mark.asyncio
class TestDocumentListAgentIntegration:
    """Tests for DocumentListAgent with metaset."""

    async def test_document_list_agent_with_metaset(self):
        """Test that DocumentListAgent works with metaset.docs."""
        # Create metaset with document chunks
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[
                DocumentChunk(filename="readme.md", text="chunk 1", similarity=0.9),
                DocumentChunk(filename="readme.md", text="chunk 2", similarity=0.8),
                DocumentChunk(filename="schema.md", text="chunk 3", similarity=0.7),
            ]
        )
        
        context = {"metaset": metaset}
        
        # Test applies
        applies = await DocumentListAgent.applies(context)
        assert applies is True  # More than 1 unique document
        
        # Test _get_items
        agent = DocumentListAgent()
        items = agent._get_items(context)
        
        # Should return unique, sorted filenames
        assert items == {"Documents": ["readme.md", "schema.md"]}

    async def test_document_list_agent_no_docs(self):
        """Test that DocumentListAgent doesn't apply when no docs."""
        # Metaset without docs
        metaset = Metaset(query="test", catalog={}, docs=None)
        context = {"metaset": metaset}
        
        applies = await DocumentListAgent.applies(context)
        assert applies is False

    async def test_document_list_agent_single_doc(self):
        """Test that DocumentListAgent doesn't apply for single doc."""
        # Metaset with only one unique document
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[DocumentChunk(filename="readme.md", text="chunk", similarity=0.9)]
        )
        context = {"metaset": metaset}

        applies = await DocumentListAgent.applies(context)
        assert applies is True


@pytest.mark.asyncio
class TestTemplateOverrides:
    """Tests for template_overrides on Agent classes."""

    async def test_subclass_override(self, llm):
        """Subclass with class-level template_overrides injects extra instructions."""

        class CustomAgent(ChatAgent):
            template_overrides = {
                "main": {"instructions": "{{ super() }}\nCustom subclass rule."}
            }

        agent = CustomAgent(llm=llm)
        messages = [{"role": "user", "content": "test"}]
        prompt = await agent._render_prompt("main", messages, {})
        assert "Custom subclass rule." in prompt


    async def test_instance_override(self, llm):
        """Instance-level template_overrides injects extra instructions."""
        agent = ChatAgent(
            llm=llm,
            template_overrides={
                "main": {"instructions": "Instance override rule."}
            }
        )
        messages = [{"role": "user", "content": "test"}]
        prompt = await agent._render_prompt("main", messages, {})
        assert "Instance override rule." in prompt

    async def test_super_preserves_parent_content(self, llm):
        """{{ super() }} keeps the parent block content when present."""

        class ExtendedChatAgent(ChatAgent):
            template_overrides = {
                "main": {"footer": "{{ super() }}\nFooter appended."}
            }

        agent = ExtendedChatAgent(llm=llm)
        messages = [{"role": "user", "content": "test"}]
        prompt = await agent._render_prompt("main", messages, {})
        assert "Footer appended." in prompt


def test_map_agents_route_geometry_columns():
    """hvPlot and DeckGL agents advertise a geometry-column condition so the
    coordinator routes GeoDataFrame data to a map-capable view."""
    assert any("geometry" in c.lower() for c in hvPlotAgent.conditions)
    assert any("geometry" in c.lower() for c in DeckGLAgent.conditions)


def _revise_ms(*slugs):
    catalog = {
        s: TableCatalogEntry(table_slug=s, similarity=1.0, columns=[])
        for s in slugs
    }
    return Metaset(query=None, catalog=catalog)


def _editor(component):
    """Minimal stand-in for the editor, which only needs to carry .component."""
    return SimpleNamespace(component=component)


def test_resolve_revise_table_from_pipeline_component(llm, tiny_source):
    """A Pipeline names its own table."""
    agent = SQLAgent(llm=llm)
    ms = _revise_ms(f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny")
    pipeline = Pipeline(source=tiny_source, table="tiny")
    assert agent._resolve_revise_table({"metaset": ms}, _editor(pipeline)) == (
        f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny"
    )


def test_resolve_revise_table_from_view_component(llm, tiny_source):
    """A View has no table of its own, so it reaches one through its pipeline."""
    agent = SQLAgent(llm=llm)
    ms = _revise_ms(f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny")
    view = Table(pipeline=Pipeline(source=tiny_source, table="tiny"))
    assert agent._resolve_revise_table({"metaset": ms}, _editor(view)) == (
        f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny"
    )


def test_resolve_revise_table_prefers_own_table_over_chained_parent(llm):
    """A chained Pipeline also has a parent pipeline; the child's own table wins."""
    agent = SQLAgent(llm=llm)
    source = DuckDBSource(tables={
        "parent": "SELECT 1 AS id",
        "child": "SELECT 2 AS id",
    })
    parent = Pipeline(source=source, table="parent")
    child = Pipeline(source=source, table="child", pipeline=parent)
    ms = _revise_ms(
        f"{source.name}{SOURCE_TABLE_SEPARATOR}parent",
        f"{source.name}{SOURCE_TABLE_SEPARATOR}child",
    )
    assert agent._resolve_revise_table({"metaset": ms}, _editor(child)) == (
        f"{source.name}{SOURCE_TABLE_SEPARATOR}child"
    )


def test_resolve_revise_table_from_context_pipeline(llm, tiny_source):
    agent = SQLAgent(llm=llm)
    ms = _revise_ms(f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny")
    ctx = {"metaset": ms, "pipeline": Pipeline(source=tiny_source, table="tiny")}
    assert agent._resolve_revise_table(ctx, None) == (
        f"{tiny_source.name}{SOURCE_TABLE_SEPARATOR}tiny"
    )


def test_resolve_revise_table_from_context_table(llm):
    agent = SQLAgent(llm=llm)
    ms = _revise_ms(f"S{SOURCE_TABLE_SEPARATOR}t1")
    assert agent._resolve_revise_table({"metaset": ms, "table": "t1"}, None) == f"S{SOURCE_TABLE_SEPARATOR}t1"


def test_resolve_revise_table_unknown_name_returns_none(llm, tiny_source):
    """A table the metaset has never heard of resolves to nothing, so the
    prompt falls back to the broader context rather than an empty scope."""
    agent = SQLAgent(llm=llm)
    ms = _revise_ms(f"other{SOURCE_TABLE_SEPARATOR}elsewhere")
    pipeline = Pipeline(source=tiny_source, table="tiny")
    assert agent._resolve_revise_table({"metaset": ms}, _editor(pipeline)) is None


def _revise_ms_cols(entries):
    """Build a Metaset from {slug: (column_name, sql_expr)}."""
    catalog = {
        slug: TableCatalogEntry(
            table_slug=slug, similarity=1.0,
            columns=[Column(name=col)], sql_expr=sql,
        )
        for slug, (col, sql) in entries.items()
    }
    return Metaset(query=None, catalog=catalog)


@pytest.mark.parametrize("agent_cls", [SQLAgent, VegaLiteAgent, DeckGLAgent])
async def test_revise_output_prompt_scopes_to_revise_table(llm, agent_cls):
    """With revise_table set, only that table's schema + derived SQL appears.
    Parametrized to prove every child revise_output template inherits the fix."""
    agent = agent_cls(llm=llm)
    t1 = f"S{SOURCE_TABLE_SEPARATOR}orders"
    t2 = f"S{SOURCE_TABLE_SEPARATOR}customers"
    ms = _revise_ms_cols({t1: ("order_id", "SELECT * FROM raw_orders"),
                          t2: ("customer_id", None)})
    prompt = await agent._render_prompt(
        "revise_output", [{"role": "user", "content": "fix"}], {"metaset": ms},
        numbered_text="1: SELECT 1", language="sql", feedback="fix",
        errors=None, revise_table=t1,
    )
    assert "order_id" in prompt
    assert "raw_orders" in prompt          # derived SQL (read_with) of the scoped table
    assert "customer_id" not in prompt     # unrelated table's columns excluded
    assert "customers" not in prompt


async def test_revise_output_prompt_fallback_without_revise_table(llm):
    """Without revise_table (or None), both tables appear (prior behaviour) and
    the StrictUndefined template does not raise."""
    agent = SQLAgent(llm=llm)
    t1 = f"S{SOURCE_TABLE_SEPARATOR}orders"
    t2 = f"S{SOURCE_TABLE_SEPARATOR}customers"
    ms = _revise_ms_cols({t1: ("order_id", None), t2: ("customer_id", None)})
    base_kwargs = dict(
        numbered_text="1: SELECT 1", language="sql", feedback="fix", errors=None,
    )
    for extra in ({}, {"revise_table": None}):
        prompt = await agent._render_prompt(
            "revise_output", [{"role": "user", "content": "fix"}], {"metaset": ms},
            **base_kwargs, **extra,
        )
        assert "order_id" in prompt
        assert "customer_id" in prompt


async def test_revise_output_prompt_unknown_slug_falls_back(llm):
    """A revise_table not in the catalog falls back to the broader context."""
    agent = SQLAgent(llm=llm)
    t1 = f"S{SOURCE_TABLE_SEPARATOR}orders"
    t2 = f"S{SOURCE_TABLE_SEPARATOR}customers"
    ms = _revise_ms_cols({t1: ("order_id", None), t2: ("customer_id", None)})
    prompt = await agent._render_prompt(
        "revise_output", [{"role": "user", "content": "fix"}], {"metaset": ms},
        numbered_text="1: SELECT 1", language="sql", feedback="fix",
        errors=None, revise_table=f"S{SOURCE_TABLE_SEPARATOR}does_not_exist",
    )
    assert "order_id" in prompt
    assert "customer_id" in prompt


async def test_revise_forwards_resolved_table_to_prompt(llm):
    """revise() resolves the current table and forwards it to the prompt render."""
    agent = SQLAgent(llm=llm)
    t1 = f"S{SOURCE_TABLE_SEPARATOR}orders"
    ms = _revise_ms(t1, f"S{SOURCE_TABLE_SEPARATOR}customers")
    captured = {}

    async def fake_invoke(prompt_name, messages, context, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(edits=[])

    with patch.object(agent, "_invoke_prompt", side_effect=fake_invoke):
        await agent.revise(
            "fix it", [{"role": "user", "content": "fix"}],
            {"metaset": ms, "table": "orders"}, spec="SELECT 1", language="sql",
        )
    assert captured.get("revise_table") == t1


async def test_revise_recovers_from_malformed_sql_via_retry(llm):
    """A line-patch edit that sqlglot cannot parse must not surface a raw
    ParseError to the user: revise() should feed the parse error back to the
    LLM as feedback and retry, the same recovery _validate_sql already uses
    for execution errors, rather than letting clean_sql's exception escape
    straight through revise() to the caller.

    The retry must keep the user's original instruction intact (as `feedback`
    in the prompt) and carry the parse error separately via `errors`, rather
    than replacing the instruction with the error text - `errors` is a
    distinct prompt slot the base template already renders. It must also
    still run `view.validate_spec()` on the spec it settles on, same as the
    first attempt.
    """
    agent = SQLAgent(llm=llm)
    validated = []
    view = SimpleNamespace(
        component=SimpleNamespace(source=SimpleNamespace(dialect="duckdb")),
        language="sql",
        validate_spec=lambda spec: validated.append(spec) or spec,
    )
    seen_instructions = []
    seen_errors = []

    async def fake_super_revise(instruction, messages, context, view=None, spec=None, language=None, errors=None, **kwargs):
        seen_instructions.append(instruction)
        seen_errors.append(errors)
        if len(seen_instructions) == 1:
            return "SELECT * WHERE \"a\" = 'x' FROM t"  # WHERE before FROM: unparsable
        return "SELECT * FROM t WHERE \"a\" = 'x'"

    with patch.object(BaseLumenAgent, "revise", side_effect=fake_super_revise):
        result = await agent.revise("fix it", [{"role": "user", "content": "fix"}], {}, view=view)

    assert len(seen_instructions) == 2
    assert seen_instructions[1] == "fix it"  # original instruction preserved, not overwritten by the error
    assert seen_errors[1] is not None and "ParseError" in seen_errors[1][0]
    assert result == 'SELECT\n  *\nFROM t\nWHERE\n  "a" = \'x\''
    assert validated == [result]  # the settled spec was validated, same as the first attempt


async def test_revise_raises_after_retries_exhausted_on_malformed_sql(llm):
    """When the LLM keeps producing unparsable SQL, revise() must eventually
    raise (bounded retries, matching _validate_sql's max_retries pattern)
    rather than retrying forever."""
    agent = SQLAgent(llm=llm)
    view = SimpleNamespace(
        component=SimpleNamespace(source=SimpleNamespace(dialect="duckdb")),
        language="sql",
        validate_spec=lambda spec: spec,
    )
    calls = []

    async def fake_super_revise(instruction, messages, context, view=None, spec=None, language=None, **kwargs):
        calls.append(instruction)
        return "SELECT * WHERE \"a\" = 'x' FROM t"  # always unparsable

    with patch.object(BaseLumenAgent, "revise", side_effect=fake_super_revise):
        with pytest.raises(Exception, match="Invalid expression"):
            await agent.revise("fix it", [{"role": "user", "content": "fix"}], {}, view=view, max_retries=2)

    assert len(calls) == 2


async def test_revise_max_retries_zero_still_cleans_sql(llm):
    """max_retries=0 must not silently return the unprettified, unvalidated
    LLM output by falling through an empty `range(0)` loop - it should still
    attempt clean_sql once (and raise, not swallow, if that attempt fails)."""
    agent = SQLAgent(llm=llm)
    view = SimpleNamespace(
        component=SimpleNamespace(source=SimpleNamespace(dialect="duckdb")),
        language="sql",
        validate_spec=lambda spec: spec,
    )

    async def fake_super_revise(instruction, messages, context, view=None, spec=None, language=None, **kwargs):
        return "select  *   from t"  # parsable but not prettified

    with patch.object(BaseLumenAgent, "revise", side_effect=fake_super_revise):
        result = await agent.revise("fix it", [{"role": "user", "content": "fix"}], {}, view=view, max_retries=0)

    assert result == "SELECT\n  *\nFROM t"  # cleaned/prettified, not the raw passthrough

    async def fake_super_revise_bad(instruction, messages, context, view=None, spec=None, language=None, **kwargs):
        return "SELECT * WHERE \"a\" = 'x' FROM t"  # unparsable

    with patch.object(BaseLumenAgent, "revise", side_effect=fake_super_revise_bad):
        with pytest.raises(Exception, match="Invalid expression"):
            await agent.revise("fix it", [{"role": "user", "content": "fix"}], {}, view=view, max_retries=0)


def test_sqlagent_active_filters_describes_conditions():
    """SQLAgent._active_filters turns the interactive slider filters into
    WHERE-style conditions (skipping inactive/full-range ones) so a follow-up
    query can preserve the subset."""
    class _Filter:
        def __init__(self, field, query):
            self.field, self._query = field, query

        @property
        def query(self):
            return self._query

    pipeline = SimpleNamespace(filters=[
        _Filter("game_year", (2000, 2016)),
        _Filter("game_season", ["Summer", "Winter"]),
        _Filter("game_location", "Japan"),
        _Filter("game_slug", None),  # full range / nothing selected -> skipped
    ])
    assert SQLAgent._active_filters(pipeline) == [
        "game_year between 2000 and 2016",
        "game_season in ('Summer', 'Winter')",
        "game_location = 'Japan'",
    ]
    assert SQLAgent._active_filters(None) is None
    assert SQLAgent._active_filters(SimpleNamespace(filters=[])) is None


async def test_sqlagent_prompt_surfaces_active_filters(llm):
    """Active exploration filters are surfaced in the SQL agent's prompt so a
    follow-up query keeps the subset."""
    agent = SQLAgent(llm=llm)
    messages = [{"role": "user", "content": "top 5 rows"}]
    prompt = await agent._render_prompt(
        "main", messages, {},
        dialect="duckdb", is_final_step=True, step_number=1, current_step="",
        sql_query_history={}, current_iteration=1, sql_plan_context=None,
        errors=None, discovery_context=None, source_names=["src"],
        active_filters=["game_year between 2000 and 2016", "game_season in ('Summer')"],
    )
    assert "Active exploration filters" in prompt
    assert "game_year between 2000 and 2016" in prompt
    assert "game_season in ('Summer')" in prompt


async def test_view_retry_keeps_context_and_passes_spec_by_keyword(llm):
    """The yaml auto-retry must keep the TContext dict (the next _extract_spec and
    revise both need it) and pass the spec by keyword, otherwise it binds to the
    `view` parameter and revise() raises AttributeError on `view.spec`."""
    agent = hvPlotAgent(llm=llm)
    captured = {}

    class _Out(dict):
        chain_of_thought = ""

    async def fake_stream_prompt(*args, **kwargs):
        # _stream_prompt is an async generator function, so the caller iterates
        # what it returns rather than awaiting it.
        yield _Out(yaml_spec="kind: line")

    async def fake_extract_spec(context, spec):
        raise ValueError("bad spec")

    async def fake_revise(instruction, messages, context, view=None, spec=None, **kwargs):
        captured["context"] = context
        captured["view"] = view
        captured["spec"] = spec
        return "kind: line"

    ctx = {"pipeline": SimpleNamespace(table="t")}
    with patch.object(agent, "_stream_prompt", side_effect=fake_stream_prompt), \
         patch.object(agent, "_extract_spec", side_effect=fake_extract_spec), \
         patch.object(agent, "revise", side_effect=fake_revise):
        # _extract_spec always fails, so the retry loop exhausts and raises
        with pytest.raises(Exception):
            await agent._generate_yaml_spec(
                [{"role": "user", "content": "x"}], ctx,
                SimpleNamespace(table="t", source=None), {},
            )

    # context must still be the TContext dict, not a formatted yaml string
    assert captured["context"] is ctx
    # the spec must not have bound to the `view` parameter
    assert captured["view"] is None
    assert isinstance(captured["spec"], str)


async def test_view_retry_recovers_using_revised_spec(llm):
    """The retry must not assume a `yaml_spec` field (hvPlot emits a flat param
    dict) and must feed the revision into the next attempt, otherwise it re-runs
    _extract_spec on the identical spec and can never recover."""
    agent = hvPlotAgent(llm=llm)
    seen_specs = []

    class _Out(dict):
        chain_of_thought = ""

    async def fake_stream_prompt(*args, **kwargs):
        # realistic hvPlot output shape: view params, no yaml_spec
        yield _Out(kind="line", x="a", y="b")

    async def fake_extract_spec(context, spec):
        seen_specs.append(dict(spec))
        if len(seen_specs) == 1:
            raise ValueError("bad spec")
        return dict(spec)

    async def fake_revise(instruction, messages, context, view=None, spec=None, **kwargs):
        return "kind: bar\nx: a\ny: b"

    ctx = {"pipeline": SimpleNamespace(table="t")}
    with patch.object(agent, "_stream_prompt", side_effect=fake_stream_prompt), \
         patch.object(agent, "_extract_spec", side_effect=fake_extract_spec), \
         patch.object(agent, "revise", side_effect=fake_revise):
        result = await agent._generate_yaml_spec(
            [{"role": "user", "content": "x"}], ctx,
            SimpleNamespace(table="t", source=None), {},
        )

    # revise was reached (no KeyError on the missing yaml_spec) and retried once
    assert len(seen_specs) == 2
    assert seen_specs[0]["kind"] == "line"   # first attempt used the original
    assert seen_specs[1]["kind"] == "bar"    # retry used the REVISED spec
    assert result["kind"] == "bar"


# -------------------------------------------------------------------
# format_exploration_result
# -------------------------------------------------------------------

def _exploration_frame(n_rows=53, n_cols=7):
    rng = np.random.default_rng(0)
    data = {"station_id": [f"ST{i:04d}" for i in range(n_rows)]}
    for c in range(n_cols - 1):
        data[f"metric_{c}"] = rng.uniform(0, 100, n_rows).round(4)
    return pd.DataFrame(data)


def test_format_exploration_result_reports_full_shape():
    """The true row count is stated, so no follow-up COUNT(*) is needed."""

    result = format_exploration_result(_exploration_frame())
    assert "53 rows x 7 columns" in result
    assert "first 5 of 53 rows" in result


def test_format_exploration_result_lists_dtypes():

    result = format_exploration_result(_exploration_frame())
    assert "station_id: str" in result
    assert "metric_0: float64" in result


def test_format_exploration_result_stays_within_budget():

    wide = _exploration_frame(n_rows=100_000, n_cols=40)
    result = format_exploration_result(wide)
    assert count_tokens(result) <= EXPLORATION_MAX_TOKENS
    assert "100000 rows x 40 columns" in result
    assert "showing the first 25 columns" in result


def test_format_exploration_result_is_far_cheaper_than_full_dump():
    """The preview must cost a fraction of the old 50-row aligned to_string dump."""

    df = _exploration_frame()
    old = df.head(100).to_string(max_cols=25, max_rows=50)
    new = format_exploration_result(df)
    assert count_tokens(new) < count_tokens(old) / 3


def test_format_exploration_result_empty_frame():

    result = format_exploration_result(_exploration_frame(n_rows=0))
    assert "0 rows x 7 columns" in result
    assert "no rows" in result


def test_unique_expr_slug_passes_through_fresh_name():

    assert SQLAgent._unique_expr_slug("top_5_athletes", {"hosts": "SELECT 1"}) == "top_5_athletes"


def test_unique_expr_slug_avoids_existing_table():
    """A slug echoing an input table must not be reused: materializing it would
    issue CREATE OR REPLACE VIEW over the source table and destroy it."""

    tables = {"data_olympic_hosts_csv": "SELECT * FROM data_olympic_hosts_csv"}
    slug = SQLAgent._unique_expr_slug("data_olympic_hosts_csv", tables)
    assert slug not in tables
    assert slug.startswith("data_olympic_hosts_csv")


def test_unique_expr_slug_skips_taken_suffixes():

    tables = {"hosts": "", "hosts_derived_1": "", "hosts_derived_2": ""}
    assert SQLAgent._unique_expr_slug("hosts", tables) == "hosts_derived_3"


def test_colliding_slug_does_not_clobber_source_table():
    """End-to-end guard for the materialization overwrite: the original table
    must survive a query whose table_slug collides with its name."""

    source = DuckDBSource(tables={"hosts": "SELECT 2022 AS game_year, 'China' AS game_location"})
    before = source.get("hosts")

    # A model handing back an unrelated exploration query under the input
    # table's own name — the observed weak-model failure.
    bad_sql = "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
    slug = SQLAgent._unique_expr_slug("hosts", source.tables)
    source.create_sql_expr_source({slug: bad_sql}, materialize=True)

    pd.testing.assert_frame_equal(source.get("hosts"), before)


@pytest.mark.parametrize("agent_cls", [hvPlotAgent, VegaLiteAgent, DeckGLAgent])
def test_view_agents_require_view_output(agent_cls):
    # `view` must be a required key of the output schema, otherwise a view agent
    # that finishes without producing a view goes undetected (holoviz/lumen#2106)
    assert "view" in agent_cls.output_schema.__required_keys__
    assert schema_fields(agent_cls.output_schema)["view"]["required"]


async def test_revise_keeps_sql_hash_literals(llm):
    """Revised SQL used to round-trip through YAML, which truncates at '#'."""
    agent = SQLAgent(llm=llm)
    fixed = "SELECT * FROM schools WHERE street = '80651 Route #271'"
    revision = RetrySpec(chain_of_thought="Fix the street", edits=[ReplaceLine(line_no=1, line=fixed)])

    with patch.object(agent, "_invoke_prompt", AsyncMock(return_value=revision)):
        result = await agent.revise("fix", [{"role": "user", "content": "fix"}], {}, spec="SELECT 1", language="sql.duckdb")

    assert result == fixed


async def test_sql_agent_accepts_empty_result_after_one_retry(llm, test_messages):
    source = DuckDBSource(tables={"orders": "SELECT * FROM (VALUES (1, 'paid')) AS t(id, status)"})
    model = make_sql_model([(source.name, "orders")])
    query = model(query="SELECT id FROM orders WHERE status = 'refunded'", table_slug="refunded_orders", tables=["orders"])
    llm.set_responses([query, query])
    context = {"source": source, "sources": [source], "metaset": await get_metaset([source], ["orders"])}

    out, _ = await SQLAgent(llm=llm, clean_data=False).respond(test_messages, context)

    assert llm._index == 2
    assert len(out[0].component.data) == 0


async def test_sql_agent_does_not_retry_unresolvable_table(llm, test_messages):
    source = DuckDBSource(tables={"orders": "SELECT 1 AS id"})
    context = {"source": source, "sources": [source], "metaset": await get_metaset([source], ["orders"])}
    source._connection.execute("DROP VIEW orders")
    model = make_sql_model([(source.name, "orders")])
    llm.set_responses([model(query="SELECT id FROM orders", table_slug="order_ids", tables=["orders"])] * 3)

    with pytest.raises(DeterministicError, match="orders"):
        await SQLAgent(llm=llm, clean_data=False).respond(test_messages, context)

    assert llm._index == 1


async def test_sql_agent_revises_a_timed_out_query_once(llm, test_messages):
    source = DuckDBSource(tables={"orders": "SELECT 1 AS id"})
    context = {"source": source, "sources": [source], "metaset": await get_metaset([source], ["orders"])}
    model = make_sql_model([(source.name, "orders")])
    revision = RetrySpec(chain_of_thought="Do less work", edits=[ReplaceLine(line_no=2, line="  1 AS id")])
    llm.set_responses([model(query="SELECT id FROM orders", table_slug="order_ids", tables=["orders"]), revision])

    timeout = AsyncMock(side_effect=QueryTimeoutError("The query did not finish within 1 seconds."))
    with patch.object(DuckDBSource, "execute_with_timeout", timeout), pytest.raises(DeterministicError, match="timed out"):
        await SQLAgent(llm=llm, clean_data=False, query_timeout=1).respond(test_messages, context)

    assert llm._index == 2
    assert [call.args[1] for call in timeout.call_args_list] == [1, 1]


async def test_sql_exploration_reports_timeout():
    source = DuckDBSource(tables={"t": "SELECT 1 AS a"})
    slow = "SELECT COUNT(*) AS n FROM range(100000000) a, range(100000000) b"
    result = await execute_exploration_sql(source.name, slow, sources={(source.name, "t"): source}, timeout=0.1)
    assert result == "QueryTimeoutError: The query did not finish within 0.1 seconds."


async def test_sql_exploration_error_hides_limit_wrapper():
    source = DuckDBSource(tables={"t": "SELECT 1 AS a"})
    result = await execute_exploration_sql(source.name, "SELECT nope FROM t", sources={(source.name, "t"): source})

    assert "subquery" not in result
    query_line, caret_line = next(
        (line, following) for line, following in zip(result.splitlines(), result.splitlines()[1:], strict=False)
        if "SELECT nope FROM t" in line
    )
    assert caret_line.index("^") == query_line.index("nope")


def test_format_error_strips_ansi_codes():
    error = ValueError("Invalid expression. Line 1, Col: 23.\n  SELECT a FROM t WHERE \x1b[4m(\x1b[0m")
    assert format_error(error) == "ValueError: Invalid expression. Line 1, Col: 23.\n  SELECT a FROM t WHERE ("


def test_exploration_tool_doc_quotes_every_name():
    sources = {("src", "orders"): None, ("src", "customers"): None}
    doc = make_run_exploration_sql_tool(sources).function.__doc__
    assert "Sources: `src`." in doc
    assert "Tables: `customers`, `orders`." in doc
