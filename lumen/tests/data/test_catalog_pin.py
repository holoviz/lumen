"""
Pins the exact text a Metaset renders for a language model, so moving or
refactoring the catalog cannot change a prompt without a test noticing.
"""
import pytest

from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.data.catalog import (
    Column, DocumentChunk, Metaset, TableCatalogEntry,
)
from lumen.data.stats import (
    EXACT, SAMPLED, TYPES_ONLY, ColumnStats, TableStats,
)

SEP = SOURCE_TABLE_SEPARATOR


def _pin_metaset():
    orders = f"shop{SEP}orders"
    customers = f"shop{SEP}customers"
    paid = f"shop{SEP}paid_orders"
    events = f"web{SEP}events"
    embeddings = f"web{SEP}embeddings"
    catalog = {
        orders: TableCatalogEntry(
            table_slug=orders, similarity=0.92,
            columns=[Column("id"), Column("customer_id"), Column("amount", "Order total in EUR"), Column("status")],
            description="One row per order placed in the web shop.",
            sql_expr="SELECT * FROM orders",
            metadata={"owner": "sales", "refresh": "daily"},
        ),
        customers: TableCatalogEntry(
            table_slug=customers, similarity=0.71,
            columns=[Column("id"), Column("name"), Column("region")],
            sql_expr="SELECT * FROM customers",
        ),
        paid: TableCatalogEntry(
            table_slug=paid, similarity=0.88,
            columns=[Column("id"), Column("amount")],
            description="Orders that were paid.",
            sql_expr="SELECT id, amount FROM orders WHERE status = 'paid'",
            derived_from=[orders], created_order=1,
        ),
        events: TableCatalogEntry(
            table_slug=events, similarity=0.4,
            columns=[Column("ts"), Column("user"), Column("kind")],
        ),
        embeddings: TableCatalogEntry(
            table_slug=embeddings, similarity=0.2,
            columns=[Column("doc")] + [Column(f"pc{i}") for i in range(1, 13)],
        ),
    }
    stats = {
        orders: TableStats(
            "orders", rows=4, rows_exact=True, method=EXACT, ranges_exact=True,
            columns=[
                ColumnStats("id", "INTEGER", "numeric", nulls=0.0, min=10, max=13, distinct=4, primary_key=True),
                ColumnStats("customer_id", "INTEGER", "numeric", nulls=0.0, min=1, max=3, distinct=3, references="customers.id"),
                ColumnStats("amount", "DOUBLE", "numeric", nulls=0.0, min=0.001234, max=100.0, distinct=4),
                ColumnStats(
                    "status", "VARCHAR", "string", nulls=0.0, distinct=2,
                    values=[["paid", 3], ["cancelled", 1]], values_complete=True,
                ),
            ],
        ),
        customers: TableStats(
            "customers", rows=1200, method=SAMPLED, sample_size=500,
            columns=[
                ColumnStats("id", "INTEGER", "numeric", min=1, max=1200, primary_key=True),
                ColumnStats("name", "VARCHAR", "string", nulls=0.02, examples=["Ada", "Bo", "Cy"]),
                ColumnStats("region", "VARCHAR", "string", values=[["east", 700], ["west", 500]]),
            ],
        ),
        events: TableStats(
            "events", method=TYPES_ONLY,
            columns=[
                ColumnStats("ts", "TIMESTAMP", "datetime"),
                ColumnStats("user", "VARCHAR", "string"),
                ColumnStats("kind", "VARCHAR", "string"),
            ],
        ),
    }
    schemas = {
        paid: {
            "id": {"min": 10, "max": 13},
            "amount": {"type": "num", "min": 0.001234, "max": 12.5},
            "__len__": 3,
        },
    }
    docs = [DocumentChunk(filename="glossary.md", text="Paid orders exclude refunds.", similarity=0.83)]
    return Metaset(query="revenue by region", catalog=catalog, schemas=schemas, docs=docs, stats=stats)


CASES = {
    "table_context": lambda ms: ms.table_context(),
    "table_context_n2_others": lambda ms: ms.table_context(n=2, n_others=5),
    "compact_context": lambda ms: ms.compact_context(),
    "compact_context_scoped_sql": lambda ms: ms.compact_context(
        schema_tables=[f"shop{SEP}orders", f"shop{SEP}paid_orders"], include_sql=True, n_others=10,
    ),
    "compact_context_docs": lambda ms: ms.compact_context(n=1, include_docs=True),
}

EXPECTED = {
    "table_context": (
        "shop ⦙ orders:\n"
        "  info: One row per order placed in the web shop.\n"
        "  metadata:\n"
        "    owner: sales\n"
        "    refresh: daily\n"
        "  columns:\n"
        "  - id\n"
        "  - customer_id\n"
        "  - amount\n"
        "  - status\n"
        "shop ⦙ paid_orders:\n"
        "  info: Orders that were paid.\n"
        "  derived_from:\n"
        "  - orders\n"
        "  step: 1\n"
        "  latest: true\n"
        "  columns:\n"
        "  - id\n"
        "  - amount\n"
        "shop ⦙ customers:\n"
        "  columns:\n"
        "  - id\n"
        "  - name\n"
        "  - region\n"
        "web ⦙ events:\n"
        "  columns:\n"
        "  - ts\n"
        "  - user\n"
        "  - kind\n"
        "web ⦙ embeddings:\n"
        "  columns:\n"
        "  - doc\n"
        "  - pc1..pc12 (12 cols)\n"
    ),
    "table_context_n2_others": (
        "shop ⦙ orders:\n"
        "  info: One row per order placed in the web shop.\n"
        "  metadata:\n"
        "    owner: sales\n"
        "    refresh: daily\n"
        "  columns:\n"
        "  - id\n"
        "  - customer_id\n"
        "  - amount\n"
        "  - status\n"
        "shop ⦙ paid_orders:\n"
        "  info: Orders that were paid.\n"
        "  derived_from:\n"
        "  - orders\n"
        "  step: 1\n"
        "  latest: true\n"
        "  columns:\n"
        "  - id\n"
        "  - amount\n"
        "\n"
        "\n"
        "Others available:\n"
        "- shop ⦙ customers\n"
        "- web ⦙ events\n"
        "- web ⦙ embeddings\n"
    ),
    "compact_context": (
        "shop ⦙ orders (4 rows)\n"
        "  info: One row per order placed in the web shop.\n"
        "  metadata: owner=sales, refresh=daily\n"
        "  id INTEGER PK 10..13\n"
        "  customer_id INTEGER -> customers.id 1..3\n"
        "  amount DOUBLE 0.001234..100 unique -- Order total in EUR\n"
        "  status VARCHAR {paid:3, cancelled:1}\n"
        "\n"
        "shop ⦙ paid_orders (3 rows)\n"
        "  info: Orders that were paid.\n"
        "  derived_from: orders (step 1, latest)\n"
        "  id 10..13\n"
        "  amount DOUBLE 0.001234..12.5\n"
        "\n"
        "shop ⦙ customers (~1200 rows; stats from a 500-row sample)\n"
        "  id INTEGER PK 1..1200\n"
        "  name VARCHAR e.g. 'Ada', 'Bo', 'Cy' nulls 2%\n"
        "  region VARCHAR {east:700, west:500, ...}\n"
        "\n"
        "web ⦙ events\n"
        "  ts TIMESTAMP\n"
        "  user VARCHAR\n"
        "  kind VARCHAR\n"
        "\n"
        "web ⦙ embeddings\n"
        "  doc\n"
        "  pc1..pc12 (12 cols)\n"
        "\n"
        "Stats from a sample or engine estimates can miss rare values; confirm literal values with distinct_values before filtering on them.\n"
        "Some tables list names and types only; call load_table_schemas for their statistics."
    ),
    "compact_context_scoped_sql": (
        "shop ⦙ orders (4 rows)\n"
        "  info: One row per order placed in the web shop.\n"
        "  metadata: owner=sales, refresh=daily\n"
        "  read_with: SELECT * FROM orders\n"
        "  id INTEGER PK 10..13\n"
        "  customer_id INTEGER -> customers.id 1..3\n"
        "  amount DOUBLE 0.001234..100 unique -- Order total in EUR\n"
        "  status VARCHAR {paid:3, cancelled:1}\n"
        "\n"
        "shop ⦙ paid_orders (3 rows)\n"
        "  info: Orders that were paid.\n"
        "  derived_from: orders (step 1, latest)\n"
        "  read_with: SELECT id, amount FROM orders WHERE status = 'paid'\n"
        "  id 10..13\n"
        "  amount DOUBLE 0.001234..12.5\n"
        "\n"
        "Others available:\n"
        "- shop ⦙ customers\n"
        "- web ⦙ events\n"
        "- web ⦙ embeddings\n"
    ),
    "compact_context_docs": (
        "shop ⦙ orders (4 rows)\n"
        "  info: One row per order placed in the web shop.\n"
        "  metadata: owner=sales, refresh=daily\n"
        "  id INTEGER PK 10..13\n"
        "  customer_id INTEGER -> customers.id 1..3\n"
        "  amount DOUBLE 0.001234..100 unique -- Order total in EUR\n"
        "  status VARCHAR {paid:3, cancelled:1}\n"
        "---\n"
        "\n"
        "<documentation>\n"
        "<doc source=\"glossary.md\">\n"
        "Paid orders exclude refunds.\n"
        "</doc>\n"
        "</documentation>"
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_rendering_is_pinned(case):
    assert CASES[case](_pin_metaset()) == EXPECTED[case]
