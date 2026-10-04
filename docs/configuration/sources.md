# :material-database: Data Sources

Connect Lumen to files, databases, or data warehouses.

**See also:** [Navigating the UI](../getting_started/navigating_the_ui.md) — Learn how to manage data sources from the interface.

## Quick start

``` bash title="Load a file"
lumen-ai serve penguins.csv
```

Works with CSV, Parquet, JSON, and URLs.

``` bash title="Multiple files"
lumen-ai serve penguins.csv earthquakes.parquet
```

``` py title="In Python"
import lumen.ai as lmai

ui = lmai.ExplorerUI(data=['penguins.csv', 'earthquakes.parquet'])
ui.servable()
```

## Supported sources

| Source | Use for |
|--------|---------|
| Files | CSV, Parquet, JSON (local or URL) |
| DuckDB | Local SQL queries on files |
| xarray | N-dimensional scientific data (NetCDF, Zarr, HDF5) |
| Snowflake | Cloud data warehouse |
| BigQuery | Google's data warehouse |
| PostgreSQL | PostgreSQL via SQLAlchemy |
| MySQL | MySQL via SQLAlchemy |
| SQLite | SQLite via SQLAlchemy |
| Oracle | Oracle via SQLAlchemy |
| MSSQL | Microsoft SQL Server via SQLAlchemy |
| Intake | Data catalogs |

## Database connections

### Snowflake

``` py title="Snowflake with SSO"
from lumen.sources.snowflake import SnowflakeSource
import lumen.ai as lmai

source = SnowflakeSource(
    account='your-account',
    database='your-database',
    authenticator='externalbrowser',  # SSO
)

ui = lmai.ExplorerUI(data=source)
ui.servable()
```

**Authentication options:**

- `authenticator='externalbrowser'` - SSO (recommended)
- `authenticator='snowflake'` - Username/password (needs `password=`)
- `authenticator='oauth'` - OAuth token (needs `token=`)

**Select specific tables:**

``` py hl_lines="4"
source = SnowflakeSource(
    account='your-account',
    database='your-database',
    tables=['CUSTOMERS', 'ORDERS']
)
```

### BigQuery

``` py title="BigQuery connection"
from lumen.sources.bigquery import BigQuerySource

source = BigQuerySource(
    project_id='your-project-id',
    tables=['dataset.table1', 'dataset.table2']
)

ui = lmai.ExplorerUI(data=source)
ui.servable()
```

**Authentication:**

``` bash
gcloud auth application-default login
```

Or set service account:

``` bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### PostgreSQL

``` py title="PostgreSQL via SQLAlchemy"
from lumen.sources.sqlalchemy import SQLAlchemySource

source = SQLAlchemySource(
    url='postgresql://user:password@localhost:5432/database'
)

ui = lmai.ExplorerUI(data=source)
ui.servable()
```

Or use individual parameters:

``` py
source = SQLAlchemySource(
    drivername='postgresql+psycopg2',
    username='user',
    password='password',
    host='localhost',
    port=5432,
    database='mydb'
)
```

### MySQL

``` py title="MySQL connection"
from lumen.sources.sqlalchemy import SQLAlchemySource

source = SQLAlchemySource(
    url='mysql+pymysql://user:password@localhost:3306/database'
)
```

### SQLite

``` py title="SQLite file"
from lumen.sources.sqlalchemy import SQLAlchemySource

source = SQLAlchemySource(url='sqlite:///data.db')
```

## Advanced file handling

### DuckDB for SQL on files

Run SQL directly on CSV/Parquet files:

``` py title="SQL on files"
from lumen.sources.duckdb import DuckDBSource

source = DuckDBSource(
    tables={
        'penguins': 'penguins.csv',
        'quakes': "read_csv('https://earthquake.usgs.gov/data.csv')",
    }
)
```

**Load remote files:**

``` py title="Remote files with DuckDB" hl_lines="3-6"
source = DuckDBSource(
    tables=['https://datasets.holoviz.org/penguins/v1/penguins.csv'],
    initializers=[
        'INSTALL httpfs;',
        'LOAD httpfs;'
    ]  # (1)!
)
```

1. Required for HTTP/S3 access

### xarray for scientific data

Query N-dimensional scientific datasets (NetCDF, Zarr, HDF5) with SQL via Apache DataFusion:

``` bash title="Install dependencies"
pip install lumen[xarray]
```

``` py title="Query a NetCDF file"
from lumen.sources.xarray_sql import XArraySQLSource
import lumen.ai as lmai

source = XArraySQLSource(uri='air_temperature.nc')

ui = lmai.ExplorerUI(data=source)
ui.servable()
```

Each data variable in the dataset becomes a separate SQL table with coordinate columns:

``` py title="Direct SQL queries"
source = XArraySQLSource(uri='air_temperature.nc')
source.get_tables()  # ['air']

df = source.execute(
    "SELECT lat, lon, AVG(air) as avg_temp "
    "FROM air GROUP BY lat, lon"
)
```

**Select specific variables:**

``` py hl_lines="3"
source = XArraySQLSource(
    uri='climate_data.nc',
    variables=['temperature', 'pressure']
)
```

### Multiple sources

``` py title="Mix sources"
from lumen.sources.snowflake import SnowflakeSource
from lumen.sources.duckdb import DuckDBSource

snowflake = SnowflakeSource(account='...', database='...')
local = DuckDBSource(tables=['local.csv'])

ui = lmai.ExplorerUI(data=[snowflake, local])
ui.servable()
```

### Custom table names

``` py title="Rename tables" hl_lines="3-4"
source = DuckDBSource(
    tables={
        'customers': 'customer_data.csv',  # (1)!
        'orders': 'order_history.parquet',
    }
)
```

1. Use 'customers' instead of 'customer_data.csv' in queries

## Table statistics

Lumen AI describes each table to the model with SQL types, keys, value ranges, null shares and frequent values. It computes these per table, starting in the background when a source is first indexed, using the cheapest accurate method the engine offers:

- Engine metadata where it is free: Postgres `pg_stats`, Parquet footers, Snowflake metadata-served aggregates, Postgres and SQLite row estimates.
- One exact aggregate pass for tables up to 1M rows.
- A seeded engine-side sample of 10,000 rows above that.

The prompt says which method was used, for example `(1056320 rows; stats from a 10000-row sample)`, or `stats from the first 10000 rows` where the engine cannot sample randomly. On BigQuery, where `LIMIT` does not reduce the bytes billed, tables defined by a SQL expression are not sampled at all, and each statistics query is capped at 10 GB billed.

Each database may spend 300 seconds per hour on statistics, and a single query may run for 30 seconds before it is cancelled. Tables profiled after the budget is spent show names and types only and are retried later. `SQLAgent` waits up to `stats_wait` seconds (default 10) for statistics before answering, and the schema block is capped at `schema_max_tokens` (default 4000) across at most `schema_tables_shown` tables (default 25):

``` py title="Tune statistics in the SQL prompt"
from lumen.ai.agents import SQLAgent

agent = SQLAgent(stats_wait=30, schema_max_tokens=6000, schema_tables_shown=40)
```

### Statistics cache

Statistics are cached in memory and, for databases with a stable identity (database files, SQLAlchemy URLs, Snowflake and BigQuery accounts), on disk under the user cache directory, for example `~/Library/Caches/lumen/table_stats` on macOS. A cached entry is discarded when the table's columns change or its modification token changes (file modification time, BigQuery `modified`, DuckDB catalog entry), and after seven days on engines that expose no such token.

!!! warning "The cache contains data values"

    Cached entries include value ranges, frequent values and example strings from your tables. On Linux and macOS the files are created readable by the current user only; on Windows the default location is inside the user profile, which only that user can read. Snowflake entries are keyed by user and role so stats computed under one role are not served to another. To keep statistics in memory only, set `LUMEN_TABLE_STATS_CACHE` to an empty string; set it to a path to move the cache.

``` bash title="Disable the on-disk cache"
export LUMEN_TABLE_STATS_CACHE=""
```

## Troubleshooting

**"Table not found"** - Table names are case-sensitive. Check exact names.

**"Connection failed"** - Verify credentials and network access.

**"File not found"** - Use absolute paths or URLs. Relative paths are relative to where you run the command.

**Slow queries** - If using DuckDB on files, it's fast. Slowness usually comes from the database or network, not Lumen.

## Best practices

**Start with files** for development. Move to databases for production.

**Use URLs** for shared datasets that don't change often.

**Limit tables** when possible - faster planning and lower LLM costs.

**Name tables clearly** - Use meaningful names instead of generic file names.
