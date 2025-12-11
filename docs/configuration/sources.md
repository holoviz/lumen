# :material-database: Data Sources

Connect Lumen to files, databases, or data warehouses.

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
| Snowflake | Cloud data warehouse |
| BigQuery | Google's data warehouse |
| PostgreSQL | PostgreSQL via SQLAlchemy |
| MySQL | MySQL via SQLAlchemy |
| SQLite | SQLite via SQLAlchemy |
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
