# Data Sources

**Connect Lumen to files, databases, or data warehouses.**

Most users start with files. Advanced users connect to databases.

## Quick start

### Load a file

```bash
lumen-ai serve penguins.csv
```

Works with CSV, Parquet, JSON, and URLs. No configuration needed.

### Load multiple files

```bash
lumen-ai serve penguins.csv earthquakes.parquet
```

Or in Python:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI(data=['penguins.csv', 'earthquakes.parquet'])
ui.servable()
```

### Upload in the UI

Start without data:

```bash
lumen-ai serve
```

Click **Manage Data** (cloud icon) to upload CSV, Parquet, or JSON files.

## Supported sources

| Source | Use for |
|--------|---------|
| **Files** | CSV, Parquet, JSON (local or URL) |
| **DuckDB** | Local SQL queries on files |
| **Snowflake** | Cloud data warehouse |
| **BigQuery** | Google's data warehouse |
| **PostgreSQL** | PostgreSQL databases |
| **Intake** | Data catalogs |

## Connect to databases

### Snowflake

```python
from lumen.sources.duckdb import DuckDBSource
from lumen.sources.snowflake import SnowflakeSource
import lumen.ai as lmai

source = SnowflakeSource(
    account='your-account',
    user='your-username',
    authenticator='externalbrowser',  # SSO login
    database='your-database',
    schema='your-schema',
)

ui = lmai.ExplorerUI(source=source)
ui.servable()
```

**Authentication options:**

- `authenticator='externalbrowser'` - SSO (recommended)
- `authenticator='snowflake'` - Username/password (needs `password=`)
- `authenticator='oauth'` - OAuth token (needs `token=`)
- Private key authentication (needs `private_key=`)

**Select specific tables:**

```python
source = SnowflakeSource(
    account='your-account',
    database='your-database',
    tables=['CUSTOMERS', 'ORDERS']  # Only these tables
)
```

### BigQuery

```python
from lumen.sources.bigquery import BigQuerySource

source = BigQuerySource(
    project_id='your-project-id',
    tables=['dataset.table1', 'dataset.table2']
)

ui = lmai.ExplorerUI(source=source)
ui.servable()
```

**Authentication:**

```bash
gcloud auth application-default login
```

Or use a service account:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

**Select all tables in a dataset:**

```python
source = BigQuerySource(
    project_id='your-project-id',
    datasets=['dataset1']  # All tables in dataset1
)
```

### PostgreSQL

```python
from lumen.sources.sql import PostgresSource

source = PostgresSource(
    connection_string='postgresql://user:password@localhost:5432/database'
)
```

## Advanced file configuration

### DuckDB for complex file queries

DuckDB lets you run SQL on files directly:

```python
from lumen.sources.duckdb import DuckDBSource

source = DuckDBSource(
    tables={
        'penguins': 'penguins.csv',
        'earthquakes': "read_csv('https://earthquake.usgs.gov/query?format=csv')",
    }
)
```

**Load remote files:**

```python
source = DuckDBSource(
    tables=['https://datasets.holoviz.org/penguins/v1/penguins.csv'],
    initializers=[
        'INSTALL httpfs;',
        'LOAD httpfs;'
    ]
)
```

**Non-standard extensions:**

```python
source = DuckDBSource(
    tables={
        'data': "read_parquet('file.parq')",  # Wrap non-standard extensions
    }
)
```

### Multiple sources

Load data from different sources:

```python
snowflake = SnowflakeSource(account='...', database='...')
bigquery = BigQuerySource(project_id='...')

ui = lmai.ExplorerUI(data=[snowflake, bigquery])
ui.servable()
```

Lumen can query across all sources.

## Common patterns

### Mix files and databases

```python
from lumen.sources.duckdb import DuckDBSource
from lumen.sources.snowflake import SnowflakeSource

local_data = DuckDBSource(tables=['local.csv'])
cloud_data = SnowflakeSource(account='...', database='...')

ui = lmai.ExplorerUI(data=[local_data, cloud_data])
ui.servable()
```

### Custom table names

```python
source = DuckDBSource(
    tables={
        'customers': 'customer_data.csv',
        'orders': 'order_history.parquet',
    }
)
```

Now ask questions using "customers" and "orders" instead of file names.

### Initialize databases

Run setup SQL when connecting:

```python
source = DuckDBSource(
    tables=['data.csv'],
    initializers=[
        'INSTALL httpfs;',
        'LOAD httpfs;',
        'SET memory_limit="8GB";'
    ]
)
```

## Troubleshooting

**"Table not found"**: Check table names match exactly (case-sensitive for some databases).

**"Connection failed"**: Verify credentials and network access to the database.

**"File not found"**: Use absolute paths or URLs. Relative paths are relative to where you run `lumen-ai serve`.

**Slow queries**: DuckDB is fast for files. If queries are slow, the database connection or query is the bottleneck, not Lumen.

## Best practices

- **Start with files** for testing and development
- **Use URLs** for shared datasets that don't change often
- **Connect databases** for production data that updates frequently
- **Limit tables** when possible (faster planning and lower costs)
