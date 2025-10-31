# Sources

Connect to and query data from various sources. Lumen AI supports local files, cloud databases, and data warehouses.

## Supported sources

- **Local and remote files** — CSV, Parquet, JSON
- **DuckDB** — Local SQL queries with file support
- **Snowflake** — Cloud data warehouse
- **BigQuery** — Google's data warehouse
- **Prometheus** — Time-series metrics database
- **Intake** — Generic data catalog

!!! info
    When you use `lumen-ai serve` with file paths or URLs, Lumen automatically uses DuckDB as the backend. You don't need to explicitly configure it.

## DuckDB

Query local and remote files with SQL support.

### Local files

```python
import lumen.ai as lmai
from lumen.sources.duckdb import DuckDBSource

source = DuckDBSource(
    tables=['penguins.csv', 'earthquakes.parquet']
)
ui = lmai.ExplorerUI(source=source)
ui.servable()
```

### Remote files

```python
source = DuckDBSource(
    tables=[
        'https://datasets.holoviz.org/penguins/v1/penguins.csv',
        'https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv'
    ]
)
```

### Custom SQL expressions

```python
# Use dictionary for custom table names
source = DuckDBSource(
    tables={
        'penguins': 'penguins.csv',
        'recent_earthquakes': "read_csv('https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv')",
    }
)
```

### Non-standard file extensions

For files with non-standard extensions like `.parq`, wrap them with `read_parquet()`:

```python
source = DuckDBSource(
    tables={
        'penguins': "read_parquet('data/penguins.parq')",
        'earthquakes': 'earthquakes.csv',
    }
)
```

See [DuckDB parquet documentation](https://duckdb.org/docs/data/parquet/overview.html) for available methods.

### Initialization statements

Run SQL statements when connecting:

```python
source = DuckDBSource(
    tables=['flights.db'],
    initializers=[
        'INSTALL httpfs;',
        'LOAD httpfs;'
    ]
)
```

## Snowflake

Connect to your Snowflake data warehouse.

### Basic connection

```python
import lumen.ai as lmai
from lumen.sources.snowflake import SnowflakeSource

source = SnowflakeSource(
    account='your-account',
    user='your-username',
    authenticator='externalbrowser',
    database='your-database',
    schema='your-schema',
)
ui = lmai.ExplorerUI(source=source)
ui.servable()
```

### Authentication methods

**External browser (SSO):**

```python
source = SnowflakeSource(
    account='your-account',
    authenticator='externalbrowser',
    database='your-database',
)
```

**Username and password:**

```python
source = SnowflakeSource(
    account='your-account',
    authenticator='snowflake',
    user='your-username',
    password='your-password',
    database='your-database',
)
```

**OAuth:**

```python
source = SnowflakeSource(
    account='your-account',
    authenticator='oauth',
    token='your-oauth-token',
    database='your-database',
)
```

**Private key:**

```python
source = SnowflakeSource(
    account='your-account',
    user='your-username',
    private_key='/path/to/private/key.p8',
    database='your-database',
)
```

### Selecting tables

Specify tables as a list:

```python
source = SnowflakeSource(
    account='your-account',
    user='your-username',
    database='your-database',
    tables=['table1', 'table2'],
)
```

Or with custom names:

```python
source = SnowflakeSource(
    account='your-account',
    user='your-username',
    database='your-database',
    tables={
        'customers': 'CUSTOMERS',
        'orders': 'ORDERS',
    }
)
```

## BigQuery

Connect to your Google BigQuery project.

### Basic connection

```python
import lumen.ai as lmai
from lumen.sources.bigquery import BigQuerySource

source = BigQuerySource(
    project_id='your-project-id',
    tables=['dataset.table1', 'dataset.table2']
)
ui = lmai.ExplorerUI(source=source)
ui.servable()
```

### Authentication

BigQuery uses Google's default authentication. Set up your credentials:

```bash
gcloud auth login
gcloud auth application-default login
```

Or provide a service account JSON file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Specifying datasets and tables

By dataset:

```python
source = BigQuerySource(
    project_id='your-project-id',
    datasets=['dataset1', 'dataset2']
)
```

By table name:

```python
source = BigQuerySource(
    project_id='your-project-id',
    tables=['dataset1.table1', 'dataset2.table2']
)
```

With custom names:

```python
source = BigQuerySource(
    project_id='your-project-id',
    tables={
        'customers': 'dataset.customers_table',
        'orders': 'dataset.orders_table'
    }
)
```

## Upload data

Upload files directly in the UI without specifying sources:

```bash
lumen-ai serve
```

Click the **Manage Data** button (cloud upload icon) to upload CSV, Parquet, or JSON files.

Or drag and drop files into the upload area.
