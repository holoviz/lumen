# :material-file-upload: Loading data with sources

Sources load data into your dashboard from files, databases, or APIs.

Sources provide raw data that can be transformed with [Pipelines](pipelines.md).

## Source fundamentals

Every Lumen dashboard needs at least one source. Sources define where data comes from and how to access it.

### Basic source syntax

```yaml
sources:
  source_name:           # Choose any name
    type: source_type    # file, duckdb, intake, rest, etc.
    # Additional parameters depend on type
```

### Available source types

| Type | Purpose | Best for |
|------|---------|----------|
| `file` | CSV, Excel, Parquet, JSON files | Local or remote file data |
| `duckdb` | DuckDB SQL queries | SQL-based data access |
| `intake` | Intake catalog entries | Data catalogs |
| `rest` | REST API endpoints | API data |
| `live` | Live website status checks | Website monitoring |

This guide focuses on `file` sources (most common). See Lumen's source reference for other types.

## File sources

FileSource loads data from files in various formats:

- **CSV** - Comma-separated values
- **Excel** - XLSX and XLS files  
- **Parquet** - Columnar storage format
- **JSON** - JavaScript Object Notation

### Local files

Load files from your local filesystem:

=== "YAML"

    ```yaml
    sources:
      local_source:
        type: file
        tables:
          my_data: data/sales.csv
          
    layouts:
      - title: Sales Data
        source: local_source
        views:
          - type: table
            table: my_data
    ```

=== "Python"

    ```python
    from lumen.pipeline import Pipeline

    pipeline = Pipeline.from_spec({
        "source": {
            "type": "file",
            "tables": {"my_data": "data/sales.csv"}
        },
    })
    
    pipeline.data  # Preview in notebook
    ```

**File paths** can be relative (to your YAML file) or absolute.

### Remote files

Load files from URLs:

=== "YAML"

    ```yaml
    sources:
      remote_source:
        type: file
        tables:
          penguins: https://datasets.holoviz.org/penguins/v1/penguins.csv

    layouts:
      - title: Penguin Data
        source: remote_source
        views:
          - type: table
            table: penguins
    ```

=== "Python"

    ```python
    from lumen.pipeline import Pipeline

    data_url = "https://datasets.holoviz.org/penguins/v1/penguins.csv"
    pipeline = Pipeline.from_spec({
        "source": {
            "type": "file",
            "tables": {"penguins": data_url}
        },
    })
    
    pipeline.data
    ```

Remote files load over HTTP/HTTPS. Lumen downloads them when needed.

### Multiple tables

Sources can provide multiple tables:

```yaml
sources:
  sales_data:
    type: file
    tables:
      customers: data/customers.csv
      orders: data/orders.csv
      products: data/products.csv
```

Reference each table independently in pipelines or layouts:

```yaml
pipelines:
  customer_pipeline:
    source: sales_data
    table: customers      # Uses the customers table
    
  order_pipeline:
    source: sales_data
    table: orders         # Uses the orders table
```

### File format options

Pass format-specific options using the `kwargs` parameter:

=== "CSV with custom delimiter"

    ```yaml
    sources:
      tsv_source:
        type: file
        tables:
          data: data.tsv
        kwargs:
          sep: "\t"        # Tab-separated
          header: 0
    ```

=== "Excel with specific sheet"

    ```yaml
    sources:
      excel_source:
        type: file
        tables:
          q1_data: sales.xlsx
        kwargs:
          sheet_name: "Q1 Sales"
    ```

=== "Parquet with engine"

    ```yaml
    sources:
      parquet_source:
        type: file
        tables:
          large_data: data.parq
        kwargs:
          engine: fastparquet
    ```

These `kwargs` pass directly to pandas read functions (`read_csv`, `read_excel`, etc.).

## Caching data

Caching saves remote data locally to speed up dashboard loading. This is especially useful for large files.

### Basic caching

Enable caching with the `cache_dir` parameter:

=== "YAML"

    ```yaml
    sources:
      large_source:
        type: file
        cache_dir: cache           # Directory for cached data
        tables:
          nyc_taxi: https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq
        kwargs:
          engine: fastparquet

    layouts:
      - title: NYC Taxi Data
        source: large_source
        views:
          - type: table
            table: nyc_taxi
    ```

=== "Python"

    ```python
    from lumen.pipeline import Pipeline

    data_url = "https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq"
    pipeline = Pipeline.from_spec({
        "source": {
            "type": "file",
            "cache_dir": "cache",
            "tables": {"nyc_taxi": data_url},
            "kwargs": {"engine": "fastparquet"},
        },
    })
    
    pipeline.data
    ```

!!! note "First load timing"
    Initial load may take several minutes for large files. Subsequent loads are much faster using the cached version.

### How caching works

1. **First load**: Lumen downloads the file and saves it to `cache_dir` as Parquet
2. **Subsequent loads**: Lumen reads from the local cache instead of downloading again
3. **Cache structure**: Lumen creates the cache directory if it doesn't exist

Cache files are stored as:

- `{cache_dir}/{table_name}.parq` - Parquet data file
- `{cache_dir}/{table_name}.json` - Schema metadata

### Cache strategies

Sources support different caching strategies via `cache_per_query`:

| Strategy | Behavior | Best for |
|----------|----------|----------|
| `cache_per_query: false` | Cache entire table | Small to medium datasets |
| `cache_per_query: true` | Cache each filtered query separately | Large datasets with many filters |

Example with per-query caching:

```yaml
sources:
  large_db:
    type: duckdb
    uri: large_database.db
    cache_dir: cache
    cache_per_query: true    # Cache filtered results separately
```

This is useful when users filter data in many different ways—each unique filter combination caches separately.

### Pre-caching

Pre-populate caches with expected queries to eliminate initial loading delays.

Define pre-cache configurations in two forms:

**Form 1: Cross-product of values**

```yaml
sources:
  weather_data:
    type: file
    cache_dir: cache
    cache_per_query: true
    tables:
      weather: weather.csv
    pre_cache:
      filters:
        region: [north, south, east, west]
        year: [2020, 2021, 2022, 2023]
```

This creates 16 cache entries (4 regions × 4 years).

**Form 2: Explicit combinations**

```yaml
sources:
  weather_data:
    type: file
    cache_dir: cache
    cache_per_query: true
    tables:
      weather: weather.csv
    pre_cache:
      - filters:
          region: north
          year: 2023
      - filters:
          region: south
          year: 2023
```

This creates only 2 specific cache entries.

To populate caches, initialize a pipeline and trigger loading:

```python
from lumen.pipeline import Pipeline

pipeline = Pipeline.from_spec({
    "source": {
        "type": "file",
        "cache_dir": "cache",
        "cache_per_query": True,
        "tables": {"weather": "weather.csv"},
        "pre_cache": {
            "filters": {
                "region": ["north", "south"],
                "year": [2022, 2023]
            }
        }
    }
})

# Trigger cache population
pipeline.populate_cache()
```

## Source parameters reference

### Common parameters

These parameters work with all source types:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `type` | string | Source type (required) |
| `tables` | dict | Mapping of table names to file paths/URLs |
| `cache_dir` | string | Directory for caching data |
| `cache_per_query` | boolean | Cache strategy (table vs query) |
| `kwargs` | dict | Format-specific options |

### FileSource-specific parameters

| Parameter | Type | Purpose | Default |
|-----------|------|---------|---------|
| `tables` | dict | Table name → file path mapping | Required |
| `root` | string | Root directory for relative paths | Current directory |
| `kwargs` | dict | Pandas read function parameters | `{}` |

## Common patterns

### Simple local file

```yaml
sources:
  data:
    type: file
    tables:
      main: data.csv
```

### Multiple remote files with caching

```yaml
sources:
  datasets:
    type: file
    cache_dir: .cache
    tables:
      penguins: https://datasets.holoviz.org/penguins/v1/penguins.csv
      iris: https://datasets.holoviz.org/iris/v1/iris.csv
      stocks: https://datasets.holoviz.org/stocks/v1/stocks.csv
```

### CSV with custom parsing

```yaml
sources:
  custom_csv:
    type: file
    tables:
      data: data.csv
    kwargs:
      sep: "|"              # Pipe-separated
      parse_dates: [date]   # Parse date column
      dtype:
        id: str             # Force ID as string
        value: float
```

### Large file with aggressive caching

```yaml
sources:
  big_data:
    type: file
    cache_dir: cache
    cache_per_query: true
    tables:
      large: https://example.com/huge_dataset.parq
    kwargs:
      engine: fastparquet
    pre_cache:
      filters:
        category: [A, B, C]
        year: [2020, 2021, 2022]
```

## Next steps

Now that you can load data:

- **[Pipelines guide](pipelines.md)** - Filter and transform your data
- **[Views guide](views.md)** - Visualize your data
- **[Variables guide](variables.md)** - Make sources dynamic with variables
