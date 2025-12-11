# Cache data

!!! important
    This guide shows how to locally cache remote data to speed up dashboard reloading.

## Caching files

For large remote files, creating a local cache saves time during reloading. Use the `cache_dir` parameter to specify a cache directory. Lumen creates the directory if needed and stores data as Parquet files with JSON schema files.

Here's an example caching a 370 MB file with nearly 12 million rows:

!!! warning
    Initial load may take several minutes as the file is downloaded and cached.

=== "YAML"

    ```yaml
    sources:
      large_source:
        type: file
        cache_dir: cache
        tables:
          large_table: https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq
        kwargs:
          engine: fastparquet

    layouts:
      - title: Table
        source: large_source
        views:
          - type: table
            table: large_table
    ```

=== "Python"

    ```python
    from lumen.pipeline import Pipeline

    data_url = "https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq"
    pipeline = Pipeline.from_spec({
        "source": {
            "type": "file",
            "cache_dir": "cache",
            "tables": {"large_table": data_url},
            "kwargs": {"engine": "fastparquet"},
        },
    })
    pipeline.data
    ```

Caching behavior varies by source type. Use the `cache_per_query` option to control whether entire tables or individual queries are cached.

!!! note
    Caching is available for all source types.

## Pre-caching

Sources that support per-query caching can pre-cache specific filter and transform combinations. Enable pre-caching by initializing a pipeline and either programmatically populating the cache or providing pre-cache configuration in your YAML.

Pre-cache definitions take one of two forms:

**Form 1: Cross-product of values**

```python
{
    'filters': {
        'filter_name': ['a', 'b', 'c'],
    },
    'variables': {
        'variable_name': [0, 2, 4],
    }
}
```

**Form 2: Explicit value combinations**

```python
[
    {
        'filters': {'filter_name': 'a'},
        'variables': {'variable_name': 0}
    },
    {
        'filters': {'filter_name': 'b'},
        'variables': {'variable_name': 2}
    },
]
```
