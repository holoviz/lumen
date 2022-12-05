# How to use cache

```{admonition} What does this guide solve?
---
class: important
---
This guide will show you how to locally cache data to speed up reloading from a remote source.
```

## Caching file

When working with large data in a non-optional file format, creating a local cache can be advantageous to save time when reloading the data. Caching a [source](../../reference/source) is done by using `cache_dir` followed by a directory. If the directory does not exist, it will be created. The data will be saved as [parquet](https://www.databricks.com/glossary/what-is-parquet) files in the cache directory and the schema will be stored as a JSON file.

Below is an example of caching a 370 MB file that consists of almost 12 million rows.

```{warning}
The initial load can take a couple of minutes as the file needs to be downloaded and cached first.
```

::::{tab-set}

:::{tab-item} YAML
:sync: yaml

```{code-block} yaml
:emphasize-lines: 4
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

:::

:::{tab-item} Python
:sync: python

```{code-block} python
---
emphasize-lines: 8
---
from lumen.pipeline import Pipeline

data_url = "https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq"
pipeline = Pipeline.from_spec(
    {
        "source": {
            "type": "file",
            "cache_dir": "cache",
            "tables": {"large_table": data_url},
            "kwargs": {"engine": "fastparquet"},
        },
    }
)
pipeline.data
```
:::

::::

Depending on the source type data caching will cache the entire table or individual queries. Using the `cache_per_query` option you can toggle this behavior.

```{note}
Lumen's [cache]() can be added to all [source types](../../reference/source).
```

## Precaching

Sources that support caching per query can be made to pre-cache specific Filter and SQLTransform combinations. To enable pre-caching you must initialize a `Pipeline` and then either programmatically request to populate the pre-cache OR supply the pre-cache configuration as part of the YAML specification.

A pre-cache definitions can take one of two forms

- A dictionary containing 'filters' and 'variables' dictionaries each containing lists of values to compute a cross-product for, e.g.

```python
{
    'filters': {
        <filter>': ['a', 'b', 'c', ...],
        ...
    },
    'variables':
        <variable>: [0, 2, 4, ...],
        ...
    }
}
```

- A list containing dictionaries of explicit values for each filter and variables.

```python
[
    {
        'filters': {<filter>: 'a'},
        'variables': {<variable>: 0}
    },
    {
        'filters': {<filter>: 'a'},
        'variables': {<variable>: 1}
    },
    ...
]
```
