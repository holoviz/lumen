# How to use cache

:::{admonition} What does this guide solve?
:class: important
This guide will go through how to cache large files for easier access.
:::

## Caching file
When working with large files, it can be very advantageous to cache it and therefore be able to save time when reloading the data.
To cache a [source](../../reference/source) is done by using `cache_dir` followed by a directory.
If the directory does not exist, it will be created.
Below is an example of caching a 370 MB file, and the table consists of almost 12 million rows.

:::{warning}
The initial load can take a couple of minutes as the file needs to be downloaded and cached first.
:::

::::{tab-set}
:::{tab-item} Specification (YAML)
:sync: yaml
``` {code-block} yaml
:emphasize-lines: 4
sources:
  large_source:
    type: file
    cache_dir: cache
    tables:
      large_table: https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq
    kwargs:
      engine: fastparquet

targets:
  - title: Table
    source: large_source
    views:
      - type: table
        table: large_table
```
_The non-emphasized lines are only there to see the table when running `lumen serve file.yaml`._

:::

:::{tab-item} Pipeline (Python)
:sync: python
``` {code-block} python
:emphasize-lines: 8
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
_The non-emphasized lines are only there to have a table to cache when running the code_
:::
::::


:::{note}
Lumen's [cache]() can be added to all sources found [here](../../reference/source).
:::
