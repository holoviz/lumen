# How to access files

```{admonition} What does this guide solve?
---
class: important
---
This guide shows you how to use a local or remote file as a source for your dashboard.
```

## Overview
Lumen can read from multiple [data source types](../../reference/source).
The [`FileSource`](../../reference/source/FileSource) type supports reading from either local or remote files of various formats: CSV, XLSX, XLS, Parquet, and JSON.

## Local files
One of the easiest ways to get a source is by using local files, such as a CSV file.
Below is an example of how to add a local file as a source.

::::{tab-set}

:::{tab-item} YAML
:sync: yaml

```{code-block} yaml
:emphasize-lines: 1-5
sources:
  local_source:
    type: file
    tables:
      local_table: local_table.csv

layouts:
  - title: Table
    source: local_source
    views:
      - type: table
        table: local_table
```
:::

:::{tab-item} Python
:sync: python

```{code-block} python
from lumen.pipeline import Pipeline

data_path = "local_table.csv"
pipeline = Pipeline.from_spec(
    {
        "source": {"type": "file", "tables": {"local_table": data_path}},
    }
)
pipeline.data  # preview the data in a notebook
```
:::

::::


## Remote files
Alternatively, you can access a remote file using a URL.

::::{tab-set}

:::{tab-item} YAML
:sync: yaml

```{code-block} yaml
:emphasize-lines: 1-5
sources:
  remote_source:
    type: file
    tables:
      remote_table: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv

layouts:
  - title: Table
    source: remote_source
    views:
      - type: table
        table: remote_table
```
:::

:::{tab-item} Python
:sync: python

```{code-block} python
from lumen.pipeline import Pipeline

data_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv"
pipeline = Pipeline.from_spec(
    {
        "source": {"type": "file", "tables": {"remote_table": data_url}},
    }
)
pipeline.data  # preview the data in a notebook
```
:::

::::
