# How to access files

:::{admonition} What does this guide solve?
:class: important
This guide will go through how to use local and remote files as sources for your dashboard and pipeline.
:::

## Local files
One of the easiest ways to get a source is by using local files, such as a CSV file.
Below is an example of how to add a local file as a source.

::::{tab-set}
:::{tab-item} Specification (YAML)
:sync: yaml
``` {code-block} yaml
:emphasize-lines: 1-5
sources:
  local_source:
    type: file
    tables:
      local_table: local_table.csv

targets:
  - title: Table
    source: local_source
    views:
      - type: table
        table: local_table
```
_The non-emphasized lines are only there to see the table when running `lumen serve file.yaml`._

:::

:::{tab-item} Pipeline (Python)
:sync: python
``` {code-block} python
from lumen.pipeline import Pipeline

data_path = "local_table.csv"
pipeline = Pipeline.from_spec(
    {
        "source": {"type": "file", "tables": {"local_table": data_path}},
    }
)
pipeline.data

```
:::
::::


## Remote files
Another way to access a file is to use an URL.


::::{tab-set}
:::{tab-item} Specification (YAML)
:sync: yaml
``` {code-block} yaml
:emphasize-lines: 1-5
sources:
  remote_source:
    type: file
    tables:
      remote_table: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv

targets:
  - title: Table
    source: remote_source
    views:
      - type: table
        table: remote_table
```
_The non-emphasized lines are only there to see the table when running `lumen serve file.yaml`._

:::

:::{tab-item} Pipeline (Python)
:sync: python
``` {code-block} python
from lumen.pipeline import Pipeline

data_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv"
pipeline = Pipeline.from_spec(
    {
        "source": {"type": "file", "tables": {"remote_table": data_url}},
    }
)
pipeline.data
```
:::
::::


## File formats
Lumen's [`FileSource`](../../reference/source/FileSource) support the following file formats: CSV, XLSX, XLS, Parquet, and JSON.
Other sources can be found [here](../../reference/source).
