# Access files

!!! important
    This guide shows how to use local or remote files as a data source for your dashboard.

## Overview

Lumen reads from multiple data source types. The FileSource type supports local or remote files in various formats: CSV, XLSX, XLS, Parquet, and JSON.

## Local files

Use local files such as CSV by specifying the file path:

=== "YAML"

    ```yaml
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

=== "Python"

    ```python
    from lumen.pipeline import Pipeline

    data_path = "local_table.csv"
    pipeline = Pipeline.from_spec({
        "source": {
            "type": "file",
            "tables": {"local_table": data_path}
        },
    })
    pipeline.data  # preview in notebook
    ```

## Remote files

Access files from remote URLs:

=== "YAML"

    ```yaml
    sources:
      remote_source:
        type: file
        tables:
          remote_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

    layouts:
      - title: Table
        source: remote_source
        views:
          - type: table
            table: remote_table
    ```

=== "Python"

    ```python
    from lumen.pipeline import Pipeline

    data_url = "https://datasets.holoviz.org/penguins/v1/penguins.csv"
    pipeline = Pipeline.from_spec({
        "source": {
            "type": "file",
            "tables": {"remote_table": data_url}
        },
    })
    pipeline.data  # preview in notebook
    ```
