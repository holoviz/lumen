# Define views

!!! important
    This guide shows how to define views on your dashboard.

## Overview

Views are the final output of your dashboard. To create a view, you need at least one data source.

Lumen includes many view types built on the [Holoviz ecosystem](https://holoviz.org/). You can use components like [hvPlot scatter plots](https://hvplot.holoviz.org/reference/pandas/scatter.html) or [Panel indicators](https://panel.holoviz.org/reference/index.html#indicators).

Here's an example showing two views of the same data—a scatter plot and a table:

=== "YAML"

    ```yaml
    sources:
      penguin_source:
        type: file
        tables:
          penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

    layouts:
      - title: Table
        source: penguin_source
        views:
          - type: hvplot
            table: penguin_table
            kind: scatter
            color: species
          - type: table
            table: penguin_table
    ```

=== "Python"

    ```python
    import panel as pn
    from lumen.pipeline import Pipeline
    from lumen.views import Table, hvPlotView

    pn.extension("tabulator")

    data_url = "https://datasets.holoviz.org/penguins/v1/penguins.csv"
    pipeline = Pipeline.from_spec({
        "source": {"type": "file", "tables": {"penguin_table": data_url}},
    })

    pn.Column(hvPlotView(pipeline=pipeline), Table(pipeline=pipeline))
    ```

To arrange views in Python, use [Panel](https://panel.holoviz.org/) as shown above.
