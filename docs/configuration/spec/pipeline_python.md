# Build pipelines in Python

!!! important
    This guide shows how to build Lumen pipelines using Python. For more details, see the [Pipelines guide](pipelines.md).

## Overview

The main object for building dashboards in Python is the Pipeline. With it, you specify data sources, filters, and transforms. You can build pipelines either **declaratively** (like YAML) or **programmatically** (step-by-step).

## Declarative approach

Use nested Python dictionaries, similar to YAML structure:

```python
import lumen as lm

data_url = 'https://datasets.holoviz.org/penguins/v1/penguins.csv'

pipeline = lm.Pipeline.from_spec({
    'source': {
        'type': 'file',
        'tables': {
            'penguins': data_url
        }
    },
    'filters': [
        {'type': 'widget', 'field': 'species'},
        {'type': 'widget', 'field': 'island'},
        {'type': 'widget', 'field': 'sex'},
        {'type': 'widget', 'field': 'year'}
    ],
    'transforms': [
        {'type': 'aggregate', 'method': 'mean', 'by': ['species', 'sex', 'year']}
    ]
})
```

## Programmatic approach

Build pipelines step-by-step using Lumen objects.

### Add a source

Start by adding a Source to your pipeline. FileSource is common for CSV, Excel, JSON, and Parquet files:

```python
from lumen.sources import FileSource

data_url = 'https://datasets.holoviz.org/penguins/v1/penguins.csv'

pipeline = lm.Pipeline(
    source=FileSource(tables={'penguins': data_url}),
    table='penguins'
)
```

!!! note
    Preview data at any point with `pipeline.data`

```python
pipeline.data.head()
```

### Add filters

Add widgets to let dashboard users filter by specific columns:

```python
pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')
```

See the documentation for all filter options.

### Add transforms

Apply transforms to the data, like computing aggregates or selecting columns:

```python
columns = ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
pipeline.add_transform('columns', columns=columns)

pipeline.data.head()
```

See the documentation for more transform options.

!!! note
    By default, pipelines update after every interaction. Set `auto_update=False` if you want manual updates via a button.

## Display the pipeline

Render your pipeline interactively in a notebook:

```python
import panel as pn

pn.extension('tabulator')

pipeline  # renders interactively
```

For local scripts or REPLs, use:

```python
pipeline.show()
```

Display just the control panel with filters:

```python
pipeline.control_panel
```

## Related guides

- [Branch pipelines](branch_pipeline.md)
- [Build dashboards with Python](dashboard_python.md)
