# Branch pipelines

!!! important
    This guide shows how to build branching pipelines, allowing views of the same source data at different processing stages.

## Overview

Branching lets you create multiple processing paths from a single source, viewing the data at different stages of transformation.

## Branching in YAML

The key to branching in YAML is using the `pipeline:` parameter to reference another pipeline.

### Create a simple dashboard

Start with a basic dashboard without branching:

```yaml
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

pipelines:
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    filters:
      - type: widget
        field: island

layouts:
  - title: Penguins
    sizing_mode: stretch_width
    views:
      - type: table
        pipeline: penguin_pipeline
        show_index: false
        height: 300
```

Serve with:

```bash
lumen serve penguins.yaml --show --autoreload
```

### Add a branch

Now create a second pipeline that branches from the first. Use the `pipeline:` parameter to reference the original pipeline:

```yaml
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

pipelines:
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    filters:
      - type: widget
        field: island
  branch_sort:
    pipeline: penguin_pipeline
    transforms:
      - type: columns
        columns: ['species', 'island', 'bill_length_mm', 'bill_depth_mm']

layouts:
  - title: Penguins
    sizing_mode: stretch_width
    views:
      - type: table
        pipeline: penguin_pipeline
        show_index: false
        height: 300
      - type: table
        pipeline: branch_sort
        show_index: false
        height: 300
```

The `branch_sort` pipeline inherits all filters and transformations from `penguin_pipeline`, then applies its own column selection.

## Branching in Python

Use the `pipeline.chain()` method to create branches in Python.

### Create initial pipeline

Start with a basic pipeline:

```python
from lumen.pipeline import Pipeline
from lumen.sources import FileSource

data_url = 'https://datasets.holoviz.org/penguins/v1/penguins.csv'

pipeline = Pipeline(
    source=FileSource(tables={'penguins': data_url}),
    table='penguins'
)

pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')

pipeline.data
```

### Branch the pipeline

Use `chain()` to create a branch with additional transforms. The branch inherits all filters and transforms from the original:

```python
from lumen.transforms import Aggregate

agg_pipeline = pipeline.chain(
    transforms=[Aggregate(method='mean', by=['species', 'year'])]
)

agg_pipeline.data
```

The `agg_pipeline` now contains both the original filters and the new aggregate transform, while the original `pipeline` remains unchanged.
