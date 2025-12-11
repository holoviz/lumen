# Building data pipelines

Lumen provides powerful abstractions that work independently of a full dashboard specification. The `Pipeline` component is particularly useful for building data transformations that can drive both analysis pipelines and visual components in notebooks or custom Panel dashboards.

## How pipelines work

Understanding pipeline execution order is essential. Pipelines distinguish between:

- Operations applied by the **Source** (via filter state and SQL transforms)
- Operations applied to **data** returned by the Source

This distinction matters because various source types support data queries. For SQL-based sources, transforms can be optimized and executed at the database level. After the source returns data as a DataFrame, declared transforms are applied in sequence.

```
┌─────────────────────────────────────────────────────┐
│ Source.get() called with Filter/SQLTransform state  │
│ (Optimized query and transforms at source level)    │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ DataFrame returned  │
     └──────────┬──────────┘
                │
                ▼
     ┌─────────────────────┐
     │ Transform stages    │
     │ applied in sequence │
     └─────────────────────┘
```

## Declaring a pipeline

Like other Lumen components, pipelines can be defined declaratively. Here's an example using the Palmer Penguins dataset:

```python
from lumen.pipeline import Pipeline

data_url = 'https://datasets.holoviz.org/penguins/v1/penguins.csv'

pipeline = Pipeline.from_spec({
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

View the current data:

```python
pipeline.data
```

Pipelines update dynamically when sources, filters, or transforms change. Access the control panel to interact with widgets:

```python
from lumen.views import Table
import panel as pn

pn.Row(pipeline.control_panel, Table(pipeline=pipeline, pagination='remote'))
```

You can also bind data to Panel components:

```python
pn.Row(
    pipeline.control_panel,
    pn.pane.DataFrame(pipeline.param.data, width=800)
)
```

## Building pipelines programmatically

Create the same pipeline without using declarative specifications:

```python
from lumen.sources import FileSource

source = FileSource(tables={'penguins': data_url})
pipeline = Pipeline(source=source, table='penguins')

pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')
```

## Automatic filters

Generate filters automatically for all available columns:

```python
Pipeline(source=source, table='penguins', filters='auto')
```

## Chaining pipelines

Build branching pipelines by chaining them together. This is useful when you need one stage to filter data and another to aggregate it:

```python
from lumen.transforms import Aggregate

agg_pipeline = pipeline.chain(
    transforms=[Aggregate(method='mean', by=['species', 'year'])]
)

agg_pipeline.data
```

Chaining allows pipelines to share computations—the filtering still occurs in the first stage.

## Building custom dashboards

One of the major benefits of pipelines is that they work outside the context of a Lumen YAML specification. This lets you build custom Panel dashboards while leveraging Lumen's power:

```python
from lumen.views import hvPlotUIView

pn.Row(
    pipeline.control_panel.servable(area='sidebar'),
    pn.Tabs(
        ('Plot', hvPlotUIView(
            pipeline=pipeline,
            kind='scatter',
            x='bill_length_mm',
            y='bill_depth_mm',
            by='species'
        )),
        ('Table', Table(pipeline=agg_pipeline))
    ).servable()
)
```

This approach combines Lumen's data transformation capabilities with Panel's flexible layout system.
