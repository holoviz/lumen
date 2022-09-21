# Build a dashboard in Python

:::{admonition} What does this guide solve?
:class: important
Although the primary interface for building a Lumen dashboard is the `YAML specification file`, this guide shows you an alternate approaches for building with Python. To learn more, visit the [Lumen in Python](../conceptual/lumen_python.md) Conceptual Guide.
:::


## Overview
When building with Lumen in Python, the main object that defines a dashboard is the `Pipeline`. With this Pipeline object, you can specify the data source, filters, and transforms. There are two approaches to add these specifications to a `Pipeline` object, declaratively or programmatically.

## Declaratively specifying a pipeline

The declarative specification approach looks similar to a YAML file hierarchy, but consists of nested Python dictionary and list objects.

```python
import lumen
from lumen.pipeline import Pipeline

data_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv'

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

pipeline.data # inspect the data
```


## Programmatically specifying a pipeline

The programmatic specification approach uses Lumen objects to build the pipeline step by step.

```python
import lumen
from lumen.pipeline import Pipeline
from lumen.sources import FileSource
from lumen.transforms import Aggregate

pipeline = Pipeline(source=FileSource(tables={'penguins': data_url}), table='penguins')

# Filters
pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')

agg_pipeline = pipeline.chain(transforms=[Aggregate(method='mean', by=['species', 'year'])])

agg_pipeline.data # inspect the data
```
:::{admonition} Chaining
:class: note
To learn more about `pipeline.chain`, visit the [Lumen in Python](../conceptual/lumen_python.md) Conceptual Guide.
:::

## Displaying the dashboard

Use the Panel package to specify the layout and then serve your dashboard within a notebook or from a Python script. See Panel's [Documentation](https://panel.holoviz.org/getting_started/index.html) to learn more about Panel components and deployment.

```python
from lumen.views import Table, hvPlotUIView

pn.Row(
    pipeline.control_panel.servable(area='sidebar'),
    pn.Tabs(
        ('Plot', hvPlotUIView(pipeline=pipeline, kind='scatter', x='bill_length_mm', y='bill_depth_mm', by='species')),
        ('Table', Table(pipeline=agg_pipeline))
    ).servable()
)
```

:::{admonition} Different approaches with Panel 
:class: note
To learn more about different Panel approaches to viewing a Lumen dashboard, visit the [Lumen in Python](../conceptual/lumen_python.md) Conceptual Guide.
:::

## 






