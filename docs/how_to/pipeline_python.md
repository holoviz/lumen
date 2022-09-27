# Build a dashboard in Python

:::{admonition} What does this guide solve?
:class: important
Although the primary interface for building a Lumen dashboard is the `YAML specification file`, this guide shows you an alternate approaches for building with Python. To learn more, visit the [Lumen in Python](../conceptual/lumen_python) Conceptual Guide.
:::


## Overview
When building with Lumen in Python, the main object that defines a dashboard is the `Pipeline`. With this Pipeline object, you can specify the data source, filters, and transforms. There are two approaches to add these specifications to a `Pipeline` object, **declaratively or programmatically**. While the declarative approach is more compact, the programmatic approach allows you to seperate the pipeline creation steps.

## Declaratively specifying a pipeline

The declarative specification approach looks similar to a YAML file hierarchy, but consists of nested Python dictionary and list objects.

```{code-block} python
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
```

## Programmatically specifying a pipeline

The programmatic specification approach uses Lumen objects to build the pipeline step by step.

### Add source
First, add a valid `Source` to your `Pipeline`. A common choice is `FileSource`, which can load CSV, Excel, JSON and Parquet files, but see the [Source Reference](../architecture//source.html#:~:text=Source%20queries%20data.-,Source%20types%23,-class%20lumen.sources) for all options.

```{code-block} python
from lumen.pipeline import Pipeline
from lumen.sources import FileSource

data_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv'

pipeline = Pipeline(source=FileSource(tables={'penguins': data_url}), table='penguins')
```

:::{admonition} Preview the data
:class: note
At any point after defining the source in your pipeline, you can inspect the data in a notebook with `pipeline.data`
:::
:::{dropdown} `pipeline.data`
:animate: fade-in-slide-down
![data preview](../_static/pipeline_data.png)
:::

### Add filter
Next, you can add `widgets` for certain columns of your source. When displaying the dashboard, these widgets will allows your dashboard users to `filter` the data. See the [Filter Reference](../architecture/filter) for all options.

```{code-block} python
:emphasize-lines: 9-12
from lumen.pipeline import Pipeline
from lumen.sources import FileSource

data_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv'

pipeline = Pipeline(source=FileSource(tables={'penguins': data_url}), table='penguins')

# Filters
pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')
```

### Add transform
Now you can apply a `transform` to the data, such as computing the mean or selecting certain columns. See the [Transform Reference](../architecture/transform) for more.

```{code-block} python
:emphasize-lines: 14-15
from lumen.pipeline import Pipeline
from lumen.sources import FileSource

data_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv'

pipeline = Pipeline(source=FileSource(tables={'penguins': data_url}), table='penguins')

# Filters
pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')

columns=['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
pipeline.add_transform('columns', columns=columns)

```
:::{dropdown} `pipeline.data`
:animate: fade-in-slide-down
![transform data preview](../_static/pipeline_transform.png)
:::

### Display the dashboard

Once your pipeline has been specified, you can use the `Panel` package to define the layout of your widgets and views, and then you can serve your dashboard within a notebook or from a Python script. See Panel's [Documentation](https://panel.holoviz.org/getting_started/index) to learn more about Panel components and deployment.

The simplest approach is to render the widgets with the `pipeline.control_panel` property alongside Lumen view components (e.g. `Table`, `hvPlotView`).

```{code-block} python
:emphasize-lines: 17-27
from lumen.pipeline import Pipeline
from lumen.sources import FileSource

data_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv'

pipeline = Pipeline(source=FileSource(tables={'penguins': data_url}), table='penguins')

# Filters
pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')

columns=['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
pipeline.add_transform('columns', columns=columns)

from lumen.views import Table, hvPlotView
import panel as pn
pn.extension('tabulator', template='fast')

pn.Row(
    pipeline.control_panel,
    pn.Tabs(
        ('Plot', hvPlotView(pipeline=pipeline, kind='scatter', x='bill_length_mm', y='bill_depth_mm', color='species')),
        ('Table', Table(pipeline=pipeline))
    )
)
```
![dashboard preview](../_static/pipeline_dash.png)

```{note} If querying data from the Source takes time use set `auto_update=False` on the Pipeline. This will require you to manually trigger an update by clicking a button.
```

Related Resources:
* [Branch a pipeline in Python](chain_python)
