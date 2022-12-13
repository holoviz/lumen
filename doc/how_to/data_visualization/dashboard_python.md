# Building a dashboard in Python

In a previous guide we discovered how we could [build data pipelines in Python](../data_processing/pipeline_python), here we will pick up where we left of and build an entire dashboard in Python.

To start with let us declare the Pipeline we will be working with again and initialize the Panel extension so we can render output and tables.

```{pyodide}
import lumen as lm
import panel as pn

pn.extension('tabulator')

pipeline = lm.Pipeline.from_spec({
    'source': {
        'type': 'file',
        'tables': {
            'penguins': 'https://datasets.holoviz.org/penguins/v1/penguins.csv'
        }
    },
    'filters': [
        {'type': 'widget', 'field': 'island'},
        {'type': 'widget', 'field': 'sex'},
        {'type': 'widget', 'field': 'year'}
    ],
    'auto_update': False
})
```

## Attaching Views

Attaching a [`View`](../../reference/view/) to a [Pipeline](../../reference/pipeline/) only requires passing the `pipeline` as an argument to the `View` constructor. The `View` will now be linked to the pipeline and update when we change it:

```{pyodide}
scatter = lm.views.hvPlotView(
    pipeline=pipeline, kind='scatter', x='bill_length_mm', y='bill_depth_mm', by='species',
    height=300, responsive=True
)

scatter
```

Now let us create one more view, a `Table`:

```{pyodide}
table = lm.views.Table(pipeline=pipeline, page_size=10, sizing_mode='stretch_width')

table
```

## Laying out views

Now we could lay these components out using Panel and publish a Panel dashboard but for now we will stick entirely with Lumen components. The Lumen `Layout` component will let us arrange one or more views. Here we will take our two views and put the in layout:

```{pyodide}
layout = lm.Layout(views={'scatter': scatter, 'table': table}, title='Palmer Penguins')

layout
```

## Building the dashboard

```{pyodide}
lm.Dashboard(layouts=[layout], config={'title': 'Palmer Penguins'})
```
