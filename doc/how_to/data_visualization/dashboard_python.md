# Building a dashboard in Python

In a previous guide we discovered how we could [build data pipelines in Python](../data_processing/pipeline_python), here we will pick up where we left of and build an entire dashboard in Python.

To start with let us declare the [Pipeline](lumen.pipeline.Pipeline) we will be working with again and initialize the Panel extension so we can render output and tables.

```{pyodide}
import lumen as lm
import panel as pn

pn.extension('tabulator', design='material')

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

pipeline
```

## Attaching Views

Attaching a [View](lumen.views.base.View) to a [Pipeline](lumen.pipeline.Pipeline) only requires passing the `pipeline` as an argument to the [View](lumen.views.base.View) constructor. The [View](lumen.views.base.View) will now be linked to the pipeline and update when we change it:

```{pyodide}
scatter = lm.views.hvPlotView(
    pipeline=pipeline, kind='scatter', x='bill_length_mm', y='bill_depth_mm', by='species',
    height=300, responsive=True
)

scatter
```

Now let us create one more view, a [Table](lumen.views.base.Table):

```{pyodide}
table = lm.views.Table(pipeline=pipeline, page_size=10, sizing_mode='stretch_width')

table
```

## Laying out views

Now we could lay these components out using Panel and publish a Panel dashboard but for now we will stick entirely with Lumen components. The Lumen [Layout](lumen.layout.Layout) component will let us arrange one or more views. Here we will take our two views and put the in layout:

```{pyodide}
layout = lm.Layout(views={'scatter': scatter, 'table': table}, title='Palmer Penguins')

layout
```

## Building the dashboard

Finally we can add our [Layout](lumen.layout.Layout) to a [Dashboard](lumen.dashboard.Dashboard) instance and give the dashboard a title via the [config](lumen.dashboard.Config) option.

```{pyodide}
lm.Dashboard(config={'title': 'Palmer Penguins'}, layouts=[layout])
```

```{admonition} Note
---
class: success
---
A [Dashboard](lumen.dashboard.Dashboard) (like most other components) can be previewed by displaying itself in notebook environments or by using `.show()` in a REPL. To serve it as a standalone application use the `.servable()` method and launch the notebook or script with `panel serve app.py`.
```
