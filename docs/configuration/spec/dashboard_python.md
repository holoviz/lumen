# Build dashboards in Python

In the previous guide, you learned how to [build pipelines in Python](pipeline_python.md). Here we'll build a complete dashboard in Python.

Start by creating a pipeline and initializing the Panel extension:

```python
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

## Attach views

Pass your pipeline to any View constructor. The view will update whenever the pipeline changes:

```python
scatter = lm.views.hvPlotView(
    pipeline=pipeline,
    kind='scatter',
    x='bill_length_mm',
    y='bill_depth_mm',
    by='species',
    height=300,
    responsive=True
)

scatter
```

Create another view, a Table:

```python
table = lm.views.Table(pipeline=pipeline, page_size=10, sizing_mode='stretch_width')

table
```

## Lay out views

Use the Lumen Layout component to arrange views:

```python
layout = lm.Layout(
    views={'scatter': scatter, 'table': table},
    title='Palmer Penguins'
)

layout
```

## Build the dashboard

Finally, add your layout to a Dashboard instance:

```python
lm.Dashboard(
    config={'title': 'Palmer Penguins'},
    layouts=[layout]
)
```

!!! note
    You can preview dashboards in notebooks by displaying them, or use `.show()` in a REPL. To serve as a standalone application, use `.servable()` and launch with `panel serve app.py`.
