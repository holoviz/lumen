# :material-language-python: Building dashboards with Python

Build Lumen dashboards programmatically using Python instead of YAML.

## Why use Python?

| Aspect | YAML | Python |
|--------|------|--------|
| **Syntax** | Simple configuration | Python code |
| **Learning curve** | Beginner-friendly | Requires Python knowledge |
| **Iteration** | Very fast (edit + refresh) | Fast (requires restart) |
| **Flexibility** | Good for standard patterns | Full programmatic control |
| **Debugging** | Limited | Full Python debugging |
| **Dynamic behavior** | Variables + templates | Complete programmatic control |
| **Best for** | Most dashboards | Complex applications, custom logic |

**Use Python when you need:**

- Complex conditional logic
- Dynamic component generation
- Custom data processing not supported by transforms
- Integration with existing Python applications
- Full programmatic control

## Python API basics

The Python API mirrors the YAML structure using Python objects:

| YAML Section | Python Class |
|--------------|--------------|
| `config` | `lumen.Config` |
| `sources` | `lumen.sources.*` classes |
| `pipelines` | `lumen.Pipeline` |
| `layouts` | `lumen.Layout` |
| `views` | `lumen.views.*` classes |

## Building pipelines in Python

Start by creating a Pipeline object. You can build declaratively (like YAML) or programmatically (step-by-step).

### Declarative approach

Use nested dictionaries mirroring YAML structure:

```python
import lumen as lm

pipeline = lm.Pipeline.from_spec({
    'source': {
        'type': 'file',
        'tables': {
            'penguins': 'https://datasets.holoviz.org/penguins/v1/penguins.csv'
        }
    },
    'filters': [
        {'type': 'widget', 'field': 'species'},
        {'type': 'widget', 'field': 'island'},
        {'type': 'widget', 'field': 'sex'},
    ],
    'transforms': [
        {'type': 'aggregate', 'method': 'mean', 'by': ['species', 'sex', 'year']}
    ]
})

# Preview the data
pipeline.data
```

### Programmatic approach

Build step-by-step using Python methods:

```python
import lumen as lm
from lumen.sources import FileSource

# Create source
source = FileSource(tables={
    'penguins': 'https://datasets.holoviz.org/penguins/v1/penguins.csv'
})

# Create pipeline
pipeline = lm.Pipeline(source=source, table='penguins')

# Add filters
pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')

# Add transforms
columns = ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
pipeline.add_transform('columns', columns=columns)

# Preview
pipeline.data.head()
```

### Preview data anytime

```python
pipeline.data          # Current filtered/transformed data
pipeline.data.head()   # First 5 rows
pipeline.data.shape    # Dimensions (rows, columns)
pipeline.data.columns  # Column names
```

### Automatic filters

Generate filters for all columns:

```python
pipeline = lm.Pipeline(
    source=source,
    table='penguins',
    filters='auto'  # Creates widget for every column
)
```

### Control updates

By default, pipelines update after every interaction. Disable for manual control:

```python
pipeline = lm.Pipeline(
    source=source,
    table='penguins',
    auto_update=False  # Manual updates only
)

# Later, trigger update manually
pipeline.update()
```

### Display in notebooks

Render pipelines interactively in Jupyter:

```python
import panel as pn

pn.extension('tabulator')

pipeline  # Shows widgets + data preview
```

Show only controls:

```python
pipeline.control_panel  # Just the filter widgets
```

## Creating views

Views visualize pipeline data. Attach views to pipelines, and they update automatically.

### Plot views

```python
from lumen.views import hvPlotView

scatter = hvPlotView(
    pipeline=pipeline,
    kind='scatter',
    x='bill_length_mm',
    y='bill_depth_mm',
    by='species',
    height=400,
    responsive=True
)

scatter  # Display in notebook
```

### Table views

```python
from lumen.views import Table

table = Table(
    pipeline=pipeline,
    page_size=10,
    show_index=False,
    sizing_mode='stretch_width'
)

table
```

### Multiple views from one pipeline

```python
# Create views
scatter = hvPlotView(pipeline=pipeline, kind='scatter', x='bill_length_mm', y='bill_depth_mm')
hist = hvPlotView(pipeline=pipeline, kind='hist', y='bill_length_mm')
table = Table(pipeline=pipeline, page_size=10)

# Display together
import panel as pn

pn.Column(scatter, pn.Row(hist, table))
```

## Building layouts

Use the `Layout` component to organize views:

```python
import lumen as lm

# Create pipeline
pipeline = lm.Pipeline.from_spec({
    'source': {
        'type': 'file',
        'tables': {'penguins': 'penguins.csv'}
    },
    'filters': [
        {'type': 'widget', 'field': 'island'},
        {'type': 'widget', 'field': 'sex'},
    ]
})

# Create views
scatter = lm.views.hvPlotView(
    pipeline=pipeline,
    kind='scatter',
    x='bill_length_mm',
    y='bill_depth_mm',
    by='species'
)

table = lm.views.Table(pipeline=pipeline, page_size=10)

# Create layout
layout = lm.Layout(
    views={'scatter': scatter, 'table': table},
    title='Palmer Penguins',
    layout=[[0], [1]]  # Scatter on top, table below
)

layout
```

## Building complete dashboards

Combine everything into a `Dashboard` object:

```python
import lumen as lm
import panel as pn

pn.extension('tabulator', design='material')

# Create pipeline
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
})

# Create views
scatter = lm.views.hvPlotView(
    pipeline=pipeline,
    kind='scatter',
    x='bill_length_mm',
    y='bill_depth_mm',
    by='species',
    responsive=True,
    height=400
)

table = lm.views.Table(
    pipeline=pipeline,
    page_size=10,
    sizing_mode='stretch_width'
)

# Create layout
layout = lm.Layout(
    views={'scatter': scatter, 'table': table},
    title='Palmer Penguins'
)

# Create dashboard
dashboard = lm.Dashboard(
    config={'title': 'Palmer Penguins', 'theme': 'dark'},
    layouts=[layout]
)

dashboard
```

### Serve the dashboard

Save as `app.py` and serve:

```python
import lumen as lm
import panel as pn

pn.extension('tabulator')

# ... (pipeline, views, layout code) ...

dashboard = lm.Dashboard(
    config={'title': 'My Dashboard'},
    layouts=[layout]
)

dashboard.servable()  # Makes it servable
```

Then run:

```bash
panel serve app.py --show
```

## Integration with Panel

Lumen components work seamlessly with Panel for maximum flexibility.

### Custom layouts with Panel

```python
import panel as pn
import lumen as lm

pn.extension('tabulator')

# Create pipeline
pipeline = lm.Pipeline.from_spec({...})

# Create views
scatter = lm.views.hvPlotView(pipeline=pipeline, kind='scatter', x='x', y='y')
table = lm.views.Table(pipeline=pipeline)

# Custom Panel layout
app = pn.template.MaterialTemplate(
    title='Custom Dashboard',
    sidebar=[
        pipeline.control_panel,
        pn.pane.Markdown('## About\nThis dashboard shows...')
    ],
    main=[
        pn.Row(scatter, table)
    ]
)

app.servable()
```

### Combine with Panel widgets

```python
import panel as pn
import lumen as lm

# Lumen pipeline
pipeline = lm.Pipeline.from_spec({...})

# Panel widget
custom_widget = pn.widgets.TextInput(name='Search', placeholder='Enter term...')

# Combine
pn.Column(
    '# My Dashboard',
    custom_widget,
    pipeline.control_panel,
    lm.views.Table(pipeline=pipeline)
)
```

### Bind to custom functions

```python
import panel as pn

@pn.depends(pipeline.param.data)
def custom_view(data):
    return pn.pane.Markdown(f"""
    ## Summary
    - Total rows: {len(data)}
    - Columns: {len(data.columns)}
    - Memory: {data.memory_usage().sum() / 1024:.2f} KB
    """)

pn.Column(
    pipeline.control_panel,
    custom_view,
    lm.views.Table(pipeline=pipeline)
)
```

### React to changes

```python
import panel as pn

def on_data_change(event):
    print(f'Data updated! New shape: {event.new.shape}')

pipeline.param.watch(on_data_change, 'data')
```

## Advanced patterns

### Dynamic pipeline creation

```python
def create_pipeline(source_file, filter_fields):
    """Create pipeline based on parameters."""
    pipeline = lm.Pipeline.from_spec({
        'source': {
            'type': 'file',
            'tables': {'data': source_file}
        },
        'filters': [
            {'type': 'widget', 'field': field}
            for field in filter_fields
        ]
    })
    return pipeline

# Use it
pipeline = create_pipeline('sales.csv', ['region', 'category', 'year'])
```

### Conditional views

```python
def create_views(pipeline, plot_type='scatter'):
    """Create different views based on type."""
    if plot_type == 'scatter':
        return lm.views.hvPlotView(
            pipeline=pipeline,
            kind='scatter',
            x='x', y='y'
        )
    elif plot_type == 'line':
        return lm.views.hvPlotView(
            pipeline=pipeline,
            kind='line',
            x='date', y='value'
        )
    else:
        return lm.views.Table(pipeline=pipeline)

view = create_views(pipeline, plot_type='scatter')
```

### Multiple pipelines

```python
# Create base pipeline
base_pipeline = lm.Pipeline.from_spec({
    'source': {'type': 'file', 'tables': {'data': 'data.csv'}},
    'filters': [{'type': 'widget', 'field': 'category'}]
})

# Create branches
agg_pipeline = base_pipeline.chain(
    transforms=[{'type': 'aggregate', 'method': 'sum', 'by': ['region']}]
)

top_pipeline = base_pipeline.chain(
    transforms=[
        {'type': 'sort', 'by': ['revenue'], 'ascending': False},
        {'type': 'query', 'query': 'index < 10'}
    ]
)

# Different views of same data
pn.Tabs(
    ('Raw', lm.views.Table(pipeline=base_pipeline)),
    ('Aggregated', lm.views.hvPlotView(pipeline=agg_pipeline, kind='bar')),
    ('Top 10', lm.views.Table(pipeline=top_pipeline))
)
```

### Class-based dashboards

```python
import lumen as lm
import panel as pn

class SalesDashboard:
    """Reusable dashboard class."""
    
    def __init__(self, data_file):
        self.pipeline = lm.Pipeline.from_spec({
            'source': {'type': 'file', 'tables': {'data': data_file}},
            'filters': [
                {'type': 'widget', 'field': 'region'},
                {'type': 'widget', 'field': 'year'}
            ]
        })
        
        self.views = {
            'scatter': lm.views.hvPlotView(
                pipeline=self.pipeline,
                kind='scatter',
                x='date', y='revenue'
            ),
            'table': lm.views.Table(pipeline=self.pipeline)
        }
    
    def show(self):
        """Display the dashboard."""
        return pn.Column(
            '# Sales Dashboard',
            self.pipeline.control_panel,
            pn.Row(self.views['scatter'], self.views['table'])
        )

# Use it
dashboard = SalesDashboard('sales.csv')
dashboard.show()
```

## Converting between YAML and Python

### YAML to Python

Load YAML specs in Python:

```python
import lumen as lm

# From file
dashboard = lm.Dashboard.from_yaml('dashboard.yaml')

# From string
yaml_spec = """
config:
  title: My Dashboard
sources:
  data:
    type: file
    tables:
      main: data.csv
"""
dashboard = lm.Dashboard.from_yaml_string(yaml_spec)
```

### Python to YAML

Export Python objects to YAML:

```python
import lumen as lm

# Create in Python
pipeline = lm.Pipeline.from_spec({...})

# Export to YAML
yaml_str = pipeline.to_yaml()
print(yaml_str)

# Save to file
with open('pipeline.yaml', 'w') as f:
    f.write(yaml_str)
```

## Debugging

### Print current data

```python
pipeline.data         # View current data
print(pipeline.data.head())
```

### Check filter values

```python
for filter in pipeline.filters:
    print(f'{filter.field}: {filter.value}')
```

### Inspect transforms

```python
for transform in pipeline.transforms:
    print(f'{transform.transform_type}: {transform.to_spec()}')
```

### Enable logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('lumen')
```

### Interactive debugging

```python
import pdb

# Add breakpoint
pdb.set_trace()

# Or use ipdb in notebooks
import ipdb; ipdb.set_trace()
```

## Best practices

### Use type hints

```python
from typing import List
import lumen as lm

def create_filters(fields: List[str]) -> List[dict]:
    """Create widget filters for fields."""
    return [
        {'type': 'widget', 'field': field}
        for field in fields
    ]

pipeline = lm.Pipeline.from_spec({
    'source': {...},
    'filters': create_filters(['region', 'year'])
})
```

### Error handling

```python
try:
    pipeline = lm.Pipeline.from_spec({
        'source': {'type': 'file', 'tables': {'data': 'missing.csv'}}
    })
except FileNotFoundError as e:
    print(f'Data file not found: {e}')
    # Fall back to alternative source
```

### Configuration management

```python
import os
from pathlib import Path

# Use environment variables
DATA_DIR = os.getenv('DATA_DIR', 'data')
CACHE_DIR = os.getenv('CACHE_DIR', '.cache')

pipeline = lm.Pipeline.from_spec({
    'source': {
        'type': 'file',
        'cache_dir': CACHE_DIR,
        'tables': {
            'data': Path(DATA_DIR) / 'sales.csv'
        }
    }
})
```

### Reusable components

```python
# config.py
DEFAULT_FILTERS = [
    {'type': 'widget', 'field': 'region'},
    {'type': 'widget', 'field': 'year'}
]

DEFAULT_PLOT_PARAMS = {
    'responsive': True,
    'height': 400
}

# app.py
from config import DEFAULT_FILTERS, DEFAULT_PLOT_PARAMS

pipeline = lm.Pipeline.from_spec({
    'source': {...},
    'filters': DEFAULT_FILTERS
})

scatter = lm.views.hvPlotView(
    pipeline=pipeline,
    kind='scatter',
    **DEFAULT_PLOT_PARAMS
)
```

## Next steps

Now that you can build dashboards in Python:

- **[Panel documentation](https://panel.holoviz.org/)** - Learn more about Panel layouts and widgets
- **[Customization guide](customization.md)** - Build custom components
- **[Deployment guide](deployment.md)** - Deploy Python dashboards
