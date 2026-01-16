# :material-pipe: Transforming data with pipelines

Pipelines filter and transform data before visualization.

Pipelines consume data from [Sources](sources.md) and prepare it for [Views](views.md).

## Pipeline fundamentals

Pipelines sit between sources and views, manipulating data as it flows through:

```
Source → Pipeline → View
         (filter + transform)
```

**Skip pipelines** if you want to display raw data without modification. Connect views directly to sources instead.

### When to use pipelines

Use pipelines to:

- **Filter data**: Show only rows matching criteria (e.g., "sales from 2023")
- **Transform data**: Modify structure or values (e.g., "compute averages by region")
- **Clean data**: Select columns, sort rows, handle missing values
- **Aggregate data**: Group and summarize (e.g., "total sales by product")

### Basic pipeline syntax

=== "YAML"

    ```yaml
    pipelines:
      pipeline_name:         # Choose any name
        source: source_name  # Which source to use
        table: table_name    # Which table from that source
        filters: [...]       # Optional
        transforms: [...]    # Optional
    ```

=== "Python"

    ```python
    from lumen.pipeline import Pipeline
    
    pipeline = Pipeline.from_spec({
        "source": {"type": "file", "tables": {"data": "data.csv"}},
        "filters": [...],
        "transforms": [...]
    })
    ```

## Understanding pipeline execution

Pipeline execution order matters. Operations occur in this sequence:

```
┌─────────────────────────────────────────────┐
│ 1. Source.get() with Filter/SQL state      │
│    (Optimized at database level if SQL)     │
└──────────────┬──────────────────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ 2. DataFrame returned│
     └──────────┬───────────┘
                │
                ▼
     ┌─────────────────────────────┐
     │ 3. Transforms applied        │
     │    (in order specified)      │
     └─────────────────────────────┘
```

**Key insight**: Filters and SQL transforms can execute at the database level (fast). Regular transforms execute on the returned DataFrame (after data is in memory).

## Working with filters

Filters let users drill down into data subsets interactively.

### Widget filters

Widget filters create interactive controls in the sidebar:

=== "YAML"

    ```yaml
    pipelines:
      filtered_data:
        source: my_source
        table: my_table
        filters:
          - type: widget
            field: category      # Column to filter
          - type: widget
            field: region
          - type: widget
            field: year
    ```

=== "Python"

    ```python
    pipeline = Pipeline(source=source, table='my_table')
    pipeline.add_filter('widget', field='category')
    pipeline.add_filter('widget', field='region')
    pipeline.add_filter('widget', field='year')
    ```

=== "Result"

    Creates dropdown widgets for each field. Users select values to filter data.

### Automatic filters

Generate filters for all columns automatically:

=== "YAML"

    ```yaml
    pipelines:
      auto_filtered:
        source: my_source
        table: my_table
        filters: auto      # Creates widgets for all columns
    ```

=== "Python"

    ```python
    pipeline = Pipeline(
        source=source,
        table='my_table',
        filters='auto'
    )
    ```

!!! tip "When to use auto filters"
    Automatic filters work well for exploring new datasets. For production dashboards, explicitly specify filters for better control.

### Filter types

| Type | Purpose | User interaction |
|------|---------|------------------|
| `widget` | Interactive dropdown/slider | Users choose values |
| `constant` | Fixed filter value | No interaction (always applied) |
| `facet` | Split data into groups | Users navigate groups |

### Constant filters

Apply fixed filters that users can't change:

```yaml
filters:
  - type: constant
    field: status
    value: active      # Always filter to active records
```

### Widget filter customization

Customize widget appearance and behavior:

```yaml
filters:
  - type: widget
    field: category
    multi: false         # Single-select instead of multi-select
    default: "Electronics"  # Pre-selected value
  - type: widget
    field: price
    kind: RangeSlider    # Use slider for numeric ranges
```

## Working with transforms

Transforms modify data structure or values.

### Column selection

Select specific columns to display:

=== "YAML"

    ```yaml
    transforms:
      - type: columns
        columns: [id, name, category, price, date]
    ```

=== "Python"

    ```python
    pipeline.add_transform('columns', columns=['id', 'name', 'category', 'price', 'date'])
    ```

### Aggregation

Group data and compute statistics:

=== "YAML"

    ```yaml
    transforms:
      - type: aggregate
        method: mean           # sum, mean, min, max, count, etc.
        by: [category, region] # Group by these columns
    ```

=== "Python"

    ```python
    from lumen.transforms import Aggregate
    
    pipeline.add_transform(Aggregate(
        method='mean',
        by=['category', 'region']
    ))
    ```

=== "Effect"

    Computes the mean of all numeric columns, grouped by category and region.

Common aggregation methods:

| Method | Computation |
|--------|-------------|
| `sum` | Total of values |
| `mean` | Average |
| `median` | Middle value |
| `min` | Minimum value |
| `max` | Maximum value |
| `count` | Number of records |
| `std` | Standard deviation |

### Sorting

Order rows by column values:

```yaml
transforms:
  - type: sort
    by: [date, revenue]
    ascending: [false, false]   # Sort descending
```

### Query (SQL-like filtering)

Filter using SQL-like expressions:

```yaml
transforms:
  - type: query
    query: "price > 100 and category == 'Electronics'"
```

### Transform types reference

| Type | Purpose | Example |
|------|---------|---------|
| `columns` | Select specific columns | Keep only needed fields |
| `aggregate` | Group and compute stats | Average sales by region |
| `sort` | Order rows | Sort by date |
| `query` | SQL-like filtering | `price > 100` |
| `astype` | Change column data types | Convert to datetime |
| `project` | Create derived columns | `total = price * quantity` |

## Building pipelines declaratively

The declarative approach uses nested dictionaries, similar to YAML.

### Complete example

=== "YAML"

    ```yaml
    sources:
      penguins:
        type: file
        tables:
          data: https://datasets.holoviz.org/penguins/v1/penguins.csv

    pipelines:
      analysis:
        source: penguins
        table: data
        filters:
          - type: widget
            field: species
          - type: widget
            field: island
          - type: widget
            field: sex
        transforms:
          - type: aggregate
            method: mean
            by: [species, sex, year]

    layouts:
      - title: Penguin Analysis
        pipeline: analysis
        views:
          - type: table
    ```

=== "Python"

    ```python
    from lumen.pipeline import Pipeline

    pipeline = Pipeline.from_spec({
        'source': {
            'type': 'file',
            'tables': {
                'data': 'https://datasets.holoviz.org/penguins/v1/penguins.csv'
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
    
    pipeline.data  # Preview the result
    ```

### Preview data

Access processed data at any point:

```python
pipeline.data           # Current filtered/transformed data
pipeline.data.head()    # First few rows
pipeline.data.shape     # Dimensions
```

## Building pipelines programmatically

The programmatic approach builds pipelines step-by-step.

### Create a pipeline

Start with a source:

```python
from lumen.sources import FileSource
from lumen.pipeline import Pipeline

source = FileSource(tables={
    'penguins': 'https://datasets.holoviz.org/penguins/v1/penguins.csv'
})

pipeline = Pipeline(source=source, table='penguins')
```

### Add filters step-by-step

```python
pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')
```

### Add transforms step-by-step

```python
# Select columns
columns = ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
pipeline.add_transform('columns', columns=columns)

# Sort by species
pipeline.add_transform('sort', by=['species'])

# Preview result
pipeline.data.head()
```

### Display in notebooks

Render pipelines interactively in Jupyter notebooks:

```python
import panel as pn

pn.extension('tabulator')

pipeline  # Renders with widgets and data preview
```

Show only the control panel:

```python
pipeline.control_panel  # Just the filter widgets
```

### Control auto-update behavior

By default, pipelines update after every interaction. Disable for manual control:

```python
pipeline = Pipeline(
    source=source,
    table='penguins',
    auto_update=False  # Require explicit update
)

# Add update button
import panel as pn

pn.Column(
    pipeline.control_panel,
    pn.widgets.Button(name='Update', on_click=lambda e: pipeline.update()),
    pipeline.data
)
```

## Chaining pipelines

Create processing stages by chaining pipelines together. This lets one pipeline build on another's output.

### Why chain pipelines?

- **Separate concerns**: Filter in one stage, aggregate in another
- **Reuse filtering**: Multiple aggregations of the same filtered data
- **Optimize performance**: Share computation between related views

### Chain in Python

Use the `.chain()` method:

```python
from lumen.sources import FileSource
from lumen.pipeline import Pipeline
from lumen.transforms import Aggregate

# Create base pipeline with filtering
source = FileSource(tables={
    'penguins': 'https://datasets.holoviz.org/penguins/v1/penguins.csv'
})

base_pipeline = Pipeline(source=source, table='penguins')
base_pipeline.add_filter('widget', field='species')
base_pipeline.add_filter('widget', field='island')

# Chain to create aggregated view
agg_pipeline = base_pipeline.chain(
    transforms=[Aggregate(method='mean', by=['species', 'year'])]
)

# Both pipelines share the same filters
# base_pipeline shows filtered raw data
# agg_pipeline shows filtered + aggregated data
```

### Chain in YAML

Reference one pipeline from another using `pipeline:` instead of `source:`:

=== "YAML"

    ```yaml
    sources:
      penguins:
        type: file
        tables:
          data: https://datasets.holoviz.org/penguins/v1/penguins.csv

    pipelines:
      # Base pipeline with filtering
      filtered:
        source: penguins
        table: data
        filters:
          - type: widget
            field: island

      # Chained pipeline adds transforms
      aggregated:
        pipeline: filtered        # Reference the other pipeline
        transforms:
          - type: aggregate
            method: mean
            by: [species, year]

    layouts:
      - title: Analysis
        views:
          - type: table
            pipeline: filtered    # Shows filtered raw data
          - type: table
            pipeline: aggregated  # Shows filtered + aggregated data
    ```

=== "Result"

    - Both tables use the same island filter
    - First table shows raw filtered data
    - Second table shows aggregated filtered data
    - Changing the filter updates both tables

### Multiple chains

Create multiple processing branches:

```python
# Base filtering
base = Pipeline(source=source, table='data')
base.add_filter('widget', field='category')

# Branch 1: Aggregated view
agg_branch = base.chain(transforms=[
    Aggregate(method='sum', by=['region'])
])

# Branch 2: Top 10 view
top_branch = base.chain(transforms=[
    Sort(by=['revenue'], ascending=False),
    {'type': 'query', 'query': 'index < 10'}
])

# All three share the category filter:
# - base: filtered raw data
# - agg_branch: filtered + aggregated
# - top_branch: filtered + top 10
```

## Branching pipelines (advanced)

Branching creates multiple views of the same source data at different processing stages.

### Simple branch example

=== "YAML"

    ```yaml
    sources:
      sales:
        type: file
        tables:
          data: sales.csv

    pipelines:
      # Base pipeline
      base:
        source: sales
        table: data
        filters:
          - type: widget
            field: region

      # Branch: adds column selection
      selected:
        pipeline: base
        transforms:
          - type: columns
            columns: [date, product, revenue]

      # Branch: adds aggregation
      summary:
        pipeline: base
        transforms:
          - type: aggregate
            method: sum
            by: [product]

    layouts:
      - title: Sales Dashboard
        views:
          - type: table
            pipeline: base       # Filtered full data
          - type: table
            pipeline: selected   # Filtered + selected columns
          - type: hvplot
            pipeline: summary    # Filtered + aggregated
            kind: bar
            x: product
            y: revenue
    ```

=== "Flow diagram"

    ```
    Source (sales.csv)
         │
         ▼
    Pipeline: base (filter by region)
         │
         ├─────────┬─────────┐
         │         │         │
         ▼         ▼         ▼
      View 1   Pipeline:  Pipeline:
               selected   summary
                  │         │
                  ▼         ▼
               View 2    View 3
    ```

All three views share the region filter from the base pipeline.

### Complex branching

Create deep processing hierarchies:

```yaml
pipelines:
  # Level 1: Base filtering
  base:
    source: data_source
    filters:
      - type: widget
        field: year
      - type: widget
        field: category

  # Level 2: Column selection
  cleaned:
    pipeline: base
    transforms:
      - type: columns
        columns: [date, product, price, quantity]

  # Level 3: Derived columns
  calculated:
    pipeline: cleaned
    transforms:
      - type: project
        columns:
          revenue: price * quantity

  # Level 3 alternate: Aggregation
  summary:
    pipeline: cleaned
    transforms:
      - type: aggregate
        method: sum
        by: [product]
```

## Using pipelines outside dashboards

Pipelines work independently of full dashboard specifications. Use them in notebooks or custom applications.

### In Jupyter notebooks

```python
from lumen.pipeline import Pipeline
from lumen.views import Table, hvPlotView
import panel as pn

pn.extension('tabulator')

# Create pipeline
pipeline = Pipeline.from_spec({
    'source': {
        'type': 'file',
        'tables': {'data': 'data.csv'}
    },
    'filters': [
        {'type': 'widget', 'field': 'category'},
        {'type': 'widget', 'field': 'region'}
    ]
})

# Display with Panel
pn.Row(
    pipeline.control_panel,
    pn.Column(
        hvPlotView(pipeline=pipeline, kind='bar', x='product', y='sales'),
        Table(pipeline=pipeline)
    )
)
```

### In custom Panel apps

Build custom applications using Panel's layout system:

```python
from lumen.pipeline import Pipeline
from lumen.views import hvPlotView, Table
import panel as pn

pn.extension('tabulator')

# Create pipeline
pipeline = Pipeline.from_spec({
    'source': {
        'type': 'file',
        'tables': {'penguins': 'penguins.csv'}
    },
    'filters': [
        {'type': 'widget', 'field': 'species'},
        {'type': 'widget', 'field': 'island'}
    ]
})

# Create views
scatter = hvPlotView(
    pipeline=pipeline,
    kind='scatter',
    x='bill_length_mm',
    y='bill_depth_mm',
    by='species'
)

table = Table(pipeline=pipeline, page_size=10)

# Custom layout
app = pn.template.MaterialTemplate(
    title='Penguin Analysis',
    sidebar=[pipeline.control_panel],
    main=[
        pn.Row(scatter, table)
    ]
)

app.servable()
```

### Binding to custom widgets

Bind pipeline data to any Panel component:

```python
import panel as pn

# Bind data to DataFrame pane
data_pane = pn.pane.DataFrame(
    pipeline.param.data,
    width=800,
    height=400
)

# Bind to custom function
@pn.depends(pipeline.param.data)
def custom_view(data):
    return pn.pane.Markdown(f"**Rows**: {len(data)}")

pn.Column(
    pipeline.control_panel,
    data_pane,
    custom_view
)
```

## Common patterns

### Filter then visualize

```yaml
pipelines:
  filtered_data:
    source: my_source
    table: my_table
    filters:
      - type: widget
        field: year
      - type: widget
        field: category

layouts:
  - title: Dashboard
    pipeline: filtered_data
    views:
      - type: hvplot
        kind: line
        x: date
        y: sales
```

### Filter, transform, then visualize

```yaml
pipelines:
  processed_data:
    source: my_source
    table: my_table
    filters:
      - type: widget
        field: region
    transforms:
      - type: columns
        columns: [date, product, revenue]
      - type: sort
        by: [date]

layouts:
  - title: Dashboard
    pipeline: processed_data
    views:
      - type: hvplot
        kind: bar
        x: product
        y: revenue
```

### Multiple aggregations of same data

```yaml
pipelines:
  base:
    source: sales
    table: data
    filters:
      - type: widget
        field: year

  by_region:
    pipeline: base
    transforms:
      - type: aggregate
        method: sum
        by: [region]

  by_product:
    pipeline: base
    transforms:
      - type: aggregate
        method: sum
        by: [product]

layouts:
  - title: Sales Analysis
    views:
      - type: hvplot
        pipeline: by_region
        kind: bar
      - type: hvplot
        pipeline: by_product
        kind: bar
```

## Next steps

Now that you understand pipelines:

- **[Views guide](views.md)** - Visualize your processed data
- **[Variables guide](variables.md)** - Make pipelines dynamic
- **[Python API guide](python-api.md)** - Build complex applications
