# :material-chart-scatter-plot: Visualizing data with views

Views display your data as plots, tables, indicators, and other visual components.

**Different from Analyses:** For computational results (statistics, calculations, tests), see [Analyses](../analyses.md). Views are for visualizing data; analyses are for computing insights from data.

## View fundamentals

Views are the final output of your dashboard. They transform data into visual representations that users can see and interact with.

### Basic view syntax

=== "YAML"

    ```yaml
    layouts:
      - title: My Dashboard
        source: my_source      # Or pipeline: my_pipeline
        views:
          - type: view_type
            # View-specific parameters
    ```

=== "Python"

    ```python
    from lumen.views import Table, hvPlotView
    
    view = hvPlotView(
        pipeline=pipeline,
        kind='scatter',
        x='x_column',
        y='y_column'
    )
    ```

### Connecting views to data

Views get data from either sources or pipelines:

=== "From source (raw data)"

    ```yaml
    layouts:
      - title: Dashboard
        source: my_source
        views:
          - type: table
            table: my_table
    ```

=== "From pipeline (processed data)"

    ```yaml
    layouts:
      - title: Dashboard
        pipeline: my_pipeline
        views:
          - type: table
    ```

Use pipelines when you need filtering or transforms. Use sources directly for raw data.

## Available view types

| View Type | Purpose | Best for |
|-----------|---------|----------|
| `hvplot` | Interactive plots | Scatter, line, bar, hist, box, etc. |
| `table` | Data tables | Displaying tabular data |
| `indicator` | Single metrics | KPIs, status values |
| `download` | Download buttons | Enabling data export |
| `hvplot_ui` | Plot with UI controls | Exploratory analysis |
| `altair` | Altair visualizations | Grammar of graphics plots |
| `plotly` | Plotly charts | 3D plots, complex interactivity |

This guide focuses on the most common types: `hvplot` and `table`.

## Plot views (hvPlot)

The `hvplot` view type creates interactive plots using [hvPlot](https://hvplot.holoviz.org/).

### Plot kinds

hvPlot supports many plot types via the `kind` parameter:

| Kind | Description | Use case |
|------|-------------|----------|
| `scatter` | Scatter plot | Relationships between variables |
| `line` | Line plot | Time series, trends |
| `bar` | Bar chart | Comparing categories |
| `hist` | Histogram | Distribution of values |
| `box` | Box plot | Statistical distribution |
| `area` | Area plot | Cumulative values over time |
| `heatmap` | Heat map | 2D data matrices |
| `violin` | Violin plot | Distribution with density |

### Scatter plots

Show relationships between two variables:

=== "YAML"

    ```yaml
    views:
      - type: hvplot
        kind: scatter
        x: bill_length_mm
        y: bill_depth_mm
        color: species           # Color by category
        size: body_mass_g        # Size by value
        responsive: true
        height: 400
    ```

=== "Python"

    ```python
    from lumen.views import hvPlotView
    
    scatter = hvPlotView(
        pipeline=pipeline,
        kind='scatter',
        x='bill_length_mm',
        y='bill_depth_mm',
        color='species',
        size='body_mass_g',
        responsive=True,
        height=400
    )
    ```

Common scatter parameters:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x` | string | Column for x-axis |
| `y` | string | Column for y-axis |
| `color` | string | Column for color coding |
| `size` | string/int | Column for point size, or fixed size |
| `alpha` | float | Transparency (0-1) |

### Line plots

Display trends over time or continuous variables:

=== "YAML"

    ```yaml
    views:
      - type: hvplot
        kind: line
        x: date
        y: revenue
        by: region              # Create separate lines
        responsive: true
        legend: 'top_right'
    ```

=== "Python"

    ```python
    line = hvPlotView(
        pipeline=pipeline,
        kind='line',
        x='date',
        y='revenue',
        by='region',
        responsive=True,
        legend='top_right'
    )
    ```

### Bar charts

Compare values across categories:

=== "YAML"

    ```yaml
    views:
      - type: hvplot
        kind: bar
        x: product
        y: sales
        color: region
        stacked: true           # Stack bars by color
        rot: 45                 # Rotate x-axis labels
    ```

=== "Python"

    ```python
    bar = hvPlotView(
        pipeline=pipeline,
        kind='bar',
        x='product',
        y='sales',
        color='region',
        stacked=True,
        rot=45
    )
    ```

### Histograms

Show distribution of values:

=== "YAML"

    ```yaml
    views:
      - type: hvplot
        kind: hist
        y: bill_length_mm
        bins: 30                # Number of bins
        alpha: 0.7
        color: '#ff6600'
    ```

=== "Python"

    ```python
    hist = hvPlotView(
        pipeline=pipeline,
        kind='hist',
        y='bill_length_mm',
        bins=30,
        alpha=0.7,
        color='#ff6600'
    )
    ```

### Box plots

Display statistical distribution:

```yaml
views:
  - type: hvplot
    kind: box
    y: bill_length_mm
    by: species              # Create box for each category
    legend: false
```

### Multiple series

Plot multiple columns on the same chart:

```yaml
views:
  - type: hvplot
    kind: line
    x: date
    y: [revenue, expenses, profit]  # Multiple y-columns
    responsive: true
```

### Plot customization

Common customization parameters:

| Parameter | Type | Purpose | Example |
|-----------|------|---------|---------|
| `title` | string | Plot title | "Sales Trends" |
| `xlabel` | string | X-axis label | "Date" |
| `ylabel` | string | Y-axis label | "Revenue ($)" |
| `height` | int | Plot height (pixels) | 400 |
| `width` | int | Plot width (pixels) | 800 |
| `responsive` | bool | Resize with browser | true |
| `legend` | string/bool | Legend position | 'top_right', false |
| `xlim` | tuple | X-axis range | (0, 100) |
| `ylim` | tuple | Y-axis range | (0, 1000) |
| `rot` | int | Label rotation (degrees) | 45 |
| `fontsize` | dict | Font sizes | {title: 16, labels: 12} |
| `grid` | bool | Show gridlines | true |
| `logx` | bool | Logarithmic x-axis | true |
| `logy` | bool | Logarithmic y-axis | true |

Example with many customizations:

```yaml
views:
  - type: hvplot
    kind: scatter
    x: gdp_per_capita
    y: life_expectancy
    color: continent
    size: population
    title: "Health vs Wealth"
    xlabel: "GDP per Capita ($)"
    ylabel: "Life Expectancy (years)"
    logx: true
    xlim: [100, 100000]
    ylim: [40, 90]
    height: 500
    responsive: true
    legend: 'top_left'
```

## Table views

Display data in interactive tables with sorting, pagination, and filtering.

### Basic table

=== "YAML"

    ```yaml
    views:
      - type: table
        show_index: false       # Hide row index
        page_size: 20           # Rows per page
        height: 400
    ```

=== "Python"

    ```python
    from lumen.views import Table
    
    table = Table(
        pipeline=pipeline,
        show_index=False,
        page_size=20,
        height=400
    )
    ```

### Table parameters

| Parameter | Type | Purpose | Default |
|-----------|------|---------|---------|
| `show_index` | bool | Display row numbers | true |
| `page_size` | int | Rows per page | 20 |
| `pagination` | string | Pagination type | 'local' |
| `height` | int | Table height (pixels) | 400 |
| `width` | int | Table width (pixels) | None |
| `theme` | string | Color theme | 'default' |
| `disabled` | bool | Disable interactions | false |
| `selectable` | bool/int | Row selection | false |

### Table themes

Available themes for table styling:

```yaml
views:
  - type: table
    theme: midnight          # Options: default, modern, simple, midnight, site
```

### Column formatting

Format specific columns:

```yaml
views:
  - type: table
    formatters:
      price: {type: money, precision: 2}
      date: {type: datetime, format: '%Y-%m-%d'}
      percent: {type: percentage, precision: 1}
```

### Selectable rows

Enable row selection:

```yaml
views:
  - type: table
    selectable: 1            # Select single row (use 'checkbox' for multiple)
```

### Remote pagination

For large datasets, use remote pagination:

```yaml
views:
  - type: table
    pagination: remote       # Load pages from server
    page_size: 50
```

## Indicator views

Display single metrics or KPIs.

### Number indicator

Show a single value:

```yaml
views:
  - type: indicator
    field: revenue           # Column to display
    title: "Total Revenue"
    format: "${value:,.0f}"  # Format as currency
```

### Gauge indicator

Display value with min/max range:

```yaml
views:
  - type: indicator
    kind: gauge
    field: completion_rate
    title: "Completion Rate"
    min_value: 0
    max_value: 100
    format: "{value:.1f}%"
```

## Download views

Enable data export:

```yaml
views:
  - type: download
    format: csv              # csv, xlsx, json
    button_type: 'primary'
```

See the [Downloads guide](downloads.md) for details.

## Arranging views

Control how views appear in your dashboard.

### Default arrangement

Views stack vertically by default:

```yaml
layouts:
  - title: Dashboard
    views:
      - type: hvplot
        kind: scatter
      - type: hvplot
        kind: hist
      - type: table
```

Result: Three views stacked vertically.

### Grid layout

Use the `layout` parameter to create grids:

```yaml
layouts:
  - title: Dashboard
    pipeline: my_pipeline
    layout: [[0, 1], [2, 3]]  # 2x2 grid
    views:
      - type: hvplot          # View 0 (top-left)
        kind: scatter
      - type: hvplot          # View 1 (top-right)
        kind: line
      - type: hvplot          # View 2 (bottom-left)
        kind: bar
      - type: table           # View 3 (bottom-right)
```

### Custom arrangements

Create complex layouts:

```yaml
layouts:
  - title: Dashboard
    layout: [[0], [1, 2], [3]]  # Full-width top, two middle, full-width bottom
    views:
      - type: hvplot            # Full width
        kind: line
      - type: hvplot            # Half width
        kind: bar
      - type: hvplot            # Half width
        kind: hist
      - type: table             # Full width
```

### Responsive sizing

Make views resize with browser window:

```yaml
layouts:
  - title: Dashboard
    sizing_mode: stretch_width  # or stretch_both, stretch_height
    views:
      - type: hvplot
        responsive: true
        height: 400
      - type: table
        height: 300
```

Sizing modes:

| Mode | Behavior |
|------|----------|
| `stretch_width` | Expand to fill width |
| `stretch_height` | Expand to fill height |
| `stretch_both` | Expand to fill both dimensions |
| `scale_width` | Maintain aspect ratio, scale to width |
| `scale_height` | Maintain aspect ratio, scale to height |
| `fixed` | Use specified width/height only |

### Tabbed layouts

Create tabs at the top level:

```yaml
config:
  layout: tabs               # Use tabs instead of single page

layouts:
  - title: Overview          # Tab 1
    views:
      - type: hvplot
        kind: line
  
  - title: Detailed Analysis # Tab 2
    views:
      - type: table
```

## Advanced view features

### Linked selections

Link multiple plots so selecting in one highlights in others:

```yaml
views:
  - type: hvplot
    kind: scatter
    x: bill_length_mm
    y: bill_depth_mm
    selection_mode: lasso    # Enable lasso selection
  - type: hvplot
    kind: hist
    y: bill_length_mm
    # Selection automatically links
```

### Dynamic colormaps

Use dynamic colormaps for continuous values:

```yaml
views:
  - type: hvplot
    kind: scatter
    x: x
    y: y
    color: temperature
    cmap: viridis            # or: plasma, inferno, magma, coolwarm
    colorbar: true
```

### Hover tooltips

Customize hover information:

```yaml
views:
  - type: hvplot
    kind: scatter
    x: x
    y: y
    hover_cols: [name, category, details]  # Show these on hover
```

### Aggregated plots

Compute aggregations within the view:

```yaml
views:
  - type: hvplot
    kind: bar
    x: category
    y: value
    aggregator: mean         # Compute mean by category
```

## Common patterns

### Dashboard with filters and multiple plots

```yaml
pipelines:
  filtered:
    source: data_source
    table: data
    filters:
      - type: widget
        field: region
      - type: widget
        field: year

layouts:
  - title: Sales Dashboard
    pipeline: filtered
    layout: [[0, 1], [2]]
    sizing_mode: stretch_width
    views:
      - type: hvplot
        kind: line
        x: date
        y: revenue
        responsive: true
        height: 300
      - type: hvplot
        kind: bar
        x: product
        y: revenue
        responsive: true
        height: 300
      - type: table
        show_index: false
        page_size: 15
        height: 400
```

### Comparison dashboard

```yaml
layouts:
  - title: Before vs After
    layout: [[0, 1]]
    views:
      - type: hvplot
        pipeline: before_pipeline
        kind: scatter
        x: x
        y: y
        title: "Before"
      - type: hvplot
        pipeline: after_pipeline
        kind: scatter
        x: x
        y: y
        title: "After"
```

### KPI dashboard with indicators

```yaml
pipelines:
  summary:
    source: data_source
    table: metrics
    transforms:
      - type: aggregate
        method: sum

layouts:
  - title: KPIs
    pipeline: summary
    layout: [[0, 1, 2], [3]]
    views:
      - type: indicator
        field: revenue
        title: "Revenue"
        format: "${value:,.0f}"
      - type: indicator
        field: orders
        title: "Orders"
        format: "{value:,}"
      - type: indicator
        field: conversion_rate
        title: "Conversion"
        format: "{value:.1f}%"
      - type: hvplot
        kind: line
        x: date
        y: revenue
```

## Next steps

Now that you can visualize data:

- **[Variables guide](variables.md)** - Make views dynamic with variables
- **[Customization guide](customization.md)** - Build custom view types
- **[Deployment guide](deployment.md)** - Deploy your dashboard

For complete plot options, see the [hvPlot reference](https://hvplot.holoviz.org/reference/index.html).
