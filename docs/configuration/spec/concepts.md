# :material-lightbulb: Core concepts

Understand how Lumen works to build your own dashboards effectively.

!!! tip "Prerequisites"
    Complete the [Build Dashboard with Spec tutorial](../../examples/tutorials/penguins_dashboard_spec.md) before reading this guide. This page generalizes those concepts so you can apply them to your own data.

## How Lumen works

Lumen transforms data through three stages:

```
┌──────────┐     ┌───────────┐     ┌────────┐
│ Sources  │ ──> │ Pipelines │ ──> │ Views  │
└──────────┘     └───────────┘     └────────┘
     │                 │                  │
     │                 │                  │
Load data      Filter/transform     Visualize
```

1. **Sources** load data from files, databases, or APIs
2. **Pipelines** filter and transform the data (optional)
3. **Views** display results as plots, tables, or indicators

Configuration settings control the overall dashboard appearance and behavior.

## YAML vs Python

Lumen supports two approaches:

| Aspect | YAML Specifications | Python API |
|--------|---------------------|------------|
| **Syntax** | Simple configuration | Python code |
| **Learning curve** | Beginner-friendly | Requires Python knowledge |
| **Iteration speed** | Very fast (edit and refresh) | Fast (requires restart) |
| **Flexibility** | Good for standard patterns | Full programmatic control |
| **Best for** | Most dashboards | Complex custom applications |
| **Guide** | This section | [Python API Guide](python-api.md) |

Most users start with YAML and only switch to Python when they need custom behavior.

## YAML specification structure

A Lumen specification uses YAML syntax. YAML represents data with **key: value** pairs and lists with leading dashes **-**. Learn more in the [YAML guide](https://learnxinyminutes.com/docs/yaml/).

Lumen specifications contain these sections:

```yaml
config:              # Dashboard settings (optional)
  title: ...
  theme: ...

sources:             # Data sources (required)
  source_name:
    type: ...
    
pipelines:           # Data processing (optional)
  pipeline_name:
    source: ...
    filters: ...
    transforms: ...
    
layouts:             # Visual output (required)
  - title: ...
    views:
      - type: ...
```

**Required sections**: You need at least one source and one layout to create a dashboard.

**Optional sections**: Config and pipelines add functionality but aren't mandatory.

## Section details

### Config: Dashboard-wide settings

The `config` section sets global options that apply to the entire dashboard:

```yaml
config:
  title: My Dashboard           # Browser title and header
  layout: tabs                  # Layout mode: tabs, grid, or column
  logo: assets/my_logo.png      # Custom logo image
  theme: dark                   # Color scheme: dark, default, etc.
```

Common config parameters:

| Parameter | Purpose | Example values |
|-----------|---------|----------------|
| `title` | Dashboard name | "Sales Dashboard" |
| `theme` | Color scheme | `dark`, `default` |
| `layout` | Top-level layout | `tabs`, `grid`, `column` |
| `logo` | Custom logo path | `assets/logo.png` |

Skip the config section to use defaults (light theme, generic title).

### Sources: Loading data

The `sources` section defines where data comes from. Each source has a name and type:

```yaml
sources:
  my_source:                    # Source name (you choose this)
    type: file                  # Source type
    tables:
      my_table: data.csv        # Table name and location
```

Common source types:

| Type | Purpose | Example |
|------|---------|---------|
| `file` | CSV, Excel, Parquet, JSON files | Local or remote files |
| `intake` | Intake catalogs | Data catalogs |
| `duckdb` | DuckDB databases | SQL queries |
| `rest` | REST APIs | API endpoints |

Example loading a remote CSV:

```yaml
sources:
  penguin_source:
    type: file
    tables:
      penguins: https://datasets.holoviz.org/penguins/v1/penguins.csv
```

See the [Sources guide](sources.md) for all source types and configuration options.

### Pipelines: Filtering and transforming

Pipelines manipulate data before visualization. Use them to filter rows or transform columns:

```yaml
pipelines:
  processed_data:               # Pipeline name (you choose this)
    source: my_source           # Which source to use
    table: my_table             # Which table from that source
    filters:                    # Filter rows (optional)
      - type: widget
        field: category
    transforms:                 # Transform data (optional)
      - type: columns
        columns: [col1, col2]
```

**Skip pipelines** if you want to display raw data without modification.

#### Filters

Filters let users drill down into data subsets:

```yaml
filters:
  - type: widget              # Interactive widget filter
    field: species            # Column to filter on
  - type: widget
    field: island
```

Common filter types:

| Type | Purpose | User control |
|------|---------|--------------|
| `widget` | Interactive dropdown/slider | Yes |
| `constant` | Fixed filter value | No |
| `facet` | Split data into groups | Yes |

#### Transforms

Transforms modify data structure or values:

```yaml
transforms:
  - type: columns             # Select specific columns
    columns: [id, name, value]
  - type: aggregate           # Group and aggregate
    method: mean
    by: [category]
```

Common transform types:

| Type | Purpose |
|------|---------|
| `columns` | Select specific columns |
| `aggregate` | Group and compute statistics |
| `sort` | Order rows |
| `query` | SQL-like filtering |

See the [Pipelines guide](pipelines.md) for complete filter and transform reference.

### Layouts: Visualizing data

The `layouts` section creates the visual dashboard. Each layout contains views:

```yaml
layouts:
  - title: My Dashboard         # Layout title
    pipeline: processed_data    # Or use source directly
    views:                      # List of visualizations
      - type: hvplot
        kind: scatter
        x: bill_length
        y: bill_depth
      - type: table
        page_size: 20
```

Link views to data using either:

- `pipeline: pipeline_name` - Use processed data
- `source: source_name` - Use raw data directly

#### View types

Common view types:

| Type | Purpose | Use case |
|------|---------|----------|
| `hvplot` | Interactive plots | Scatter, line, bar, hist, box, etc. |
| `table` | Data tables | Displaying tabular data |
| `indicator` | Single metrics | KPIs, status indicators |
| `download` | Download buttons | Data export |

Example with multiple views:

```yaml
layouts:
  - title: Analysis
    pipeline: my_pipeline
    layout: [[0], [1, 2]]       # Custom arrangement
    views:
      - type: hvplot            # View 0: Scatter plot
        kind: scatter
        x: x_col
        y: y_col
      - type: hvplot            # View 1: Histogram
        kind: hist
        y: x_col
      - type: table             # View 2: Data table
        page_size: 20
```

The `layout` parameter controls view arrangement:

- `[[0], [1, 2]]` - View 0 on top, views 1 and 2 side-by-side below
- `[[0, 1], [2, 3]]` - 2x2 grid

See the [Views guide](views.md) for all view types and layout options.

## Advanced sections

### Defaults: Parameter overrides

Set default values for components throughout your specification:

```yaml
defaults:
  filters:
    - type: widget
      multi: false            # Single-select by default
  views:
    - type: hvplot
      responsive: true        # All plots responsive
```

This prevents repetition when many components share settings.

### Variables: Dynamic configuration

Create reusable variables that can be referenced anywhere:

```yaml
variables:
  ticker:
    type: widget
    kind: TextInput
    value: AAPL

sources:
  stock_data:
    type: file
    tables:
      data: $variables.ticker  # References the variable
```

Variables make dashboards dynamic. See the [Variables guide](variables.md).

### Authentication

Restrict dashboard access to authorized users:

```yaml
auth:
  type: basic
  users:
    - alice
    - bob
```

See the [Authentication guide](authentication.md) for setup details.

## Execution flow

Understanding how Lumen executes helps debug issues:

1. **Initialization**: Lumen reads your YAML file
2. **Source loading**: Data loads from configured sources
3. **Pipeline execution**: 
   - Filters apply (based on widget values)
   - Transforms execute in order
4. **View rendering**: Views display the processed data
5. **User interaction**: Widget changes trigger re-execution from step 3

Pipeline execution is lazy—data only loads when needed for display.

## YAML syntax tips

### Indentation matters

YAML uses whitespace for structure. Use consistent indentation (2 spaces recommended):

```yaml
# Correct
sources:
  my_source:
    type: file
    
# Wrong - inconsistent indentation
sources:
my_source:
      type: file
```

### Lists use dashes

Create lists with leading dashes:

```yaml
filters:
  - type: widget
    field: category
  - type: widget
    field: region
```

### Quotes for special characters

Use quotes when values contain special characters:

```yaml
title: "Sales: Q4 Results"   # Colon requires quotes
table: my-table               # Hyphens are fine without quotes
```

## Common patterns

### Basic dashboard (source + view)

Minimal dashboard showing raw data:

```yaml
sources:
  data:
    type: file
    tables:
      table: data.csv

layouts:
  - title: Data
    source: data
    views:
      - type: table
        table: table
```

### Dashboard with filtering

Add interactive filters:

```yaml
sources:
  data:
    type: file
    tables:
      table: data.csv

pipelines:
  filtered:
    source: data
    table: table
    filters:
      - type: widget
        field: category

layouts:
  - title: Filtered Data
    pipeline: filtered
    views:
      - type: hvplot
        kind: bar
        x: name
        y: value
```

### Multi-view dashboard

Show data in multiple ways:

```yaml
sources:
  data:
    type: file
    tables:
      table: data.csv

pipelines:
  processed:
    source: data
    table: table
    filters:
      - type: widget
        field: region

layouts:
  - title: Dashboard
    pipeline: processed
    layout: [[0, 1], [2]]
    views:
      - type: hvplot
        kind: line
        x: date
        y: sales
      - type: hvplot
        kind: bar
        x: product
        y: sales
      - type: table
        page_size: 10
```

## Next steps

Now that you understand Lumen's structure:

1. **[Sources guide](sources.md)** - Learn to load data from different sources
2. **[Pipelines guide](pipelines.md)** - Master filters and transforms
3. **[Views guide](views.md)** - Create compelling visualizations
4. **[Variables guide](variables.md)** - Build dynamic dashboards

**Building a specific dashboard?** Use these concepts as reference while following individual guides for your specific needs.
