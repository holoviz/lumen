# Core concepts

!!! important
    This page helps you generalize what you learned in [Build your first dashboard](build_dashboard.md). After understanding these concepts, start building your own dashboard. Refer to the other guides in this section as needed.

## Overview

Lumen is a framework for building powerful data-driven dashboards. The power comes from leveraging the Python data science ecosystem—without coding. The ease comes from a simple **YAML** file interface that reads like a recipe.

Unlike traditional code-based approaches, Lumen makes it easy to iterate between the specification and the final dashboard. Modify your YAML, refresh your browser, and see the results instantly.

## YAML specification

A Lumen specification is a human-readable YAML file. YAML uses consistent whitespace for structure—lines either create associations with **key: value** or lists with a leading **-**. Learn more in the [YAML guide](https://learnxinyminutes.com/docs/yaml/).

The four primary sections of a Lumen specification are:

**config**
: Apply settings for the whole dashboard

**sources**
: List your data sources

**pipelines**
: Specify data manipulation (filtering and transforming)

**layouts**
: Create views (tables, plots) for your dashboard

These sections should not be indented in your YAML:

```yaml
config:
  ...: ...
sources:
  ...:
    ...: ...
pipelines:
  ...:
    ...: ...
layouts:
  - ...: ...
    ...: ...
```

## Config

The config section provides settings that apply to the entire dashboard. Use it to set the title, layout, theme, and other global options:

```yaml
config:
  title: My dashboard
  layout: tabs
  logo: assets/my_logo.png
  theme: dark
```

In the [Build a dashboard](build_dashboard.md) tutorial, you used the config section to set a title and enable dark mode:

```yaml
config:
  title: Palmer Penguins
  theme: dark
```

See the Config documentation for all available parameters.

## Sources

The `sources` section defines where your data comes from. Each source follows this pattern:

```yaml
sources:
  source_name:
    type: Type of source
    ...: Additional parameters
```

A common choice is FileSource, which loads CSV, Excel, JSON, and Parquet files from local or remote locations:

```yaml
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv
```

See the documentation for available source types and parameters.

## Pipelines (data processing)

The `pipelines` section defines how to filter and transform your data. If you don't need data manipulation, you can omit this section.

### Filters

Filters allow dashboard viewers to drill down into subsets of data:

```yaml
pipelines:
  pipeline_name:
    source: source_name
    table: table_name
    filters:
      - type: Type of filter
        ...: Filter parameters
```

A useful filter is `widget`, which exposes filtering capability in the dashboard. From the tutorial, these filters let viewers filter by sex and island:

```yaml
pipelines:
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    filters:
      - type: widget
        field: sex
      - type: widget
        field: island
```

See the documentation for other filter types.

### Transforms

Transforms manipulate data—for example, selecting specific columns:

```yaml
pipelines:
  pipeline_name:
    source: source_name
    transforms:
      - type: Type of transform
        ...: Transform parameters
```

Here's the complete pipeline from the tutorial, combining filters and a transform:

```yaml
pipelines:
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    filters:
      - type: widget
        field: sex
      - type: widget
        field: island
    transforms:
      - type: columns
        columns: ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
```

See the documentation for other transform types.

## Layouts (views)

The `layouts` section declares the visual content of your dashboard:

```yaml
layouts:
  - title: Dashboard title
    pipeline: Pipeline driving these views
    views:
      - type: View type
        ...: View parameters
```

At minimum, each layout needs a `title` and a list of `views`. A good starting point is hvPlotView, which lets you create many [plot types](https://hvplot.holoviz.org/reference/index.html) by setting the `kind` parameter.

From the tutorial, the final dashboard used two view types—scatter and histogram:

```yaml
layouts:
  - title: Penguins
    pipeline: penguin_pipeline
    layout: [[0], [1, 2]]
    sizing_mode: stretch_width
    height: 800
    views:
      - type: hvplot
        x: bill_length_mm
        y: bill_depth_mm
        kind: scatter
        color: species
        responsive: true
        height: 400
      - type: hvplot
        kind: hist
        y: bill_length_mm
        responsive: true
        height: 300
      - type: table
        show_index: false
        height: 300
```

See the documentation for other view types.

## Advanced functionality

Lumen supports advanced features—explore the other guides in this section for complete recipes.

### Defaults

The defaults section lets you override parameter defaults for filters, sources, transforms, and views:

```yaml
defaults:
  filters:
    - type: widget
      multi: false
  views:
    - type: hvplot
      responsive: true
```

### Variables

The variables section lets you create parameters that can be referenced throughout your specification using `$variables.<variable_name>`:

```yaml
variables:
  ticker:
    type: widget
    kind: TextInput
    default: AAPL.csv

sources:
  stock_data:
    type: file
    tables:
      ticker: $variables.ticker
```

### Source references

You can reference sources and their fields using the `$` syntax. This is useful when data from one source feeds another:

```yaml
sources:
  csv:
    type: file
    files: [websites.csv]
  live:
    type: live
    urls: $csv.websites.url
```

### External variables (templating)

Pass variables from environment variables, shell commands, cookies, headers, or OAuth tokens without hardcoding them in your specification:

- `{{env("USER")}}`: environment variable
- `{{shell("get_login thisuser -t")}}`: shell command output
- `{{cookie("USER")}}`: HTTP request cookie
- `{{header("USER")}}`: HTTP request header
- `{{oauth("USER")}}`: OAuth token variable
- `{{USER}}`: command-line argument via `lumen serve --template-vars="{'USER': 'value'}"`

### Authentication

The auth field validates users against configured providers. For example, allow only specific GitHub users:

```yaml
auth:
  login: [philippjfr, otheruser]
```

See the [Panel authentication documentation](https://panel.holoviz.org/user_guide/Authentication.html) for more details.

## Next steps

Now that you understand these concepts, start building your own dashboard. Refer to the other guides in this section as needed.
