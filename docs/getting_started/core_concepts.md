# {octicon}`mortar-board;2em;sd-mr-1` Core Concepts

:::{admonition} What is the purpose of this page?
:class: important
This conceptual overview of creating a Lumen dashboard is meant to help you start generalizing what you acheived in the tutorial ([Build a dashboard](build_dashboard)). After this page you should start building your own dashboard. As you build, consult the relavant `How-to` guides or `Reference` pages, as needed.
:::

## Overview
Lumen is a framework for easily building powerful data-driven dashboards. The *ease* of Lumen comes from the primary interface being a simple `YAML` file, which reads like a cooking recipe. The *power* of Lumen comes from the ability to leverage different data intake, data processing, and data visualization libraries that are available in the Python datascience ecosystem, but without having to code!

## YAML specification
The YAML file is a simple and human readable recipe for building a dashboard. Briefly, YAML uses consistent whitespace to denote structure, and most lines either create an association (with `key: value`), or a list (with a leading hyphen ` - `). For more on YAML, we recommend this [quick YAML guide](https://learnxinyminutes.com/docs/yaml/).

The three primary sections of a Lumen specification file are:

::::{grid} 1
:gutter: 3

:::{grid-item-card} `sources`
to list your data sources
:::

::::

::::{grid} 1
:gutter: 3

:::{grid-item-card} `pipeline`
to specify how you want the data to be manipulated (`filtered` and `transformed`)
:::

::::

::::{grid} 1
:gutter: 3

:::{grid-item-card} `targets`
to create the views (e.g. table, plot) for your dashboard
:::

::::

These core sections should be unindented in the YAML file, as they are at the top of the hierarchy:

```{code-block} YAML
sources:
  ...:
    ...: ...
pipelines:
  ...:
    ...: ...
targets:
  - ...: ...
    ...: ...
```

There are some other top level sections that you may or may not need. We will get to these later on, but just to give you a taste:

::::{grid} 2
:gutter: 3

:::{grid-item-card} `config`
to apply settings for the whole dashboard
:::

:::{grid-item-card} `defaults`
to provide default parameters for other sections
:::

::::

::::{grid} 2
:gutter: 3

:::{grid-item-card} `variables`
to create variables to reference throughout the YAML
:::

:::{grid-item-card} `auth`
to add authentication to your dashboard
:::

::::

## Data sources
The `sources` section defines the [Source](lumen.sources.Source) of your data. Depending on your source type, the specification may look a bit different, but in general it will follow this pattern:

```{code-block} YAML
sources:
  source_name:
    type: Type of source
    ...: Additional source parameters
```

A common choice for a source type is [FileSource](lumen.sources.FileSource), which can load CSV, Excel, JSON and Parquet files from either local (filepaths) or remote (URL) locations. You might remember an implementation of this from what you created in the [Build an App](build_an_app) tutorial:

```{code-block} YAML
sources:
  penguin_source:
    type: file
    tables:
      table_penguin: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv
```

 See the [Source Reference](../architecture//source.html#:~:text=Source%20queries%20data.-,Source%20types%23,-class%20lumen.sources) for other source `types` and for the relevant `parameters`.

## Pipelines to manipulate data
The `pipelines` section is where you list all the ways that you want the data to be filtered or transformed. If you ever don't want the data to be manipulated, just exclude this section.

### Filters
A filter allows you or your dashboard's viewers to drill down into just a subset of the data.

```{code-block} YAML
pipelines:
  name:
    source: ...
    filters:
      - type: Type of filter
        ...: Filter Parameters
```

A very useful filter type is `widget`, which is used to expose filtering capability in the dashboard view. Here is the implementation from your tutorial, which specifies two widgets for dashboard viewers.

```{code-block} YAML
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

See the [Filter Reference](../architecture/filter) for other filter `types` and for the relevant `parameters`.

### Transforms
Within the pipeline section, you can also apply a `transform` to the data, such as selecting only certain columns of the data.

```{code-block} YAML
pipelines:
  name:
    source: ...
    transforms:
      - type: Type of transform
        ...: Transform Parameters
```

Here is the complete `pipelines` section from your tutorial, which include both filters and a transform:

```{code-block} YAML
:emphasize-lines: 10-12
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

See the [Transform Reference](../architecture/transform) for other transform `types` and for the relevant `parameters`.

## Dashboard views

The `targets` section defines how the dashboard is going to look. The essential structure of a `targets` section is as follows:

```{code-block} YAML
targets:
  - title: Dashboard title
    pipeline: The pipeline driving these views
    ...: Dashboard parameters
    views:
      - type: View type
        ...: View parameters
```

Each view can be of a different type, but a good starting point is the `hvPlotView`. This view type is allows you to produce [many different types of plots](https://hvplot.holoviz.org/reference/index.html) available from the [hvPlot](https://hvplot.holoviz.org/) library, just by specifying the `kind` parameter.

In your tutorial, the final dashboard included two `kinds` - scatter and histogram:

```{code-block} YAML
targets:
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
        height: 450
      - type: hvplot
        kind: hist
        y: bill_length_mm
        responsive: true
        height: 350
      - type: hvplot
        kind: hist
        y: bill_depth_mm
        responsive: true
        height: 350
```

See the [View Reference](../architecture/view) for other view `types` and for the relevant `parameters`.


## Other top level sections

### Config
Apply settings for the whole dashboard

### Defaults
Provide default parameters for other sections

### Variables
Create variables to reference throughout the YAML

### Authentication
Add authentication to your dashboard



## Where to go from here?
As mentioned at the beginning, you should now start building your own dashboards. As you build, consult the relavant `How-to` guides or `Reference` pages, as needed. At any point, if you want deeper discussion of Lumen to solidify your understanding, peruse the `Background` section.
