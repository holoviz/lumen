# {octicon}`mortar-board;2em;sd-mr-1` Core Concepts

```{admonition} What is the purpose of this page?
:class: important
This conceptual overview of creating a Lumen dashboard is meant to help you start generalizing what you achieved in the tutorial ([Build a dashboard](build_dashboard)). After this page you should start building your own dashboard. As you build, consult the relevant [How-to](../how_to/) guides or [Reference](../reference/) pages, as needed.
```

## Overview

Lumen is a framework for easily building powerful data-driven dashboards. The *ease* of Lumen comes from the primary interface being a simple **YAML** file, which reads like a cooking recipe. We call this the **specification**. Unlike a recipe Lumen makes it very easy to go back and forth between the specification and the final product. This makes it easy to interactively go back and forth and iteratively improve the specification. The *power* of Lumen comes from the ability to leverage different data intake, data processing, and data visualization libraries that are available in the Python data science ecosystem, but without having to code!

## YAML specification

The YAML file is a simple and human readable recipe for building a dashboard. Briefly, YAML uses consistent whitespace to denote structure, and most lines either create an association (with **key: value**), or a list (with a leading hyphen **-**). For more on YAML, we recommend this [quick YAML guide](https://learnxinyminutes.com/docs/yaml/).

The four primary sections of a Lumen specification file are:

::::{grid} 1
:gutter: 3

:::{grid-item-card} `config`
To apply settings for the whole dashboard
:::

::::

::::{grid} 1
:gutter: 3

:::{grid-item-card} `sources`
To list your data sources
:::

::::

::::{grid} 1
:gutter: 3

:::{grid-item-card} `pipelines`
To specify how you want the data to be manipulated (filtered and transformed)
:::

::::

::::{grid} 1
:gutter: 3

:::{grid-item-card} `layouts`
To create the views (e.g. table, plot) for your dashboard
:::

::::

These core sections should be not be indented in the YAML file, as they are at the top of the hierarchy:

```{code-block} YAML
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

In addition to these core sections, there is plenty of advanced functionality that we will preview towards the bottom of this page.

## Config

The `config` section provides general settings which apply to the whole dashboard to control things like the title, overall layout and theme. The structure is very simple:

```{code-block} YAML
config:
  title: My dashboard
  layout: tabs
  logo: assets/my_logo.png
  ...
```

You might remember an implementation of this from what you created in the [Build a dashboard](build_dashboard) tutorial. Your config section was used to add a title and dark theme to your dashboard:

```{code-block} YAML
config:
  title: Palmer Penguins
  theme: dark
```

See the [Config Reference](../reference/Config.html) for a description of all parameters.

## Sources

The `sources` section defines the source of your data. Depending on your source type, the specification may look a bit different, but in general it will follow this pattern:

```{code-block} YAML
sources:
  source_name:
    type: Type of source
    ...: Additional source parameters
```

A common choice for a source type is [FileSource](../reference/source/FileSource), which can load CSV, Excel, JSON and Parquet files from either local (filepaths) or remote (URL) locations. In your tutorial, you use a remote CSV source:

```{code-block} YAML
sources:
  penguin_source:
    type: file
    tables:
      table_penguin: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv
```

See the [Source Reference](../reference/source/index) for other source types and for the relevant parameters.

## Pipelines (data processing)

The `pipelines` section is where you list all the ways that you want the data to be **filtered** or **transformed**. If you don't need the data to be manipulated, you can just exclude this section.

### Filters

The `filters` of a `Pipeline` allows you or your dashboard's viewers to drill down into just a subset of the data.

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

See the [Filter Reference](../reference/filter/index) for other filter types and for the relevant parameters.

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

See the [Transform Reference](../reference/transform/index) for other transform types and for the relevant parameters.

## Layouts (views)

The `layouts` section declares the actual content in the rendered application. The essential structure of a `layouts` section is as follows:

```{code-block} YAML
layouts:
  - title: Dashboard title
    pipeline: The pipeline driving these views
    ...: Dashboard parameters
    views:
      - type: View type
        ...: View parameters
```

:::{dropdown} Expand this dropdown for a more complex example structure of `layouts`
:animate: fade-in-slide-down
```{code-block} yaml
layouts:
  - title: The title of the monitoring endpoint
    download:
      format: When specified adds a section to the sidebar allowing users to download the filtered dataset
      kwargs: Additional keyword arguments to pass to the pandas/dask to_<format> method
      tables: Allows declaring a subset of tables to download
    pipeline: The pipeline driving the views of this layout. Each View can independently declare a pipeline or all use the shared pipeline defined at the layout level
    views: A list of metrics to monitor and display on the endpoint
      - pipeline: The Pipeline driving the View
        type: The type of View to use for rendering the table
        ...: Additional parameters for the View
    layout: The layout inside the card(s), e.g. 'row', 'column' or 'grid'
    facet:
      by: List of fields to facet by
      sort: List of fields to sort by
      reverse: Whether to reverse the sort order
    refresh_rate: How frequently to poll for updates in milliseconds
    ...: Additional parameters passed to the Card layout(s), e.g. width or height
```
:::

At minimum each `Layout` must declare a `title` and a set of `views`. Each view can be of a different type, but a good starting point is the `hvPlotView`. This view type allows you to produce [many different types of plots](https://hvplot.holoviz.org/reference/index.html) available from the [hvPlot](https://hvplot.holoviz.org/) library, just by specifying the `kind` parameter.

In your tutorial, the final dashboard included two `kinds` - scatter and histogram:

```{code-block} YAML
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

See the [View Reference](../reference/view/index) for other view types and for the relevant parameters.

## Advanced Functionality

The following sections are meant to introduce you some of Lumen's advanced functionality. If you want to implement some of these features, explore their associated [How to](../how_to/index) guides for complete recipes.

### Defaults

The `defaults` section allows overriding parameter defaults on the [Filter](../reference/filter/index), [Source](../reference/source/index), [Transform](../reference/transform/index) and [View](../reference/view/index) objects.

```{code-block} YAML
defaults:
  filters:
    - type: Type to override the default on
      ...: Parameter to override
  sources:
    - type: Type to override the default on
      ...: Parameter to override
  transforms:
    - type: Type to override the default on
      ...: Parameter to override
  views:
    - type: Type to override the default on
      ...: Parameter to override
```

You haven't seen an implementation of this yet, but as an example, we could have used the `defaults` section to override the default `WidgetFilter.multi` value because perhaps we wanted the widget to filter with a specific value rather than multiple values:

```{code-block} YAML
defaults:
  filters:
    - type: widget
      multi: false
```

For more on `defaults`, check out the `How to override parameter defaults` guide.

### Variables

The `variables` sections allow you to link settings across your entire application. Once a variable has been declared in the `variables` section, you can reference it throughout your specification using the `$variables.<variable_name>` syntax.

```{code-block} YAML
variables:
  variable_name:
    - type: ...
      ...: Variable parameters
```

 For example, you might have a `TextInput` widget where a dashboard user could enter the name of a CSV file with stock data. Establishing such a widget as a `Variable` would allow the dashboard to dynamically update with the new data source.

```{code-block} YAML
:emphasize-lines: 1-5, 11
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

For more on variables, check out the [How to use variables and references](../how_to/general/variables_and_references) guide.

### Sources as variables

In addition to the `variables` section, in which you can create arbitrary types of variables, you can also refer to sources with a similar syntax. In some scenarios you might want to refer to a `Source`, a table on a `Source` or a field on a table from elsewhere in the yaml specification.

As an example you may have local CSV file which contains a column of URLs to monitor and feed that information to a `WebsiteSource` which reports whether those URLs are live. Using the `$` syntax we can easily establish such references.

```yaml
sources:
  csv:
    type: file
    files: [websites.csv]
  live:
    type: live
    urls: $csv.websites.url
```

The `$csv.websites.url` syntax will look up a `Source` called 'csv', request a table called 'websites' and then feed the 'url' column in that table to the `urls` parameter of the `WebsiteSource`.

For more on referring to sources, check out the [How to use variables and references](../how_to/variables_and_references) guide.

### External variables (templating)

In many cases you do not want to hardcode variables inside the yaml specification instead passing in variables from an environment variable, a shell command, a CLI argument, a HTTP request header or cookie, or a OAuth token variable. This can be achieved using the following templating syntax:

- `{{env("USER")}}`: look in the set environment variables for the named variable

- `{{shell("get_login thisuser -t")}}`: execute the command, and use the output as the value. The output will be trimmed of any trailing whitespace.

- `{{cookie("USER")}}`: look in the HTTP request cookies of the served application

- `{{header("USER")}}`: look in the HTTP request headers of the served application

- `{{oauth("USER")}}`: look in the OAuth user token

- `{{USER}}`: Arguments passed in using `--template-vars="{'USER': 'lumen_user'}"` when using `lumen serve` on the commandline.

For more on templating, check out the [How to use variables and references](../how_to/variables_and_references) guide.

### Authentication

The `auth` field may provide a dictionary of any number of fields which are validated against the user information provided the the Auth provider, which is made available by Panel in the `panel.state.user_info` dictionary. To discover how to configure an Auth provider with Panel/Lumen see the [Panel documentation](https://panel.holoviz.org/user_guide/Authentication.html).

As an example the GitHub OAuth provider returns the login of the user that is visiting the dashboard. If we add the following field to the yaml:

```yaml
auth:
  login: [philippjfr]
```

Lumen will check the current user `login` against all user logins listed here. For a more generic Auth mechanism many Auth providers, such as Okta, make it possible to configure a list of groups a user belongs to in which case you could list the allowed groups in the auth field.

For more on Authentication, check out the `How to set up authentication` guide.

## Where to go from here?

As mentioned at top of this page, you should now start building your own dashboards. As you build, consult the relevant `How-to` guides or `Reference` pages, as needed. At any point, if you want deeper discussion of Lumen to solidify your understanding, take a look at the `Background` section.
