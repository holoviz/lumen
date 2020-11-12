# Dashboard specification

The Lumen dashboard can be configured using a `dashboard.yml`
file. The core design principle behind the yaml specification is that
it instantiates instances of the four main types, i.e.
[Filter](lumen.filters.Filter), [Source](lumen.sources.Source),
[Transform](lumen.transforms.Transform) and [View](lumen.views.View)
objects.

For example the `type` declaration of a `Source` is matched against
`source_type` and all the other keywords are passed as arguments to
the class, instantiating the object with the provided configuration.

## Sections

We will start by breaking down the specification into individual sections.

### `config`

The `config` section provides general settings which apply to the whole dashboard.

```yaml
config:
  title: The title of the overall application
  layout: The layout to put the targets in ('grid', 'tabs', 'column')
  logo: A URL or local path to an image file
  template: The template to use for the monitoring application
  ncols: The number of columns to use in the grid of cards
```

A simple example `config` might look like this:

```yaml
config:
  title: "Example Application"
  layout: grid
  logo: example.png
  ncols: 3
```

### `defaults`

The `defaults` section allows overriding parameter defaults on the
[Filter](lumen.filters.Filter), [Source](lumen.sources.Source),
[Transform](lumen.transforms.Transform) and [View](lumen.views.View)
objects.

```yaml
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

As an example we may want to override the default `WidgetFilter.multi`
value, because we want to query with a specific value rather than
multiple values:

```yaml
defaults:
  filters:
    - type: widget
      multi: false
```

### `sources`

The `sources` section allows defining [Source](lumen.sources.Source)
instances which can be referenced from the monitoring targets. This is
useful when reusing a `Source` across multiple targets.

```yaml
sources:
  - type: The type of Source to instantiate
    ...: Additional parameters for the Source
```

As an example we may want to load a table of population data from a
local CSV file using the [FileSource](lumen.sources.FileSource):

```yaml
sources:
  population:
    type: file
    files: [population.csv]
```

This will declare a `Source` called `population` which publishes a table called `population`.

### `targets`

The targets section defines the actual monitoring targets and
therefore makes up the meat of the declaration.

```yaml
targets: This is the list of targets to monitor
  - title: The title of the monitoring endpoint
    source: The Source used to monitor an endpoint (may also reference a Source in the sources section
      type: The type of Source to use, e.g. 'rest' or 'live'
      ...: Additional parameters for the Source
    views: A list of metrics to monitor and display on the endpoint
      - table: The name of the table to visualize
        type: The type of View to use for rendering the table
        ...: Additional parameters for the View
    filters: A list of Filter types to select a subset of the data
      - field: The name of the filter
        type: The type of the Filter to use, e.g. 'constant', 'widget' or 'facet'
        ...: Additional parameters for the Filter
    layout: The layout of the card(s), e.g. 'row', 'column' or 'grid'
    refresh_rate: How frequently to poll for updates in milliseconds
    ...: Additional parameters passed to the Card layout(s), e.g. width or height
```

## Special Syntax

To avoid repeating yourself the yaml specification supports some special syntax.

### Source References

In some scenarios you might want to refer to a `Source`, a table on a
`Source` or a field on a table from elsewhere in the yaml
specification.

As an example you may have local CSV file which contains a column of
URLs to monitor and feed that information to a `WebsiteSource` which
reports whether those URLs are live. Using the `@` syntax we can
easily establish such references.

```
sources:
  csv:
    type: file
    files: [websites.csv]
  live:
    type: live
    urls: "@csv.websites.url"
```

The `@csv.websites.url` syntax will look up a `Source` called 'csv',
request a table called 'websites' and then feed the 'url' column in
that table to the `urls` parameter of the `WebsiteSource`.
