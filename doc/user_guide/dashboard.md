# Dashboard Specification

The Lumen dashboard can be configured using a `dashboard.yml` file. The core design principle behind the yaml specification is that it instantiates instances of the four main types, i.e. [Filter](lumen.filters.Filter), [Source](lumen.sources.Source), [Transform](lumen.transforms.Transform) and [View](lumen.views.View) objects.

For example the `type` declaration of a `Source` is matched against`source_type` and all the other keywords are passed as arguments to the class, instantiating the object with the provided configuration.

## Sections

We will start by breaking down the specification into individual sections.

### `config`

The `config` section provides general settings which apply to the whole dashboard.

```yaml
config:
  title: The title of the overall application
  layout: The layout to arrange the (sub-)layouts in ('grid', 'tabs', 'column')
  logo: A URL or local path to an image file
  sync_with_url: Whether to sync app state with URL
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

The `defaults` section allows overriding parameter defaults on the [Filter](lumen.filters.Filter), [Source](lumen.sources.Source), [Transform](lumen.transforms.Transform) and [View](lumen.views.View) objects.

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

As an example we may want to override the default `WidgetFilter.multi` value, because we want to query with a specific value rather than multiple values:

```yaml
defaults:
  filters:
    - type: widget
      multi: false
```

### `variables`

Variables are powerful components for declaring values that you want to use throughout your application by referencing throughout the YAML.


```yaml
variables:
  user:
    - type: url
  password:
    - type: env
      key: MY_PASSWORD_ENV_VAR
```

Once declared variables can be referenced through the rest of your dashboard specification by using the reference syntax (`$`) described below.

### `sources`

The `sources` section allows defining [Source](lumen.sources.Source) instances which can be referenced by the pipelines.

```yaml
sources:
  - type: The type of Source to instantiate
    ...: Additional parameters for the Source
```

As an example we may want to load a table of population data from a local CSV file using the [FileSource](lumen.sources.FileSource):

```yaml
sources:
  population:
    type: file
    files: [population.csv]
```

This will declare a `Source` called `population` which publishes a table called `population`.

It is also possible to declare filters associated with a global `Source`.

```yaml
sources:
  population:
    type: file
    files: [population.csv]
```

### `pipelines`

The `pipelines` section allows declaring reusable pipelines that encapsulate a [Source](lumen.sources.Source) along with any number of [Filter](lumen.filters.Filter) and [Transform](lumen.transforms.Transform) components. A pipeline can be declared in this global section or the definition can be inlined on each Layout or View.

```yaml
pipelines:
  population:
    source: population
    filters:
      - type: widget
        field: country
    transforms:
      - type: columns
        columns: [year, country, population]
```

### `layouts`

The layouts section defines the actual visual representations on the page and therefore makes up the meat of the declaration.

```yaml
layouts: This is the list of visual layouts to render.
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

### `auth`

The `auth` field may provide a dictionary of any number of fields which are validated against the user information provided the the Auth provider, which is made available by Panel in the `panel.state.user_info` dictionary. To discover how to configure an Auth provider with Panel/Lumen see the [Panel documentation](https://panel.holoviz.org/user_guide/Authentication.html).

As an example the GitHub OAuth provider returns the login of the user that is visiting the dashboard. If we add the following field to the yaml:

```yaml
auth:
  login: [philippjfr]
```

Lumen will check the current user `login` against all user logins listed here. For a more generic Auth mechanism many Auth providers, such as Okta, make it possible to configure a list of groups a user belongs to in which case you could list the allowed groups in the auth field.

## Special Syntax

To avoid repeating yourself the yaml specification supports some special syntax.

### References

#### Source References

In some scenarios you might want to refer to a `Source`, a table on a `Source` or a field on a table from elsewhere in the yaml specification.

As an example you may have local CSV file which contains a column of URLs to monitor and feed that information to a `WebsiteSource` which reports whether those URLs are live. Using the `$` syntax we can easily establish such references.

```yaml
sources:
  csv:
    type: file
    files: [websites.csv]
  live:
    type: live
    urls: "$csv.websites.url"
```

The `$csv.websites.url` syntax will look up a `Source` called 'csv', request a table called 'websites' and then feed the 'url' column in that table to the `urls` parameter of the `WebsiteSource`.

### Variable References

Variables are powerful components that allow you to link settings across your entire application. Once a `Variable` has been declared in the `variables:` section of the specification you can reference it throughout your application using the `$variables.<variabl-name>` syntax, e.g. you might have a `url` widget `Variable` that allows entering a URL using a `TextInput` widget:

```yaml
variables:
  url:
    type: widget
    kind: TextInput
    default: AAPL.csv
```

Once declared you can reference this variable using the `$variables.` syntax:

```yaml
sources:
  stock_data:
    type: file
    tables:
      ticker: $variables.url
```

Whenever the `url` variable is updated the `Source` will be refreshed and any views attached to that `Source` will be updated.

### Templating

In many cases you do not want to hardcode variables inside the yaml specification instead passing in variables from an environment variable, a shell command, a CLI argument, a HTTP request header or cookie, or a OAuth token variable. This can be achieved using the following templating syntax:

- `{{env("USER")}}`: look in the set environment variables for the named variable

- `{{shell("get_login thisuser -t")}}`: execute the command, and use the output as the value. The output will be trimmed of any trailing whitespace.

- `{{cookie("USER")}}`: look in the HTTP request cookies of the served application

- `{{header("USER")}}`: look in the HTTP request headers of the served application

- `{{oauth("USER")}}`: look in the OAuth user token

- `{{USER}}`: Arguments passed in using `--template-vars="{'USER': 'lumen_user'}"` when using `lumen serve` on the commandline.

## Local components

While Lumen ships with a wide range of components users may also define custom components in files which live alongside the `dashboard.yml` file. Specifically Lumen will automatically import `filters.py`, `sources.py`, `transforms.py` and `views.py` if these files exist alongside the dashboard specification.
