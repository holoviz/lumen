# Use variables and references

Variables are one of Lumen's most powerful features. This guide shows three ways to use them.

## Variables section

Define variables in the `variables` section and reference them with `$variables.NAME`. This example uses a MultiSelect widget to control which data columns display:

=== "YAML"

    ```yaml
    variables:
      columns:
        type: widget
        kind: MultiSelect
        value: [Open, High, Low, Close, Volume, Adj Close]
        options: [Open, High, Low, Close, Volume, Adj Close]
        size: 7

    sources:
      stock_data:
        type: file
        tables:
          ticker: https://raw.githubusercontent.com/matplotlib/sample_data/master/aapl.csv
        kwargs:
          index_col: Date
          parse_dates: [Date]

    pipelines:
      ticker_pipe:
        source: stock_data
        table: ticker
        transforms:
          - type: columns
            columns: $variables.columns

    layouts:
      - title: Plot
        pipeline: ticker_pipe
        views:
          - type: hvplot
            table: ticker
    ```

=== "All columns"

    Showing all data columns

=== "Selected columns"

    Showing only the selected columns

## Source references

Reference sources using `$source_name` syntax. You can reference:

1. The source itself: `$csv`
2. A table on the source: `$csv.websites`
3. Unique values in a field: `$csv.websites.url`

This is useful when one source feeds another. Here, website URLs from a CSV feed a live status checker:

=== "YAML"

    ```yaml
    sources:
      csv:
        type: file
        files: [websites.csv]
      live:
        type: live
        urls: $csv.websites.url

    layouts:
      - title: Status of websites
        source: live
        views:
          - type: table
    ```

=== "websites.csv"

    ```csv
    url
    https://google.com
    https://python.org
    https://anaconda.com
    not-alive
    ```

## External variables (templating)

Pass variables from environment, CLI, cookies, headers, or OAuth tokens using Jinja2 syntax:

| Method | Syntax |
|--------|--------|
| Environment variable | `{{ env("USER") }}` |
| Request cookie | `{{ cookie("USER") }}` |
| Request header | `{{ header("USER") }}` |
| OAuth token | `{{ oauth("USER") }}` |

### CLI arguments

Pass variables via command-line with `--template-vars`:

=== "YAML"

    ```yaml
    sources:
      source:
        type: file
        tables:
          table: https://datasets.holoviz.org/penguins/v1/penguins.csv

    layouts:
      - title: Hello {{ USER }}
        source: source
        views:
          - type: table
            table: table
    ```

=== "Command"

    ```bash
    lumen serve dashboard.yaml --template-vars="{'USER': 'lumen_user'}"
    ```

### Shell commands

Execute shell commands with `{{ shell(COMMAND) }}`:

=== "YAML"

    ```yaml
    sources:
      source:
        type: file
        tables:
          table: https://datasets.holoviz.org/penguins/v1/penguins.csv

    layouts:
      - title: {{ shell("echo hello from the shell") }}
        source: source
        views:
          - type: table
            table: table
    ```

!!! tip
    Use `$` for internal variable references (resolved dynamically), and `{{ }}` for external variables (resolved on initialization).
