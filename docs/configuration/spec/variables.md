# Dynamic configuration with variables

Variables make dashboards dynamic by parameterizing configuration values.

## Variable types

Lumen supports three types of variables:

| Type | Syntax | Scope | Evaluated |
|------|--------|-------|-----------|
| **Internal variables** | `$variables.name` | Within spec | Dynamically at runtime |
| **Source references** | `$source.table.field` | Within spec | Dynamically at runtime |
| **External variables** | `{{env("VAR")}}` | From environment | Once at initialization |

Use `$` for internal dynamic references. Use `{{}}` for external values resolved at startup.

## Internal variables

Internal variables create reusable parameters within your specification.

### Variables section

Define variables in the `variables` section:

```yaml
variables:
  my_variable:
    type: widget              # Makes it interactive
    kind: Select              # Widget type
    value: default_value      # Initial value
    options: [opt1, opt2]     # Available choices
```

Reference them anywhere using `$variables.name`:

```yaml
sources:
  data:
    type: file
    tables:
      table: $variables.my_variable  # Uses the variable value
```

### Widget-driven variables

Create interactive variables with widgets:

=== "YAML"

    ```yaml
    variables:
      columns:
        type: widget
        kind: MultiSelect
        value: [Open, High, Low, Close]
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
            columns: $variables.columns    # Uses selected columns

    layouts:
      - title: Stock Data
        pipeline: ticker_pipe
        views:
          - type: hvplot
            kind: line
    ```

=== "Effect"

    - Widget appears in sidebar for selecting columns
    - Changing selection updates the plot
    - Only selected columns pass through pipeline

### Variable widget types

Common widget types for variables:

| Widget | Purpose | Example |
|--------|---------|---------|
| `TextInput` | Free text entry | File names, search terms |
| `Select` | Single selection | Category, region |
| `MultiSelect` | Multiple selections | Columns, tags |
| `IntSlider` | Integer range | Years, counts |
| `FloatSlider` | Float range | Thresholds, ratios |
| `DatePicker` | Date selection | Start/end dates |
| `Checkbox` | Boolean toggle | Enable/disable features |

### Text input variable

```yaml
variables:
  ticker:
    type: widget
    kind: TextInput
    value: AAPL.csv
    placeholder: Enter ticker symbol

sources:
  stock_data:
    type: file
    tables:
      data: $variables.ticker  # Uses entered filename
```

### Slider variable

```yaml
variables:
  year:
    type: widget
    kind: IntSlider
    value: 2023
    start: 2020
    end: 2025
    step: 1

pipelines:
  filtered:
    source: data_source
    table: data
    filters:
      - type: constant
        field: year
        value: $variables.year  # Filters by slider value
```

### Date picker variable

```yaml
variables:
  start_date:
    type: widget
    kind: DatePicker
    value: '2023-01-01'
  end_date:
    type: widget
    kind: DatePicker
    value: '2023-12-31'

pipelines:
  date_filtered:
    source: data_source
    table: data
    filters:
      - type: constant
        field: date
        value: {start: $variables.start_date, end: $variables.end_date}
```

## Source references

Reference data from sources to create dynamic relationships.

### Reference syntax

Source references use `$` notation:

| Syntax | References |
|--------|------------|
| `$source_name` | The entire source object |
| `$source_name.table_name` | A specific table from the source |
| `$source_name.table_name.column` | Unique values from a column |

### Column value references

Reference unique values from a column:

=== "YAML"

    ```yaml
    sources:
      websites_csv:
        type: file
        tables:
          websites: websites.csv
      
      live_checker:
        type: live
        urls: $websites_csv.websites.url  # Gets unique URLs from CSV

    layouts:
      - title: Website Status
        source: live_checker
        views:
          - type: table
    ```

=== "websites.csv"

    ```csv
    url
    https://google.com
    https://python.org
    https://anaconda.com
    ```

=== "Effect"

    The `live` source checks status of all URLs found in the CSV's `url` column.

### Chaining references

Use one source's data to configure another:

```yaml
sources:
  config:
    type: file
    tables:
      settings: config.csv

  data:
    type: file
    tables:
      main: $config.settings.data_path  # Path from config
    cache_dir: $config.settings.cache_dir
```

### Dynamic source selection

Reference sources in pipelines:

```yaml
variables:
  data_source:
    type: widget
    kind: Select
    value: source_a
    options: [source_a, source_b, source_c]

pipelines:
  dynamic:
    source: $variables.data_source  # Uses selected source
    table: data
```

## External variables (templating)

External variables inject values from outside the specification using Jinja2 syntax.

### Environment variables

Read from system environment:

=== "YAML"

    ```yaml
    sources:
      database:
        type: duckdb
        uri: {{ env("DATABASE_URL") }}  # From environment

    layouts:
      - title: Data for {{ env("USER") }}
        source: database
        views:
          - type: table
    ```

=== "Setting environment"

    ```bash
    export DATABASE_URL="postgresql://localhost/mydb"
    export USER="alice"
    lumen serve dashboard.yaml
    ```

### Command-line arguments

Pass variables when serving:

=== "YAML"

    ```yaml
    sources:
      data:
        type: file
        tables:
          table: {{ DATA_FILE }}  # From command line

    layouts:
      - title: Dashboard for {{ USER }}
        source: data
        views:
          - type: table
    ```

=== "Command"

    ```bash
    lumen serve dashboard.yaml --template-vars="{'DATA_FILE': 'sales.csv', 'USER': 'Bob'}"
    ```

### Shell commands

Execute shell commands for values:

=== "YAML"

    ```yaml
    sources:
      data:
        type: file
        tables:
          table: data.csv

    layouts:
      - title: {{ shell("echo 'Dashboard created on $(date +%Y-%m-%d)'") }}
        source: data
        views:
          - type: table
    ```

=== "Result"

    Title becomes: "Dashboard created on 2024-01-15"

### HTTP cookies

Read from request cookies:

```yaml
layouts:
  - title: Dashboard for {{ cookie("username") }}
    source: data
    views:
      - type: table
```

### HTTP headers

Read from request headers:

```yaml
sources:
  data:
    type: file
    tables:
      table: data_{{ header("X-Region") }}.csv  # Region-specific file
```

### OAuth tokens

Access OAuth user information:

```yaml
config:
  auth:
    type: oauth

layouts:
  - title: Dashboard for {{ oauth("user") }}
    source: data
    views:
      - type: table
```

## Variable scope and resolution

### Resolution timing

| Variable Type | When Resolved |
|---------------|---------------|
| External (`{{}}`) | Once at dashboard initialization |
| Internal (`$`) | Dynamically on each update |

Example:

```yaml
variables:
  threshold:
    type: widget
    value: 100

sources:
  data:
    type: file
    tables:
      # Resolved once at startup
      table: {{ env("DATA_FILE") }}

pipelines:
  filtered:
    source: data
    table: table
    transforms:
      - type: query
        # Resolved dynamically when threshold changes
        query: "value > $variables.threshold"
```

### Variable precedence

When multiple variable sources exist:

1. External variables resolve first (initialization)
2. Internal variables resolve during execution
3. Later references can use earlier variables

```yaml
# This works:
variables:
  base_path:
    type: widget
    value: {{ env("DATA_DIR") }}  # External first

sources:
  data:
    type: file
    tables:
      table: $variables.base_path/data.csv  # Then internal
```

## Common patterns

### User-selectable data file

```yaml
variables:
  dataset:
    type: widget
    kind: Select
    value: penguins
    options: [penguins, iris, stocks]

sources:
  data:
    type: file
    tables:
      table: https://datasets.holoviz.org/$variables.dataset/v1/$variables.dataset.csv

layouts:
  - title: $variables.dataset Analysis
    source: data
    views:
      - type: table
        table: table
```

### Dynamic filtering threshold

```yaml
variables:
  min_value:
    type: widget
    kind: IntSlider
    value: 100
    start: 0
    end: 1000
    step: 10

pipelines:
  filtered:
    source: data_source
    table: data
    transforms:
      - type: query
        query: "sales >= $variables.min_value"

layouts:
  - title: Sales Above $variables.min_value
    pipeline: filtered
    views:
      - type: hvplot
        kind: bar
```

### Configuration from file

=== "config.csv"

    ```csv
    setting,value
    data_path,/data/sales.csv
    cache_enabled,true
    theme,dark
    ```

=== "dashboard.yaml"

    ```yaml
    sources:
      config:
        type: file
        tables:
          settings: config.csv
      
      data:
        type: file
        tables:
          sales: $config.settings.data_path
        cache_dir: cache

    config:
      theme: $config.settings.theme

    layouts:
      - title: Sales Dashboard
        source: data
        views:
          - type: table
    ```

### Multi-source federation

```yaml
sources:
  source_list:
    type: file
    tables:
      sources: sources.csv  # Contains: name, url

  federated:
    type: file
    tables: $source_list.sources.name  # Table per row
    urls: $source_list.sources.url     # URLs from CSV

layouts:
  - title: All Sources
    source: federated
    views:
      - type: table
```

### Environment-specific configuration

```yaml
config:
  title: {{ env("APP_NAME", "My Dashboard") }}  # With default
  
sources:
  data:
    type: {{ env("DB_TYPE", "file") }}
    uri: {{ env("DATABASE_URL") }}
    cache_dir: {{ env("CACHE_DIR", ".cache") }}

layouts:
  - title: {{ env("ENVIRONMENT") }} Environment
    source: data
    views:
      - type: table
```

## Best practices

### When to use each type

| Use Case | Variable Type | Why |
|----------|---------------|-----|
| User selections | Internal (`$variables`) | Dynamic updates |
| Data relationships | Source references (`$source`) | Automatic synchronization |
| Deployment config | External (`{{env()}}`) | Separation of concerns |
| Secrets | External (`{{env()}}`) | Security |
| Dynamic paths | Both | Flexibility |

### Security considerations

!!! danger "Never hardcode secrets"
    Don't put API keys, passwords, or tokens directly in YAML:
    
    ```yaml
    # ❌ Bad - secret in file
    sources:
      api:
        token: "secret-api-key-12345"
    
    # ✅ Good - secret from environment
    sources:
      api:
        token: {{ env("API_KEY") }}
    ```

Set secrets via environment variables:

```bash
export API_KEY="secret-api-key-12345"
lumen serve dashboard.yaml
```

### Variable naming

Use clear, descriptive names:

```yaml
# ❌ Bad - unclear
variables:
  v1:
    type: widget
    value: 100

# ✅ Good - descriptive
variables:
  sales_threshold:
    type: widget
    value: 100
```

### Default values

Always provide sensible defaults:

```yaml
variables:
  year:
    type: widget
    kind: IntSlider
    value: 2023        # Default value
    start: 2020
    end: 2025
```

## Troubleshooting

### Variable not updating

**Problem**: Dashboard doesn't reflect variable changes

**Solutions**:
- Check reference uses `$` not `{{}}` for dynamic updates
- Verify variable name matches exactly (case-sensitive)
- Ensure variable is in `variables` section
- Check browser console for errors

### Template variable not found

**Problem**: `{{env("VAR")}}` fails with "variable not found"

**Solutions**:
- Set environment variable before running
- Provide default: `{{env("VAR", "default")}}`
- Check variable name spelling
- On Windows, use `set VAR=value` instead of `export`

### Circular references

**Problem**: Variables reference each other in a loop

**Solution**: Restructure to avoid circles:

```yaml
# ❌ Bad - circular
variables:
  a: $variables.b
  b: $variables.a

# ✅ Good - linear dependency
variables:
  base: 100
  derived: $variables.base * 2
```

### Source reference empty

**Problem**: `$source.table.column` returns no values

**Solutions**:
- Verify source loads data successfully
- Check table and column names are correct
- Ensure column has non-null values
- Preview source data first

## Next steps

Now that you can use variables:

- **[Customization guide](customization.md)** - Build custom variable types
- **[Deployment guide](deployment.md)** - Manage variables in production
- **[Python API guide](python-api.md)** - Work with variables programmatically
