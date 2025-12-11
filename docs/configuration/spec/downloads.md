# :material-download: Enabling data downloads

Let users export dashboard data in various formats.

## Download locations

Add download capability in three ways:

| Location | Configuration | Description |
|----------|---------------|-------------|
| **Sidebar** | `download:` in layout | Button in sidebar |
| **View** | `download` view type | Dedicated download component |
| **Table** | `download:` in table view | Integrated with table |

## Sidebar download

Add a download button to the layout sidebar:

```yaml
sources:
  penguins:
    type: file
    tables:
      data: https://datasets.holoviz.org/penguins/v1/penguins.csv

layouts:
  - title: Penguin Data
    source: penguins
    download: csv              # Adds download button to sidebar
    views:
      - type: table
        table: data
```

Users click the download button in the sidebar to export current (filtered) data.

## Download view

Create a dedicated download component:

```yaml
sources:
  penguins:
    type: file
    tables:
      data: https://datasets.holoviz.org/penguins/v1/penguins.csv

layouts:
  - title: Penguin Data
    source: penguins
    views:
      - type: download         # Download view
        format: csv
      - type: table
        table: data
```

The download view appears as a button within the main content area.

## Table download

Integrate download directly into table views:

```yaml
sources:
  penguins:
    type: file
    tables:
      data: https://datasets.holoviz.org/penguins/v1/penguins.csv

layouts:
  - title: Penguin Data
    source: penguins
    views:
      - type: table
        table: data
        download: csv          # Adds download to table controls
```

Download button appears in the table's control bar.

## Download formats

Supported export formats:

| Format | Extension | Best for |
|--------|-----------|----------|
| `csv` | .csv | Spreadsheets, most tools |
| `xlsx` | .xlsx | Excel workbooks |
| `json` | .json | Web APIs, JavaScript |

### CSV downloads

```yaml
layouts:
  - title: Data
    source: my_source
    download: csv
    views:
      - type: download
        format: csv
```

### Excel downloads

```yaml
layouts:
  - title: Data
    source: my_source
    download: xlsx
    views:
      - type: download
        format: xlsx
```

### JSON downloads

```yaml
layouts:
  - title: Data
    source: my_source
    download: json
    views:
      - type: download
        format: json
```

## Customizing download buttons

Configure button appearance and behavior:

```yaml
views:
  - type: download
    format: csv
    button_type: primary       # primary, success, warning, danger
    button_text: "Export Data"
    filename: "my_data.csv"    # Custom filename
```

Button types:

| Type | Appearance |
|------|------------|
| `default` | Standard button |
| `primary` | Blue/highlighted |
| `success` | Green |
| `warning` | Orange |
| `danger` | Red |

## Combined approach

Use multiple download methods together:

```yaml
sources:
  sales:
    type: file
    tables:
      data: sales.csv

pipelines:
  filtered:
    source: sales
    table: data
    filters:
      - type: widget
        field: region
      - type: widget
        field: year

layouts:
  - title: Sales Dashboard
    pipeline: filtered
    download: csv              # Sidebar download
    views:
      - type: download         # Dedicated download view
        format: xlsx
        button_type: success
        button_text: "Export to Excel"
      - type: table
        show_index: false
        download: csv          # Table download
      - type: hvplot
        kind: bar
        x: product
        y: revenue
```

This provides three download options:
1. CSV from sidebar
2. Excel from dedicated button  
3. CSV from table

## Download behavior

### What gets downloaded

Downloads export the **current filtered data**:

- If users applied filters, only filtered data exports
- If pipeline has transforms, transformed data exports
- Raw source data is never directly accessible

### File naming

Default filename format: `{table_name}_{timestamp}.{format}`

Custom filenames:

```yaml
views:
  - type: download
    format: csv
    filename: "sales_report_2024.csv"
```

## Python API

Enable downloads programmatically:

```python
from lumen.views import Download
import lumen as lm

pipeline = lm.Pipeline.from_spec({...})

download_view = Download(
    pipeline=pipeline,
    format='csv',
    button_type='primary',
    button_text='Export Data'
)

# Include in layout
import panel as pn

pn.Column(
    pipeline.control_panel,
    download_view,
    lm.views.Table(pipeline=pipeline)
)
```

## Next steps

- **[Views guide](views.md)** - Learn about table views and other visualizations
- **[Pipelines guide](pipelines.md)** - Filter data before download
- **[Deployment guide](deployment.md)** - Deploy dashboards with downloads
