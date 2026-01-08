# :material-file-document-edit: Building a Dashboard with YAML

Build a complete Lumen dashboard using YAML specifications. This tutorial creates an interactive penguin data explorer with filters and multiple visualizations.

## Final result

![Build dashboard final result](../../assets/build_app_07.png)

**Time**: 15 minutes

## What you'll build

An interactive dashboard with filtering, multiple chart types, and responsive layout. The tutorial follows nine steps:

1. **Create a YAML file** - Set up the specification
2. **Add a data source** - Load the Palmer Penguins dataset
3. **Add a table view** - Display raw data
4. **Create a plot** - Basic visualization
5. **Configure a scatter plot** - Specific axes and colors
6. **Add filters and transforms** - Interactive filtering and column selection
7. **Add multiple views** - Histogram and table alongside scatter plot
8. **Improve the layout** - Responsive sizing and positioning
9. **Add title and theme** - Final polish

Each step shows the YAML code and resulting output.

## Prerequisites

Install Lumen:

```bash
pip install lumen
```

!!! tip "Why YAML specs?"
    **Lumen AI generates these specs automatically** through conversation. Learn specs to:
    
    - Customize AI outputs
    - Version control dashboards
    - Understand how Lumen works
    - Access features not yet in AI
    
    Prefer AI? See [Using Lumen AI](../../getting_started/using_lumen_ai.md) instead.

## 1. Create a YAML file

Create `penguins.yaml` in your text editor.

**Time**: 30 seconds

## 2. Add a data source

Load the Palmer Penguins dataset:

``` yaml title="penguins.yaml" linenums="1"
sources:
  penguin_source: # (1)!
    type: file # (2)!
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv # (3)!
```

1. Source name - reference this in layouts
2. File type handles CSV, Parquet, JSON
3. Remote URL or local file path

Launch the dashboard:

```bash
lumen serve penguins.yaml --show --autoreload
```

The `--autoreload` flag refreshes automatically when you save changes.

Your browser opens, but the dashboard is empty—you haven't added views yet.

**Time**: 2 minutes

## 3. Add a table view

Display the raw data:

``` yaml title="penguins.yaml" linenums="1" hl_lines="7-12"
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

layouts:
  - title: Penguins
    source: penguin_source
    views:
      - type: table
        table: penguin_table
```

![Table view added](../../assets/build_app_01.png)

The table shows all columns. Note the column names for later plots.

**Time**: 3 minutes

## 4. Create a plot

Replace the table with a plot:

``` yaml title="penguins.yaml" hl_lines="11-12"
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

layouts:
  - title: Penguins
    source: penguin_source
    views:
      - type: hvplot # (1)!
        table: penguin_table
```

1. hvPlot automatically chooses visualization

![hvPlot view added](../../assets/build_app_02.png)

This default plot shows too much. Let's make it specific.

**Time**: 4 minutes

## 5. Configure a scatter plot

Create a scatter plot with specific axes:

``` yaml title="penguins.yaml" hl_lines="12-15"
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

layouts:
  - title: Penguins
    source: penguin_source
    views:
      - type: hvplot
        table: penguin_table
        kind: scatter # (1)!
        x: bill_length_mm # (2)!
        y: bill_depth_mm
        color: species # (3)!
```

1. Scatter plot type
2. X and Y axes from table columns
3. Color points by species

![Scatter plot created](../../assets/build_app_03.png)

Now you see the relationship between bill length and depth, colored by species.

**Time**: 6 minutes

## 6. Add filters and transforms

Add interactive filters and select columns:

``` yaml title="penguins.yaml" linenums="1" hl_lines="7-19"
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

pipelines: # (1)!
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    filters: # (2)!
      - type: widget
        field: sex
      - type: widget
        field: island
    transforms: # (3)!
      - type: columns
        columns: ['species', 'island', 'sex', 'year', 
                  'bill_length_mm', 'bill_depth_mm']

layouts:
  - title: Penguins
    pipeline: penguin_pipeline # (4)!
    views:
      - type: hvplot
        x: bill_length_mm
        y: bill_depth_mm
        kind: scatter
        color: species
```

1. Pipelines connect sources to views with filters and transforms
2. Widget filters create interactive dropdowns
3. Column transform selects specific columns
4. Reference pipeline instead of source

![Filters and transforms added](../../assets/build_app_04.png)

Users can now filter by sex and island using sidebar widgets.

**Time**: 8 minutes

## 7. Add multiple views

Add histogram and table views:

``` yaml title="penguins.yaml" hl_lines="26-31"
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

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
        columns: ['species', 'island', 'sex', 'year', 
                  'bill_length_mm', 'bill_depth_mm']

layouts:
  - title: Penguins
    pipeline: penguin_pipeline
    views:
      - type: hvplot
        x: bill_length_mm
        y: bill_depth_mm
        kind: scatter
        color: species
      - type: hvplot
        kind: hist
        y: bill_length_mm
      - type: table
        show_index: false
```

![Multiple views added](../../assets/build_app_05.png)

All three views update when you change filters.

**Time**: 10 minutes

## 8. Improve the layout

Configure positioning and sizing:

``` yaml title="penguins.yaml" hl_lines="23-25 31 36 40"
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

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
        columns: ['species', 'island', 'sex', 'year', 
                  'bill_length_mm', 'bill_depth_mm']

layouts:
  - title: Penguins
    pipeline: penguin_pipeline
    layout: [[0], [1, 2]] # (1)!
    sizing_mode: stretch_width # (2)!
    height: 800
    views:
      - type: hvplot
        x: bill_length_mm
        y: bill_depth_mm
        kind: scatter
        color: species
        responsive: true # (3)!
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

1. Scatter on top row, histogram and table side-by-side on bottom
2. Views expand to fill width
3. Plots resize with browser window

![Layout customized](../../assets/build_app_06.png)

**Time**: 12 minutes

## 9. Add title and theme

Final polish:

``` yaml title="penguins.yaml" hl_lines="1-3 43"
config:
  title: Palmer Penguins
  theme: dark # (1)!

sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

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
        columns: ['species', 'island', 'sex', 'year', 
                  'bill_length_mm', 'bill_depth_mm']

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
        theme: midnight
```

1. Dark theme for the entire dashboard

![Final dashboard](../../assets/build_app_07.png)

🎉 **Done!** You've built a complete interactive dashboard.

**Total time**: 15 minutes

## What you learned

- Create YAML specifications
- Load data from remote sources
- Add interactive filters
- Create multiple visualization types
- Configure responsive layouts
- Apply themes and styling

## Next steps

**Learn the system:**

- [Core Concepts](../../configuration/spec/concepts.md) - Understand sources, pipelines, views
- [Sources Guide](../../configuration/spec/sources.md) - Load from databases and APIs
- [Pipelines Guide](../../configuration/spec/pipelines.md) - All filters and transforms
- [Views Guide](../../configuration/spec/views.md) - All visualization types

**Try these challenges:**

- 📊 Replace histogram with box plot (`kind: box`)
- 🎨 Color scatter by different field
- 🔍 Add year filter
- 📥 Enable downloads (see [Downloads guide](../../configuration/spec/downloads.md))
