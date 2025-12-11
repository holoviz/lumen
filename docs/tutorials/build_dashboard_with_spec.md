# Build dashboard with spec

Build a complete Lumen dashboard in under 15 minutes using YAML specifications.

!!! info "Why learn specs when AI exists?"
    **Lumen AI generates these specs automatically** through conversation. So why learn to write them manually?
    
    - **Customize AI outputs** - Edit and enhance what the AI creates
    - **Reproducibility** - AI-generated dashboards are saved as specs you can version control
    - **Understanding** - Know how Lumen works under the hood
    - **Advanced features** - Some features aren't yet available through AI
    
    **Historically**, users wrote these specs by hand before LLMs. Now AI does the heavy lifting, but understanding specs lets you fine-tune results.
    
    **Choose your approach:**
    
    - Prefer AI assistance? → [Use Lumen AI](../getting_started/using_lumen_ai.md) instead
    - Want to understand the system? → Continue with this tutorial
    - AI generated something, want to customize it? → This tutorial will help

!!! tip "What you'll learn"
    - Create a YAML specification from scratch
    - Load data from a remote source
    - Add interactive filters
    - Create multiple visualizations
    - Customize layout and styling

## Final result

By the end of this tutorial, you'll have built this dashboard:

![Build dashboard final result](../assets/build_app_07.png)

---

## Step 1: Create a YAML file

Open your text editor and create a file called `penguins.yaml`.

**Time: 30 seconds**

---

## Step 2: Add a data source

Load the Palmer Penguins dataset from a remote URL:

=== "YAML"

    ```yaml
    sources:
      penguin_source:
        type: file
        tables:
          penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv
    ```

=== "What this does"

    - Creates a source named `penguin_source`
    - Uses the `file` type to load CSV data
    - Names the table `penguin_table`
    - Points to a remote dataset URL

Save the file, then open a terminal and navigate to its location.

Launch the dashboard:

```bash
lumen serve penguins.yaml --show --autoreload
```

!!! note "About --autoreload"
    The `--autoreload` flag automatically refreshes your browser when you save changes to the YAML file.

Your browser opens, but the dashboard is empty—you haven't added any views yet:

![Data source added](../assets/build_app_00.png)

**Time: 2 minutes**

---

## Step 3: Add a table view

Display the raw data in a table to preview available columns:

=== "YAML"

    ```yaml
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

=== "Preview"

    ![Table view added](../assets/build_app_01.png)

The table shows all columns in the dataset. You'll use these column names later for plots.

**Time: 3 minutes**

---

## Step 4: Create a plot

Replace the table with an interactive plot:

=== "YAML"

    ```yaml
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
    ```

=== "Preview"

    ![hvPlot view added](../assets/build_app_02.png)

This default plot shows too much information at once. Let's make it more specific.

**Time: 4 minutes**

---

## Step 5: Configure a scatter plot

Create a scatter plot with specific axes and colors:

=== "YAML"

    ```yaml
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
            kind: scatter
            x: bill_length_mm
            y: bill_depth_mm
            color: species
    ```

=== "Preview"

    ![Scatter plot created](../assets/build_app_03.png)

Now you can see the relationship between bill length and depth, colored by species.

**Time: 6 minutes**

---

## Step 6: Add filters and transforms

Add interactive filters and select specific columns:

=== "YAML"

    ```yaml
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
            columns: ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']

    layouts:
      - title: Penguins
        pipeline: penguin_pipeline
        views:
          - type: hvplot
            x: bill_length_mm
            y: bill_depth_mm
            kind: scatter
            color: species
    ```

=== "Preview"

    ![Filters and transforms added](../assets/build_app_04.png)

!!! success "Interactive filtering"
    Users can now filter by sex and island using the widgets in the sidebar!

**Time: 8 minutes**

---

## Step 7: Add multiple views

Expand your dashboard with a histogram and table:

=== "YAML"

    ```yaml
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
            columns: ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']

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

=== "Preview"

    ![Multiple views added](../assets/build_app_05.png)

The dashboard now shows three different views of the same filtered data.

**Time: 10 minutes**

---

## Step 8: Improve the layout

Make the dashboard responsive and set proper sizing:

=== "YAML"

    ```yaml
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
            columns: ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']

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

=== "Preview"

    ![Layout customized](../assets/build_app_06.png)

=== "What changed"

    - `layout: [[0], [1, 2]]` - Scatter on top row, histogram and table on bottom row
    - `sizing_mode: stretch_width` - Views expand to fill available width
    - `responsive: true` - Plots resize when browser window changes
    - Individual `height` settings control each view's height

**Time: 12 minutes**

---

## Step 9: Add title and theme

Finish with a descriptive title and dark theme:

=== "YAML"

    ```yaml
    config:
      title: Palmer Penguins
      theme: dark

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
            columns: ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']

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

=== "Preview"

    ![Final dashboard](../assets/build_app_07.png)

🎉 **Congratulations!** You've built a complete interactive dashboard.

**Total time: 15 minutes**

---

## What you learned

In this tutorial, you:

- ✅ Created a YAML specification file
- ✅ Loaded data from a remote source
- ✅ Added interactive filter widgets
- ✅ Created multiple visualization types
- ✅ Configured responsive layouts
- ✅ Applied themes and styling

## Next steps

Now that you've built your first dashboard, deepen your understanding:

1. **[Core Concepts](../configuration/spec/concepts.md)** - Understand the YAML structure and how Lumen works
2. **[Loading Data](../configuration/spec/sources.md)** - Load data from different sources (files, databases, APIs)
3. **[Transforming Data](../configuration/spec/pipelines.md)** - Learn all available filters and transforms
4. **[Visualizing Data](../configuration/spec/views.md)** - Explore different visualization types

**Want more?** Browse the [complete spec documentation](../configuration/spec/index.md) for advanced features.

## Try these challenges

Apply what you learned:

- 📊 Replace the histogram with a box plot (`kind: box`)
- 🎨 Color the scatter plot by a different field
- 🔍 Add a filter for the `year` column
- 📥 Enable data downloads (see [Downloads guide](../configuration/spec/downloads.md))
