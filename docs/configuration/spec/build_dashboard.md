# Build your first dashboard

!!! important
    This tutorial is meant to get you hands-on building a Lumen dashboard. Don't worry about understanding everything just yet—your only goal is to follow the steps as directed.

You will build a simple Lumen dashboard and deploy it in your browser. The result will look like this:

![Build dashboard final result](../../assets/build_app_07.png)

## 1. Create a YAML file

Open your favorite text editor and create an empty file called `penguins.yaml`.

## 2. Add a data source

The first thing you need is a source of data. Add the text below to create a remote FileSource. This tells Lumen to fetch the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/):

```yaml
sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv
```

Save the file, then open a terminal and navigate to the file's location.

Launch the dashboard with:

```bash
lumen serve penguins.yaml --show --autoreload
```

The `--autoreload` flag automatically updates the dashboard when you modify the YAML file.

Your browser should now display something like this:

=== "YAML"

    ```yaml
    sources:
      penguin_source:
        type: file
        tables:
          penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv
    ```

=== "Preview"

    ![Data source added](../../assets/build_app_00.png)

So far you have an empty dashboard because you haven't specified a view yet. Let's add one!

## 3. Specify a table view

The simplest view to add is a table showing the raw data. This gives you a preview of what you're working with and shows available fields.

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

    ![Table view added](../../assets/build_app_01.png)

## 4. Create a plot view

The table shows the raw data, but to see patterns you need visualization. Add a plot view by replacing the `table` type with `hvplot`:

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

    ![hvPlot view added](../../assets/build_app_02.png)

## 5. Make a scatter plot

This plot includes too much data at once. Instead, create a scatter plot with specific axes and colors:

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

    ![Scatter plot created](../../assets/build_app_03.png)

## 6. Manipulate the data

Add two filter widgets for 'sex' and 'island', and use a transform to select only the columns you need:

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

    ![Filters and transforms added](../../assets/build_app_04.png)

## 7. Add multiple view types

Expand your dashboard with a histogram and a table:

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

    ![Multiple views added](../../assets/build_app_05.png)

## 8. Customize layout and appearance

The default layout cuts off plots and doesn't resize responsively. Improve this with `sizing_mode` and custom layout settings:

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

    ![Layout customized](../../assets/build_app_06.png)

## 9. Add a title and theme

Finish by giving your dashboard a descriptive title and dark theme:

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

    ![Final dashboard](../../assets/build_app_07.png)

## Next steps

Congratulations! You've created your first Lumen dashboard. To generalize these steps and build your own dashboards, review the [core concepts](core_concepts.md) and explore the other topics in this section.
