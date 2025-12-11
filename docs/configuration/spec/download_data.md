# Enable data download

Let dashboard viewers download data in various formats.

## Add download capability

Add download buttons in three different ways:

=== "Sidebar Download"

    ```yaml
    sources:
      penguins:
        type: file
        tables:
          penguins: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv

    layouts:
      - title: Table
        source: penguins
        download: csv
        views:
          - type: download
            format: csv
          - type: table
            table: penguins
            download: csv
    ```

    Creates a download button in the sidebar

=== "View Download"

    ```yaml
    sources:
      penguins:
        type: file
        tables:
          penguins: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv

    layouts:
      - title: Table
        source: penguins
        download: csv
        views:
          - type: download
            format: csv
          - type: table
            table: penguins
            download: csv
    ```

    Creates a dedicated download view component

=== "Table Download"

    ```yaml
    sources:
      penguins:
        type: file
        tables:
          penguins: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv

    layouts:
      - title: Table
        source: penguins
        download: csv
        views:
          - type: download
            format: csv
          - type: table
            table: penguins
            download: csv
    ```

    Adds download to the table view itself

!!! note
    All three approaches use the same YAML specification. The emphasized lines show which configuration creates each download option.
