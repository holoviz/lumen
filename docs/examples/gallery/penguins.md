# :material-penguin: Palmer Penguins

Interactive dashboard exploring Palmer Station penguin measurements with linked selections.

![Penguins Dashboard](https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/penguins.png)

## Features

- **Linked selections** - Select points in scatter plot to filter histograms
- **Multiple filters** - Species, island, and sex
- **Interactive table** - Shows selected data with pagination

## YAML Specification

```yaml title="penguins.yaml" linenums="1"
--8<-- "examples/gallery/penguins/penguins.yaml"
```

## Run this example

Save the YAML above as `penguins.yaml` and run:

```bash
lumen serve penguins.yaml --show
```

Or explore with AI:

```bash
lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv
```

[Download YAML](https://github.com/holoviz/lumen/blob/main/examples/gallery/penguins/penguins.yaml){ .md-button }
