
# :material-map-marker-multiple: Geospatial Data Explorer

Explore geospatial point data using an interactive map, filters, linked table, and histograms. This example uses a local CSV dataset of illustrative city service locations.

![Geospatial Data Explorer Dashboard](../../assets/gallery/geospatial_data_explorer.png)

## Features

- **Geospatial visualization** — Display locations using longitude and latitude on an interactive map.
- **Category comparison** — Color locations by service category.
- **Interactive filtering** — Filter records by category, visitors per day, and rating.
- **Linked selections** — Select map points and inspect the corresponding records in a table.
- **Attribute distributions** — Explore visitor counts and ratings using histograms.
- **Local CSV source** — Run the example without relying on an external data API.

## Demo

<video controls width="100%" preload="metadata">
  <source src="../../assets/gallery/geospatial_data_explorer.mp4" type="video/mp4">
  Your browser does not support embedded videos.
</video>

## YAML Specification

```yaml title="geospatial_data_explorer.yaml" linenums="1"
--8<-- "examples/gallery/geospatial_data_explorer/geospatial_data_explorer.yaml"
```

## Run this example

From the Lumen repository root, run:

```bash
lumen serve examples/gallery/geospatial_data_explorer/geospatial_data_explorer.yaml --show
```

[Download YAML](https://github.com/holoviz/lumen/blob/main/examples/gallery/geospatial_data_explorer/geospatial_data_explorer.yaml){ .md-button }

## Dataset

The example uses `city_service_locations.csv`, which contains location names, categories, coordinates, visitor counts, and ratings. The included Mumbai records are illustrative sample data, not verified real-world facilities or statistics.
