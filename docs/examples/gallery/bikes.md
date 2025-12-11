# :material-bike: London Bike Points

Real-time bike sharing data from Transport for London with interactive maps and linked selections.

![London Bike Points Dashboard](https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/bikes.png)

## Features

- **Live API data** - Real-time bike availability from TfL API
- **Geographic visualization** - Interactive map with bike counts
- **Linked selections** - Select stations on map to filter charts
- **Multiple data sources** - Joins station metadata with occupancy data

## YAML Specification

```yaml title="bikes.yaml" linenums="1"
--8<-- "examples/gallery/bikes/bikes.yaml"
```

## Run this example

Save the YAML above as `bikes.yaml` and run:

```bash
lumen serve bikes.yaml --show
```

[Download YAML](https://github.com/holoviz/lumen/blob/main/examples/gallery/bikes/bikes.yaml){ .md-button }

## Key concepts

This example demonstrates:

- **JSON sources** - Loading data from REST APIs
- **Variables** - Using API keys securely
- **Source joins** - Combining multiple data sources
- **Geographic plots** - Maps with tiles and hover information
- **Linked selections** - Cross-filtering between views
