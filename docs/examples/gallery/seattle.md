# :material-weather-partly-cloudy: Seattle Weather

Historical weather patterns from 2012-2015 with multi-panel time series and heatmap visualizations.

![Seattle Weather Dashboard](https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/seattle.png)

## Features

- **Temperature heatmap** - Daily highs by month and day
- **Time series** - Maximum temperature with precipitation size encoding
- **Weather distribution** - Bar chart of weather type frequency
- **URL syncing** - Filters sync with browser URL

## YAML Specification

```yaml title="seattle.yaml" linenums="1"
--8<-- "examples/gallery/seattle/weather.yaml"
```

## Run this example

Save the YAML above as `seattle.yaml` and run:

```bash
lumen serve seattle.yaml --show
```

Or explore with AI:

```bash
lumen-ai serve https://raw.githubusercontent.com/vega/vega/main/docs/data/seattle-weather.csv
```

[Download YAML](https://github.com/holoviz/lumen/blob/main/examples/gallery/seattle/weather.yaml){ .md-button }

## Key concepts

This example demonstrates:

- **Altair views** - Declarative Vega-Lite visualizations
- **Date filtering** - Filter by date ranges
- **Multiple encodings** - Color and size channels
- **URL synchronization** - Shareable filtered states
