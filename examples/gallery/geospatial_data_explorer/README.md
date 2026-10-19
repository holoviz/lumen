
# Geospatial Data Explorer

An interactive Lumen dashboard for exploring geospatial point data alongside related attributes. This example demonstrates how to load a local CSV file, visualize geographic locations, filter records, and connect multiple views through selections.

## Features

- **Local data source:** Load geospatial records from a CSV file.
- **Interactive map:** Display locations using longitude and latitude, with points colored by category.
- **Filtering:** Explore records by category, visitors per day, and rating.
- **Linked table:** Inspect records associated with selected map locations.
- **Histograms:** Explore the distributions of visitor counts and ratings.

## Files

- `geospatial_data_explorer.yaml` — Lumen dashboard configuration.
- `city_service_locations.csv` — Sample dataset used by the dashboard.

The included Mumbai service-location records are illustrative sample data. They do not represent verified real-world facilities or statistics.

## Run the Example

From the Lumen repository root, run:

```bash
lumen serve examples/gallery/geospatial_data_explorer/geospatial_data_explorer.yaml --show
```

## Data Format

The example uses a local CSV with one row per location. The dataset includes longitude and latitude alongside category, visitor count, and rating.

To use your own data, replace the sample CSV while preserving the column names expected by the YAML configuration, or update the configuration to match your dataset.
