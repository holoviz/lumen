# :material-taxi: NYC Taxi

Million-row taxi trip analysis with pickup/dropoff maps and fare distribution.

![NYC Taxi Dashboard](https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/nyc_taxi.png)

## Features

- **Large dataset handling** - Efficiently processes millions of taxi trips
- **Geographic maps** - Pickup and dropoff locations
- **Fare analysis** - Distribution of trip costs
- **Time filtering** - Filter by pickup time

## YAML Specification

```yaml title="nyc_taxi.yaml" linenums="1"
--8<-- "examples/gallery/nyc_taxi/nyc_taxi.yaml"
```

## Run this example

Save the YAML above as `nyc_taxi.yaml` and run:

```bash
lumen serve nyc_taxi.yaml --show
```

[Download YAML](https://github.com/holoviz/lumen/blob/main/examples/gallery/nyc_taxi/nyc_taxi.yaml){ .md-button }
