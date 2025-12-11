# :material-wind-turbine: Wind Turbines

Wind turbine performance metrics and geographic distribution across the US.

![Wind Turbines Dashboard](https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/windturbines.png)

## Features

- **Geographic distribution** - Turbine locations across the US
- **Performance metrics** - Capacity and output analysis
- **Manufacturer comparison** - Compare different turbine makers
- **Interactive filtering** - Filter by state, manufacturer, capacity

## YAML Specification

```yaml title="windturbines.yaml" linenums="1"
--8<-- "examples/gallery/windturbines/windturbines.yaml"
```

## Run this example

Save the YAML above as `windturbines.yaml` and run:

```bash
lumen serve windturbines.yaml --show
```

[Download YAML](https://github.com/holoviz/lumen/blob/main/examples/gallery/windturbines/windturbines.yaml){ .md-button }
