# :material-weather-partly-rainy: Building a Weather API Explorer with OpenAPI

Build a data exploration application that auto-discovers endpoints from an OpenAPI specification — no manual endpoint definitions needed.

This tutorial connects to the [National Weather Service API](https://www.weather.gov/documentation/services-web-api) using `OpenAPISourceControls`, which parses the published OpenAPI spec and generates UI widgets for each endpoint automatically.

## Final result

A chat interface with auto-discovered weather endpoints (forecasts, alerts, radar stations, and more) that users can query through dropdowns and text inputs, then explore with natural language questions.

**Time**: 10-15 minutes

## What you'll build

An API explorer that fetches its endpoint definitions from a live OpenAPI spec at startup. The tutorial follows three steps:

1. **Start with a minimal example** - Point `OpenAPISourceControls` at a spec URL and get a working app in ~20 lines
2. **Understand how auto-discovery works** - Learn how the spec becomes widgets and HTTP requests
3. **Filter and customize** - Use `include_paths` and `exclude_paths` to control which endpoints appear

For a detailed reference on source controls, see the [Source Controls](../../configuration/controls.md) documentation.

## Why OpenAPISourceControls?

Many public APIs publish an [OpenAPI specification](https://www.openapis.org/) — a machine-readable JSON file that describes every endpoint, parameter, and response. `OpenAPISourceControls` reads this spec and:

- **Auto-discovers all endpoints** - No manual endpoint definitions needed
- **Generates typed widgets** - Parameter types, enums, and defaults become UI controls
- **Resolves `$ref` references** - Handles the full OpenAPI 3.x spec format
- **Stays in sync** - Re-fetching the spec picks up new endpoints automatically

This is ideal for public APIs like weather.gov, data.gov, or any service that publishes an OpenAPI spec.

## Prerequisites

Install the required packages:

```bash
pip install lumen-ai
```

No API key is needed — the National Weather Service API is free and open.

## 1. Minimal runnable example

Copy this to `weather_openapi.py` and run with `panel serve weather_openapi.py --show`:

``` py title="weather_openapi.py" linenums="1"
from lumen.ai.agents import SourceAgent
from lumen.ai.controls import OpenAPISourceControls, UploadSourceControls
from lumen.ai.ui import ExplorerUI


controls = OpenAPISourceControls(
    spec_url="https://api.weather.gov/openapi.json",  # (1)!
    headers={"User-Agent": "LumenWeatherExplorer/1.0"},  # (2)!
    include_paths=["/alerts", "/stations", "/points"],  # (3)!
    label=(
        '<span class="material-icons" style="vertical-align: middle;">'
        "cloud"
        "</span> Weather.gov"
    ),
)

ui = ExplorerUI(
    agents=[SourceAgent()],
    source_controls=[controls, UploadSourceControls()],
    title="Weather API Explorer",
    log_level="DEBUG",
)

ui.servable()
```

1. The OpenAPI 3.x spec URL — fetched and parsed at startup
2. weather.gov requires a `User-Agent` header on all requests
3. Filter to just alerts, stations, and points endpoints — the full spec has dozens

When the app starts, `OpenAPISourceControls` fetches the spec, parses every path that matches your filters, and registers each as a selectable action with auto-generated widgets. Click "Sources" in the sidebar, pick an endpoint from the Action dropdown, fill in parameters, and click "Fetch Data".

Once data loads, try asking:

- "How many active alerts are there?"
- "Show me the stations as a table"
- "What forecast zones have active alerts?"

## 2. Understanding how auto-discovery works

`OpenAPISourceControls` converts an OpenAPI spec into callable actions in four steps.

### Spec fetch and parse

At startup, the control fetches the JSON spec and extracts:

- **`servers[0].url`** → base URL for all requests (e.g. `https://api.weather.gov`)
- **`paths`** → each path/method pair becomes an action
- **`parameters`** → each parameter becomes a UI widget
- **`$ref` references** → resolved against the `components` section

### Path filtering

The weather.gov spec has many endpoints. `include_paths` and `exclude_paths` control which ones appear:

```python
# Only include these path prefixes
include_paths=["/alerts", "/stations", "/points"]

# Or exclude specific paths
exclude_paths=["/icons", "/thumbnails"]

# Omit both to include everything
```

### Parameter type mapping

OpenAPI types are mapped to Python types for proper widget rendering:

| OpenAPI type | OpenAPI format | Python type | Widget |
|-------------|---------------|-------------|--------|
| `string` | — | `str` | Text input |
| `string` | `date` | `datetime.date` | Date picker |
| `string` | `date-time` | `datetime.datetime` | Datetime picker |
| `integer` | — | `int` | Number input |
| `number` | `float` | `float` | Number input |
| `boolean` | — | `bool` | Checkbox |
| `string` with `enum` | — | `Literal[...]` | Dropdown |

### Action naming

Each endpoint gets a display name from the spec, in priority order:

1. The `summary` field (e.g. "Returns active alerts")
2. The `operationId` field, converted to title case
3. A fallback of `METHOD /path` (e.g. "GET /alerts/active")

## 3. Filtering and customizing

### Show all endpoints

Remove `include_paths` to see everything the spec offers:

```python
controls = OpenAPISourceControls(
    spec_url="https://api.weather.gov/openapi.json",
    headers={"User-Agent": "LumenWeatherExplorer/1.0"},
    # No include_paths — all endpoints appear
)
```

### Override the base URL

Some specs define a base URL you want to change (e.g. staging vs production):

```python
controls = OpenAPISourceControls(
    spec_url="https://api.weather.gov/openapi.json",
    base_url="https://api.weather.gov",  # Override what's in the spec
    headers={"User-Agent": "LumenWeatherExplorer/1.0"},
)
```

### Combine with other controls

`OpenAPISourceControls` works alongside other source controls. Add file uploads, manual REST endpoints, or SDK-wrapped controls as separate tabs:

```python
ui = ExplorerUI(
    agents=[SourceAgent()],
    source_controls=[
        OpenAPISourceControls(spec_url="https://api.weather.gov/openapi.json", ...),
        UploadSourceControls(),
    ],
)
```

## Full example

Here's a complete implementation with filtered endpoints and customized labels:

``` py title="weather_openapi_full.py" linenums="1"
"""
Weather API Explorer - Full Example
Auto-discovered from the National Weather Service OpenAPI spec
"""

from lumen.ai.agents import SourceAgent
from lumen.ai.controls import OpenAPISourceControls, UploadSourceControls
from lumen.ai.ui import ExplorerUI


controls = OpenAPISourceControls(
    spec_url="https://api.weather.gov/openapi.json",
    headers={"User-Agent": "LumenWeatherExplorer/1.0"},
    include_paths=[
        "/alerts",       # Active weather alerts
        "/stations",     # Observation stations
        "/points",       # Forecast by lat/lon
        "/gridpoints",   # Raw forecast grids
        "/radar",        # Radar stations
    ],
    label=(
        '<span class="material-icons" style="vertical-align: middle;">'
        "cloud"
        "</span> Weather.gov API"
    ),
)

ui = ExplorerUI(
    agents=[SourceAgent()],
    source_controls=[controls, UploadSourceControls()],
    title="Weather API Explorer",
    log_level="DEBUG",
)

ui.servable()
```

## Next steps

Extend this example by:

- **Try other OpenAPI specs** - Many public APIs publish specs (e.g. [data.gov](https://api.data.gov), [GitHub](https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json))
- **Combine with SDK controls** - Add a `CodeSourceControls` for APIs that have Python clients (see [Stock Market Explorer](massive_stock_explorer.md))
- **Add custom analyses** - Create specialized weather visualizations (see [Analyses configuration](../../configuration/analyses.md))
- **Subclass for preprocessing** - Override `_fetch_data` to rename columns or convert types

## See also

- [Source Controls](../../configuration/controls.md) — Complete guide to source controls including `OpenAPISourceControls`
- [Stock Market Explorer](massive_stock_explorer.md) — CodeSourceControls tutorial wrapping a Python SDK
- [Census Data Explorer](census_data_ai_explorer.md) — CodeSourceControls tutorial with reactive options
- [Mesonet Weather Explorer](mesonet_weather_explorer.md) — URLSourceControls tutorial with preprocessing
