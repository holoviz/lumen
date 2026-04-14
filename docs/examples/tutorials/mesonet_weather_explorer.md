# :material-weather-partly-cloudy: Building a Weather Data Explorer

<!-- TODO: Add screenshot at docs/assets/tutorials/mesonet_full.png -->

Build a data exploration application that fetches weather observations from the Iowa Environmental Mesonet (IEM) using URL-based source controls.

This tutorial creates a custom `URLSourceControls` subclass that lets users select weather stations, date ranges, and networks through a simple interface.

## Final result

A chat interface that fetches daily weather summaries from ASOS/AWOS stations and lets users explore temperature, precipitation, and wind data through natural language queries.

**Time**: 15-20 minutes

## What you'll build

A URL-based source control that fetches weather data from the IEM API with automatic parameter preprocessing. The tutorial follows three steps:

1. **Start with a minimal example** - Build a basic URL control with ~40 lines of runnable code
2. **Understand URL template interpolation** - Learn how class-level params map to URL placeholders
3. **Add preprocessing** - Override `_fetch_data` to handle user input variations

For a detailed reference on source controls, see the [Source Controls](../../configuration/controls.md) documentation.

## Why URLSourceControls?

When your data source is a REST API with query parameters, `URLSourceControls` provides a declarative pattern:

- **Declare params as class attributes** - They automatically become UI widgets
- **Define a URL template** - Parameter values are interpolated at fetch time
- **Override for preprocessing** - Transform user input before the API call

This is simpler than `BaseSourceControls` when you just need to fetch from a URL with parameters.

## Prerequisites

Install the required packages:

```bash
pip install lumen-ai
```

## 1. Minimal runnable example

Copy this complete example to `mesonet_explorer.py` and run it with `panel serve mesonet_explorer.py --show`:

``` py title="mesonet_explorer.py" linenums="1"
import datetime

import param

from lumen.ai.agents import SourceAgent
from lumen.ai.controls import URLSourceControls, UploadSourceControls
from lumen.ai.ui import ExplorerUI


class MesonetDailyControls(URLSourceControls):
    """Fetch daily weather observations from the Iowa Environmental Mesonet."""

    url_template = (  # (1)!
        "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
        "?stations={stations}&network={network}&sts={sts}&ets={ets}&format=csv"
    )

    stations = param.String(  # (2)!
        default="SEA",
        doc="FAA station identifier(s), comma-separated",
    )

    network = param.Selector(  # (3)!
        default="WA_ASOS",
        objects=["CA_ASOS", "IL_ASOS", "NY_ASOS", "TX_ASOS", "WA_ASOS"],
        doc="IEM network code for the state ASOS network",
    )

    sts = param.CalendarDate(  # (4)!
        default=datetime.date.today() - datetime.timedelta(days=7),
        doc="Start date",
    )

    ets = param.CalendarDate(
        default=datetime.date.today() - datetime.timedelta(days=1),
        doc="End date",
    )

    label = '<span class="material-icons">thermostat</span> IEM Mesonet Daily'


ui = ExplorerUI(
    agents=[SourceAgent()],
    source_controls=[MesonetDailyControls(), UploadSourceControls()],
    title="Weather Data Explorer",
    log_level="DEBUG",
)

ui.servable()
```

1. URL template with `{param_name}` placeholders matching the class attributes below
2. String parameter - renders as a text input
3. Selector parameter - renders as a dropdown menu
4. CalendarDate parameter - renders as a date picker

This ~45 line example is immediately runnable. Try clicking "Sources" in the sidebar, selecting a network and date range, then clicking "Fetch Data".

Once the data loads, you can ask questions like:

- "What was the highest temperature recorded?"
- "Show me a chart of daily high and low temperatures"
- "Which day had the most precipitation?"

## 2. Understanding URL template interpolation

The key insight is that **class-level param attributes become both UI widgets and URL parameters**.

### How it works

When the user clicks "Fetch Data", `URLSourceControls`:

1. **Collects parameter values** from the widgets
2. **Interpolates them into `url_template`** using Python's `str.format()`
3. **Downloads the result** and parses it as CSV, JSON, or Parquet

For example, with these values:

- `stations = "SEA"`
- `network = "WA_ASOS"`
- `sts = 2024-01-01`
- `ets = 2024-01-07`

The URL becomes:

```
https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?stations=SEA&network=WA_ASOS&sts=2024-01-01&ets=2024-01-07&format=csv
```

### Parameter types and widgets

| Param type | Widget rendered | Example |
|------------|-----------------|---------|
| `param.String` | Text input | Station codes |
| `param.Selector` | Dropdown | Network selection |
| `param.Integer` | Number input | Year |
| `param.CalendarDate` | Date picker | Start/end dates |
| `param.Boolean` | Checkbox | Include metadata |

## 3. Adding preprocessing

The IEM API uses 3-letter FAA station codes (like `SEA` for Seattle), but users often enter 4-letter ICAO codes (like `KSEA`). We can handle this by overriding `_fetch_data`:

``` py title="mesonet_explorer.py (with preprocessing)" linenums="1" hl_lines="17-24"
class MesonetDailyControls(URLSourceControls):
    """Fetch daily weather observations from the Iowa Environmental Mesonet."""

    url_template = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
        "?stations={stations}&network={network}&sts={sts}&ets={ets}&format=csv"
    )

    stations = param.String(
        default="SEA",
        doc="FAA station identifier(s). A leading 'K' ICAO prefix is stripped automatically.",
    )

    # ... other params ...

    async def _fetch_data(self, action_name: str, **params) -> "SourceResult":
        # IEM uses 3-letter FAA codes; strip the ICAO 'K' prefix users often add
        raw = params.get("stations", "")
        params["stations"] = ",".join(
            s[1:] if len(s) == 4 and s.startswith("K") else s
            for s in (t.strip() for t in raw.split(","))
        )
        return await super()._fetch_data(action_name, **params)
```

Now users can enter either `SEA` or `KSEA` and both will work correctly.

### When to override `_fetch_data`

Override `_fetch_data` when you need to:

- **Transform user input** - Normalize codes, convert units, validate ranges
- **Add computed parameters** - Calculate values based on other params
- **Handle API quirks** - Modify parameters for specific API requirements
- **Add custom error handling** - Check for specific error conditions

## Full example

Here's the complete implementation with preprocessing and improved labels:

``` py title="mesonet_explorer_full.py" linenums="1"
"""
Mesonet Weather Explorer - Full Example
Fetches daily weather data from Iowa Environmental Mesonet
"""

import datetime

import param

from lumen.ai.agents import SourceAgent
from lumen.ai.controls import URLSourceControls, UploadSourceControls
from lumen.ai.ui import ExplorerUI


class MesonetDailyControls(URLSourceControls):
    """
    Fetch daily weather observations from the Iowa Environmental Mesonet
    for one or more ASOS/AWOS stations.

    Data includes high/low temperature, precipitation, average wind speed,
    and other daily summary fields.
    """

    url_template = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
        "?stations={stations}&network={network}&sts={sts}&ets={ets}&format=csv"
    )

    stations = param.String(
        default="SEA",
        doc="FAA station identifier(s), comma-separated (e.g. 'ORD', 'SEA,PDX'). "
            "A leading 'K' ICAO prefix is stripped automatically.",
    )

    network = param.Selector(
        default="WA_ASOS",
        objects=[
            "CA_ASOS", "CO_ASOS", "FL_ASOS", "GA_ASOS", "IL_ASOS",
            "MA_ASOS", "MI_ASOS", "MN_ASOS", "NY_ASOS", "OH_ASOS",
            "OR_ASOS", "PA_ASOS", "TX_ASOS", "WA_ASOS",
        ],
        doc="IEM network code for the state ASOS network.",
    )

    sts = param.CalendarDate(
        default=datetime.date.today() - datetime.timedelta(days=7),
        doc="Start date for the data request.",
    )

    ets = param.CalendarDate(
        default=datetime.date.today() - datetime.timedelta(days=1),
        doc="End date for the data request (exclusive). "
            "Note: today's daily summary may not be available until tomorrow.",
    )

    label = (
        '<span class="material-icons" style="vertical-align: middle;">'
        "thermostat"
        "</span> IEM Mesonet Daily"
    )

    async def _fetch_data(self, action_name: str, **params) -> "SourceResult":
        # IEM uses 3-letter FAA codes; strip the ICAO 'K' prefix users often add
        raw = params.get("stations", "")
        params["stations"] = ",".join(
            s[1:] if len(s) == 4 and s.startswith("K") else s
            for s in (t.strip() for t in raw.split(","))
        )
        return await super()._fetch_data(action_name, **params)


ui = ExplorerUI(
    agents=[SourceAgent()],
    source_controls=[MesonetDailyControls(), UploadSourceControls()],
    title="Weather Data Explorer",
    log_level="DEBUG",
)

ui.servable()
```

## Understanding the data

The IEM daily summary includes these columns:

| Column | Description |
|--------|-------------|
| `station` | Station identifier |
| `day` | Date of observation |
| `max_tmpf` | Maximum temperature (F) |
| `min_tmpf` | Minimum temperature (F) |
| `precip` | Precipitation (inches) |
| `avg_sknt` | Average wind speed (knots) |
| `max_sknt` | Maximum wind speed (knots) |

Try asking questions like:

- "What was the temperature range for each day?"
- "Create a line chart of max and min temperatures over time"
- "Which day had the strongest winds?"
- "Calculate the total precipitation for the week"

## Next steps

Extend this example by:

- **Add more networks** - Include all 50 state ASOS networks
- **Add hourly data** - Create a separate control for hourly observations
- **Combine with other sources** - Join weather data with your own datasets
- **Add custom analyses** - Create specialized weather visualizations

## See also

- [Source Controls](../../configuration/controls.md) — Complete guide to creating custom controls
- [Census Data Explorer](census_data_ai_explorer.md) — BaseSourceControls tutorial with dynamic options
- [Agents](../../configuration/agents.md) — Configuring SourceAgent and other agents
