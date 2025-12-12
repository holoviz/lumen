# :material-weather-partly-cloudy: Building a Weather Data AI Explorer

Build a domain-specific data exploration application using Lumen AI. This tutorial creates a weather data explorer that analyzes atmospheric soundings and displays Skew-T diagrams.

## Final result

<img width="1505" height="1185" alt="image" src="https://github.com/user-attachments/assets/7e1cd684-14d2-4d4c-b409-db82e286f43d" />

**Time**: 15-20 minutes

## What you'll build

A chat interface that understands meteorological concepts and generates specialized visualizations. The tutorial follows five steps:

1. **Create a custom analysis** - Build `SkewTAnalysis` to render Skew-T diagrams
2. **Configure the data source** - Set up DuckDB to load atmospheric sounding data
3. **Add domain expertise** - Teach agents meteorological concepts through template overrides
4. **Build the interface** - Create the UI with helpful suggestions
5. **Run and test** - Launch the application and interact with it

Each step introduces key Lumen AI concepts with links to detailed configuration guides.

## Prerequisites

Install the required packages:

```bash
pip install lumen-ai metpy panel matplotlib
```

## 1. Create a custom analysis

Analyses allow developers to create deterministic flows for specialized visualizations. While agents handle data queries and interpretation, analyses ensure visualizations render correctly every time. Create `SkewTAnalysis` to generate Skew-T diagrams:

**Key concepts:**

- **Custom analyses** extend Lumen AI by subclassing `lmai.analysis.Analysis` (see [Analyses configuration](../../configuration/analyses.md))
- **Required columns** specify data requirements using the `columns` attribute
- **Parameters** make analyses configurable using [Param](https://param.holoviz.org/)
- **Autorun** determines if the analysis executes automatically when conditions are met

``` py title="weather_sounding.py" linenums="1"
import param
import lumen.ai as lmai
import pandas as pd
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc

class SkewTAnalysis(lmai.analysis.Analysis):
    """
    Creates a Skew-T log-P diagram from upper air sounding data.
    Shows temperature, dew point, and wind profiles.
    """

    autorun = param.Boolean(default=True) # (1)!

    barbs_interval = param.Integer(
        default=3, 
        doc="Interval for plotting wind barbs to avoid crowding"
    )

    columns = ["validUTC", "pressure_mb", "tmpc", "dwpc"] # (2)!
```

1. `autorun=True` runs this analysis automatically when the required columns are present
2. The `columns` attribute tells Lumen AI which data columns this analysis needs

### Implement the visualization

The `__call__` method contains the core visualization logic:

``` py title="weather_sounding.py" linenums="25"
    def __call__(self, pipeline, *args, **kwargs):
        df = pipeline.data.copy()
        df["validUTC"] = pd.to_datetime(df["validUTC"])
        latest_time = df["validUTC"].max()
        sounding = df[df["validUTC"] == latest_time].copy() # (1)!

        # Extract variables with units
        pressure = sounding["pressure_mb"].values * units.hPa # (2)!
        temperature = sounding["tmpc"].values * units.degC
        dewpoint = sounding["dwpc"].values * units.degC

        # Create Skew-T plot
        fig = plt.figure(figsize=(7, 7))
        skew = SkewT(fig, rotation=45) # (3)!
        skew.plot_dry_adiabats(alpha=0.25, color="orangered")
        skew.plot_moist_adiabats(alpha=0.25, color="tab:green")
        skew.plot_mixing_lines(alpha=0.25, color="tab:blue")
        skew.plot(pressure, temperature, "r", linewidth=2, label="Temperature")
        skew.plot(pressure, dewpoint, "g", linewidth=2, label="Dew Point")

        # Add wind barbs if available
        if "drct" in sounding.columns and "speed_kts" in sounding.columns:
            wind_data = sounding[["speed_kts", "drct"]].apply(pd.to_numeric, errors="coerce")
            wind_speed = wind_data["speed_kts"].values * units.knots
            wind_direction = wind_data["drct"].values * units.degrees
            u_wind, v_wind = mpcalc.wind_components(wind_speed, wind_direction)
            skew.plot_barbs(
                pressure[:: self.barbs_interval], 
                u_wind[:: self.barbs_interval], 
                v_wind[:: self.barbs_interval]
            )

        time_str = latest_time.strftime("%Y-%m-%d %H:%M UTC")
        plt.title(
            f"Atmospheric Sounding - Oakland, CA\n{time_str}", 
            fontsize=14, 
            fontweight="bold"
        )
        return pn.pane.Matplotlib(fig, sizing_mode="stretch_both") # (4)!
```

1. Filter to the most recent sounding
2. MetPy requires units for meteorological calculations
3. Skew-T uses a 45° rotation coordinate system
4. Return a Panel pane for the Lumen UI

## 2. Configure the data source

Set up DuckDB to load weather data from GitHub:

``` py title="weather_sounding.py"
from lumen.sources.duckdb import DuckDBSource

source = DuckDBSource(
    uri=":memory:",
    tables={
        "raob_soundings": """
            SELECT * FROM read_csv_auto(
                'https://raw.githubusercontent.com/holoviz/lumen/main/examples/raob.csv'
            )
        """,
    },
)
```

Data comes from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) with atmospheric measurements at different pressure levels.

## 3. Add domain expertise to agents

Template overrides inject domain knowledge into agent system prompts. This customizes how agents understand and respond to domain-specific queries.

**Key concepts:**

- **Template overrides** inject specialized knowledge using the `template_overrides` attribute (see [Agents configuration](../configuration/agents.md))
- **Global context** shares knowledge across all agents by setting overrides on the `Actor` base class
- **Agent-specific overrides** customize individual agent types like `SQLAgent` or `AnalystAgent`
- **Jinja2 templates** use `{{ super() }}` to preserve original content while adding customizations

### Share meteorological knowledge globally

Template overrides inject domain-specific knowledge into agents' system prompts. This teaches all agents about inversion analysis without requiring users to explain it in every query.

``` py title="weather_sounding.py"
global_context = """
{{ super() }} # (1)!

To perform inversion analysis, extract temperatures at different pressure levels and
complete a difference calculation. Then use AnalystAgent to determine if the temperature 
increases with decreasing pressure in a layer, which indicates an inversion.
"""
lmai.actor.Actor.template_overrides = {"main": {"global": global_context}} # (2)!
```

1. `{{ super() }}` preserves the original template
2. Apply to `Actor` base class to affect all agents

### Guide the SQL agent

The SQL agent needs to know which columns are required for specific visualizations. This footer ensures it queries for complete data when users request Skew-T diagrams.

``` py title="weather_sounding.py"
sql_footer = """
To show a Skew-T diagram, you must have at least the following columns:
- pressure_mb
- tmpc
- dwpc
- speed_kts
- drct
- validUTC
"""
lmai.agents.SQLAgent.template_overrides = {"main": {"footer": sql_footer}}
```

### Make the analyst speak like a meteorologist

Specialized instructions make the analyst agent use proper meteorological terminology and explain concepts accurately, creating a more professional domain-specific assistant.

``` py title="weather_sounding.py"
analyst_instructions = """
{{ super() }}

You are a meteorologist. Use proper meteorological terminology and explain 
atmospheric concepts clearly.
"""
lmai.agents.AnalystAgent.template_overrides = {
    "main": {"instructions": analyst_instructions}
}
```

## 4. Build the interface

Create the explorer with suggestions:

``` py title="weather_sounding.py"
import panel as pn

pn.extension()

ui = lmai.ExplorerUI(
    data=source,
    analyses=[SkewTAnalysis],
    title="Weather Data Explorer",
    suggestions=[
        ("question_answer", "What is a Skew-T diagram?"),
        ("vertical_align_top", "Generate a Skew-T diagram."),
        ("question_mark", "Is there an inversion layer today?"),
        ("search", "What's the surface temperature?"),
    ],
    log_level="DEBUG",
)

ui.servable()
```

## 5. Run the application

Start the server:

```bash
panel serve weather_sounding.py --show
```

Your browser opens to the weather explorer. Try these interactions:

- **"What is a Skew-T diagram?"** - Get an explanation
- **"Generate a Skew-T diagram."** - See the visualization
- **"Is there an inversion layer today?"** - Perform analysis
- **"What's the surface temperature?"** - Query the data

## Complete code

``` py title="weather_sounding.py"
"""
Weather Data Explorer with Atmospheric Soundings
"""

import param
import lumen.ai as lmai
from lumen.sources.duckdb import DuckDBSource
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import panel as pn
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc

pn.extension()
mpl.use("agg")


class SkewTAnalysis(lmai.analysis.Analysis):
    """
    Creates a Skew-T log-P diagram from upper air sounding data.
    Shows temperature, dew point, and wind profiles.
    """

    autorun = param.Boolean(default=True)
    barbs_interval = param.Integer(default=3, doc="Interval for plotting wind barbs to avoid crowding")
    columns = ["validUTC", "pressure_mb", "tmpc", "dwpc"]

    def __call__(self, pipeline, *args, **kwargs):
        df = pipeline.data.copy()
        df["validUTC"] = pd.to_datetime(df["validUTC"])
        latest_time = df["validUTC"].max()
        sounding = df[df["validUTC"] == latest_time].copy()

        pressure = sounding["pressure_mb"].values * units.hPa
        temperature = sounding["tmpc"].values * units.degC
        dewpoint = sounding["dwpc"].values * units.degC

        fig = plt.figure(figsize=(7, 7))
        skew = SkewT(fig, rotation=45)
        skew.plot_dry_adiabats(alpha=0.25, color="orangered")
        skew.plot_moist_adiabats(alpha=0.25, color="tab:green")
        skew.plot_mixing_lines(alpha=0.25, color="tab:blue")
        skew.plot(pressure, temperature, "r", linewidth=2, label="Temperature")
        skew.plot(pressure, dewpoint, "g", linewidth=2, label="Dew Point")

        if "drct" in sounding.columns and "speed_kts" in sounding.columns:
            wind_data = sounding[["speed_kts", "drct"]].apply(pd.to_numeric, errors="coerce")
            wind_speed = wind_data["speed_kts"].values * units.knots
            wind_direction = wind_data["drct"].values * units.degrees
            u_wind, v_wind = mpcalc.wind_components(wind_speed, wind_direction)
            skew.plot_barbs(pressure[:: self.barbs_interval], u_wind[:: self.barbs_interval], v_wind[:: self.barbs_interval])

        time_str = latest_time.strftime("%Y-%m-%d %H:%M UTC")
        plt.title(f"Atmospheric Sounding - Oakland, CA\n{time_str}", fontsize=14, fontweight="bold")
        return pn.pane.Matplotlib(fig, sizing_mode="stretch_both")


source = DuckDBSource(
    uri=":memory:",
    tables={
        "raob_soundings": """
            SELECT * FROM read_csv_auto(
                'https://raw.githubusercontent.com/holoviz/lumen/main/examples/raob.csv'
            )
        """,
    },
)

global_context = """
{{ super() }}

To perform inversion analysis, extract temperatures at different pressure levels and
complete a difference calculation. Then use AnalystAgent to determine if the temperature increases with decreasing pressure in a layer,
which indicates an inversion.
"""
lmai.actor.Actor.template_overrides = {"main": {"global": global_context}}

sql_footer = """
To show a Skew-T diagram, you must have at least the following columns:
- pressure_mb
- tmpc
- dwpc
- speed_kts
- drct
- validUTC
"""
lmai.agents.SQLAgent.template_overrides = {"main": {"footer": sql_footer}}

analyst_instructions = """
{{ super() }}

You are a meteorologist. Use proper meteorological terminology and explain atmospheric concepts clearly.
"""
lmai.agents.AnalystAgent.template_overrides = {"main": {"instructions": analyst_instructions}}

ui = lmai.ExplorerUI(
    data=source,
    analyses=[SkewTAnalysis],
    title="Weather Data Explorer",
    suggestions=[
        ("question_answer", "What is a Skew-T diagram?"),
        ("vertical_align_top", "Generate a Skew-T diagram."),
        ("question_mark", "Is there an inversion layer today?"),
        ("search", "What's the surface temperature?"),
    ],
    log_level="DEBUG",
)

ui.servable()
```

## Run the application

Start the server:

```bash
panel serve weather_sounding.py --show
```

Your browser opens to the weather explorer. Try these interactions:

- **"What is a Skew-T diagram?"** - Get an explanation
- **"Generate a Skew-T diagram."** - See the visualization
- **"Is there an inversion layer today?"** - Perform analysis
- **"What's the surface temperature?"** - Query the data

## Next steps

Extend this example:

- **Add calculations**: Implement CAPE, CIN, or lifted index (see [custom analyses](../../configuration/analyses.md))
- **Create diagrams**: Build hodographs or time-height sections
- **Integrate forecasts**: Combine observations with model predictions using [multiple data sources](../../configuration/sources.md)
- **Add validation**: Include quality control for meteorological data
- **Customize agents**: Learn more about [agent configuration](../../configuration/agents.md) and [prompts](../../configuration/prompts.md)
