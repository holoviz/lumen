# :material-chart-line: Analyses

**Analyses are functions that run deterministically with code instead of LLM generation.**

Like tools or hooks, analyses give you control over specific calculations. Use them when you need reliable, repeatable results—financial metrics, scientific formulas, or statistical tests that must be calculated exactly the same way every time.

## Quick example

``` py title="Summary statistics analysis"
import lumen.ai as lmai
import pandas as pd
import panel as pn

class SummaryStats(lmai.Analysis):
    """Calculate summary statistics."""
    
    columns = ["value"]  # (1)!
    
    def __call__(self, pipeline, *args, **kwargs):
        df = pipeline.data
        stats = df.describe()
        return pn.widgets.Tabulator(stats)

# Must use AnalysisAgent to enable analyses
analysis_agent = lmai.agents.AnalysisAgent(analyses=[SummaryStats])

ui = lmai.ExplorerUI(
    data='data.csv',
    agents=[analysis_agent]
)
ui.servable()
```

1. Analysis only runs when data has a "value" column

When the data has a "value" column, users can ask "Show me summary statistics" and Lumen runs your analysis.

## When to use analyses

**Use analyses for:**

- Domain-specific calculations (wind speed from components, ROI formulas)
- Statistical tests (t-tests, ANOVA, correlation)
- Custom visualizations (domain-specific charts)
- Repeated calculations across datasets

**Don't use analyses for:**

- One-off queries (just ask in chat)
- Simple aggregations (SQL handles these)
- Generic operations (built-in agents cover common tasks)

## Basic structure

``` py title="Analysis structure"
import param
import lumen.ai as lmai

class MyAnalysis(lmai.Analysis):
    """What this analysis does."""
    
    autorun = param.Boolean(default=True)  # (1)!
    
    columns = param.List(default=["col1", "col2"])  # (2)!
    
    def __call__(self, pipeline, *args, **kwargs):
        # Your calculation logic
        df = pipeline.data
        result = df["col1"] * df["col2"]
        
        # Return any Panel viewable
        return result.hvplot(kind="line")
```

1. Whether to run automatically when conditions are met
2. Required columns - analysis only applies when these exist

## Complete example: Wind analysis

``` py title="Wind pattern analysis" linenums="1" hl_lines="7-11 18-19 26-27"
import numpy as np
import pandas as pd
import param
import panel as pn
import lumen.ai as lmai
from lumen.transforms import Transform
import hvplot.pandas

class WindSpeedDirection(Transform):
    """Calculate wind speed and direction from u,v components."""
    
    u_component = param.String(default="u")
    v_component = param.String(default="v")
    
    def apply(self, df):
        df["wind_speed"] = np.sqrt(
            df[self.u_component]**2 + df[self.v_component]**2
        )  # (1)!
        df["wind_direction"] = np.degrees(
            np.arctan2(df[self.v_component], df[self.u_component])
        )
        df["wind_direction"] = (df["wind_direction"] + 360) % 360
        return df

class WindAnalysis(lmai.Analysis):
    """Analyze wind patterns from u,v components."""
    
    autorun = param.Boolean(default=True)
    
    columns = param.List(default=["time", "u", "v"])  # (2)!
    
    def __call__(self, pipeline, *args, **kwargs):
        # Apply calculation
        wind_pipeline = pipeline.chain(transforms=[WindSpeedDirection()])
        
        # Create visualizations
        speed_plot = wind_pipeline.data.hvplot(
            x="time", y="wind_speed", title="Wind Speed"
        )
        direction_plot = wind_pipeline.data.hvplot(
            x="time", y="wind_direction", kind="scatter", title="Wind Direction"
        )
        
        # Return combined view
        return pn.Column(speed_plot, direction_plot)

# Use with AnalysisAgent
analysis_agent = lmai.agents.AnalysisAgent(analyses=[WindAnalysis])

ui = lmai.ExplorerUI(
    data='weather.csv',
    agents=[analysis_agent]
)
ui.servable()
```

1. Deterministic calculation - same inputs always give same outputs
2. Only runs when data has time, u, and v columns

Now users can ask "Analyze wind patterns" and get both charts automatically.

## Real-world example: Skew-T diagram

This atmospheric science example creates meteorological diagrams:

``` py title="Atmospheric sounding analysis" linenums="1"
import param
import panel as pn
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import lumen.ai as lmai

class SkewTAnalysis(lmai.Analysis):
    """
    Creates a Skew-T log-P diagram from upper air sounding data.
    Shows temperature, dew point, and wind profiles.
    """
    
    autorun = param.Boolean(default=True)
    
    barbs_interval = param.Integer(default=3)  # (1)!
    
    columns = param.List(default=["validUTC", "pressure_mb", "tmpc", "dwpc"])
    
    def __call__(self, pipeline, *args, **kwargs):
        df = pipeline.data.copy()
        df["validUTC"] = pd.to_datetime(df["validUTC"])
        latest_time = df["validUTC"].max()
        sounding = df[df["validUTC"] == latest_time].copy()
        
        # Extract variables with units
        pressure = sounding["pressure_mb"].values * units.hPa
        temperature = sounding["tmpc"].values * units.degC
        dewpoint = sounding["dwpc"].values * units.degC
        
        # Create Skew-T plot
        fig = plt.figure(figsize=(7, 7))
        skew = SkewT(fig, rotation=45)
        skew.plot_dry_adiabats(alpha=0.25, color="orangered")
        skew.plot_moist_adiabats(alpha=0.25, color="tab:green")
        skew.plot_mixing_lines(alpha=0.25, color="tab:blue")
        skew.plot(pressure, temperature, "r", linewidth=2)
        skew.plot(pressure, dewpoint, "g", linewidth=2)
        
        # Add wind barbs if available
        if "drct" in sounding.columns and "speed_kts" in sounding.columns:
            wind_speed = sounding["speed_kts"].values * units.knots
            wind_direction = sounding["drct"].values * units.degrees
            u_wind, v_wind = mpcalc.wind_components(wind_speed, wind_direction)
            skew.plot_barbs(
                pressure[::self.barbs_interval],
                u_wind[::self.barbs_interval],
                v_wind[::self.barbs_interval]
            )
        
        plt.title(f"Atmospheric Sounding\n{latest_time}", fontsize=14)
        return pn.pane.Matplotlib(fig, sizing_mode="stretch_both")

# Use it
analysis_agent = lmai.agents.AnalysisAgent(analyses=[SkewTAnalysis])

ui = lmai.ExplorerUI(
    data='soundings.csv',
    agents=[analysis_agent]
)
ui.servable()
```

1. Configurable parameter - users can adjust wind barb density

## Control when analyses run

``` py title="Manual trigger only" hl_lines="5"
class ExpensiveAnalysis(lmai.Analysis):
    """Complex calculation."""
    
    columns = param.List(default=["data"])
    autorun = param.Boolean(default=False)  # (1)!
    
    def __call__(self, pipeline, *args, **kwargs):
        # Expensive computation
        return result
```

1. Only runs when user explicitly requests it

Set `autorun = False` for analyses that take a long time or should only run on explicit request.

## Add configurable parameters

``` py title="Analysis with parameters" hl_lines="6-7"
import param

class ThresholdAnalysis(lmai.Analysis):
    """Filter data by threshold."""
    
    columns = param.List(default=["value"])
    threshold = param.Number(default=50)  # (1)!
    
    def __call__(self, pipeline, *args, **kwargs):
        df = pipeline.data
        filtered = df[df["value"] > self.threshold]
        
        return filtered.hvplot(y="value", kind="hist")
```

1. User can adjust the threshold value through UI controls

## Enable analyses with AnalysisAgent

!!! warning "Must use AnalysisAgent"
    Analyses don't work automatically. You must explicitly add `AnalysisAgent` with your analyses:

    ``` py hl_lines="2 5"
    # Create AnalysisAgent with your analyses
    analysis_agent = lmai.agents.AnalysisAgent(analyses=[MyAnalysis])

    ui = lmai.ExplorerUI(
        data='data.csv',
        agents=[analysis_agent]  # (1)!
    )
    ui.servable()
    ```

    1. Adds AnalysisAgent alongside default agents

## Multiple analyses

``` py title="Multiple analyses"
wind_agent = lmai.agents.AnalysisAgent(analyses=[
    WindAnalysis(),
    TemperatureAnalysis(),
    PressureAnalysis(),
])

ui = lmai.ExplorerUI(
    data='data.csv',
    agents=[wind_agent]
)
ui.servable()
```

Lumen automatically picks the right analysis based on available columns and the user's question.

## Common patterns

=== "Statistical Analysis"

    ``` py title="Correlation analysis"
    class CorrelationAnalysis(lmai.Analysis):
        """Show correlation between variables."""
        
        columns = param.List(default=["x", "y"])
        
        def __call__(self, pipeline, *args, **kwargs):
            df = pipeline.data
            correlation = df["x"].corr(df["y"])
            
            plot = df.hvplot.scatter(
                x="x", y="y",
                title=f"Correlation: {correlation:.2f}"
            )
            return plot
    ```

=== "Financial Calculations"

    ``` py title="ROI analysis"
    class ROIAnalysis(lmai.Analysis):
        """Calculate return on investment."""
        
        columns = param.List(default=["revenue", "cost"])
        
        def __call__(self, pipeline, *args, **kwargs):
            df = pipeline.data.copy()
            df["roi"] = ((df["revenue"] - df["cost"]) / df["cost"]) * 100
            
            return df.hvplot.hist(y="roi", title="ROI Distribution")
    ```

=== "Data Quality"

    ``` py title="Quality check"
    import pandas as pd
    import panel as pn
    
    class QualityCheck(lmai.Analysis):
        """Check for missing and duplicate data."""
        
        columns = param.List(default=[])  # Works with any columns
        
        def __call__(self, pipeline, *args, **kwargs):
            df = pipeline.data
            
            report = pd.DataFrame([{
                "total_rows": len(df),
                "missing_values": df.isnull().sum().sum(),
                "duplicates": df.duplicated().sum(),
            }])
            
            return pn.widgets.Tabulator(report)
    ```

## Return types

Analyses can return any Panel viewable:

``` py title="Different return types"
def __call__(self, pipeline, *args, **kwargs):
    df = pipeline.data
    
    # HoloViews/hvPlot charts
    return df.hvplot(x="time", y="value")
    
    # Panel widgets
    return pn.widgets.Tabulator(df)
    
    # Matplotlib figures
    fig, ax = plt.subplots()
    ax.plot(df["x"], df["y"])
    return pn.pane.Matplotlib(fig)
    
    # Multiple components
    return pn.Column(chart, table, summary)
```

## Troubleshooting

### Analysis never runs

Check that required columns exist in your data with exact names (case-sensitive).

``` py title="Debug available columns"
def __call__(self, pipeline, *args, **kwargs):
    print("Available columns:", pipeline.data.columns.tolist())
    # Your logic
```

### Analysis fails silently

Add error handling:

``` py title="Error handling in analysis"
def __call__(self, pipeline, *args, **kwargs):
    try:
        df = pipeline.data
        # Your logic
        return result
    except Exception as e:
        return pn.pane.Alert(f"Analysis failed: {e}", alert_type="danger")
```

### Wrong columns

Column names are case-sensitive. Check your data:

``` py
# If data has "Value" (capital V)
columns = param.List(default=["value"])  # ❌ Won't match
columns = param.List(default=["Value"])  # ✅ Matches
```

### Forgot AnalysisAgent

Analyses require AnalysisAgent to work:

``` py hl_lines="2 5"
# Create agent with your analyses
analysis_agent = lmai.agents.AnalysisAgent(analyses=[MyAnalysis])

ui = lmai.ExplorerUI(
    data='data.csv',
    agents=[analysis_agent]  # Don't forget this!
)
```

## Best practices

**Keep analyses focused.** One analysis should perform one calculation or produce one type of output.

**Declare required columns.** Use `param.List(default=[...])` to specify what columns your analysis needs.

**Use param for configurability.** Add `param.Number`, `param.Integer`, etc. for values users might want to adjust.

**Return Panel viewables.** Use `pn.widgets.Tabulator`, `hvplot`, `pn.pane.Matplotlib`, or `pn.Column` to combine multiple outputs.

**Handle missing data.** Check for NaN values and handle them appropriately in your calculations.

**Test with real data.** Different datasets may have different distributions or edge cases.

**Document what it does.** Write clear docstrings explaining the calculation and when to use it.

## Working with global context

You can add domain knowledge that all agents see by setting template overrides on the base Actor class:

``` py title="Global context for all agents" hl_lines="1-7"
global_context = """
{{ super() }}

Domain knowledge:
- Inversions occur when temperature increases with altitude
- Standard lapse rate is 6.5°C per km
"""
lmai.actor.Actor.template_overrides = {"main": {"global": global_context}}

ui = lmai.ExplorerUI(data='weather.csv', agents=[analysis_agent])
ui.servable()
```

This adds the context to all agents' prompts, helping them understand your domain.

## Per-agent customization with analyses

Combine analyses with agent-specific template overrides:

``` py title="Domain-specific agent behavior" linenums="1"
# Add meteorology context to SQLAgent
sql_footer = """
To show a Skew-T diagram, you must have these columns:
- pressure_mb, tmpc, dwpc, speed_kts, drct, validUTC
"""
lmai.agents.SQLAgent.template_overrides = {"main": {"footer": sql_footer}}

# Make AnalystAgent use meteorological terminology
analyst_instructions = """
{{ super() }}

You are a meteorologist. Use proper meteorological terminology 
and explain atmospheric concepts clearly.
"""
lmai.agents.AnalystAgent.template_overrides = {
    "main": {"instructions": analyst_instructions}
}

# Create analysis
analysis_agent = lmai.agents.AnalysisAgent(analyses=[SkewTAnalysis])

ui = lmai.ExplorerUI(
    data='soundings.csv',
    agents=[analysis_agent],
    suggestions=[
        ("question_answer", "What is a Skew-T diagram?"),
        ("vertical_align_top", "Generate a Skew-T diagram."),
    ]
)
ui.servable()
```

This creates a specialized meteorology assistant with domain-specific analyses and terminology.
