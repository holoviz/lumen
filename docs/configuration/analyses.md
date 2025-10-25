# Analyses

Analyses let you define fixed, domain-specific calculations and visualizations that the LLM can invoke when appropriate. Instead of asking the LLM to generate queries every time, you specify analyses once—wind calculations, statistical tests, custom visualizations—and let the AnalysisAgent automatically apply them when the user's data matches.

## What are analyses?

Analyses are reusable, pre-built computations tailored to specific domains or use cases. Each analysis:

- Declares which columns it requires
- Checks if those columns exist in the current dataset
- Performs custom calculations or transformations
- Returns interactive visualizations or data tables

When a user asks a question, the AnalysisAgent checks available analyses, matches their required columns against the current data, and invokes the relevant ones.

## Using analyses

### Enable the AnalysisAgent

Include `AnalysisAgent` in your UI and provide custom analyses:

```python
import lumen.ai as lmai

class MyAnalysis(lmai.Analysis):
    columns = ["time", "value"]
    
    def __call__(self, pipeline):
        # Perform analysis and return result
        return pipeline

ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[lmai.agents.AnalysisAgent(analyses=[MyAnalysis()])]
)
ui.servable()
```

### Use multiple analyses

Pass multiple analyses to the agent:

```python
analyses_list = [
    WindAnalysis(),
    TemperatureAnalysis(),
    StatisticalTestAnalysis(),
]

ui = lmai.ExplorerUI(
    data='weather_data.csv',
    agents=[lmai.agents.AnalysisAgent(analyses=analyses_list)]
)
ui.servable()
```

### Automatic invocation

The AnalysisAgent automatically invokes analyses when:

1. User's query is relevant to the analysis
2. All required columns exist in the current dataset
3. The analysis hasn't been run recently

No explicit user request needed—the agent decides intelligently.

## Creating custom analyses

### Core concepts

**`columns`**

List of column names required for the analysis. If any are missing, the analysis is skipped.

```python
columns = ["time", "value"]  # Analysis requires both columns
```

**`__call__(pipeline)`**

Method that performs the analysis. Receives a data pipeline and returns a visualization or result.

```python
def __call__(self, pipeline):
    # Access data, perform calculations, return view
    return view
```

**`applies(pipeline)`**

Class method that checks if the analysis applies. Checks column existence by default.

```python
@classmethod
async def applies(cls, pipeline) -> bool:
    # Return True if analysis should be available
    return await super().applies(pipeline)
```

### Analysis structure

Create custom analyses by subclassing `lmai.Analysis`:

```python
import lumen.ai as lmai
from lumen.views import Table, hvPlotView

class CustomAnalysis(lmai.Analysis):
    """Description of what this analysis does."""
    
    columns = ["column1", "column2"]  # Required columns
    autorun = True  # Run automatically when columns are available
    
    def __call__(self, pipeline):
        # Perform calculations
        # Return visualization or component
        return Table(pipeline=pipeline)
```

### Build your first analysis

**Step 1: Define required columns**

Specify which columns the analysis needs:

```python
columns = ["species", "body_mass_g"]
```

**Step 2: Import visualization tools**

Use Lumen's view components to create outputs:

```python
from lumen.views import Table, hvPlotView, hvOverlayView
from lumen.layout import Layout
from lumen.transforms import Transform
import numpy as np
```

**Step 3: Create transforms if needed**

Preprocess data with custom transforms:

```python
class BodyMassAnalysis(Transform):
    """Calculate BMI categories."""
    
    def apply(self, df):
        df["mass_category"] = pd.cut(
            df["body_mass_g"],
            bins=[0, 3500, 4500, 6500],
            labels=["Light", "Medium", "Heavy"]
        )
        return df
```

**Step 4: Implement `__call__`**

Chain transforms and create views:

```python
def __call__(self, pipeline):
    # Apply custom transform
    analyzed_pipeline = pipeline.chain(
        transforms=[BodyMassAnalysis()]
    )
    
    # Create visualizations
    view1 = hvPlotView(
        pipeline=analyzed_pipeline,
        x="species",
        y="body_mass_g",
        kind="box"
    )
    
    view2 = Table(pipeline=analyzed_pipeline)
    
    # Return layout with multiple views
    return Layout(
        views=[view1, view2],
        layout=[[0], [1]]
    )
```

### Example: Wind Analysis

Here's a complete meteorology analysis:

```python
import numpy as np
import pandas as pd
import lumen.ai as lmai
from lumen.layout import Layout
from lumen.transforms import Transform
from lumen.views import hvPlotView, hvOverlayView, Table

class WindSpeedDirection(Transform):
    """Calculate wind speed and direction from u,v components."""
    
    u_component = param.String(default="u")
    v_component = param.String(default="v")
    
    def apply(self, df):
        # Calculate wind speed from components
        df["wind_speed"] = np.sqrt(
            df[self.u_component] ** 2 + df[self.v_component] ** 2
        )
        
        # Calculate wind direction (degrees from north)
        df["wind_direction"] = np.degrees(
            np.arctan2(df[self.v_component], df[self.u_component])
        )
        df["wind_direction"] = (df["wind_direction"] + 360) % 360
        
        return df

class WindAnalysis(lmai.Analysis):
    """Analyzes wind patterns from u,v components."""
    
    columns = ["time", "u", "v"]
    autorun = True
    
    def __call__(self, pipeline):
        # Apply wind calculations
        wind_pipeline = pipeline.chain(
            transforms=[WindSpeedDirection()]
        )
        
        # Create wind speed time series
        wind_speed_view = hvPlotView(
            pipeline=wind_pipeline,
            x="time",
            y="wind_speed",
            title="Wind Speed"
        )
        
        # Overlay wind direction
        wind_direction_view = hvPlotView(
            pipeline=wind_pipeline,
            x="time",
            y="wind_direction",
            title="Wind Direction"
        )
        
        # Data table
        wind_table = Table(pipeline=wind_pipeline)
        
        # Combine views
        return Layout(
            views=[
                hvOverlayView(
                    layers=[wind_speed_view, wind_direction_view]
                ),
                wind_table
            ],
            layout=[[0], [1]]
        )
```

Usage:

```python
import lumen.ai as lmai

# Create sample data
uv_df = pd.DataFrame({
    "time": pd.date_range('2024-11-11', '2024-11-22'),
    "u": np.random.rand(12),
    "v": np.random.rand(12)
})

source = lmai.memory["source"] = DuckDBSource.from_df({"uv_df": uv_df})

# Use analysis
ui = lmai.ExplorerUI(
    agents=[lmai.agents.AnalysisAgent(analyses=[WindAnalysis()])]
)
ui.servable()
```

### Control when analyses run

Use `autorun` to control automatic execution:

```python
class ExpensiveAnalysis(lmai.Analysis):
    """Complex analysis that shouldn't run automatically."""
    
    columns = ["col1", "col2"]
    autorun = False  # Only run when explicitly requested
    
    def __call__(self, pipeline):
        # Expensive computation
        return result
```

### Add interactive controls

Analyses can have configurable parameters:

```python
class ParameterizedAnalysis(lmai.Analysis):
    
    columns = ["time", "value"]
    
    threshold = param.Number(default=50)  # User-configurable
    method = param.Selector(default="mean", objects=["mean", "median"])
    
    def controls(self):
        """Return UI controls for parameters."""
        import panel as pn
        return pn.Param(self.param, parameters=["threshold", "method"])
    
    def __call__(self, pipeline):
        # Use self.threshold and self.method
        return result
```

## Best practices

**One analysis, one purpose.** Each analysis should solve a specific problem. Create multiple analyses for different tasks rather than one omnibus analysis.

**Declare columns precisely.** List exactly which columns are required. Incomplete declarations cause analyses to skip silently.

**Test with real data.** Verify your analysis works with the actual data distributions you'll encounter.

**Cache expensive operations.** If an analysis is computationally expensive, check if results can be reused.

**Keep transforms simple.** Complex transforms make analyses harder to debug. Keep each transform focused.

**Return informative visualizations.** Don't just return data—add context with titles, labels, and annotations.

**Handle edge cases.** Check for empty data, null values, and outliers. Provide graceful error messages.

**Document requirements clearly.** Explain what columns are needed and what the analysis computes in docstrings.
