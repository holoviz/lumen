# Analyses

**Analyses are functions that run deterministically with code instead of LLM generation.**

Like tools or hooks, analyses give you control over specific calculations. Use them when you need reliable, repeatable results—financial metrics, scientific formulas, or statistical tests that must be calculated exactly the same way every time.

## Quick example

```python
import lumen.ai as lmai
from lumen.views import Table

class SummaryStats(lmai.Analysis):
    """Calculate summary statistics."""
    
    columns = ["value"]  # Required columns
    
    def __call__(self, pipeline):
        df = pipeline.data
        stats = df.describe()
        return Table(data=stats)

ui = lmai.ExplorerUI(
    data='data.csv',
    analyses=[SummaryStats()]
)
ui.servable()
```

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

```python
import lumen.ai as lmai
from lumen.views import hvPlotView

class MyAnalysis(lmai.Analysis):
    """What this analysis does."""
    
    columns = ["col1", "col2"]  # Required columns
    
    def __call__(self, pipeline):
        # Your calculation logic
        df = pipeline.data
        result = df["col1"] * df["col2"]
        
        # Return a view
        return hvPlotView(
            data=result,
            kind="line"
        )
```

## Complete example: Wind analysis

```python
import numpy as np
import lumen.ai as lmai
from lumen.transforms import Transform
from lumen.views import hvPlotView
from lumen.layout import Layout

class WindCalculation(Transform):
    """Convert u,v components to speed and direction."""
    
    def apply(self, df):
        df["wind_speed"] = np.sqrt(df["u"]**2 + df["v"]**2)
        df["wind_direction"] = np.degrees(np.arctan2(df["v"], df["u"]))
        return df

class WindAnalysis(lmai.Analysis):
    """Analyze wind patterns from u,v components."""
    
    columns = ["time", "u", "v"]
    
    def __call__(self, pipeline):
        # Apply calculation
        wind_pipeline = pipeline.chain(transforms=[WindCalculation()])
        
        # Create speed chart
        speed_chart = hvPlotView(
            pipeline=wind_pipeline,
            x="time",
            y="wind_speed",
            title="Wind Speed"
        )
        
        # Create direction chart
        direction_chart = hvPlotView(
            pipeline=wind_pipeline,
            x="time",
            y="wind_direction",
            title="Wind Direction"
        )
        
        return Layout(views=[speed_chart, direction_chart], layout=[[0], [1]])

# Use it
ui = lmai.ExplorerUI(
    data='weather.csv',
    analyses=[WindAnalysis()]
)
ui.servable()
```

Now users can ask "Analyze wind patterns" and get both charts automatically.

## Control when analyses run

```python
class ExpensiveAnalysis(lmai.Analysis):
    """Complex calculation."""
    
    columns = ["data"]
    autorun = False  # Only run when explicitly requested
    
    def __call__(self, pipeline):
        # Expensive computation
        return result
```

Set `autorun = False` for analyses that:

- Take a long time to compute
- Should only run on explicit request
- Have expensive operations

## Add configurable parameters

```python
import param
from lumen.views import hvPlotView

class ThresholdAnalysis(lmai.Analysis):
    
    columns = ["value"]
    threshold = param.Number(default=50)  # User can adjust
    
    def __call__(self, pipeline):
        df = pipeline.data
        filtered = df[df["value"] > self.threshold]
        
        return hvPlotView(
            data=filtered,
            y="value",
            kind="hist"
        )
```

## Multiple analyses

```python
ui = lmai.ExplorerUI(
    data='data.csv',
    analyses=[
        WindAnalysis(),
        TemperatureAnalysis(),
        PressureAnalysis(),
    ]
)
ui.servable()
```

Lumen automatically picks the right analysis based on:

- Available columns in current data
- User's question
- Analysis descriptions

## Common patterns

### Statistical analysis

```python
class CorrelationAnalysis(lmai.Analysis):
    """Show correlation between variables."""
    
    columns = ["x", "y"]
    
    def __call__(self, pipeline):
        df = pipeline.data
        correlation = df["x"].corr(df["y"])
        
        return hvPlotView(
            pipeline=pipeline,
            x="x",
            y="y",
            kind="scatter",
            title=f"Correlation: {correlation:.2f}"
        )
```

### Financial calculations

```python
class ROIAnalysis(lmai.Analysis):
    """Calculate return on investment."""
    
    columns = ["revenue", "cost"]
    
    def __call__(self, pipeline):
        df = pipeline.data
        df["roi"] = ((df["revenue"] - df["cost"]) / df["cost"]) * 100
        
        return hvPlotView(
            pipeline=pipeline.chain(transforms=[]),
            y="roi",
            kind="hist",
            title="ROI Distribution"
        )
```

### Data quality checks

```python
class QualityCheck(lmai.Analysis):
    """Check for missing and duplicate data."""
    
    columns = []  # Works with any columns
    
    def __call__(self, pipeline):
        df = pipeline.data
        
        report = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
        }
        
        return Table(data=pd.DataFrame([report]))
```

## Troubleshooting

**Analysis never runs:**

Check that required columns exist in your data with exact names (case-sensitive).

**Analysis fails silently:**

Add error handling:

```python
def __call__(self, pipeline):
    try:
        # Your logic
        return result
    except Exception as e:
        print(f"Analysis failed: {e}")
        return Table(data=pd.DataFrame([{"error": str(e)}]))
```

**Can't find analysis columns:**

Print available columns during development:

```python
def __call__(self, pipeline):
    print("Available columns:", pipeline.data.columns.tolist())
    # Your logic
```

## Best practices

- Keep analyses focused on one calculation
- Declare required columns explicitly
- Return informative visualizations with titles
- Handle missing data gracefully
- Test with real data distributions
- Document what the analysis does in docstrings
