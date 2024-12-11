
# Custom Analyses

Sometimes, users may want to output views of fixed analyses that the LLMs may not be able to reproduce on its own.

To achieve this, the user can write their own custom `Analysis` to perform custom actions tailored to a particular use case and/or domain.

The `AnalysisAgent` will then invoke the custom `Analysis` when needed based on relevance to the user's query and whether the current exploration's dataset contains all the required `columns`.

As a basic example, the user may be a meteorologist and want to perform a `WindAnalysis`.

```python
import param
import numpy as np
import pandas as pd
import lumen.ai as lmai
from lumen.layout import Layout
from lumen.transforms import Transform
from lumen.sources.duckdb import DuckDBSource
from lumen.views import hvPlotView, Table


class WindSpeedDirection(Transform):

    u_component = param.String(default="u", doc="Column name of the zonal component")

    v_component = param.String(default="v", doc="Column name of the meridional component")

    def apply(self, df):
        # Calculate wind speed
        df["wind_speed"] = np.sqrt(
            df[self.u_component] ** 2 + df[self.v_component] ** 2
        )

        # Calculate wind direction in degrees (from north)
        df["wind_direction"] = np.degrees(
            np.arctan2(df[self.v_component], df[self.u_component])
        )

        # Ensure wind direction is in the range [0, 360)
        df["wind_direction"] = (df["wind_direction"] + 360) % 360
        return df


class WindAnalysis(lmai.Analysis):
    """
    Calculates the wind speed and direction from the u and v components of wind,
    displaying both the table and meteogram.
    """

    columns = param.List(default=["time", "u", "v"])

    def __call__(self, pipeline):
        wind_pipeline = pipeline.chain(transforms=[WindSpeedDirection()])
        wind_speed_view = hvPlotView(
            pipeline=wind_pipeline,
            title="Wind Speed",
            x="time",
            y="wind_speed",
        )
        wind_direction_view = hvPlotView(
            pipeline=wind_pipeline,
            title="Wind Speed",
            x="time",
            y="wind_speed",
            text="wind_direction",
            kind="labels",
        )
        wind_table = Table(wind_pipeline)
        return Layout(
            views=[
                wind_speed_view,
                wind_direction_view,
                wind_table,
            ],
            layout=[[0, 1], [2]],
        )


llm = lmai.llm.Llama()
uv_df = pd.DataFrame({
    "time": pd.date_range('2024-11-11', '2024-11-22'),
    "u": np.random.rand(12),
    "v": np.random.rand(12)
})
source = lmai.memory["current_source"] = DuckDBSource.from_df({"uv_df": uv_df})
analysis_agent = lmai.agents.AnalysisAgent(analyses=[WindAnalysis])
ui = lmai.ExplorerUI(llm=llm, agents=[analysis_agent])
ui.servable()
```
