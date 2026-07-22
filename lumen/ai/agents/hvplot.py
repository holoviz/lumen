from typing import Any, Literal

import param

from pydantic import BaseModel, Field, create_model

from ...views import hvPlotUIView
from ..config import PROMPTS_DIR
from ..context import TContext
from ..utils import get_data
from .base_view import BaseViewAgent


def resolve_cmap(cmap: str) -> str:
    """
    Map a semantic colormap name onto a concrete one.

    hvPlot understands 'linear', 'diverging', 'categorical' and 'cyclic' in its
    converter, but `hvplot.ui.Colormapping.cmap` is a Selector over concrete
    colormap names and rejects them, so they have to be resolved before the
    spec reaches the view. The categorical default resolves to a list of
    colours, which the Selector also rejects, hence the named fallback.
    """
    from hvplot.ui import CMAPS, DEFAULT_CMAPS

    resolved = DEFAULT_CMAPS.get(cmap, cmap)
    if not isinstance(resolved, str) or resolved not in CMAPS:
        return "glasbey"
    return resolved


class hvPlotSpec(BaseModel):
    """Fields that do not depend on the data schema."""

    chain_of_thought: str = Field(
        description="Your thought process behind the plot."
    )

    cmap: Literal["linear", "diverging", "categorical", "cyclic"] | None = Field(
        default=None,
        description=(
            "Colormap to use when a column is mapped to colour. Pick 'diverging' "
            "for values around a meaningful centre such as anomalies or deviations, "
            "'categorical' for unordered categories, 'cyclic' for wrapping values "
            "such as wind direction or hour of day, and 'linear' otherwise."
        ),
    )

    cnorm: Literal["linear", "log", "eq_hist"] | None = Field(
        default=None,
        description="Colour scale normalization. Use 'log' for values spanning orders of magnitude.",
    )

    colorbar: bool | None = Field(
        default=None,
        description="Whether to show a colorbar. Leave unset to let hvPlot decide.",
    )

    geo: bool = Field(
        default=False,
        description="Whether the data is geographic and should be plotted on a map.",
    )

    limit: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of rows to plot. Leave unset to plot all rows.",
    )

    title: str | None = Field(
        default=None,
        description="Title describing what the plot shows.",
    )


def make_hvplot_model(view_type: type[hvPlotUIView], schema: dict[str, Any]) -> type[BaseModel]:
    """
    Build the structured-output model for a given view type and data schema.

    The column fields are restricted to the columns actually present so the
    model cannot reference a column that does not exist, and `kind` is derived
    from the view's own Selector so the two cannot drift apart.
    """
    kinds = tuple(view_type.param["kind"].objects)
    columns = tuple(schema)
    # Literal[()] is not a valid annotation, so fall back to a plain string
    # when the schema is empty.
    column = Literal[columns] if columns else str

    return create_model(
        "hvPlotSpecWithColumns",
        by=(
            list[column] | None,
            Field(default=None, description="Columns to split into separate series."),
        ),
        color=(
            column | None,
            Field(default=None, description="Column to map onto colour."),
        ),
        groupby=(
            list[column] | None,
            Field(default=None, description="Columns to group by into widgets."),
        ),
        kind=(
            Literal[kinds],
            Field(description="The kind of plot to generate."),
        ),
        x=(
            column | None,
            Field(default=None, description="Column to plot on the x-axis."),
        ),
        y=(
            column | None,
            Field(default=None, description="Column to plot on the y-axis."),
        ),
        __base__=hvPlotSpec,
    )


class hvPlotAgent(BaseViewAgent):

    conditions = param.List(
        default=[
            "Use for exploratory data analysis, interactive plots, and dynamic filtering",
            "Use for quick, iterative data visualization during analysis",
            "Can render a geometry column (GeoDataFrame polygons/lines/points) as a "
            "2D choropleth, shading shapes by a value column without a basemap; fits "
            "geometry whose schema is not 'geographic' (a projected or absent CRS) or "
            "when a basemap adds no context",
        ]
    )

    purpose = param.String(default="Generates a plot of the data given a user prompt.")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "hvPlotAgent" / "main.jinja2"},
        }
    )

    view_type = hvPlotUIView

    def _get_model(self, prompt_name: str, schema: dict[str, Any]) -> type[BaseModel]:
        return make_hvplot_model(self.view_type, schema)

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        pipeline = context["pipeline"]
        spec = {key: val for key, val in spec.items() if val is not None}
        # Reasoning is streamed to the UI, it is not part of the view spec.
        spec.pop("chain_of_thought", None)
        spec["type"] = "hvplot_ui"
        self.view_type.validate(spec)
        spec.pop("type", None)

        # Add defaults
        spec["responsive"] = True
        data = await get_data(pipeline)

        if "cmap" in spec:
            spec["cmap"] = resolve_cmap(spec["cmap"])

        # Colouring by a non-numeric column makes hvPlot default the colormap to
        # a list of colours, which the explorer's Selector rejects, so name one.
        if (color := spec.get("color")) in data and data[color].dtype.kind in "OSU":
            spec.setdefault("cmap", resolve_cmap("categorical"))

        if len(data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            spec["rasterize"] = True
            spec.setdefault("cnorm", "log")
            spec.setdefault("cmap", resolve_cmap("linear"))
        return spec
