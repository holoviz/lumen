import importlib.util
import re

from typing import Any

import pandas as pd
import param

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ...util import is_geodataframe
from ...views import hvPlotView
from ...views.base import BAR_KINDS, GRIDDED_KINDS, VALUE_AGGREGATORS
from ..config import PROMPTS_DIR
from ..context import TContext
from ..translate import param_to_pydantic
from ..utils import get_data
from .base_view import BaseViewAgent

# Rows (grid cells) past which a gridded plot is rasterized, as Bokeh slows
# down drawing a quadmesh or image cell by cell
GRID_RASTERIZE_CELLS = 250_000

LON_NAMES, LAT_NAMES = {"lon", "longitude", "lng"}, {"lat", "latitude"}


# Words in a column name that mark a cyclic quantity, which wraps around
# (359 degrees sits next to 0), and a signed one, read from zero
CYCLIC_WORDS = {"direction", "dir", "bearing", "heading", "azimuth", "angle", "phase", "hour", "month", "doy", "dayofyear"}
SIGNED_WORDS = {"anomaly", "anom", "diff", "difference", "change", "delta", "trend", "bias", "residual", "departure"}


def _default_cmap(column: pd.Series) -> str | None:
    """
    A colorcet colormap for a colour column whose name says what it holds,
    or None to leave the choice to hvPlot, which already turns diverging
    when the values cross zero.
    """
    if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
        return None
    words = set(re.split(r"[^a-z]+", str(column.name).lower()))
    if words & CYCLIC_WORDS:
        return "colorwheel"
    if words & SIGNED_WORDS:
        return "coolwarm"
    return None


def _is_lon_lat(spec: dict[str, Any], data: Any) -> bool:
    """Whether the plot draws longitude and latitude, as geometry or as x and y."""
    if is_geodataframe(data):
        return data.crs is not None and data.crs.is_geographic
    x, y = str(spec.get("x", "")).lower(), str(spec.get("y", "")).lower()
    return x in LON_NAMES and y in LAT_NAMES


class hvPlotAgent(BaseViewAgent):

    conditions = param.List(
        default=[
            "Use for exploratory data analysis, interactive plots, and dynamic filtering",
            "Use for quick, iterative data visualization during analysis",
            "Prefer over Vega-Lite for maps and gridded data",
            "Can render a geometry column (GeoDataFrame polygons/lines/points) as a "
            "2D map or choropleth, shading shapes by a value column; geometry whose "
            "schema is 'geographic' (a lat/lon CRS) and lon/lat points are drawn on "
            "basemap tiles, projected geometry without one",
        ]
    )

    purpose = param.String(default="Generates a plot of the data given a user prompt.")

    user = param.String(default="hvPlot")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "hvPlotAgent" / "main.jinja2"},
        }
    )

    view_type = hvPlotView

    def _get_model(self, prompt_name: str, schema: dict[str, Any] | None = None) -> type[BaseModel]:
        # Only the main prompt describes the view. Every other prompt, revise
        # among them, carries its own response model and is passed no schema,
        # and revise is what runs when a spec fails to render.
        if schema is None:
            return super()._get_model(prompt_name)

        # Find parameters
        excluded = self.view_type._internal_params + [
            "controls",
            "type",
            "source",
            "pipeline",
            "transforms",
            "download",
            "field",
            "selection_group",
            # Runtime state and HoloViews-level escape hatches: nothing a plot
            # request asks for, and nothing the model could fill sensibly.
            "operations",
            "opts",
            "selection_expr",
            "streaming",
        ]
        model = param_to_pydantic(
            self.view_type,
            base_model=BaseModel,
            excluded=excluded,
            schema=schema,
            extra_fields={
                "chain_of_thought": (str, FieldInfo(description="Your thought process behind the plot.")),
            },
            # Only this one view is being described. Expanding subclasses is for
            # callers that want a union over a taxonomy, and here it would walk
            # every subclass of every base, down into Panel and Bokeh objects
            # that have no JSON schema and nothing to do with a plot.
            process_subclasses=False,
        )
        return model[self.view_type.__name__]

    @staticmethod
    def _drop_conflicting_axes(spec: dict[str, Any], geometry: bool = False) -> None:
        """Remove axis assignments that contradict each other.

        The prompt asks for x, y, by and groupby to name distinct columns, and
        the model does not always oblige. A groupby repeating x or y raises
        while the plot is built, and one repeating by is worse than that: it
        pages each category into its own frame, so a datashaded plot renders
        without complaint and blends nothing.
        """
        taken = {spec.get("x"), spec.get("y"), *(spec.get("by") or [])}
        groupby = [col for col in (spec.get("groupby") or []) if col not in taken]
        if groupby:
            spec["groupby"] = groupby
        else:
            spec.pop("groupby", None)
        # z belongs to the gridded kinds, plus heatmap, which takes the same
        # column as C, and to points and maps, which are coloured by it.
        # Elsewhere hvPlot only warns that it is unused, which is a warning
        # nobody reads.
        kind = spec.get("kind")
        if kind not in (*GRIDDED_KINDS, "heatmap", "points", "scatter") and not geometry:
            spec.pop("z", None)
        # Reducing a value column needs one named, and the spec has no field for
        # it, so an aggregator asking for that has nothing to work on.
        if spec.get("aggregator") in VALUE_AGGREGATORS:
            spec.pop("aggregator", None)

    @staticmethod
    def _drop_unknown_columns(spec: dict[str, Any], columns: set[str]) -> None:
        """Drop axes naming a column the query did not produce.

        The model is offered the columns of the table it was given, but it can
        still answer with one it expected the query to create. hvPlot only
        finds out while drawing, where it raises a bare KeyError and the whole
        chart is lost; dropping the axis leaves hvPlot to infer one instead.
        """
        for field in ("x", "y", "z", "size"):
            if spec.get(field) is not None and spec[field] not in columns:
                spec.pop(field)
        for field in ("by", "groupby"):
            known = [column for column in spec.get(field) or [] if column in columns]
            if known:
                spec[field] = known
            else:
                spec.pop(field, None)

    @staticmethod
    def _complete_axes(spec: dict[str, Any], data: pd.DataFrame) -> None:
        """Fill in what the model left out where the data settles it."""
        x, y = spec.get("x"), spec.get("y")
        # Without x a bar chart is drawn against the row index; with one text
        # column there is only one thing x can mean
        labels = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
        if spec.get("kind") in BAR_KINDS and x is None and len(labels) == 1:
            spec["x"] = labels[0]
        # Longitude and latitude covering every pair once are a grid, which
        # points draw as a cloud of dots with no value on them
        if (
            spec.get("kind") in ("points", "scatter") and _is_lon_lat(spec, data)
            and not is_geodataframe(data) and len(data) > 1
            and data[x].nunique() * data[y].nunique() == len(data)
        ):
            spec["kind"] = "quadmesh"
        values = [
            c for c in data.columns
            if c not in (x, y) and pd.api.types.is_numeric_dtype(data[c])
        ]
        if spec.get("kind") in GRIDDED_KINDS and "z" not in spec and len(values) == 1:
            spec["z"] = values[0]

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        pipeline = context["pipeline"]
        spec = {key: val for key, val in spec.items() if val is not None}
        # Asked for to make the model reason about the plot, not to be plotted;
        # hvPlotExplorer rejects any keyword none of its controls claims.
        spec.pop("chain_of_thought", None)
        data = await get_data(pipeline)
        self._drop_conflicting_axes(spec, geometry=is_geodataframe(data))
        spec["type"] = "hvplot"
        self.view_type.validate(spec)
        spec.pop("type", None)

        # Add defaults
        spec["responsive"] = True
        # hvPlot turns Bokeh's grid lines off, and values are hard to read
        # off a chart without them
        spec.setdefault("grid", True)
        self._drop_unknown_columns(spec, set(data.columns))
        self._complete_axes(spec, data)
        colour = next((spec[k] for k in ("z", "c", "color") if spec.get(k) in data.columns), None)
        if colour is not None and "cmap" not in spec and (cmap := _default_cmap(data[colour])):
            spec["cmap"] = cmap
            if cmap == "coolwarm":
                # Centred on zero, so the colours split where the sign does
                spec.setdefault("symmetric", True)
        if _is_lon_lat(spec, data):
            # Longitude and latitude read best on a basemap
            spec.setdefault("geo", True)
            if spec["geo"]:
                spec.setdefault("tiles", "OSM")
        if spec.get("geo") and spec.get("kind") in GRIDDED_KINDS:
            # A grid hides the tiles under it and Web Mercator stretches its
            # polar rows; coastlines on longitude and latitude show it whole
            spec.pop("tiles", None)
            spec.setdefault("coastline", True)
            # Projecting an image resamples it, which needs scipy or pykdtree;
            # a quadmesh projects its cell corners and needs neither
            if spec["kind"] == "image":
                spec["kind"] = "quadmesh"
        if spec.get("geo"):
            # A map keeps the aspect of its data, which a responsive plot can
            # only meet by zooming far out or in to fill its container; a fixed
            # height lets the width follow the data instead
            spec.pop("responsive")
            spec.setdefault("frame_height", 400)
        # Rasterizing and datashading need datashader, which is not a dependency
        datashader = importlib.util.find_spec("datashader") is not None
        if datashader and spec.get("kind") in GRIDDED_KINDS and len(data) > GRID_RASTERIZE_CELLS:
            spec["rasterize"] = True
        if datashader and len(data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            if spec.get("by"):
                # rasterize reduces to a single number per pixel, which throws
                # away the category; datashade blends the ones sharing a pixel.
                spec["datashade"] = True
            else:
                spec["rasterize"] = True
                spec["cnorm"] = "log"
        return spec
