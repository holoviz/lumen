from typing import Any

import param

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ...views import hvPlotView
from ...views.hvplot import GRIDDED_KINDS, VALUE_AGGREGATORS
from ..config import PROMPTS_DIR
from ..context import TContext
from ..translate import param_to_pydantic
from ..utils import get_data
from .base_view import BaseViewAgent


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
    def _drop_conflicting_axes(spec: dict[str, Any]) -> None:
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
        # column as C. Elsewhere hvPlot only warns that it is unused, which is
        # a warning nobody reads.
        kind = spec.get("kind")
        if kind not in GRIDDED_KINDS and kind != "heatmap":
            spec.pop("z", None)
        # Reducing a value column needs one named, and the spec has no field for
        # it, so an aggregator asking for that has nothing to work on.
        if spec.get("aggregator") in VALUE_AGGREGATORS:
            spec.pop("aggregator", None)

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        pipeline = context["pipeline"]
        spec = {key: val for key, val in spec.items() if val is not None}
        # Asked for to make the model reason about the plot, not to be plotted;
        # hvPlotExplorer rejects any keyword none of its controls claims.
        spec.pop("chain_of_thought", None)
        self._drop_conflicting_axes(spec)
        spec["type"] = "hvplot"
        self.view_type.validate(spec)
        spec.pop("type", None)

        # Add defaults
        spec["responsive"] = True
        data = await get_data(pipeline)
        if len(data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            if spec.get("by"):
                # rasterize reduces to a single number per pixel, which throws
                # away the category; datashade blends the ones sharing a pixel.
                spec["datashade"] = True
            else:
                spec["rasterize"] = True
                spec["cnorm"] = "log"
        return spec
