
from typing import Any

import param

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ...views import hvPlotUIView
from ..config import PROMPTS_DIR
from ..context import TContext
from ..shared.models import PartialBaseModel
from ..translate import param_to_pydantic
from ..utils import get_data
from .base_view import BaseViewAgent


class hvPlotAgent(BaseViewAgent):

    conditions = param.List(
        default=[
            "Use for exploratory data analysis, interactive plots, and dynamic filtering",
            "Use for quick, iterative data visualization during analysis",
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
        ]
        model = param_to_pydantic(
            self.view_type,
            base_model=PartialBaseModel,
            excluded=excluded,
            schema=schema,
            extra_fields={
                "chain_of_thought": (str, FieldInfo(description="Your thought process behind the plot.")),
            },
        )
        return model[self.view_type.__name__]

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        pipeline = context["pipeline"]
        spec = {key: val for key, val in spec.items() if val is not None}
        spec["type"] = "hvplot_ui"
        self.view_type.validate(spec)
        spec.pop("type", None)

        # Add defaults
        spec["responsive"] = True
        data = await get_data(pipeline)
        if len(data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            spec["rasterize"] = True
            spec["cnorm"] = "log"
        return spec
