"""
DeckGL Agent for generating 3D map visualizations.

This agent generates deck.gl visualizations either:
- Directly as JSON specs (when code_execution is disabled)
- Via PyDeck Python code execution (when code_execution is enabled)
"""
from __future__ import annotations

import json

from typing import TYPE_CHECKING, Any

import param

from pydantic import Field

from ...views.base import DeckGLView
from ..code_executor import CodeSafetyCheck, PyDeckExecutor
from ..config import PROMPTS_DIR, UserCancelledError
from ..models import EscapeBaseModel, PartialBaseModel, RetrySpec
from ..utils import (
    get_data, get_schema, retry_llm_output, sanitize_column_names,
)
from ..views import DeckGLOutput
from .base_code import BaseCodeAgent

if TYPE_CHECKING:
    from ...pipeline import Pipeline
    from ..context import TContext
    from ..llm import Message


class DeckGLSpec(EscapeBaseModel):
    """Response model for DeckGL JSON spec generation (declarative mode)."""

    chain_of_thought: str = Field(
        description="""Explain your design choices for the 3D visualization:
        - What geographic patterns does this data reveal?
        - Which layer type best represents the data (HexagonLayer, ScatterplotLayer, etc.)?
        - What visual encodings (elevation, color, radius) highlight key insights?
        Keep response to 1-2 sentences.""",
        examples=[
            "The data shows population density clustering around urban centers—HexagonLayer with elevation encoding count will create an intuitive 3D cityscape effect.",
            "Point locations with varying magnitudes suggest ScatterplotLayer with radius encoding value and color gradient for intensity."
        ]
    )
    json_spec: str = Field(
        description="""A DeckGL JSON specification. Must include:
        - initialViewState with latitude, longitude, zoom, pitch, bearing
        - layers array with @@type for each layer
        Do NOT include 'data' in layers - it will be injected automatically.
        Use @@= syntax for accessors, e.g. "getPosition": "@@=[longitude, latitude]"
        """
    )


class PyDeckSpec(PartialBaseModel):
    """Response model for PyDeck code generation."""

    chain_of_thought: str = Field(
        default="",
        description="""Explain your design choices for the 3D visualization:
        - What geographic patterns does this data reveal?
        - Which layer type best represents the data?
        - What visual encodings highlight key insights?
        Keep response to 1-2 sentences.""",
        examples=[
            "The data shows CO2 emissions by plant location—HexagonLayer aggregating emissions with elevation shows concentration hotspots.",
            "Wind turbine locations with capacity data—ScatterplotLayer with radius proportional to capacity reveals infrastructure distribution."
        ]
    )
    code: str = Field(
        description="""Python code that creates a PyDeck visualization.
        Requirements:
        - Import pydeck as `pdk`
        - Data is available as `df` (pandas DataFrame)
        - Column names have spaces replaced with underscores and non-alphanumeric chars removed
        - Must assign final deck to variable `deck`
        - Do NOT call .to_html(), .show(), or any I/O methods
        ```
        """
    )


class DeckGLAgent(BaseCodeAgent):
    """Agent for generating DeckGL 3D map visualizations.

    Supports two generation modes:
    - When code_execution is 'disabled': Generate DeckGL JSON specs directly
    - When code_execution is enabled: Generate PyDeck Python code, execute safely, convert to spec
    """

    conditions = param.List(
        default=[
            "Use for 3D geographic visualizations, map-based data, or when user requests DeckGL/deck.gl",
            "Use for large-scale geospatial data with latitude/longitude coordinates",
            "Use for hexbin aggregations, heatmaps, or 3D extruded visualizations on maps",
        ]
    )

    purpose = param.String(
        default="Generates DeckGL 3D map visualizations from geographic data."
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": DeckGLSpec,
                "template": PROMPTS_DIR / "DeckGLAgent" / "main.jinja2",
            },
            "main_pydeck": {
                "response_model": PyDeckSpec,
                "template": PROMPTS_DIR / "DeckGLAgent" / "main_pydeck.jinja2",
            },
            "code_safety": {
                "response_model": CodeSafetyCheck,
                "template": PROMPTS_DIR / "DeckGLAgent" / "code_safety.jinja2",
            },
            "revise_output": {
                "response_model": RetrySpec,
                "template": PROMPTS_DIR / "DeckGLAgent" / "revise_output.jinja2",
            },
        }
    )

    user = param.String(default="DeckGL")

    # Default map style (CartoDB, no API key needed)
    default_map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

    view_type = DeckGLView

    _executor_class = PyDeckExecutor

    _extensions = ("deckgl",)

    _output_type = DeckGLOutput

    def __init__(self, **params):
        self._last_output = None
        super().__init__(**params)

    @retry_llm_output()
    async def _generate_yaml_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        errors: list | None = None
    ) -> dict[str, Any]:
        """Generate DeckGL spec via YAML/JSON (declarative mode)."""
        errors_context = self._build_errors_context(pipeline, context, errors)

        with self._add_step(title="Generating DeckGL specification", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "main",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                **errors_context,
            )

            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=DeckGLSpec,
            )

            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            step.stream(f"\n```json\n{output.json_spec}\n```", replace=False)
            step.success_title = "DeckGL specification created"

        self._last_output = {"json_spec": output.json_spec}
        return await self._extract_spec(context, {"json_spec": output.json_spec})

    @retry_llm_output()
    async def _generate_code_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        errors: list | None = None,
    ) -> dict[str, Any] | None:
        """Generate spec via PyDeck code execution."""
        errors_context = self._build_errors_context(pipeline, context, errors)

        with self._add_step(title="Generating PyDeck code", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "main_pydeck",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                **errors_context,
            )

            model_spec = self.prompts["main_pydeck"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=PyDeckSpec,
            )

            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            step.stream(f"\n```python\n{output.code}\n```\n", replace=False)

            # Get LLM system prompt for safety validation if needed
            system = None
            if self.code_execution == "llm":
                system = await self._render_prompt(
                    "code_safety",
                    messages,
                    context,
                    code=output.code,
                )

            # Execute code using base class method (handles AST validation, LLM validation, user prompt)
            df = await get_data(pipeline)
            df = sanitize_column_names(df)
            deck = await self._execute_code(output.code, df, system=system, step=step)

        if deck is None:
            raise UserCancelledError("Code execution rejected by user.")

        # Convert PyDeck Deck to JSON spec
        spec = deck.to_json()
        if isinstance(spec, str):
            spec = json.loads(spec)

        self._last_output = {"json_spec": spec}
        return await self._extract_spec(context, {"json_spec": spec})

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate DeckGL spec from PyDeck output or JSON spec."""

        # Handle both json_spec (from either mode) and direct spec dict
        if json_spec := spec.get("json_spec"):
            if isinstance(json_spec, str):
                deckgl_spec = json.loads(json_spec)
            else:
                deckgl_spec = json_spec
        else:
            deckgl_spec = spec

        # Handle map style - PyDeck uses 'map_style', DeckGL uses 'mapStyle'
        # Also replace Mapbox styles (which require API key) with CartoDB
        map_style = deckgl_spec.pop("map_style", None) or deckgl_spec.get("mapStyle")
        if not map_style or (isinstance(map_style, str) and map_style.startswith("mapbox://")):
            deckgl_spec["mapStyle"] = self.default_map_style
        else:
            deckgl_spec["mapStyle"] = map_style

        # CRITICAL: Remove data from all layers - it will be injected by DeckGLView at render time
        # This keeps the spec compact and avoids serializing potentially large datasets
        if "layers" in deckgl_spec:
            for layer in deckgl_spec["layers"]:
                # Remove data completely - do not include in spec
                layer.pop("data", None)
                # Also remove any nested data structures that PyDeck might add
                if isinstance(layer, dict):
                    for key in list(layer.keys()):
                        if key.lower() == "data" or key.lower().endswith("_data"):
                            layer.pop(key, None)

        # Extract tooltips if present
        tooltips = deckgl_spec.pop("tooltip", True)

        # Validate spec structure
        self._output_type.validate_spec(deckgl_spec)

        return {
            "spec": deckgl_spec,
            "tooltips": tooltips,
            "sizing_mode": "stretch_both",
            "min_height": 400,
        }

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """Generate a DeckGL visualization."""
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__

        # Generate spec based on code_execution mode
        if self.code_execution_enabled:
            # Use PyDeck code execution
            full_dict = await self._generate_code_spec(
                messages, context, pipeline, doc
            )
            if full_dict is None:
                # User rejected code execution
                return [], {}
        else:
            # Use declarative JSON spec generation
            full_dict = await self._generate_yaml_spec(
                messages, context, pipeline, doc
            )

        # Create view and output
        view = self.view_type(pipeline=pipeline, **full_dict)
        out = self._output_type(component=view, title=step_title)

        return [out], {"view": dict(full_dict, type=view.view_type)}
