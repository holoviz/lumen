"""
DeckGL Agent for generating 3D map visualizations.

This agent generates deck.gl specifications either directly as JSON
or via PyDeck Python code execution.
"""
from typing import Any

import param

from pydantic import Field

from ...config import load_yaml
from ...pipeline import Pipeline
from ...views import DeckGLView
from ..code_executor import CodeSafetyCheck, PyDeckExecutor
from ..config import PROMPTS_DIR
from ..context import TContext
from ..llm import Message
from ..models import EscapeBaseModel, RetrySpec
from ..utils import (
    get_data, get_schema, retry_llm_output, sanitize_column_names,
)
from ..views import DeckGLOutput
from .base_view import BaseViewAgent


class DeckGLSpec(EscapeBaseModel):
    """Response model for DeckGL JSON spec generation."""

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
        - views array with MapView

        Do NOT include 'data' in layers - it will be injected automatically.
        Use @@= syntax for accessors, e.g. "getPosition": "@@=[longitude, latitude]"
        """
    )


class PyDeckSpec(EscapeBaseModel):
    """Response model for PyDeck code generation."""

    chain_of_thought: str = Field(
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
    pydeck_code: str = Field(
        description="""Python code that creates a PyDeck visualization.

        Requirements:
        - Import pydeck as `pdk`
        - Data is available as `df` (pandas DataFrame)
        - Column names have spaces replaced with underscores and non-alphanumeric chars removed
        - Must assign final deck to variable `deck`
        - Do NOT call .to_html(), .show(), or any I/O methods

        Example:
        ```python
        import pydeck as pdk

        layer = pdk.Layer(
            "HexagonLayer",
            data=df,
            get_position=["longitude", "latitude"],
            radius=10000,
            elevation_scale=50,
            elevation_range=[0, 3000],
            extruded=True,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=37.8,
            longitude=-122.4,
            zoom=6,
            pitch=45,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/dark-v10",
        )
        ```
        """
    )


class DeckGLAgent(BaseViewAgent):
    """Agent for generating DeckGL 3D map visualizations.

    Supports two generation modes:
    - 'deckgl': Generate DeckGL JSON specs directly
    - 'pydeck': Generate PyDeck Python code, execute safely, convert to spec
    """

    conditions = param.List(
        default=[
            "Use for 3D geographic visualizations, map-based data, or when user requests DeckGL/deck.gl",
            "Use for large-scale geospatial data with latitude/longitude coordinates",
            "Use for hexbin aggregations, heatmaps, or 3D extruded visualizations on maps",
        ]
    )

    generation_mode = param.Selector(
        default="pydeck",
        objects=["deckgl", "pydeck", "pydeck_llm_validated"],
        doc="""How to generate the visualization:
        - 'deckgl': Generate DeckGL JSON spec directly
        - 'pydeck': Generate PyDeck Python code, execute safely
        - 'pydeck_llm_validated': Same as 'pydeck' but adds LLM security review"""
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

    view_type = DeckGLView

    _extensions = ("deckgl",)

    _output_type = DeckGLOutput

    # Default map style
    DEFAULT_MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

    @retry_llm_output()
    async def _generate_deckgl_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        errors: list | None = None
    ) -> dict[str, Any]:
        """Generate DeckGL spec directly via JSON."""
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

        return await self._extract_spec(context, {"json_spec": output.json_spec})

    @retry_llm_output()
    async def _generate_pydeck_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        errors: list | None = None,
        llm_validated: bool = False,
    ) -> dict[str, Any]:
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

            step.stream(f"\n```python\n{output.pydeck_code}\n```", replace=False)

        # Validate code safety with AST
        with self._add_step(title="Validating code safety (AST)", steps_layout=self._steps_layout) as step:
            validation_errors = PyDeckExecutor.validate(output.pydeck_code)
            if validation_errors:
                error_msg = "; ".join(validation_errors)
                step.failed_title = f"Unsafe code: {error_msg}"
                raise ValueError(f"Unsafe code detected: {error_msg}")
            step.success_title = "AST validation passed"

        # Optional LLM-based security review
        if llm_validated:
            with self._add_step(title="LLM security review", steps_layout=self._steps_layout) as step:
                system_prompt = await self._render_prompt(
                    "code_safety",
                    messages,
                    context,
                    code=output.pydeck_code,
                )
                is_safe, reasoning = await PyDeckExecutor.validate_with_llm(
                    output.pydeck_code, self.llm, system_prompt, self.llm_spec_key
                )
                step.stream(f"Analysis: {reasoning}")
                if not is_safe:
                    step.failed_title = "LLM flagged code as unsafe"
                    raise ValueError(f"LLM security review failed: {reasoning}")
                step.success_title = "LLM security review passed"

        # Execute and convert
        with self._add_step(title="Executing PyDeck code", steps_layout=self._steps_layout) as step:
            df = await get_data(pipeline)
            df = sanitize_column_names(df)

            deck = PyDeckExecutor.execute(output.pydeck_code, df)

            # Convert to JSON spec
            spec = deck.to_json()
            if isinstance(spec, str):
                import json
                spec = json.loads(spec)

            step.success_title = "PyDeck visualization created"

        return await self._extract_spec(context, {"json_spec": spec})

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate DeckGL spec."""
        import json

        if json_spec := spec.get("json_spec"):
            if isinstance(json_spec, str):
                deckgl_spec = json.loads(json_spec)
            else:
                deckgl_spec = json_spec
        elif yaml_spec := spec.get("yaml_spec"):
            deckgl_spec = load_yaml(yaml_spec)
        else:
            raise ValueError("No valid spec provided")

        # Handle map style - PyDeck uses 'map_style', DeckGL uses 'mapStyle'
        # Also replace Mapbox styles (which require API key) with CartoDB
        map_style = deckgl_spec.pop("map_style", None) or deckgl_spec.get("mapStyle")
        if not map_style or map_style.startswith("mapbox://"):
            # Use CartoDB style which doesn't require API key
            deckgl_spec["mapStyle"] = self.DEFAULT_MAP_STYLE
        else:
            deckgl_spec["mapStyle"] = map_style

        # Remove data from layers - it will be injected by DeckGLView at render time
        # This keeps the spec compact and avoids serializing potentially large datasets
        if "layers" in deckgl_spec:
            for layer in deckgl_spec["layers"]:
                layer.pop("data", None)

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

        # Generate spec based on mode
        if self.generation_mode in ("pydeck", "pydeck_llm_validated"):
            llm_validated = self.generation_mode == "pydeck_llm_validated"
            full_dict = await self._generate_pydeck_spec(
                messages, context, pipeline, doc, llm_validated=llm_validated
            )
        else:
            full_dict = await self._generate_deckgl_spec(
                messages, context, pipeline, doc
            )

        # Create view and output
        view = self.view_type(pipeline=pipeline, **full_dict)
        out = self._output_type(component=view, title=step_title)

        return [out], {"view": dict(full_dict, type=view.view_type)}
