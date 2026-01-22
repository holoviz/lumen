from typing import Any

import param
import requests

from pydantic import BaseModel, Field

from ...config import dump_yaml, load_yaml
from ...pipeline import Pipeline
from ...views import VegaLiteView
from ..code_executor import AltairExecutor, CodeSafetyCheck
from ..config import (
    LUMEN_CACHE_DIR, PROMPTS_DIR, VECTOR_STORE_ASSETS_URL,
    VEGA_LITE_EXAMPLES_NUMPY_DB_FILE, VEGA_LITE_EXAMPLES_OPENAI_DB_FILE,
    VEGA_MAP_LAYER, VEGA_ZOOMABLE_MAP_ITEMS, UserCancelledError,
)
from ..context import TContext
from ..llm import Message, OpenAI
from ..models import EscapeBaseModel, PartialBaseModel, RetrySpec
from ..utils import (
    get_data, get_schema, load_json, log_debug, retry_llm_output,
)
from ..vector_store import DuckDBVectorStore
from ..views import LumenOutput, VegaLiteOutput
from .base_code import BaseCodeAgent


class VegaLiteSpec(EscapeBaseModel):

    chain_of_thought: str = Field(
        description="""Explain your design choices based on visualization theory:
        - What story does this data tell?
        - What's the most compelling insight or trend (for the title)?
        - What additional context adds value without repeating the title (for the subtitle)?
        - Which visual encodings (position, color, size) best reveal patterns?
        - Should color highlight specific insights or remain neutral?
        - What makes this plot engaging and useful for the user?
        Keep response to 1-2 sentences.""",
        examples=[
            "The data reveals US dominance in Winter Olympic hosting (4 times vs France's 3)—title should emphasize this leadership. Position encoding via horizontal bars sorted descending makes comparison immediate, neutral blue keeps focus on counts rather than categories, and the subtitle can note the 23-country spread to add context without redundancy.",
            "This time series shows a 40% revenue spike in Q3 2024—the key trend for the title. A line chart with position encoding (time→x, revenue→y) reveals the pattern, endpoint labels eliminate need for constant grid reference making it cleaner, and color remains neutral since there's one series; the subtitle should explain what drove the spike (e.g., 'Three offshore projects') to add insight."
        ]
    )
    yaml_spec: str = Field(
        description="A basic vega-lite YAML specification with core plot elements only (data, mark, basic x/y encoding)."
    )


class VegaLiteSpecUpdate(BaseModel):
    chain_of_thought: str = Field(
        description="Explain what changes you're making to the Vega-Lite spec and why. Keep to 1-2 sentences.",
        examples=[
            "Adding tooltips to show exact values on hover for better interactivity.",
            "Swapping x and y axes to create horizontal bars as requested."
        ]
    )
    yaml_update: str = Field(
        description="""Partial YAML with ONLY modified properties (unchanged values omitted).
        Respect your step's scope; don't override previous steps."""
    )


class AltairSpec(PartialBaseModel):
    """Response model for Altair code generation."""

    chain_of_thought: str = Field(
        default="",
        description="""Explain your design choices based on visualization theory:
        - What story does this data tell?
        - What's the most compelling insight or trend (for the title)?
        - Which visual encodings (position, color, size) best reveal patterns?
        Keep response to 1-2 sentences.""",
        examples=[
            "The data reveals US dominance in Winter Olympic hosting—a horizontal bar chart sorted descending makes comparison immediate, with the leader highlighted in a distinct color.",
            "This time series shows a 40% revenue spike in Q3 2024—a line chart with point markers reveals the trend clearly."
        ]
    )
    code: str = Field(
        description="""Python code that creates an Altair chart.
        Requirements:
        - Import altair as `alt`
        - Data is available as `df` (pandas DataFrame)
        - Must assign final chart to variable `chart`
        - Do NOT call .to_dict(), .save(), .display() or any I/O methods
        - Use 'container' for width to make charts responsive
        """
    )


class VegaLiteAgent(BaseCodeAgent):

    conditions = param.List(
        default=[
            "Use for publication-ready visualizations or when user specifically requests Vega-Lite charts",
            "Use for polished charts intended for presentation or sharing",
        ]
    )

    purpose = param.String(default="Generates a vega-lite plot specification from the input data pipeline.")

    prompts = param.Dict(
        default={
            "main": {"response_model": VegaLiteSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2"},
            "main_altair": {"response_model": AltairSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "main_altair.jinja2"},
            "code_safety": {"response_model": CodeSafetyCheck, "template": PROMPTS_DIR / "VegaLiteAgent" / "code_safety.jinja2"},
            "interaction_polish": {"response_model": VegaLiteSpecUpdate, "template": PROMPTS_DIR / "VegaLiteAgent" / "interaction_polish.jinja2"},
            "annotate_plot": {"response_model": VegaLiteSpecUpdate, "template": PROMPTS_DIR / "VegaLiteAgent" / "annotate_plot.jinja2"},
            "revise_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "revise_output.jinja2"},
        }
    )

    user = param.String(default="Vega")

    vector_store_path = param.Path(default=None, check_exists=False, doc="""
        Path to a custom vector store for storing and retrieving Vega-Lite examples;
        if not provided a default store will be used depending on the LLM--
        OpenAIEmbeddings for OpenAI LLM or NumpyEmbeddings for all others.""")

    view_type = VegaLiteView

    _executor_class = AltairExecutor

    _extensions = ("vega",)

    _output_type = VegaLiteOutput

    def __init__(self, **params):
        self._vector_store: DuckDBVectorStore | None = None
        super().__init__(**params)

    def _get_vector_store(self):
        """Get or initialize the vector store (lazy initialization)."""
        if self._vector_store is not None:
            return self._vector_store

        if self.vector_store_path:
            uri = self.vector_store_path
        else:
            db_file = VEGA_LITE_EXAMPLES_OPENAI_DB_FILE if isinstance(self.llm, OpenAI) else VEGA_LITE_EXAMPLES_NUMPY_DB_FILE
            uri = LUMEN_CACHE_DIR / db_file
            if not uri.exists():
                response = requests.get(f"{VECTOR_STORE_ASSETS_URL}{db_file}", timeout=5)
                response.raise_for_status()
                uri.write_bytes(response.content)
        # Use a read-only connection to avoid lock conflicts
        self._vector_store = DuckDBVectorStore(uri=str(uri), read_only=True)
        return self._vector_store

    def _deep_merge_dicts(self, base_dict: dict[str, Any], update_dict: dict[str, Any] | list) -> dict[str, Any]:
        """Deep merge two dictionaries, with update_dict taking precedence.

        Special handling:
        - If update_dict is a list, treat it as a list of layers to append
        - If update_dict contains 'layer', append new annotation layers rather than merging
        - Background marks (rect, area) are automatically prepended to render behind other layers
        - When merging layers, ensure each layer has both 'mark' and 'encoding'
        """
        # Mark types that should be rendered as backgrounds
        BACKGROUND_MARKS = {'rect', 'area'}

        if not update_dict:
            return base_dict

        # Handle case where update_dict is a list of layers (e.g., annotation layers)
        if isinstance(update_dict, list):
            update_dict = {"layer": update_dict}

        result = base_dict.copy()

        # Special handling for layer arrays
        if "layer" in update_dict and "layer" in result:
            base_layers = result["layer"]
            update_layers = update_dict["layer"]

            # Separate background and foreground layers from updates
            background_updates = []
            foreground_updates = []

            for layer in update_layers:
                mark_type = self._get_layer_mark_type(layer)
                if mark_type in BACKGROUND_MARKS:
                    background_updates.append(self._normalize_layer_mark(layer.copy()))
                else:
                    foreground_updates.append(layer)

            # Check if remaining foreground updates are new annotations or layer updates
            is_append_operation = False
            if foreground_updates and base_layers:
                first_update_mark = self._get_layer_mark_type(foreground_updates[0])
                first_base_mark = self._get_layer_mark_type(base_layers[0])

                # If marks are different, or if update has marks like 'rule', 'text'
                # which are typically annotations, treat as append
                annotation_marks = {'rule', 'text'}
                if first_update_mark != first_base_mark or first_update_mark in annotation_marks:
                    is_append_operation = True

            if is_append_operation:
                # Append foreground layers as new annotation layers
                normalized_foreground = [self._normalize_layer_mark(layer.copy()) for layer in foreground_updates]
                # Prepend backgrounds, keep base, append foregrounds
                result["layer"] = background_updates + base_layers + normalized_foreground
            else:
                # Original merge behavior for updating existing layers
                merged_layers = []
                for i, update_layer in enumerate(foreground_updates):
                    if i < len(base_layers):
                        # Merge with corresponding base layer
                        base_layer = base_layers[i]
                        merged_layer = self._deep_merge_dicts(base_layer, update_layer)

                        # Ensure layer has mark (carry over from base if not in update)
                        if "mark" not in merged_layer and "mark" in base_layer:
                            merged_layer["mark"] = base_layer["mark"]

                        merged_layers.append(merged_layer)
                    else:
                        # New layer added by update
                        merged_layers.append(self._normalize_layer_mark(update_layer.copy()))

                # Keep any remaining base layers not updated
                merged_layers.extend(base_layers[len(foreground_updates):])
                # Prepend backgrounds to all merged layers
                result["layer"] = background_updates + merged_layers
        else:
            # Standard recursive merge for non-layer properties
            for key, value in update_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge_dicts(result[key], value)
                else:
                    result[key] = value

        # If we're merging in a 'layer', remove conflicting top-level properties
        if "layer" in update_dict:
            result.pop("mark", None)
            result.pop("encoding", None)

        return result

    def _get_layer_mark_type(self, layer: dict) -> str | None:
        """Extract mark type from a layer, handling both dict and string formats."""
        if "mark" in layer:
            mark = layer["mark"]
            if isinstance(mark, dict):
                return mark.get("type")
            return mark
        return None

    def _normalize_layer_mark(self, layer: dict) -> dict:
        """Ensure layer has a properly formatted single mark property.

        Handles cases where:
        - Mark appears multiple times (duplicate keys in parsed YAML/dict)
        - Mark is a string but needs to be an object with 'type'
        - Layer might have conflicting mark definitions
        """
        if "mark" not in layer:
            return layer

        mark = layer["mark"]

        # If mark is a string, convert to object with type
        if isinstance(mark, str):
            layer["mark"] = {"type": mark}
        # If mark is a dict, ensure it has 'type'
        elif isinstance(mark, dict) and "type" not in mark:
            # Malformed mark without type - try to infer or leave as-is
            # This shouldn't happen in valid Vega-Lite specs
            pass

        return layer

    async def _update_spec_step(
        self,
        step_name: str,
        step_desc: str,
        vega_spec: dict[str, Any],
        prompt_name: str,
        messages: list[Message],
        context: TContext,
        doc: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Update a Vega-Lite spec with incremental changes for a specific step."""
        with self._add_step(title=step_desc, steps_layout=self._steps_layout) as step:
            vega_spec = dump_yaml(vega_spec, default_flow_style=False)
            system_prompt = await self._render_prompt(
                prompt_name,
                messages,
                context,
                vega_spec=vega_spec,
                doc=doc,
                table=context["pipeline"].table,
            )

            model_spec = self.prompts.get(prompt_name, {}).get("llm_spec", self.llm_spec_key)
            result = await self.llm.invoke(
                messages=messages,
                system=system_prompt,
                response_model=VegaLiteSpecUpdate,
                model_spec=model_spec,
            )

            step.stream(f"Reasoning: {result.chain_of_thought}")
            step.stream(f"Update:\n```yaml\n{result.yaml_update}\n```", replace=False)
            update_dict = load_yaml(result.yaml_update)
        return step_name, update_dict

    def _add_zoom_params(self, vega_spec: dict) -> None:
        """Add zoom parameters to vega spec."""
        if "params" not in vega_spec:
            vega_spec["params"] = []

        existing_param_names = {
            p.get("name") for p in vega_spec["params"]
            if isinstance(p, dict) and "name" in p
        }

        for p in VEGA_ZOOMABLE_MAP_ITEMS["params"]:
            if p.get("name") not in existing_param_names:
                vega_spec["params"].append(p)

    def _setup_projection(self, vega_spec: dict) -> None:
        """Setup map projection settings."""
        if "projection" not in vega_spec:
            vega_spec["projection"] = {"type": "mercator"}
        vega_spec["projection"].update(VEGA_ZOOMABLE_MAP_ITEMS["projection"])

    def _handle_map_compatibility(self, vega_spec: dict, vega_spec_str: str) -> None:
        """Handle map projection compatibility and add geographic outlines."""
        has_world_map = "world-110m.json" in vega_spec_str
        uses_albers_usa = vega_spec["projection"]["type"] == "albersUsa"

        if has_world_map and uses_albers_usa:
            # albersUsa incompatible with world map
            vega_spec["projection"] = "mercator"
        elif not has_world_map and not uses_albers_usa:
            # Add world map outlines if needed
            vega_spec["layer"].append(VEGA_MAP_LAYER["world"])

    def _add_geographic_items(self, vega_spec: dict, vega_spec_str: str) -> dict:
        """Add geographic visualization items to vega spec."""
        self._add_zoom_params(vega_spec)
        self._setup_projection(vega_spec)

        # Remove projection from individual layers to prevent conflicts
        # All layers must inherit the top-level projection for zoom/pan to work correctly
        if "layer" in vega_spec:
            for layer in vega_spec["layer"]:
                if isinstance(layer, dict) and "projection" in layer:
                    del layer["projection"]

        self._handle_map_compatibility(vega_spec, vega_spec_str)
        return vega_spec

    @classmethod
    def _extract_as_keys(cls, transforms: list[dict]) -> list[str]:
        """
        Extracts all 'as' field names from a list of Vega-Lite transform definitions.

        Parameters
        ----------
        transforms : list[dict]
            A list of Vega-Lite transform objects.

        Returns
        -------
        list[str]
            A list of field names from 'as' keys (flattened, deduplicated).
        """
        as_fields = []
        for t in transforms:
            # Top-level 'as'
            if "as" in t:
                if isinstance(t["as"], list):
                    as_fields.extend(t["as"])
                elif isinstance(t["as"], str):
                    as_fields.append(t["as"])
            for key in ("aggregate", "joinaggregate", "window"):
                if key in t and isinstance(t[key], list):
                    for entry in t[key]:
                        if "as" in entry:
                            as_fields.append(entry["as"])

        return list(dict.fromkeys(as_fields))

    @retry_llm_output()
    async def _generate_basic_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc_examples: list,
        doc: str,
        errors: list | None = None
    ) -> dict[str, Any]:
        """Generate the basic VegaLite spec structure."""
        errors_context = self._build_errors_context(pipeline, context, errors)
        with self._add_step(title="Creating basic plot structure", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "main",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                doc_examples=doc_examples,
                **errors_context,
            )
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=VegaLiteSpec,
            )

            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            current_spec = await self._extract_spec(context, {"yaml_spec": output.yaml_spec})
            step.success_title = "Complete visualization with titles and colors created"
        return current_spec

    @retry_llm_output()
    async def _generate_code_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc_examples: list,
        doc: str,
        errors: list | None = None,
    ) -> dict[str, Any] | None:
        """Generate spec via Altair code execution."""
        errors_context = self._build_errors_context(pipeline, context, errors)

        with self._add_step(title="Generating Altair code", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "main_altair",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                doc_examples=doc_examples,
                **errors_context,
            )

            model_spec = self.prompts["main_altair"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=AltairSpec,
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

            # Execute code using mixin (handles AST validation, LLM validation, user prompt)
            df = await get_data(pipeline)
            chart = await self._execute_code(output.code, df, system=system, step=step)

        if chart is None:
            raise UserCancelledError("Code execution rejected by user.")

        # Convert to Vega-Lite spec
        spec = chart.to_dict()
        spec.pop("datasets", None)  # Remove inline data
        spec["data"] = {"name": pipeline.table}  # Use named data source

        return await self._extract_spec(context, {"yaml_spec": dump_yaml(spec)})

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        # .encode().decode('unicode_escape') fixes a JSONDecodeError in Python
        # where it's expecting property names enclosed in double quotes
        # by properly handling the escaped characters in your JSON string
        if yaml_spec := spec.get("yaml_spec"):
            vega_spec = load_yaml(yaml_spec)
        elif json_spec := spec.get("json_spec"):
            vega_spec = load_json(json_spec)

        # Remove wrapper properties that aren't part of Vega-Lite spec
        for key in ['sizing_mode', 'min_height', 'type']:
            vega_spec.pop(key, None)

        if "$schema" not in vega_spec:
            vega_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"

        # Check if this is a compound chart (hconcat, vconcat, concat, facet, repeat)
        is_compound = any(key in vega_spec for key in ['hconcat', 'vconcat', 'concat', 'facet', 'repeat'])

        if is_compound:
            # Remove invalid top-level width/height for compound charts
            # Altair's config.view.continuousWidth/Height handles sub-chart sizing
            vega_spec.pop('width', None)
            vega_spec.pop('height', None)
        else:
            if "width" not in vega_spec:
                vega_spec["width"] = "container"
            if "height" not in vega_spec:
                vega_spec["height"] = "container"

        self._output_type.validate_spec(vega_spec)

        # using string comparison because these keys could be in different nested levels
        vega_spec_str = dump_yaml(vega_spec)
        # Handle different types of interactive controls based on chart type
        if "latitude:" in vega_spec_str or "longitude:" in vega_spec_str:
            vega_spec = self._add_geographic_items(vega_spec, vega_spec_str)
        # elif ("point: true" not in vega_spec_str or "params" not in vega_spec) and vega_spec_str.count("encoding:") == 1:
        #     # add pan/zoom controls to all plots except geographic ones and points overlaid on line plots
        #     # because those result in an blank plot without error
        #     vega_spec["params"] = [{"bind": "scales", "name": "grid", "select": "interval"}]
        return {"spec": vega_spec, "sizing_mode": "stretch_both", "min_height": 100}

    async def _get_doc_examples(self, user_query: str) -> list[str]:
        # Query vector store for relevant examples
        doc_examples = []
        vector_store = self._get_vector_store()
        if vector_store:
            if user_query:
                doc_results = await vector_store.query(user_query, top_k=5)
                # Extract text and specs from results
                k = 0
                for result in doc_results:
                    if "metadata" in result and "spec" in result["metadata"]:
                        # Include the text description/title before the spec
                        text_description = result.get("text", "")
                        spec = result["metadata"]["spec"]
                        if "hconcat" in spec or "vconcat" in spec or "repeat" in spec or "params" in spec or "values:" in spec:
                            # Skip complex multi-view specs for simplicity
                            continue
                        doc_examples.append(f"{text_description}\n```yaml\n{spec}\n```")
                        k += 1
                    if k >= 1:  # Limit to top 1 example
                        break
        return doc_examples

    async def revise(
        self,
        feedback: str,
        messages: list[Message],
        context: TContext,
        view: LumenOutput | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        **kwargs
    ) -> str:
        if errors is not None:
            kwargs["errors"] = errors
        doc_examples = await self._get_doc_examples(feedback)
        context["doc_examples"] = doc_examples
        return await super().revise(
            feedback, messages, context, view=view, spec=spec, language=language, **kwargs
        )

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """
        Generates a VegaLite visualization using progressive building approach with real-time updates.
        """
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        user_query = messages[-1].get("content", "") if messages[-1].get("role") == "user" else ""
        try:
            doc_examples = await self._get_doc_examples(user_query)
        except Exception:
            doc_examples = []

        # Step 1: Generate basic spec
        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        # Produces {"spec": {$schema: ..., ...}, "sizing_mode": ..., ...}
        if self.code_execution in ("prompt", "llm", "allow"):
            full_dict = await self._generate_code_spec(
                messages, context, pipeline, doc_examples, doc,
            )
            if full_dict is None:
                # User rejected code execution, exit gracefully
                return [], {}
        else:
            full_dict = await self._generate_basic_spec(messages, context, pipeline, doc_examples, doc)

        # Step 2: Show complete plot immediately
        view = self.view_type(pipeline=pipeline, **full_dict)
        out = self._output_type(component=view, title=step_title)

        # Step 3: enhancements (LLM-driven creative decisions)
        # Skip for altair modes since interactivity is built into the generated code
        if self.code_execution == "disabled":
            steps = {
                "interaction_polish": "Add helpful tooltips and ensure responsive, accessible user experience",
            }

            # Execute steps sequentially
            for step_name, step_desc in steps.items():
                # Only pass the vega lite 'spec' portion to prevent ballooning context
                step_name, update_dict = await self._update_spec_step(
                    step_name, step_desc, out.spec, step_name, messages, context, doc=doc
                )
                try:
                    # Validate merged spec
                    test_spec = self._deep_merge_dicts(full_dict["spec"], update_dict)
                    await self._extract_spec(context, {"yaml_spec": dump_yaml(test_spec)})
                except Exception as e:
                    log_debug(f"Skipping invalid {step_name} update due to error: {e}")
                    continue
                spec = self._deep_merge_dicts(full_dict["spec"], update_dict)
                out.spec = dump_yaml(spec)
                log_debug(f"📊 Applied {step_name} updates and refreshed visualization")

        return [out], {"view": dict(full_dict, type=view.view_type)}

    async def annotate(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        spec: dict
    ) -> str:
        """
        Apply annotations based on user request.

        Parameters
        ----------
        instruction: str
            User's description of what to annotate
        messages: list[Message]
            Chat history for context
        context: TContext
            Session context
        spec : dict
            The current VegaLite specification (full dict with 'spec' key)

        Returns
        -------
        str
            Updated specification with annotations
        """
        # Add user's annotation request to messages context
        annotation_messages = messages + [{
            "role": "user",
            "content": f"Add annotations: {instruction}"
        }]

        vega_spec = dump_yaml(spec["spec"], default_flow_style=False)
        system_prompt = await self._render_prompt(
            "annotate_plot",
            annotation_messages,
            context,
            vega_spec=vega_spec,
        )

        model_spec = self.prompts.get("annotate_plot", {}).get("llm_spec", self.llm_spec_key)
        result = await self.llm.invoke(
            messages=annotation_messages,
            system=system_prompt,
            response_model=VegaLiteSpecUpdate,
            model_spec=model_spec,
        )
        update_dict = load_yaml(result.yaml_update)

        # Merge and validate
        final_dict = spec.copy()

        try:
            final_dict["spec"] = self._deep_merge_dicts(final_dict["spec"], update_dict)
            spec = await self._extract_spec(context, {"yaml_spec": dump_yaml(final_dict["spec"])})
        except Exception as e:
            log_debug(f"Skipping invalid annotation update due to error: {e}")
            raise e
        return dump_yaml(spec["spec"])
