import asyncio

from typing import Any

import param
import requests

from panel.pane.image import Image
from pydantic import Field

from ...config import dump_yaml, load_yaml
from ...pipeline import Pipeline
from ...views import VegaLiteView
from ..config import (
    LUMEN_CACHE_DIR, PROMPTS_DIR, VECTOR_STORE_ASSETS_URL,
    VEGA_LITE_EXAMPLES_NUMPY_DB_FILE, VEGA_LITE_EXAMPLES_OPENAI_DB_FILE,
    VEGA_MAP_LAYER, VEGA_ZOOMABLE_MAP_ITEMS,
)
from ..context import TContext
from ..llm import Message, OpenAI
from ..models import EscapeBaseModel, SearchReplaceSpec
from ..utils import (
    apply_search_replace, generate_diff, get_schema, load_json, log_debug,
    retry_llm_output,
)
from ..vector_store import DuckDBVectorStore
from ..views import LumenOutput, VegaLiteOutput
from .base_view import BaseViewAgent


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


class VegaLiteAgent(BaseViewAgent):

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
            "polish_ux": {"response_model": SearchReplaceSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "polish_ux.jinja2"},
            "annotate_plot": {"response_model": SearchReplaceSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "annotate_plot.jinja2"},
            "revise_output": {"response_model": SearchReplaceSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "revise_output.jinja2"},
        }
    )

    polish_enabled = param.Boolean(default=True, allow_refs=True, doc="""
        Whether to apply visual polish to the generated Vega-Lite charts.""")

    user = param.String(default="Vega")

    vector_store_path = param.Path(default=None, check_exists=False, doc="""
        Path to a custom vector store for storing and retrieving Vega-Lite examples;
        if not provided a default store will be used depending on the LLM--
        OpenAIEmbeddings for OpenAI LLM or NumpyEmbeddings for all others.""")

    view_type = VegaLiteView

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

    def _add_zoom_params(self, vega_spec: dict) -> None:
        """Add zoom parameters to vega spec."""
        if "params" not in vega_spec:
            vega_spec["params"] = []

        existing_param_names = {
            param.get("name") for param in vega_spec["params"]
            if isinstance(param, dict) and "name" in param
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
        return {"spec": vega_spec, "sizing_mode": "stretch_both", "min_height": 300}

    async def _get_doc_examples(self, user_query: str) -> list[str]:
        # Query vector store for relevant examples
        doc_examples = []
        vector_store = self._get_vector_store()
        if vector_store:
            if user_query:
                doc_results = await vector_store.query(user_query, top_k=5, threshold=0.2)
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

    async def _update_spec(
        self,
        prompt_name: str,
        spec: dict[str, Any],
        messages: list[Message],
        context: TContext,
        image_data: bytes | None = None,
        errors: list[str] | None = None,
        **extra_context
    ) -> dict[str, Any]:
        yaml_spec = dump_yaml(spec, default_flow_style=False)

        pipeline = context.get("pipeline")
        errors_context = self._build_errors_context(pipeline, context, errors) if pipeline else {}

        system_prompt = await self._render_prompt(
            prompt_name,
            messages,
            context,
            spec=yaml_spec,
            language="yaml",
            **errors_context,
            **extra_context
        )

        model_spec = self.prompts.get(prompt_name, {}).get("llm_spec", self.llm_spec_key)
        invoke_messages = [{"role": "user", "content": [Image(image_data)]}] if image_data else messages

        result = await self.llm.invoke(
            messages=invoke_messages,
            system=system_prompt,
            response_model=SearchReplaceSpec,
            model_spec=model_spec,
        )

        updated_yaml = apply_search_replace(yaml_spec, result.edits)
        return load_yaml(updated_yaml)

    async def _safe_update_spec(
        self,
        prompt_name: str,
        spec: dict[str, Any],
        messages: list[Message],
        context: TContext,
        image_data: bytes | None = None,
        errors: list[str] | None = None,
        **extra_context
    ) -> tuple[dict[str, Any], str | None]:
        """
        Apply edits and validate. Returns (new_spec, diff) on success, (old_spec, None) on failure.

        This is the safe wrapper around _update_spec that validates changes before accepting them.
        If validation fails, the original spec is returned unchanged.
        """
        old_yaml = dump_yaml(spec)
        try:
            updated_spec = await self._update_spec(
                prompt_name, spec, messages, context,
                image_data=image_data, errors=errors, **extra_context
            )
            validated = await self._extract_spec(context, {"yaml_spec": dump_yaml(updated_spec)})
            new_spec = validated["spec"]
            new_yaml = dump_yaml(new_spec)
            diff = generate_diff(old_yaml, new_yaml, filename="vega_spec.yaml")
            return new_spec, diff
        except Exception as e:
            log_debug(f"{prompt_name} failed validation: {e}")
            return spec, None

    def _export_plot_image(self, out: VegaLiteOutput) -> bytes | None:
        """Export plot as PNG for vision-based polish."""
        try:
            image_io = out.export("png")
            log_debug("Successfully exported plot image for vision analysis")
            return image_io.getvalue()
        except Exception as e:
            log_debug(f"Failed to export plot image: {e}")
            return None

    def _schedule_background_polish(
        self,
        out: VegaLiteOutput,
        messages: list[Message],
        context: TContext,
        image_data: bytes | None,
        doc: str,
        table: str,
    ):
        """Schedule background polish passes to run after initial render."""
        steps_layout = self._steps_layout
        interface = self.interface

        async def apply_polish():
            if not self.polish_enabled:
                return

            await asyncio.sleep(0.1)  # Let UI render first
            current_spec = load_yaml(out.spec)

            # Run both polish passes
            pass_titles = [
                ("polish_ux", "Polishing layout & style", "Polish the layout, styling, and interaction for better UX."),
            ]
            for pass_name, title, feedback_msg in pass_titles:
                with out.param.update(loading=True):
                    with interface.add_step(title=title, steps_layout=steps_layout) as step:
                        current_spec, diff = await self._safe_update_spec(
                            pass_name, current_spec, messages, context,
                            image_data=image_data, doc=doc, table=table,
                            feedback=feedback_msg,
                        )
                        if diff:
                            out.spec = dump_yaml(current_spec)
                            step.stream(f"```diff\n{diff}\n```", replace=True)
                            step.success_title = f"{title} ✓"
                        else:
                            step.stream("No changes needed")
                            step.success_title = f"{title} (no changes)"

        param.parameterized.async_executor(apply_polish)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """Generate a VegaLite visualization with background polish passes."""
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")
        if not await get_schema(pipeline):
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        # Gather examples for few-shot prompting
        user_query = messages[-1].get("content", "") if messages[-1].get("role") == "user" else ""
        try:
            doc_examples = await self._get_doc_examples(user_query)
        except Exception:
            doc_examples = []

        # Generate initial spec
        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        spec_dict = await self._generate_basic_spec(messages, context, pipeline, doc_examples, doc)

        # Create output for immediate display
        view = self.view_type(pipeline=pipeline, **spec_dict)
        out = self._output_type(component=view, title=step_title)

        # Schedule background polish
        image_data = self._export_plot_image(out)
        self._schedule_background_polish(out, messages, context, image_data, doc, pipeline.table)

        return [out], {"view": dict(spec_dict, type=view.view_type)}

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
        """
        Revise the visualization spec based on user feedback.

        This method uses _update_spec instead of the base class implementation
        to support image-based feedback and maintain consistency with other
        spec modification methods.
        """
        # Get the spec to revise
        if view is not None:
            spec_dict = load_yaml(view.spec)
        elif spec is not None:
            spec_dict = load_yaml(spec) if isinstance(spec, str) else spec
        else:
            raise ValueError("Must provide previous spec to revise.")

        image_data = self._export_plot_image(view)
        doc_examples = await self._get_doc_examples(feedback)
        new_spec, diff = await self._safe_update_spec(
            "revise_output",
            spec_dict,
            messages,
            context,
            image_data=image_data,
            errors=errors,
            doc_examples=doc_examples,
            feedback=feedback,
            **kwargs
        )
        if diff:
            return dump_yaml(new_spec)
        else:
            raise ValueError("Annotation failed validation")


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
        annotation_messages = messages + [{
            "role": "user",
            "content": f"Add annotations: {instruction}"
        }]

        new_spec, diff = await self._safe_update_spec(
            "annotate_plot",
            spec["spec"],
            annotation_messages,
            context,
        )
        if diff:
            return dump_yaml(new_spec)
        else:
            raise ValueError("Annotation failed validation")
