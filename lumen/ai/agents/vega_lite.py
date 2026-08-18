import base64

from collections import Counter
from functools import partial
from typing import Any

import param
import requests

from instructor import Image
from panel import bind
from panel.io import state
from panel.layout import Column
from panel.param import ParamFunction
from pydantic import BaseModel, Field

from ...config import dump_yaml, load_yaml
from ...pipeline import Pipeline
from ...views import Panel, VegaLiteView
from ..code_executor import AltairExecutor, CodeSafetyCheck
from ..config import (
    LUMEN_CACHE_DIR, PROMPTS_DIR, UNRECOVERABLE_ERRORS,
    VECTOR_STORE_ASSETS_URL, VEGA_LITE_DOCS_NUMPY_DB_FILE,
    VEGA_LITE_DOCS_OPENAI_DB_FILE, UserCancelledError,
)
from ..context import TContext
from ..editors import LumenEditor, MultiChartEditor, VegaLiteEditor
from ..embeddings import NumpyEmbeddings, OpenAIEmbeddings
from ..llm import Message, OpenAI
from ..models import EscapeBaseModel, RetrySpec
from ..utils import (
    category_palette, get_data, get_gridded_metadata, get_schema,
    has_categorical_color, load_json, log_debug, normalize_vegalite_spec,
    retry_llm_output, subset_gridded_to_2d,
)
from ..vector_store import DuckDBVectorStore
from .base_code import BaseCodeAgent


class ChartSpec(BaseModel):
    """A single Vega-Lite chart within a (potentially multi-chart) response."""

    title: str = Field(
        default="",
        description="Short, human-readable label for this chart, used as its tab or section heading when several charts are returned."
    )
    yaml_spec: str = Field(
        description="A basic vega-lite YAML specification with core plot elements only (mark, basic x/y encoding). Skip $schema and data fields."
    )


class VegaLiteSpec(EscapeBaseModel):

    chain_of_thought: str = Field(
        description="""Explain your design choices based on visualization theory:
        - What story does this data tell?
        - What's the most compelling insight or trend (for the title)?
        - What additional context adds value without repeating the title (for the subtitle)?
        - Which visual encodings (position, color, size) best reveal patterns?
        - Should color highlight specific insights or remain neutral?
        - When several charts are warranted, why each one earns its place.
        Keep response to 1-2 sentences.""",
        examples=[
            "The data reveals US dominance in Winter Olympic hosting (4 times vs France's 3)—title should emphasize this leadership. Position encoding via horizontal bars sorted descending makes comparison immediate, neutral blue keeps focus on counts rather than categories, and the subtitle can note the 23-country spread to add context without redundancy.",
            "This time series shows a 40% revenue spike in Q3 2024—the key trend for the title. A line chart with position encoding (time→x, revenue→y) reveals the pattern, endpoint labels eliminate need for constant grid reference making it cleaner, and color remains neutral since there's one series; the subtitle should explain what drove the spike (e.g., 'Three offshore projects') to add insight."
        ]
    )
    charts: list[ChartSpec] = Field(
        min_length=1,
        description="One or more chart specifications. Return several charts only when the request spans distinct metrics or relationships that should not share a single plot; otherwise return exactly one."
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


class AltairChartSpec(BaseModel):
    """A single Altair chart within a (potentially multi-chart) response."""

    title: str = Field(
        default="",
        description="Short, human-readable label for this chart, used as its tab or section heading when several charts are returned."
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


class AltairSpec(BaseModel):
    """Response model for Altair code generation."""

    chain_of_thought: str = Field(
        default="",
        description="""Explain your design choices based on visualization theory:
        - What story does this data tell?
        - What's the most compelling insight or trend (for the title)?
        - Which visual encodings (position, color, size) best reveal patterns?
        - When several charts are warranted, why each one earns its place.
        Keep response to 1-2 sentences.""",
        examples=[
            "The data reveals US dominance in Winter Olympic hosting—a horizontal bar chart sorted descending makes comparison immediate, with the leader highlighted in a distinct color.",
            "This time series shows a 40% revenue spike in Q3 2024—a line chart with point markers reveals the trend clearly."
        ]
    )
    charts: list[AltairChartSpec] = Field(
        min_length=1,
        description="One or more chart specifications. Return several charts only when the request spans distinct metrics or relationships that should not share a single plot; otherwise return exactly one."
    )


class VegaLiteAgent(BaseCodeAgent):

    conditions = param.List(
        default=[
            "Use for publication-ready visualizations or when user specifically requests Vega-Lite charts",
            "Use for polished charts intended for presentation or sharing",
            "Use for a choropleth whose table names places (states, countries) but holds no geometry or coordinates, since the boundaries can be looked up and joined by name",
        ]
    )

    purpose = param.String(default="Generates one or more vega-lite plot specifications from the input data pipeline.")

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

    exclude_spec_constructs = param.List(default=[
        "facet", "repeat", "concat", "hconcat", "vconcat", "row", "column",
    ], item_type=str, doc="""
        Vega-Lite composition keys that disqualify a retrieved example spec.
        The declarative prompt tells the model not to pack several plots into one
        entry with these, so injecting a reference spec that uses them puts the
        documentation at odds with the instructions. Set to [] to keep every
        retrieved spec -- appropriate for the Altair path, whose prompt permits
        composition within a single chart entry.""")

    n_doc_pages = param.Integer(default=5, bounds=(0, None), doc="""
        Number of documentation sections to inject into the prompt.
        Set to 0 to disable documentation lookup entirely.""")

    reserve_base_form = param.Boolean(default=True, doc="""
        Guarantee one retrieved section is the base form of a chart type, when
        the candidate pool contains one. A bare request like "plot a bar chart"
        matches every bar variant about equally -- similarity across the top
        candidates spans less than the noise floor -- so the plain section loses
        to specialisations that simply have more text to match on. Sections are
        flagged deterministically at build time from their title, which is why
        this reserves a slot rather than nudging a score.""")

    doc_pages_per_slug = param.Integer(default=2, bounds=(1, None), doc="""
        Maximum number of retrieved sections allowed from any single
        documentation page. Without a cap, similarity collapses onto one page:
        a "bar chart" query returns five near-identical mark-bar variants while
        the transform and encoding pages that would broaden the answer sit just
        below the cut.""")

    vector_store_path = param.Path(default=None, check_exists=False, doc="""
        Path to a custom vector store for storing and retrieving Vega-Lite examples;
        if not provided a default store will be used depending on the LLM--
        OpenAIEmbeddings for OpenAI LLM or NumpyEmbeddings for all others.""")

    view_type = VegaLiteView

    _executor_class = AltairExecutor

    _extensions = ("vega",)

    _editor_type = VegaLiteEditor

    # Over-fetch multiplier: the per-slug cap needs alternatives below the cut
    # to promote, otherwise capping just returns fewer results.
    _doc_page_pool = 6

    # Marks that annotate an existing view rather than plot data in their own
    # right. A tooltip on a value label is noise, so their absence must not force
    # a polish pass over a chart whose data layers are already covered.
    _annotation_marks = frozenset({"text", "rule"})

    def __init__(self, **params):
        self._vector_store: DuckDBVectorStore | None = None
        super().__init__(**params)

    def _get_vector_store(self):
        """Get or initialize the vector store (lazy initialization)."""
        if self._vector_store is not None:
            return self._vector_store

        # The embeddings must match whatever the store was built with, or
        # DuckDBVectorStore._verify_parameters raises on first query and
        # _get_doc_pages silently returns no documentation at all.
        if isinstance(self.llm, OpenAI):
            db_file, embeddings = VEGA_LITE_DOCS_OPENAI_DB_FILE, OpenAIEmbeddings()
        else:
            db_file, embeddings = VEGA_LITE_DOCS_NUMPY_DB_FILE, NumpyEmbeddings()

        if self.vector_store_path:
            uri = self.vector_store_path
        else:
            uri = LUMEN_CACHE_DIR / db_file
            if not uri.exists():
                response = requests.get(f"{VECTOR_STORE_ASSETS_URL}{db_file}", timeout=5)
                response.raise_for_status()
                uri.write_bytes(response.content)
        # Use a read-only connection to avoid lock conflicts
        self._vector_store = DuckDBVectorStore(
            uri=str(uri), embeddings=embeddings, read_only=True
        )
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

    def _export_plot_image(self, out: VegaLiteEditor) -> bytes | None:
        """Export plot as PNG for vision-based polish.

        Parameters
        ----------
        out : VegaLiteEditor
            The VegaLite editor containing the plot to export.

        Returns
        -------
        bytes | None
            PNG image bytes if export succeeds, None otherwise.
        """
        try:
            image_io = out.export("png")
            log_debug("Successfully exported plot image for vision analysis")
            return image_io.getvalue()
        except Exception as e:
            log_debug(f"Failed to export plot image: {e}")
            return None

    def _prepare_vision_messages(
        self, messages: list[Message], out: LumenEditor | None, content: str
    ) -> list[Message]:
        """Add plot image to messages for LLM vision analysis.

        If vision is unavailable or image export fails we append the
        content as a plain-text message.
        """
        fallback = messages + [{"role": "user", "content": content}]
        if not self.llm._supports_vision:
            return fallback
        if out is None or not isinstance(out, VegaLiteEditor):
            return fallback

        image_bytes = self._export_plot_image(out)
        if image_bytes is None:
            return fallback

        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        plot_image = Image.from_raw_base64(base64_str)
        log_debug("Added plot image to messages for LLM vision analysis")
        return messages + [{
            "role": "user",
            "content": [content, plot_image]
        }]

    async def _update_spec_step(
        self,
        step_name: str,
        step_desc: str,
        vega_spec: dict[str, Any] | str,
        prompt_name: str,
        messages: list[Message],
        context: TContext,
        doc: str | None = None,
        out: VegaLiteEditor | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Update a Vega-Lite spec with incremental changes for a specific step.

        Parameters
        ----------
        step_name : str
            Name identifier for this step.
        step_desc : str
            Human-readable description of this step.
        vega_spec : dict[str, Any] | str
            The current Vega-Lite specification.
        prompt_name : str
            Name of the prompt template to use.
        messages : list[Message]
            Chat message history.
        context : TContext
            Session context.
        doc : str | None
            Optional documentation string.
        out : VegaLiteEditor | None
            Optional VegaLite editor to export plot image from for vision analysis.
        """
        with self._add_step(title=step_desc, steps_layout=self._steps_layout) as step:
            if not isinstance(vega_spec, str):
                vega_spec = dump_yaml(vega_spec, default_flow_style=False)
            invoke_messages = self._prepare_vision_messages(messages, out, "Current chart to polish:")
            result = await self._invoke_prompt(
                prompt_name,
                invoke_messages,
                context,
                vega_spec=vega_spec,
                doc=doc,
                table=context["pipeline"].table,
            )
            step.stream(f"Reasoning: {result.chain_of_thought}")
            step.stream(f"Update:\n```yaml\n{result.yaml_update}\n```", replace=False)
            update_dict = load_yaml(result.yaml_update)
        return step_name, update_dict

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
    async def _generate_yaml_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        doc_pages: list[dict] | None = None,
        errors: list | None = None
    ) -> dict[str, Any]:
        """Generate one or more VegaLite specs via YAML (declarative mode)."""
        errors_context = self._build_errors_context(pipeline, context, errors)
        gridded = get_gridded_metadata(pipeline)
        with self._add_step(title="Creating basic plot structure", steps_layout=self._steps_layout) as step:
            response = self._stream_prompt(
                "main",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                doc_pages=doc_pages,
                gridded=gridded,
                **errors_context,
            )
            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            # Extract each chart independently so one malformed spec does not
            # discard the charts that did parse.
            charts, chart_errors = [], []
            for chart_spec in output.charts:
                try:
                    spec = await self._extract_spec(context, {"yaml_spec": chart_spec.yaml_spec})
                except UNRECOVERABLE_ERRORS:
                    # e.g. missing context or a cancelled run; retrying cannot help.
                    raise
                except Exception as e:
                    log_debug(f"Skipping chart {chart_spec.title!r} that failed to parse: {e}")
                    chart_errors.append(f"{chart_spec.title or 'chart'}: {e}")
                    continue
                charts.append((spec, chart_spec.title))
            if not charts:
                # Nothing parsed; report every failure so the regenerated specs
                # from retry_llm_output have something concrete to fix.
                raise ValueError(
                    "None of the generated chart specifications could be parsed. "
                    + "; ".join(chart_errors)
                )
            skipped = len(output.charts) - len(charts)
            if skipped:
                # Surface the drop to the user instead of only logging it.
                step.stream(f"\n\nSkipped {skipped} chart(s) whose specification could not be parsed.")
            step.success_title = "Complete visualization with titles and colors created"
        return {"charts": charts}

    @retry_llm_output()
    async def _generate_code_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        doc_pages: list[dict] | None = None,
        errors: list | None = None,
    ) -> dict[str, Any] | None:
        """Generate one or more specs via Altair code execution."""
        errors_context = self._build_errors_context(pipeline, context, errors)
        gridded = get_gridded_metadata(pipeline)

        with self._add_step(title="Generating Altair code", steps_layout=self._steps_layout) as step:
            response = self._stream_prompt(
                "main_altair",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                doc_pages=doc_pages,
                gridded=gridded,
                **errors_context,
            )
            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            # Execute each generated code block into its own chart object
            df = await get_data(pipeline)
            chart_objects = []
            for chart_spec in output.charts:
                step.stream(f"\n```python\n{chart_spec.code}\n```\n", replace=False)
                # Get LLM system prompt for safety validation if needed
                system = None
                if self.code_execution == "llm":
                    system = await self._render_prompt(
                        "code_safety",
                        messages,
                        context,
                        code=chart_spec.code,
                    )
                # Execute code using mixin (handles AST validation, LLM validation, user prompt)
                chart = await self._execute_code(chart_spec.code, df, system=system, step=step)
                if chart is None:
                    raise UserCancelledError("Code execution rejected by user.")
                chart_objects.append((chart, chart_spec.title))

        charts = []
        for chart, title in chart_objects:
            # Convert to Vega-Lite spec
            spec = chart.to_dict()
            spec.pop("datasets", None)  # Remove inline data
            spec["data"] = {"name": pipeline.table}  # Use named data source
            charts.append((await self._extract_spec(context, {"yaml_spec": dump_yaml(spec)}), title))

        return {"charts": charts}

    async def _extract_spec(self, context: TContext, spec: dict[str, Any], apply_defaults: bool = True):
        # .encode().decode('unicode_escape') fixes a JSONDecodeError in Python
        # where it's expecting property names enclosed in double quotes
        # by properly handling the escaped characters in your JSON string
        if yaml_spec := spec.get("yaml_spec"):
            vega_spec = load_yaml(yaml_spec)
        elif json_spec := spec.get("json_spec"):
            vega_spec = load_json(json_spec)
        # Supply the palette as a default the model can override, so charts in a
        # single report share colors unless a chart asked for something else.
        # Only on the way in: re-applying it on a later edit would undo a
        # palette the user had deliberately removed.
        if apply_defaults and has_categorical_color(vega_spec):
            vega_spec = self._deep_merge_dicts(
                {"config": {"range": {"category": category_palette()}}}, vega_spec
            )
        if apply_defaults:
            # Kills scientific notation in axis and legend labels globally.
            # Per-encoding `axis: {format: ','}` is not sufficient: in a layered
            # spec the axes of a shared scale are merged, so a format set on one
            # layer and absent on another does not survive, and the axis falls
            # back to a default that renders 6000 as 6e+3.
            vega_spec = self._deep_merge_dicts({"config": {"numberFormat": ","}}, vega_spec)
        return normalize_vegalite_spec(vega_spec, editor_type=self._editor_type)

    def _last_user_query(self, messages: list[Message], step_title: str | None = None) -> str:
        """Text to retrieve documentation against.

        The final message is not reliably the user's: when this agent runs as a
        step inside a plan, the last entry is typically an assistant or tool
        message, which silently disabled documentation lookup altogether. Scan
        back for the most recent user turn and fall back to the step title.

        Multimodal turns carry a list `content` (see _prepare_vision_messages),
        so keep only the text parts -- the store expects a string.
        """
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            else:
                text = " ".join(part for part in content if isinstance(part, str))
            if text.strip():
                return text
        return step_title or ""

    def _diversify_doc_pages(self, candidates: list[dict]) -> list[dict]:
        """Cap how many sections any single documentation page contributes.

        Similarity alone collapses onto one page. A "bar chart of athlete count
        by country" query returned five mark-bar variants -- stacked, normalized
        stacked, histogram, discrete-temporal, faceted -- separated by 0.015
        similarity, none of them a plain bar chart, while nothing about
        transforms or sorting made the cut.

        Candidates arrive sorted by similarity, so taking them in order keeps
        the ranking intact within each page. When `reserve_base_form` is set, the
        best-matching base-form section is admitted first and the remaining slots
        filled around it; the result is still returned in similarity order, since
        the reservation is about inclusion rather than position.
        """
        if not self.n_doc_pages:
            return []
        reserved = None
        if self.reserve_base_form:
            reserved = next(
                (c for c in candidates if (c.get("metadata") or {}).get("base_form")), None
            )
        kept: list[dict] = []
        seen: Counter = Counter()
        if reserved is not None:
            kept.append(reserved)
            seen[(reserved.get("metadata") or {}).get("slug")] += 1
        for candidate in candidates:
            if len(kept) >= self.n_doc_pages:
                break
            if candidate is reserved:
                continue
            slug = (candidate.get("metadata") or {}).get("slug")
            if slug is not None and seen[slug] >= self.doc_pages_per_slug:
                continue
            seen[slug] += 1
            kept.append(candidate)
        position = {id(c): i for i, c in enumerate(candidates)}
        return sorted(kept, key=lambda c: position[id(c)])

    def _dedupe_examples(self, doc_pages: list[dict]) -> list[dict]:
        """Drop example specs already carried by an earlier retrieved section.

        Vega-Lite embeds the same live example on several pages -- 46 of 297 spec
        names are cited more than once -- so two sections can inject byte-identical
        YAML. One production prompt carried stacked_bar_normalize twice, from
        mark-bar and transform-stack, for ~120 tokens of exact repetition.

        Candidates are already in similarity order, so the copy that survives is
        the one on the better-matching section. Pages are copied rather than
        edited in place: the dicts come from the vector store.
        """
        seen: set[str] = set()
        deduped: list[dict] = []
        for page in doc_pages:
            metadata = page.get("metadata") or {}
            examples = metadata.get("examples") or []
            unique = [e for e in examples if e.get("name") not in seen]
            seen.update(e.get("name") for e in unique)
            if len(unique) == len(examples):
                deduped.append(page)
                continue
            # A section stripped of every spec still contributes its prose, so it
            # stays in the payload rather than freeing up an n_doc_pages slot.
            deduped.append({**page, "metadata": {**metadata, "examples": unique}})
        return deduped

    def _drop_forbidden_examples(self, doc_pages: list[dict]) -> list[dict]:
        """Drop example specs built on constructs the prompt forbids.

        Retrieval has twice supplied a reference spec that contradicts the
        instructions in the same prompt -- bar_grouped_facet faceting on `column`,
        repeat_layer using `repeat` -- which leaves the model to guess which half
        to believe. The keys are tagged at build time by
        scripts/create_vega_lite_docs_embeddings.py, so this is a set test rather
        than a judgment.

        Stores built before that tagging carry no `constructs`, in which case
        nothing is dropped.
        """
        forbidden = set(self.exclude_spec_constructs or ())
        if not forbidden:
            return doc_pages
        filtered: list[dict] = []
        for page in doc_pages:
            metadata = page.get("metadata") or {}
            examples = metadata.get("examples") or []
            kept = [
                e for e in examples
                if not forbidden.intersection(e.get("constructs") or ())
            ]
            if len(kept) == len(examples):
                filtered.append(page)
                continue
            # Prose still describes the idiom even with its spec withheld, so the
            # section stays rather than surrendering an n_doc_pages slot.
            filtered.append({**page, "metadata": {**metadata, "examples": kept}})
        return filtered

    async def _get_doc_pages(self, user_query: str) -> list[dict]:
        if not self.n_doc_pages:
            log_debug("Vega-Lite docs lookup disabled (n_doc_pages=0)")
            return []
        if not user_query:
            log_debug("Vega-Lite docs lookup skipped: no query text")
            return []
        try:
            vector_store = self._get_vector_store()
        except Exception as e:
            # Log rather than swallow: a mismatched or missing store otherwise
            # degrades to "no documentation" with no indication anything failed.
            log_debug(f"Vega-Lite docs store unavailable: {e!r}")
            return []
        if not vector_store:
            return []
        try:
            candidates = await vector_store.query(
                user_query, top_k=self.n_doc_pages * self._doc_page_pool
            )
        except Exception as e:
            log_debug(f"Vega-Lite docs lookup failed for {user_query!r}: {e!r}")
            return []
        doc_pages = self._diversify_doc_pages(candidates)
        before = sum(len(page.get("metadata", {}).get("examples") or ()) for page in doc_pages)
        doc_pages = self._dedupe_examples(doc_pages)
        deduped = sum(len(page.get("metadata", {}).get("examples") or ()) for page in doc_pages)
        doc_pages = self._drop_forbidden_examples(doc_pages)
        specs = sum(len(page.get("metadata", {}).get("examples") or ()) for page in doc_pages)
        sources = len({page.get("metadata", {}).get("slug") for page in doc_pages})
        base_forms = sum(1 for page in doc_pages if (page.get("metadata") or {}).get("base_form"))
        # Similarity of what was kept, against the pool it was drawn from. Ranking
        # here has repeatedly proved to sit inside the noise floor -- the section
        # that actually helped has landed anywhere from 2nd to 4th -- so record the
        # numbers needed to decide whether a relevance floor could separate an
        # off-topic payload from a useful one, rather than guessing a threshold.
        scores = [c["similarity"] for c in doc_pages if "similarity" in c]
        pool_scores = [c["similarity"] for c in candidates if "similarity" in c]
        spread = ""
        if scores:
            spread = (
                f", similarity {min(scores):.3f}-{max(scores):.3f}"
                f" (spread {max(scores) - min(scores):.3f}"
                f", pool floor {min(pool_scores):.3f})"
            )
        dropped = "".join([
            f", {base_forms} base-form section(s)" if base_forms else "",
            f", {before - deduped} duplicate spec(s) dropped" if before != deduped else "",
            f", {deduped - specs} spec(s) dropped on forbidden constructs" if deduped != specs else "",
        ])
        log_debug(
            f"Vega-Lite docs lookup returned {len(doc_pages)} section(s) from "
            f"{sources} page(s) of {len(candidates)} candidate(s), {specs} example spec(s)"
            f"{dropped}{spread}"
        )
        # Per-section detail: which slug scored what, and whether it brought a
        # spec. Deciding between a relevance floor, a prose-only filter and a
        # reranker needs the score attached to the section it belongs to.
        for page in doc_pages:
            metadata = page.get("metadata") or {}
            score = page.get("similarity")
            score_text = f"{score:.3f}" if isinstance(score, float) else "  -  "
            log_debug(
                f"    {score_text} {metadata.get('slug')} / {metadata.get('section_title')} "
                f"[{metadata.get('kind')}, {len(metadata.get('examples') or ())} spec(s)]"
            )
        return doc_pages

    async def revise(
        self,
        feedback: str,
        messages: list[Message],
        context: TContext,
        view: LumenEditor | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        **kwargs
    ) -> str:
        """Revise a VegaLite specification based on user feedback.

        This override adds:
        1. Doc examples from vector store
        2. Plot image for vision analysis (if view is a VegaLiteEditor)

        Parameters
        ----------
        feedback : str
            User's feedback or instruction for revision.
        messages : list[Message]
            Chat message history.
        context : TContext
            Session context.
        view : LumenEditor | None
            The editor containing the current spec to revise.
        spec : str | None
            The spec string (used if view is None).
        language : str | None
            The spec language (used if view is None).
        errors : list[str] | None
            List of errors to include in context.
        **kwargs
            Additional arguments passed to parent revise.

        Returns
        -------
        str
            The revised YAML specification.
        """
        if errors is not None:
            kwargs["errors"] = errors
        doc_pages = await self._get_doc_pages(feedback)
        context["doc_pages"] = doc_pages

        messages = self._prepare_vision_messages(messages, view, f"Revise this chart: {feedback!r}")

        return await super().revise(
            feedback, messages, context, view=view, spec=spec, language=language, **kwargs
        )

    @classmethod
    def _is_annotation_view(cls, view: dict) -> bool:
        """Whether a layer only annotates the chart rather than plotting data."""
        mark = view.get("mark")
        mark_type = mark.get("type") if isinstance(mark, dict) else mark
        return mark_type in cls._annotation_marks

    @classmethod
    def _spec_has_tooltips(cls, spec: dict) -> bool:
        """Whether every data-drawing view in a spec already sets tooltips.

        `interaction_polish` exists only to add tooltips, and costs a full LLM
        round trip plus a rendered plot image. The main pass now emits tooltips
        itself on most charts, so the step is often a no-op that rewrites the spec
        it was handed. Composition containers are walked because a spec is only
        fully covered when every leaf view is.
        """
        for key in ("layer", "concat", "hconcat", "vconcat"):
            views = spec.get(key)
            if isinstance(views, list):
                data_views = [v for v in views if not cls._is_annotation_view(v)]
                return bool(data_views) and all(cls._spec_has_tooltips(v) for v in data_views)
        if (spec.get("encoding") or {}).get("tooltip") is not None:
            return True
        mark = spec.get("mark")
        return isinstance(mark, dict) and mark.get("tooltip") is not None

    async def _polish_plot(self, out: VegaLiteEditor, messages: list[Message], context: TContext, doc: str | None = None):
        steps = {
            "interaction_polish": "Add helpful tooltips and ensure responsive, accessible user experience",
        }
        if self._spec_has_tooltips(out._spec_dict.get("spec") or {}):
            log_debug("Skipping interaction_polish: every view already defines tooltips")
            return
        with out.param.update(loading=True):
            for step_name, step_desc in steps.items():
                # Only pass the vega lite 'spec' portion to prevent ballooning context
                # Include the VegaLiteEditor so the LLM can see the current plot
                step_name, update_dict = await self._update_spec_step(
                    step_name, step_desc, out.spec, step_name, messages, context, doc=doc, out=out
                )
                try:
                    # Validate merged spec, and keep what normalization added to it
                    merged_spec = self._deep_merge_dicts(out._spec_dict["spec"], update_dict)
                    normalized = await self._extract_spec(
                        context, {"yaml_spec": dump_yaml(merged_spec)}, apply_defaults=False
                    )
                except Exception as e:
                    log_debug(f"Skipping invalid {step_name} update due to error: {e}")
                    continue
                out.spec = dump_yaml(normalized["spec"])
            log_debug(f"📊 Applied {step_name} updates and refreshed visualization")

    @staticmethod
    def _overview_item(editor: VegaLiteEditor) -> ParamFunction:
        """A plot for the "All" overview that tracks an editor's component.

        Binding to the editor's ``component`` means the overview re-renders when
        the chart is edited or polished, instead of showing a stale snapshot.
        """
        render = bind(
            lambda component: component.get_panel() if component is not None else None,
            editor.param.component,
        )
        return ParamFunction(render, sizing_mode="stretch_width")

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """
        Generates one or more VegaLite visualizations using a progressive building
        approach with real-time updates.
        """
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        user_query = self._last_user_query(messages, step_title)
        # _get_doc_pages already degrades to [] and logs why.
        doc_pages = await self._get_doc_pages(user_query)

        # Step 1: Generate one or more specs
        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        # Produces {"charts": [({"spec": {...}, ...}, title), ...]}
        result = await self._generate_spec(
            messages, context, pipeline, doc, doc_pages=doc_pages
        )
        if result is None:
            # User rejected code execution
            return [], {}

        # Step 2: One editable editor per chart. The UI renders each as its own
        # tab with a code editor beside the plot. A lone chart keeps the task's
        # own title; several charts are labelled individually so their tabs are
        # distinguishable. Vega-Lite cannot page a dimension, so per chart
        # collapse every gridded dim its spec does not reference to a single
        # slice (each chart may pick different axes).
        charts = result["charts"]
        editors = [
            self._editor_type(
                component=self.view_type(
                    pipeline=subset_gridded_to_2d(pipeline, spec.get("spec", {}), "vega-lite"),
                    **spec,
                ),
                title=step_title if len(charts) == 1 else (title or f"Chart {i}"),
            )
            for i, (spec, title) in enumerate(charts, start=1)
        ]

        # Step 3: For multiple charts, append an "All" tab that stacks every plot
        # together as an overview, with one code editor sub-tab per chart. It is
        # added last so the UI opens it by default as the final tab. Each plot
        # re-renders from its editor's component, so edits (or the polish pass)
        # to a chart tab stay in sync with the overview.
        outs = editors
        if len(editors) > 1:
            # Left at its natural height so it overflows the editor's view,
            # which is what gives that view something to scroll.
            stacked = Column(
                *(self._overview_item(editor) for editor in editors),
                sizing_mode="stretch_width",
            )
            overview = MultiChartEditor(
                component=Panel(object=stacked), title="All", chart_editors=editors
            )
            outs = [*editors, overview]

        # Step 4: enhancements (LLM-driven creative decisions), per editable chart
        if not self.code_execution_enabled:
            for editor in editors:
                state.execute(partial(self._polish_plot, editor, messages, context, doc))

        out_context = await editors[-1].render_context()
        return outs, out_context

    async def annotate(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        spec: dict,
        view: VegaLiteEditor | None = None,
    ) -> str:
        """
        Apply annotations based on user request.

        Parameters
        ----------
        instruction : str
            User's description of what to annotate
        messages : list[Message]
            Chat history for context
        context : TContext
            Session context
        spec : dict
            The current VegaLite specification (full dict with 'spec' key)
        view : VegaLiteEditor | None
            Optional VegaLite editor to export plot image from for vision analysis.

        Returns
        -------
        str
            Updated specification with annotations
        """
        messages = self._prepare_vision_messages(messages, view, f"Annotate this chart: {instruction!r}")

        vega_spec = dump_yaml(spec["spec"], default_flow_style=False)
        result = await self._invoke_prompt(
            "annotate_plot",
            messages,
            context,
            vega_spec=vega_spec,
        )
        update_dict = load_yaml(result.yaml_update)

        # Merge and validate
        final_dict = spec.copy()

        try:
            final_dict["spec"] = self._deep_merge_dicts(final_dict["spec"], update_dict)
            spec = await self._extract_spec(
                context, {"yaml_spec": dump_yaml(final_dict["spec"])}, apply_defaults=False
            )
        except Exception as e:
            log_debug(f"Skipping invalid annotation update due to error: {e}")
            raise e
        return dump_yaml(spec["spec"])
