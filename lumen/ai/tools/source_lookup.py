"""
SourceLookup — vector-search discovery tool for source control actions.

Modeled on :class:`MetadataLookup`: indexes action names + docstrings from
``ParametricSourceControls`` instances in ``context["source_controls"]``,
then returns a formatted listing of relevant actions with parameter schemas.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NotRequired

import param

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..controls import CatalogSourceControls, ParametricSourceControls
from ..llm import Message
from ..translate import doc_descriptions, function_to_model
from ..utils import deterministic_hash, log_debug
from .vector_lookup import VectorLookupTool, make_refined_query_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _control_hash(ctrl: ParametricSourceControls | CatalogSourceControls) -> str:
    """Stable hash for a control based on its class name + action signatures.

    Unlike ``id(ctrl)``, this survives garbage-collection and
    re-instantiation (e.g. across page reloads).
    """
    parts = [type(ctrl).__qualname__]
    for name, func in ctrl.as_tools():
        sig = getattr(func, "__signature__", None)
        parts.append(f"{name}:{sig}")
    return format(deterministic_hash("|".join(parts)), "08x")


# ---------------------------------------------------------------------------
# Context models
# ---------------------------------------------------------------------------

class SourceLookupInputs(ContextModel):
    source_controls: NotRequired[list]


class SourceLookupOutputs(ContextModel):
    source_actions: dict[str, Any]


# ---------------------------------------------------------------------------
# SourceLookup
# ---------------------------------------------------------------------------

class SourceLookup(VectorLookupTool):
    """
    Discovers available external-data-source actions via vector search.

    On ``sync(context)``, iterates ``context["source_controls"]`` and embeds
    each ``ParametricSourceControls`` action (name + docstring + parameter
    descriptions) into the shared vector store.  On ``respond()``, queries
    the vector store for the user's query and returns a formatted listing of
    available actions with their parameter schemas — analogous to how
    :class:`MetadataLookup` returns relevant tables.
    """

    always_use = param.Boolean(default=True)

    conditions = param.List(
        default=[
            "Best paired with SourceAgent for fetching external data",
            "Use when no tables are loaded but external source controls are configured",
            "Provides available API endpoints and callable actions, not table metadata",
            "Not useful for querying already-loaded data",
        ]
    )

    exclusions = param.List(default=[])

    not_with = param.List(default=[])

    prompts = param.Dict(
        default={
            "refine_query": {
                "template": PROMPTS_DIR / "VectorLookupTool" / "refine_query.jinja2",
                "response_model": make_refined_query_model,
            },
        },
    )

    purpose = param.String(
        default="""
        Discovers relevant external data source actions (API endpoints,
        Python callables, URL templates) using vector search, providing
        context for SourceAgent to fetch new data into the application."""
    )

    min_similarity = param.Number(default=0)

    n = param.Integer(default=10, bounds=(1, None))

    _item_type_name = "data_sources"

    _ready = param.Boolean(default=False, allow_None=True)

    input_schema = SourceLookupInputs
    output_schema = SourceLookupOutputs

    def __init__(self, **params):
        super().__init__(**params)
        self._indexed_controls: set[str] = set()

    async def prepare(self, context: TContext):
        """Override base to call sync instead of _update_vector_store."""
        await self.sync(context)

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        controls = context.get("source_controls", [])
        return any(
            isinstance(c, (ParametricSourceControls, CatalogSourceControls))
            for c in controls
        )

    # ------------------------------------------------------------------
    # Sync / index
    # ------------------------------------------------------------------

    async def sync(self, context: TContext):
        """Index actions from all ParametricSourceControls in context."""
        controls = context.get("source_controls", [])
        if not controls:
            self._ready = True
            return

        entries = []
        for ctrl in controls:
            if not isinstance(ctrl, (ParametricSourceControls, CatalogSourceControls)):
                continue
            ctrl_key = _control_hash(ctrl)
            if ctrl_key in self._indexed_controls:
                continue
            self._indexed_controls.add(ctrl_key)

            for action_name, func in ctrl.as_tools():
                text = self._build_action_text(action_name, func)
                entries.append({
                    "text": text,
                    "metadata": {
                        "type": self._item_type_name,
                        "action_name": action_name,
                        "control_key": ctrl_key,
                    },
                })

        if entries:
            log_debug(f"[SourceLookup] Upserting {len(entries)} action entries")
            await self.vector_store.upsert(entries)

        self._ready = True

    @staticmethod
    def _build_action_text(action_name: str, func: Callable) -> str:
        """Build a rich text representation for embedding."""
        description, param_docs = doc_descriptions(func)
        parts = [f"Action: {action_name}"]
        if description:
            parts.append(f"Description: {description}")
        if param_docs:
            param_lines = []
            for pname, pdesc in param_docs.items():
                param_lines.append(f"  - {pname}: {pdesc}")
            parts.append("Parameters:\n" + "\n".join(param_lines))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Respond — gather + format
    # ------------------------------------------------------------------

    def _format_results_for_refinement(
        self, results: list[dict[str, Any]], context: TContext
    ) -> str:
        """Format action search results for the refinement prompt."""
        formatted = []
        for result in results:
            text = result["text"]
            formatted.append(
                f"- {text} (Similarity: {result.get('similarity', 0):.3f})"
            )
        return "\n".join(formatted)

    async def _gather_info(
        self, messages: list[Message], context: TContext
    ) -> SourceLookupOutputs:
        query = messages[-1]["content"]
        filters = {"type": self._item_type_name}

        # Count total actions
        total_actions = sum(
            len(ctrl.as_tools())
            for ctrl in context.get("source_controls", [])
            if isinstance(ctrl, (ParametricSourceControls, CatalogSourceControls))
        )

        if total_actions < 5:
            results = await self.vector_store.query(query, top_k=self.n, filters=filters)
        else:
            results = await self._perform_search_with_refinement(
                query, context, filters=filters
            )

        # Build action info mapping
        controls = context.get("source_controls", [])
        ctrl_map = {
            _control_hash(c): c for c in controls
            if isinstance(c, (ParametricSourceControls, CatalogSourceControls))
        }

        source_actions: dict[str, Any] = {}
        for result in results:
            if result.get("similarity", 0) < self.min_similarity:
                continue
            meta = result["metadata"]
            action_name = meta["action_name"]
            ctrl_key = meta.get("control_key")
            ctrl = ctrl_map.get(ctrl_key)
            if ctrl is None:
                continue

            tools_dict = dict(ctrl.as_tools())
            func = tools_dict.get(action_name)
            if func is None:
                continue

            description, param_docs = doc_descriptions(func)

            # Build parameter schema from function_to_model
            try:
                model = function_to_model(func)
                param_schema = {
                    name: {
                        "type": str(field.annotation),
                        "required": field.is_required(),
                        **(
                            {"default": repr(field.default)}
                            if field.default is not None
                            else {}
                        ),
                        **(
                            {"description": param_docs.get(name, "")}
                            if param_docs.get(name)
                            else {}
                        ),
                    }
                    for name, field in model.model_fields.items()
                }
            except Exception:
                param_schema = {}

            source_actions[action_name] = {
                "description": description or "(no description)",
                "parameters": param_schema,
                "similarity": result.get("similarity", 0),
                "control_key": ctrl_key,
            }

        return {"source_actions": source_actions}

    def _format_context(self, outputs: SourceLookupOutputs) -> str:
        """Format the discovered actions for inclusion in LLM context."""
        source_actions = outputs.get("source_actions", {})
        if not source_actions:
            return "No relevant external data source actions found."

        lines = ["Available external data source actions:"]
        for action_name, info in source_actions.items():
            lines.append(f"\n## {action_name}")
            lines.append(f"Description: {info['description']}")
            if info.get("parameters"):
                lines.append("Parameters:")
                for pname, pinfo in info["parameters"].items():
                    parts = [f"  - {pname} ({pinfo.get('type', 'Any')})"]
                    if not pinfo.get("required", True) and pinfo.get("default"):
                        parts.append(f"default={pinfo['default']}")
                    if pinfo.get("description"):
                        parts.append(f"— {pinfo['description']}")
                    lines.append(" ".join(parts))
        return "\n".join(lines)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        **kwargs: dict[str, Any],
    ) -> tuple[list[Any], SourceLookupOutputs]:
        if not messages:
            return [], {"source_actions": {}}

        out_model = await self._gather_info(messages, context)
        return [self._format_context(out_model)], out_model
