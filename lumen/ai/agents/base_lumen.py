import param

from ...config import dump_yaml, load_yaml
from ..config import PROMPTS_DIR
from ..context import TContext
from ..editors import LumenEditor
from ..llm import Message
from ..models import RetrySpec
from ..schemas import slug_to_table_name
from ..utils import apply_changes, retry_llm_output
from .base import Agent

SPEC_TYPE_MAP = {
    "sql": "query",
    "yaml": "specification",
    "json": "specification",
    "vega-lite": "visualization spec",
}


class BaseLumenAgent(Agent):

    prompts = param.Dict(
        default={
            "explain_output": {"template": PROMPTS_DIR / "BaseLumenAgent" / "explain_output.jinja2"},
            "revise_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "BaseLumenAgent" / "revise_output.jinja2"},
        }
    )

    user = param.String(default="Lumen")

    _max_width = None
    _editor_type = LumenEditor

    def _resolve_revise_table(self, context: TContext, view: LumenEditor | None) -> str | None:
        """Slug of the table the output being revised uses, so the revise
        prompt can scope its schema context to just that table. Returns None
        when it cannot be determined (e.g. SQL generation before a pipeline
        exists), in which case the prompt falls back to the broader context.
        """
        metaset = context.get("metaset") if isinstance(context, dict) else None
        if metaset is None:
            return None
        # Prefer the component being revised (unambiguous), then the context.
        # Table-first: a Pipeline component exposes its own .table, while a View
        # component has no .table so we fall back to its .pipeline.table. A
        # chained Pipeline also has a .pipeline (its parent), whose table is NOT
        # the one being revised, so .table must win over .pipeline.
        candidate = None
        component = getattr(view, "component", None)
        if component is not None:
            candidate = getattr(component, "table", None)
            if candidate is None:
                pipeline = getattr(component, "pipeline", None)
                candidate = getattr(pipeline, "table", None) if pipeline is not None else None
        if candidate is None:
            candidate = getattr(context.get("pipeline"), "table", None) or context.get("table")
        slug = metaset._resolve_table_slug(candidate)
        if slug is None:
            return None
        # Prefer the active (newest) generation so _generate_context's
        # active-slug filter does not empty the scoped context.
        active = set(metaset._deduplicated_slugs())
        if slug not in active:
            name = slug_to_table_name(slug)
            slug = next((s for s in active if slug_to_table_name(s) == name), slug)
        return slug

    @retry_llm_output()
    async def revise(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        view: LumenEditor | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        **kwargs
    ) -> str:
        """
        Retry the output by line, allowing the user to provide instruction on why the output was not satisfactory, or an error.
        """
        if view is not None:
            spec = view.spec
            language = view.language
        if spec is None:
            raise ValueError("Must provide previous spec to revise.")
        lines = spec.splitlines()
        numbered_text = "\n".join(f"{i:2d}: {line}" for i, line in enumerate(lines, 1))
        revise_table = kwargs.pop("revise_table", None) or self._resolve_revise_table(context, view)
        result = await self._invoke_prompt(
            "revise_output",
            messages,
            context,
            model_spec="edit",
            numbered_text=numbered_text,
            language=language,
            feedback=instruction,
            errors=errors,
            revise_table=revise_table,
            **kwargs
        )
        new_spec_raw = apply_changes(lines, result.edits)
        spec = load_yaml(new_spec_raw)
        if view is not None:
            view.validate_spec(spec)
        if isinstance(spec, str):
            yaml_spec = spec
        else:
            yaml_spec = dump_yaml(spec)
        return yaml_spec

    async def explain(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        view: LumenEditor | None = None,
        spec: str | None = None,
        language: str | None = None,
    ):
        """
        Explain the current spec in plain language, streaming chunks.
        """
        if view is not None:
            spec = view.spec
            language = view.language
        if not spec or not spec.strip():
            return

        spec_type = SPEC_TYPE_MAP.get(language, "code")

        if instruction:
            user_content = f"Explain: {instruction}"
        else:
            user_content = "Explain what this does."
        messages = messages + [{"role": "user", "content": user_content}]

        async for chunk in self._stream_prompt(
            "explain_output",
            messages,
            context,
            spec=spec,
            language=language,
            spec_type=spec_type,
            user_question=instruction,
        ):
            yield chunk
