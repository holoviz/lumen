import param

from ...config import dump_yaml, load_yaml
from ..config import PROMPTS_DIR
from ..context import TContext
from ..editors import LumenEditor
from ..llm import Message
from ..models import RetrySpec
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
        result = await self._invoke_prompt(
            "revise_output",
            messages,
            context,
            model_spec="edit",
            numbered_text=numbered_text,
            language=language,
            feedback=instruction,
            errors=errors,
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
