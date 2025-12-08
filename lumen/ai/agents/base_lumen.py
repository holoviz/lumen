

import param

from ...config import dump_yaml, load_yaml
from ..config import PROMPTS_DIR
from ..context import TContext
from ..llm import Message
from ..models import RetrySpec
from ..utils import apply_changes, retry_llm_output
from ..views import LumenOutput
from .base import Agent


class BaseLumenAgent(Agent):

    prompts = param.Dict(
        default={
            "revise_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "BaseLumenAgent" / "revise_output.jinja2"},
        }
    )

    user = param.String(default="Lumen")

    _max_width = None
    _output_type = LumenOutput

    @retry_llm_output()
    async def revise(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        view: LumenOutput | None = None,
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
        system = await self._render_prompt(
            "revise_output",
            messages,
            context,
            numbered_text=numbered_text,
            language=language,
            feedback=instruction,
            errors=errors,
            **kwargs
        )
        retry_model = self._lookup_prompt_key("revise_output", "response_model")
        result = await self.llm.invoke(
            messages,
            system=system,
            response_model=retry_model,
            model_spec="edit"
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
