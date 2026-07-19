from __future__ import annotations

import param
import yaml

from pydantic import BaseModel, Field

from ...pipeline import Pipeline
from ..actor import LLMUser
from ..config import PROMPTS_DIR
from ..editors import LumenEditor
from ..utils import describe_data_sync, get_data

_MAX_SPEC_CHARS = 1500
_MAX_DATA_CHARS = 2000


class StoryBlock(BaseModel):

    prose: str = Field(
        default="",
        description=(
            "A narrative paragraph in Markdown. Leave empty when this block "
            "places a chart or table instead."
        ),
    )

    view: int | None = Field(
        default=None,
        description=(
            "The number of the chart or table to place here, taken from the "
            "catalog. Leave null for a prose block."
        ),
    )


class Story(BaseModel):

    chain_of_thought: str = Field(
        description="In 1-2 sentences, outline the narrative arc that ties the charts and tables together."
    )

    title: str = Field(
        description="A short, descriptive title for the report story."
    )

    blocks: list[StoryBlock] = Field(
        description=(
            "The report as a flowing blog post: alternating prose blocks and "
            "chart/table blocks, in reading order."
        )
    )


class ProseEdit(BaseModel):

    chain_of_thought: str = Field(
        description="In one sentence, how the paragraph should change to satisfy the instruction."
    )

    prose: str = Field(
        description="The rewritten paragraph in Markdown, keeping any facts that are still accurate."
    )


class StoryAgent(LLMUser):
    """
    Writes a "story" that reads like a blog post: short prose paragraphs
    interleaved with the charts and tables a user selected for export.
    """

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "StoryAgent" / "main.jinja2",
                "response_model": Story,
            },
            # Named "rewrite" rather than "edit": model_kwargs maps an "edit"
            # spec to a pricier model on some providers, and this is a small job.
            "rewrite": {
                "template": PROMPTS_DIR / "StoryAgent" / "rewrite.jinja2",
                "response_model": ProseEdit,
            },
        }
    )

    async def rewrite_prose(self, prose: str, instruction: str, catalog: str = "") -> ProseEdit:
        """Rewrite a single paragraph of the story following the user's instruction."""
        return await self._invoke_prompt(
            "rewrite",
            messages=[{"role": "user", "content": "Rewrite the paragraph."}],
            context={},
            prose=prose,
            instruction=instruction,
            catalog=catalog,
        )

    async def write_story(self, catalog: str, guidance: str = "", title: str = "") -> Story:
        """Write the blog-post story that weaves prose around the catalogued views."""
        return await self._invoke_prompt(
            "main",
            messages=[{"role": "user", "content": "Write the report story."}],
            context={},
            catalog=catalog,
            guidance=guidance,
            report_title=title,
        )


async def _describe_editor(view: LumenEditor) -> str:
    """Summarize a chart/table editor as spec + data summary for the prompt."""
    parts = []
    component = view.component
    try:
        spec = yaml.dump(component.to_spec(), sort_keys=False)
        parts.append(f"Specification:\n{spec[:_MAX_SPEC_CHARS]}")
    except (TypeError, ValueError, AttributeError, yaml.YAMLError):
        pass
    pipeline = component if isinstance(component, Pipeline) else component.pipeline
    if pipeline is not None:
        try:
            data = await get_data(pipeline)
            parts.append(f"Data summary:\n{describe_data_sync(data)[:_MAX_DATA_CHARS]}")
        except Exception:
            # Best-effort: a view we cannot summarize should not abort the story.
            pass
    return "\n".join(parts) or (view.title or type(component).__name__)


async def build_catalog(views) -> str:
    """Number each chart/table so the story can reference them by their catalog number.

    Deliberately omits the view's title: it is the pipeline step name (e.g.
    "Prepare data", "Create pie chart"), which would leak into the story. The
    data summary describes what each figure actually shows instead.
    """
    blocks = [
        f"### View {i}\n{await _describe_editor(view)}"
        for i, view in enumerate(views, start=1)
    ]
    return "\n\n".join(blocks)
