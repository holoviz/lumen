from __future__ import annotations

import param
import yaml

from panel.pane import Markdown
from panel_material_ui import ChatMessage, Typography
from pydantic import BaseModel, Field

from ...pipeline import Pipeline
from ..actor import LLMUser
from ..config import PROMPTS_DIR
from ..editors import LumenEditor
from ..utils import describe_data_sync, get_data

_MAX_SPEC_CHARS = 1500
_MAX_DATA_CHARS = 2000


class Story(BaseModel):

    chain_of_thought: str = Field(
        description="In 1-2 sentences, outline the narrative arc across the selected tables and charts."
    )

    title: str = Field(
        description="A short, descriptive title for the report story."
    )

    markdown: str = Field(
        description="The narrative in Markdown, referencing the selected tables and charts and their key findings."
    )


class SectionNote(BaseModel):

    chain_of_thought: str = Field(
        description="In 1-2 sentences, the key insight for this section."
    )

    markdown: str = Field(
        description="A concise Markdown note (2-4 sentences) annotating this section's table(s) and chart(s)."
    )


class StoryAgent(LLMUser):
    """
    Writes a natural-language "story" annotating the tables and charts a user
    selected for export: one overall narrative plus per-section notes.
    """

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "StoryAgent" / "main.jinja2",
                "response_model": Story,
            },
            "section": {
                "template": PROMPTS_DIR / "StoryAgent" / "section.jinja2",
                "response_model": SectionNote,
            },
        }
    )

    async def write_story(self, report_context: str, title: str = "") -> Story:
        """Write the overall narrative covering all selected sections."""
        return await self._invoke_prompt(
            "main",
            messages=[{"role": "user", "content": "Write the report story."}],
            context={},
            report_context=report_context,
            report_title=title,
        )

    async def write_note(self, section_context: str, title: str = "") -> SectionNote:
        """Write a short note annotating a single section."""
        return await self._invoke_prompt(
            "section",
            messages=[{"role": "user", "content": "Write the section note."}],
            context={},
            section_context=section_context,
            section_title=title,
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
    pipeline = component if isinstance(component, Pipeline) else getattr(component, "pipeline", None)
    if pipeline is not None:
        try:
            data = await get_data(pipeline)
            parts.append(f"Data summary:\n{describe_data_sync(data)[:_MAX_DATA_CHARS]}")
        except Exception:
            # Best-effort: a view we cannot summarize should not abort the story.
            pass
    return "\n".join(parts) or (view.title or type(component).__name__)


async def build_view_context(view) -> str | None:
    """Describe a single report view (text, table, or chart) for the story prompt."""
    if isinstance(view, (Typography, Markdown)):
        return view.object
    if isinstance(view, str):
        return view
    if isinstance(view, ChatMessage):
        obj = view.object
        return obj.object if isinstance(obj, Markdown) else (obj if isinstance(obj, str) else None)
    if isinstance(view, LumenEditor):
        return await _describe_editor(view)
    return None


async def build_section_context(section) -> str:
    """Compact context describing one section's tables, charts, and notes."""
    described = [await build_view_context(view) for view in section.views]
    return "\n\n".join(text.strip() for text in described if text and text.strip())


async def build_report_context(sections) -> str:
    """Context describing every selected section for the overall story."""
    blocks = [
        f"## {section.title or 'Section'}\n{await build_section_context(section)}"
        for section in sections
    ]
    return "\n\n".join(blocks)
