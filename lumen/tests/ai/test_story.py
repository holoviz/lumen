import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown

from lumen.ai.report import Action, Report, Section


class A(Action):

    order = param.List()

    async def _execute(self, context, **kwargs):
        return [Markdown("A done")], {'text': 'A'}


class B(Action):

    order = param.List()

    async def _execute(self, context, **kwargs):
        return [Markdown("B done")], {'text': 'B'}


async def test_story_agent_write_story(llm):
    from lumen.ai.agents.story import Story, StoryAgent

    llm.set_responses([Story(chain_of_thought="arc", title="Olympics", markdown="# Olympics\nBody")])
    agent = StoryAgent(llm=llm)

    result = await agent.write_story("Section: Medals\n<data summary>", title="Olympic Report")

    assert isinstance(result, Story)
    assert result.title == "Olympics"
    assert result.markdown == "# Olympics\nBody"


async def test_story_agent_write_note(llm):
    from lumen.ai.agents.story import SectionNote, StoryAgent

    llm.set_responses([SectionNote(chain_of_thought="key", markdown="Men-only sports dominate.")])
    agent = StoryAgent(llm=llm)

    result = await agent.write_note("Section: Gender\n<data summary>", title="Gender Exclusive")

    assert isinstance(result, SectionNote)
    assert "Men-only" in result.markdown


async def test_build_report_context_includes_titles_and_text():
    from lumen.ai.agents.story import (
        build_report_context, build_section_context,
    )

    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='My Report',
    )
    await report.execute()

    ctx = await build_report_context([s for s in report if s.include_in_export])
    assert "Section A" in ctx
    assert "A done" in ctx
    assert "Section B" in ctx
    assert "B done" in ctx

    section_ctx = await build_section_context(report[0])
    assert "A done" in section_ctx
    assert "B done" not in section_ctx


async def test_build_section_context_summarizes_table_pipeline(tiny_source):
    from lumen.ai.agents.story import build_view_context
    from lumen.ai.editors import LumenEditor
    from lumen.pipeline import Pipeline

    pipeline = Pipeline(source=tiny_source, table='tiny')
    editor = LumenEditor(component=pipeline, title='Tiny table')

    ctx = await build_view_context(editor)

    # The table's columns should surface in the summary sent to the LLM.
    assert "category" in ctx
    assert "value" in ctx
