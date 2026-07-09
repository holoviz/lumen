import json

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


def _nb_text(report):
    return "".join(
        "".join(cell["source"]) for cell in json.loads(report.to_notebook())["cells"]
    )


async def test_report_annotate_inserts_story_and_notes(llm):
    from lumen.ai.agents.story import SectionNote, Story

    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='My Report',
        llm=llm,
    )
    await report.execute()
    llm.set_responses([
        Story(chain_of_thought="c", title="The Big Picture", markdown="Overall narrative."),
        SectionNote(chain_of_thought="c", markdown="Note for A."),
        SectionNote(chain_of_thought="c", markdown="Note for B."),
    ])

    await report._annotate_report()

    nb = _nb_text(report)
    assert "The Big Picture" in nb
    assert "Overall narrative." in nb
    assert "Note for A." in nb
    assert "Note for B." in nb

    # The overall story and the notes must both survive HTML export too.
    html = report.to_html()
    assert "The Big Picture" in html
    assert "Overall narrative." in html
    assert "Note for A." in html
    assert "Note for B." in html


async def test_report_annotate_only_selected_sections(llm):
    from lumen.ai.agents.story import SectionNote, Story

    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='My Report',
        llm=llm,
    )
    await report.execute()
    report[1].include_in_export = False
    llm.set_responses([
        Story(chain_of_thought="c", title="S", markdown="overall"),
        SectionNote(chain_of_thought="c", markdown="Note for A."),
    ])

    await report._annotate_report()

    nb = _nb_text(report)
    assert "Note for A." in nb
    assert "Note for B." not in nb


async def test_report_annotate_is_idempotent(llm):
    from lumen.ai.agents.story import SectionNote, Story

    report = Report(Section(A(order=[]), title='Section A'), title='My Report', llm=llm)
    await report.execute()

    llm.set_responses([Story(chain_of_thought="c", title="S1", markdown="v1"), SectionNote(chain_of_thought="c", markdown="n1")])
    await report._annotate_report()
    llm.set_responses([Story(chain_of_thought="c", title="S2", markdown="v2"), SectionNote(chain_of_thought="c", markdown="n2")])
    await report._annotate_report()

    nb = _nb_text(report)
    assert "v2" in nb and "n2" in nb
    assert "v1" not in nb and "n1" not in nb


async def test_report_annotate_no_llm_is_noop():
    report = Report(Section(A(order=[]), title='Section A'), title='My Report')
    await report.execute()

    await report._annotate_report()  # must not raise

    nb = _nb_text(report)
    assert "A done" in nb


async def test_report_outline_defaults_empty():
    report = Report(Section(A(order=[]), title='Section A'), title='R')
    assert report._story_outline == []


async def test_report_default_outline_from_sections():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='R',
    )
    await report.execute()
    assert report._build_default_outline() == [
        {"section": "Section A"}, {"section": "Section B"}
    ]


async def test_report_outline_reorders_export():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='R',
    )
    await report.execute()

    report._story_outline = [{"section": "Section B"}, {"section": "Section A"}]

    nb = _nb_text(report)
    assert nb.index("B done") < nb.index("A done")


async def test_report_outline_inserts_headings():
    report = Report(Section(A(order=[]), title='Section A'), title='R')
    await report.execute()

    report._story_outline = [{"heading": "Introduction", "level": 1}, {"section": "Section A"}]

    nb = _nb_text(report)
    assert "Introduction" in nb
    assert nb.index("Introduction") < nb.index("A done")


async def test_report_outline_reorders_html_export():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='R',
    )
    await report.execute()

    report._story_outline = [
        {"heading": "Key Findings", "level": 1},
        {"section": "Section B"},
        {"section": "Section A"},
    ]

    html = report.to_html()
    assert "Key Findings" in html
    assert html.index("B done") < html.index("A done")


async def test_report_arrange_editor_seeds_and_binds_outline():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='R',
    )
    await report.execute()

    # Opening the arrange dialog seeds the editor from the current sections.
    report._open_arrange_dialog()
    assert report._outline_editor.value == [{"section": "Section A"}, {"section": "Section B"}]
    assert report._outline_dialog.open is True

    # Editing the JSONEditor updates the outline, which reorders the export.
    report._outline_editor.value = [{"section": "Section B"}, {"section": "Section A"}]
    assert report._story_outline == [{"section": "Section B"}, {"section": "Section A"}]
    nb = _nb_text(report)
    assert nb.index("B done") < nb.index("A done")
