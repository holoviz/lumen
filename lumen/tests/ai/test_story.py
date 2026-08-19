import json

import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown
from panel.widgets import TextEditor

from lumen.ai.agents.story import (
    ProseEdit, Story, StoryAgent, StoryBlock, build_catalog,
)
from lumen.ai.editors import LumenEditor
from lumen.ai.report import Action, Section
from lumen.ai.story import StoryReport, _prose_to_html, _prose_to_markdown
from lumen.pipeline import Pipeline


class A(Action):

    order = param.List()

    async def _execute(self, context, **kwargs):
        return [Markdown("A done")], {'text': 'A'}


class B(Action):

    order = param.List()

    async def _execute(self, context, **kwargs):
        return [Markdown("B done")], {'text': 'B'}


class ChartAction(Action):
    """Emits a pipeline-backed LumenEditor so the story catalog has real views."""

    source = param.Parameter()

    label = param.String(default="Chart")

    async def _execute(self, context, **kwargs):
        pipeline = Pipeline(source=self.source, table='tiny')
        return [LumenEditor(component=pipeline, title=self.label)], {'text': self.label}


def _nb_text(report):
    return "".join(
        "".join(cell["source"]) for cell in json.loads(report.to_notebook())["cells"]
    )


def _kinds(report):
    return [kind for kind, _ in report._story.blocks]


def _prose_editors(report):
    """Every editable prose block rendered in the story, at any nesting."""
    found = []

    def walk(obj):
        if isinstance(obj, TextEditor):
            found.append(obj)
            return
        for child in (getattr(obj, 'objects', None) or []):
            walk(child[1] if isinstance(child, tuple) else child)

    walk(report._story_column)
    return found


async def test_story_agent_write_story(llm):
    llm.set_responses([Story(
        chain_of_thought="arc", title="Olympics",
        blocks=[StoryBlock(prose="Intro."), StoryBlock(view=1), StoryBlock(prose="Wrap up.")],
    )])
    agent = StoryAgent(llm=llm)

    result = await agent.write_story(
        "### View 1: Medals\n<summary>", guidance="focus on trends", title="Olympic Report"
    )

    assert isinstance(result, Story)
    assert result.title == "Olympics"
    assert [block.view for block in result.blocks] == [None, 1, None]


async def test_build_catalog_numbers_and_summarizes(tiny_source):
    editor = LumenEditor(component=Pipeline(source=tiny_source, table='tiny'), title='Tiny table')

    ctx = await build_catalog([editor])

    # Views are numbered so the story can reference them, and the data surfaces.
    assert "View 1" in ctx
    assert "category" in ctx
    assert "value" in ctx


async def test_report_annotate_interleaves_prose_and_view(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='My Report',
        llm=llm,
    )
    await report.execute()
    llm.set_responses([Story(chain_of_thought="c", title="The Big Picture", blocks=[
        StoryBlock(prose="Lead-in prose."),
        StoryBlock(view=1),
        StoryBlock(prose="Closing prose."),
    ])])

    await report._annotate_report()

    nb = _nb_text(report)
    assert "The Big Picture" in nb
    # Prose is interleaved around the chart's reconstructed spec, blog-post style.
    assert nb.index("Lead-in prose.") < nb.index("lm.Pipeline.from_spec") < nb.index("Closing prose.")

    # The flow must survive HTML export too.
    html = report.to_html()
    assert "The Big Picture" in html
    assert "Lead-in prose." in html
    assert "Closing prose." in html


async def test_report_annotate_appends_unreferenced_views(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        Section(ChartAction(source=tiny_source, label='Chart B'), title='Section B'),
        title='R',
        llm=llm,
    )
    await report.execute()
    # The story only references the second chart; the first must not be dropped.
    llm.set_responses([Story(chain_of_thought="c", title="S", blocks=[
        StoryBlock(prose="Only about the second."),
        StoryBlock(view=2),
    ])])

    await report._annotate_report()

    assert _kinds(report).count("view") == 2


async def test_report_annotate_only_selected_sections(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        Section(ChartAction(source=tiny_source, label='Chart B'), title='Section B'),
        title='R',
        llm=llm,
    )
    await report.execute()
    report[1].include_in_export = False
    llm.set_responses([Story(chain_of_thought="c", title="S", blocks=[
        StoryBlock(prose="p"), StoryBlock(view=1),
    ])])

    await report._annotate_report()

    # Only the selected section's chart is catalogued.
    assert _kinds(report).count("view") == 1


async def test_report_annotate_is_idempotent(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()

    llm.set_responses([Story(chain_of_thought="c", title="S1", blocks=[
        StoryBlock(prose="v1"), StoryBlock(view=1),
    ])])
    await report._annotate_report()
    llm.set_responses([Story(chain_of_thought="c", title="S2", blocks=[
        StoryBlock(prose="v2"), StoryBlock(view=1),
    ])])
    await report._annotate_report()

    nb = _nb_text(report)
    assert "v2" in nb and "v1" not in nb
    assert "S2" in nb and "S1" not in nb


async def test_report_annotate_no_llm_is_noop(tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R',
    )
    await report.execute()

    await report._annotate_report()  # must not raise

    # Without a story the chart still exports on its own.
    assert "lm.Pipeline.from_spec" in _nb_text(report)


async def test_report_story_dialog_presets_and_generate(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()

    # Clicking Annotate opens the guidance dialog instead of generating immediately.
    report._open_story_dialog()
    assert report._story_dialog.open is True

    # A tone preset fills the guidance box.
    report._apply_story_preset(text="Executive summary tone.")
    assert report._story_guidance.value == "Executive summary tone."

    # Generate runs the annotation with that guidance and closes the dialog.
    llm.set_responses([Story(chain_of_thought="c", title="S", blocks=[
        StoryBlock(prose="p"), StoryBlock(view=1),
    ])])
    await report._generate_story()

    assert report._story_dialog.open is False
    assert "prose" in _kinds(report)


async def test_report_story_tab_added_and_export_follows_active_tab(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()

    # Before annotating there is only the Report tab.
    assert len(report._tabs) == 1
    assert report._show_story is False

    llm.set_responses([Story(chain_of_thought="c", title="Headline", blocks=[
        StoryBlock(prose="Story prose."), StoryBlock(view=1),
    ])])
    await report._annotate_report()

    # Annotating adds a Story tab and switches to it, and the export follows.
    assert len(report._tabs) == 2
    assert report._tabs.active == 1
    assert report._show_story is True
    assert "Story prose." in _nb_text(report)

    # Selecting the Report tab keeps the story but exports the sections.
    report._tabs.active = 0
    assert report._show_story is False
    assert report._story.blocks  # still remembered
    nb_sections = _nb_text(report)
    assert "Story prose." not in nb_sections
    assert "lm.Pipeline.from_spec" in nb_sections

    # Back to the Story tab: the story returns without regenerating.
    report._tabs.active = 1
    assert report._show_story is True
    assert "Story prose." in _nb_text(report)


async def test_story_prose_is_editable_and_edits_reach_the_exports(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()
    llm.set_responses([Story(chain_of_thought="c", title="T", blocks=[
        StoryBlock(prose="Original prose."), StoryBlock(view=1),
    ])])
    await report._annotate_report()

    # Prose is kept as Markdown and rendered into a rich text editor on screen.
    assert report._story.blocks[0] == ("prose", "Original prose.")
    editors = _prose_editors(report)
    assert len(editors) == 1
    assert "<p>Original prose.</p>" in editors[0].value

    # The editor speaks HTML; what lands in the story, and so in every export,
    # is Markdown.
    editors[0].value = "<p>Edited <strong>by hand</strong>.</p>"
    assert report._story.blocks[0] == ("prose", "Edited **by hand**.")
    nb = _nb_text(report)
    assert "Edited **by hand**." in nb
    assert "Original prose." not in nb
    assert "<strong>" not in nb
    assert "by hand" in report.to_html()

    # The edit survives re-rendering the story (e.g. switching tabs).
    report._render_story()
    assert "<strong>by hand</strong>" in _prose_editors(report)[0].value


async def test_rendering_the_story_does_not_rewrite_the_blocks(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()
    llm.set_responses([Story(chain_of_thought="c", title="T", blocks=[
        StoryBlock(prose="## Findings\n\n- one\n- two"), StoryBlock(view=1),
    ])])
    await report._annotate_report()

    # Building the editors must not round-trip the prose back over itself, or
    # the Markdown the LLM wrote would drift every time the tab is rebuilt.
    before = list(report._story.blocks)
    report._render_story()
    assert report._story.blocks == before


@pytest.mark.parametrize("markdown", [
    "**Pop** leads with *1,252* units.",
    "## Heading",
    "### Deeper heading",
    "- one\n- two",
    "1. first\n2. second",
    "> a quoted insight",
    "Text with a [link](https://example.com) in it.",
    "Text with `inline code` in it.",
    "| a | b |\n| --- | --- |\n| 1 | 2 |",
    "Line one\nLine two",
    "## Heading\n\nAnd a paragraph below it.",
])
def test_prose_round_trips_through_html(markdown):
    # The editor shows HTML but the story stores Markdown, so what the user
    # never touched must come back out exactly as the LLM wrote it.
    assert _prose_to_markdown(_prose_to_html(markdown)) == markdown


def test_prose_round_trip_is_a_fixed_point():
    text = "## Heading\n\n- one\n- two\n\nWith **bold** and a [link](https://example.com)."
    once = _prose_to_markdown(_prose_to_html(text))
    assert _prose_to_markdown(_prose_to_html(once)) == once


def test_prose_normalizes_non_breaking_spaces():
    # contenteditable inserts &nbsp; for runs of spaces; they must not leak
    # into the stored Markdown or the exports.
    assert _prose_to_markdown("<p>a&nbsp;b</p>") == "a b"


async def test_story_prose_rewritten_by_ai(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()
    llm.set_responses([Story(chain_of_thought="c", title="T", blocks=[
        StoryBlock(prose="Original prose."), StoryBlock(view=1),
    ])])
    await report._annotate_report()

    # Asking the AI to rewrite one paragraph only touches that block.
    report._open_edit_dialog(0)
    assert report._edit_dialog.open is True
    report._edit_instruction.value = "make it punchier"
    llm.set_responses([ProseEdit(chain_of_thought="c", prose="Punchier prose.")])
    await report._rewrite_prose()

    assert report._edit_dialog.open is False
    assert report._story.blocks[0] == ("prose", "Punchier prose.")
    assert _kinds(report) == ["prose", "view"]  # the chart block is untouched
    assert "Punchier prose." in _nb_text(report)


async def test_regenerating_keeps_the_previous_version_and_its_edits(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()

    llm.set_responses([Story(chain_of_thought="c", title="V1", blocks=[
        StoryBlock(prose="v1 prose"), StoryBlock(view=1),
    ])])
    await report._annotate_report()
    assert len(report._story.versions) == 1

    # The user edits the first version by hand.
    _prose_editors(report)[0].value = "my edit in v1"

    # Regenerating adds a second version rather than overwriting the first.
    llm.set_responses([Story(chain_of_thought="c", title="V2", blocks=[
        StoryBlock(prose="v2 prose"), StoryBlock(view=1),
    ])])
    await report._annotate_report()
    assert len(report._story.versions) == 2
    assert report._story.version == 1
    assert "v2 prose" in _nb_text(report)

    # Going back to the first version restores it with the edit intact.
    report._select_version(0)
    assert report._story.title == "V1"
    nb = _nb_text(report)
    assert "my edit in v1" in nb
    assert "v2 prose" not in nb


async def test_report_reset_discards_story(llm, tiny_source):
    report = StoryReport(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()
    llm.set_responses([Story(chain_of_thought="c", title="H", blocks=[
        StoryBlock(prose="p"), StoryBlock(view=1),
    ])])
    await report._annotate_report()
    assert report._story.blocks

    report.reset()

    assert report._story.blocks == []
    assert report._show_story is False
    # The Story tab is removed along with the story.
    assert len(report._tabs) == 1
    assert report._tabs.active == 0


async def test_report_outline_defaults_empty():
    report = StoryReport(Section(A(order=[]), title='Section A'), title='R')
    assert report._story.outline == []


async def test_report_default_outline_from_sections():
    report = StoryReport(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='R',
    )
    await report.execute()
    assert report._build_default_outline() == [
        {"section": "Section A"}, {"section": "Section B"}
    ]


async def test_report_outline_reorders_export():
    report = StoryReport(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='R',
    )
    await report.execute()

    report._story.outline = [{"section": "Section B"}, {"section": "Section A"}]

    nb = _nb_text(report)
    assert nb.index("B done") < nb.index("A done")


async def test_report_outline_inserts_headings():
    report = StoryReport(Section(A(order=[]), title='Section A'), title='R')
    await report.execute()

    report._story.outline = [{"heading": "Introduction", "level": 1}, {"section": "Section A"}]

    nb = _nb_text(report)
    assert "Introduction" in nb
    assert nb.index("Introduction") < nb.index("A done")


async def test_report_outline_reorders_html_export():
    report = StoryReport(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='R',
    )
    await report.execute()

    report._story.outline = [
        {"heading": "Key Findings", "level": 1},
        {"section": "Section B"},
        {"section": "Section A"},
    ]

    html = report.to_html()
    assert "Key Findings" in html
    assert html.index("B done") < html.index("A done")


async def test_report_arrange_editor_seeds_and_binds_outline():
    report = StoryReport(
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
    assert report._story.outline == [{"section": "Section B"}, {"section": "Section A"}]
    nb = _nb_text(report)
    assert nb.index("B done") < nb.index("A done")
