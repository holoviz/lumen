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


class ChartAction(Action):
    """Emits a pipeline-backed LumenEditor so the story catalog has real views."""

    source = param.Parameter()

    label = param.String(default="Chart")

    async def _execute(self, context, **kwargs):
        from lumen.ai.editors import LumenEditor
        from lumen.pipeline import Pipeline

        pipeline = Pipeline(source=self.source, table='tiny')
        return [LumenEditor(component=pipeline, title=self.label)], {'text': self.label}


def _nb_text(report):
    return "".join(
        "".join(cell["source"]) for cell in json.loads(report.to_notebook())["cells"]
    )


def _kinds(report):
    return [kind for kind, _ in report._story_blocks]


async def test_story_agent_write_story(llm):
    from lumen.ai.agents.story import Story, StoryAgent, StoryBlock

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
    from lumen.ai.agents.story import build_catalog
    from lumen.ai.editors import LumenEditor
    from lumen.pipeline import Pipeline

    editor = LumenEditor(component=Pipeline(source=tiny_source, table='tiny'), title='Tiny table')

    ctx = await build_catalog([editor])

    # Views are numbered so the story can reference them, and the data surfaces.
    assert "View 1" in ctx
    assert "category" in ctx
    assert "value" in ctx


async def test_report_annotate_interleaves_prose_and_view(llm, tiny_source):
    from lumen.ai.agents.story import Story, StoryBlock

    report = Report(
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
    from lumen.ai.agents.story import Story, StoryBlock

    report = Report(
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
    from lumen.ai.agents.story import Story, StoryBlock

    report = Report(
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
    from lumen.ai.agents.story import Story, StoryBlock

    report = Report(
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
    report = Report(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R',
    )
    await report.execute()

    await report._annotate_report()  # must not raise

    # Without a story the chart still exports on its own.
    assert "lm.Pipeline.from_spec" in _nb_text(report)


async def test_report_story_dialog_presets_and_generate(llm, tiny_source):
    from lumen.ai.agents.story import Story, StoryBlock

    report = Report(
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


async def test_report_view_toggle_switches_story_and_sections(llm, tiny_source):
    from lumen.ai.agents.story import Story, StoryBlock

    report = Report(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()
    llm.set_responses([Story(chain_of_thought="c", title="Headline", blocks=[
        StoryBlock(prose="Story prose."), StoryBlock(view=1),
    ])])
    await report._annotate_report()

    # After annotate the story is shown and exported, and the toggle is visible.
    assert report._show_story is True
    assert report._view_toggle.visible is True
    assert "Story prose." in _nb_text(report)

    # Toggle to the section view: the story is kept, export follows the sections.
    report._toggle_view()
    assert report._show_story is False
    assert report._story_blocks  # still remembered
    nb_sections = _nb_text(report)
    assert "Story prose." not in nb_sections
    assert "lm.Pipeline.from_spec" in nb_sections

    # Toggle again: the story returns without regenerating.
    report._toggle_view()
    assert report._show_story is True
    assert "Story prose." in _nb_text(report)


async def test_report_reset_discards_story(llm, tiny_source):
    from lumen.ai.agents.story import Story, StoryBlock

    report = Report(
        Section(ChartAction(source=tiny_source, label='Chart A'), title='Section A'),
        title='R', llm=llm,
    )
    await report.execute()
    llm.set_responses([Story(chain_of_thought="c", title="H", blocks=[
        StoryBlock(prose="p"), StoryBlock(view=1),
    ])])
    await report._annotate_report()
    assert report._story_blocks

    report.reset()

    assert report._story_blocks == []
    assert report._show_story is False
    assert report._view_toggle.visible is False


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
