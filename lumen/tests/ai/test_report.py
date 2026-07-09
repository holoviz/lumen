import asyncio
import json

import panel as pn
import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.layout import Column, Row
from panel.pane import Markdown
from typing_extensions import NotRequired

from lumen.ai.actor import ContextModel
from lumen.ai.report import (
    Action, Report, Section, TaskGroup, Typography,
)

try:
    from panel_material_ui import ChatMessage, Checkbox
except ImportError:
    ChatMessage = None
    Checkbox = None


class HelloAction(Action):

    async def _execute(self, context, **kwargs):
        return [Markdown("**Hello**")], {'text': 'Hello'}


class A(Action):

    order = param.List()

    async def _execute(self, context, **kwargs):
        self.order.append("A")
        return [Markdown("A done")], {'text': 'A'}


class B(Action):

    order = param.List()

    async def _execute(self, context, **kwargs):
        self.order.append("B")
        return [Markdown("B done")], {'text': 'B'}


class AError(A):
    async def _execute(self, context, **kwargs):
        self.order.append("A")
        raise Exception()


class AInputs(ContextModel):

    input: NotRequired[str]

class AOutputs(ContextModel):

    a: str

class ASchema(Action):

    input_schema = AInputs
    output_schema = AOutputs

    async def _execute(self, context, **kwargs):
        return [], {'a': context.get("input", "A")}

class BInputs(ContextModel):

    a: str

class BOutputs(ContextModel):

    b: str

class BSchema(Action):

    input_schema = BInputs
    output_schema = BOutputs

    async def _execute(self, context, **kwargs):
        a = context["a"]
        return [Markdown(a)], {"b": a*2}


async def test_action_execute_renders_outputs():
    action = HelloAction(title="A1")

    outs, ctx = await action.execute()

    assert action.views == outs
    assert len(outs) == 2
    out1, out2 = outs
    assert isinstance(out1, Typography)
    assert out1.object == "### A1"
    assert isinstance(out1, Markdown)
    assert out2.object == "**Hello**"

    assert len(action._view) == 1
    assert action._view[0] is out2

async def test_action_execute_updates_status():
    action = HelloAction(title="A1")

    running, status = [], []

    action.param.watch(lambda e: status.append(e.new), "status")
    action.param.watch(lambda e: running.append(e.new), "running")

    outs, ctx = await action.execute()

    assert status == ["running", "success"]
    assert running == [True, False]

async def test_action_reset():
    action = HelloAction(title="A1")
    outs = await action.execute()

    action.reset()

    assert len(action._view) == 0
    assert len(action.views) == 0
    assert action.status == "idle"

async def test_taskgroup_sequences_actions():
    order = []

    tg = TaskGroup(A(order=order), B(order=order), title="Seq")
    outs, ctx = await tg.execute()

    assert order == ["A", "B"]

    assert len(tg.views) == 3
    out1, out2, out3 = tg.views
    assert isinstance(out1, Typography)
    assert out1.object == "### Seq"
    assert isinstance(out2, Markdown) and out2.object == "A done"
    assert isinstance(out3, Markdown) and out3.object == "B done"
    assert outs == tg.views

    assert len(tg._view) == 2
    v1, v2 = tg._view
    assert isinstance(v1, Column)
    assert len(v1.objects) == 1
    assert v1.objects[0].object == "A done"
    assert isinstance(v2, Column)
    assert len(v2.objects) == 1
    assert v2.objects[0].object == "B done"

async def test_taskgroup_status():
    tg = TaskGroup(A(), B(), title="Seq")
    outs, ctx = await tg.execute()

    assert tg.status == "success"
    assert tg[0].status == "success"
    assert tg[1].status == "success"

async def test_taskgroup_append_action():
    order = []

    tg = TaskGroup(A(order=order), title="Seq")

    await tg.execute()

    assert order == ["A"]

    tg.append(B(order=order))

    outs, ctx = await tg.execute()

    assert order == ["A", "B"]

    assert len(tg.views) == 3
    out1, out2, out3 = tg.views
    assert isinstance(out1, Typography)
    assert out1.object == "### Seq"
    assert isinstance(out2, Markdown) and out2.object == "A done"
    assert isinstance(out3, Markdown) and out3.object == "B done"
    assert outs == tg.views

    assert len(tg._view) == 3
    v1, v2, v3 = tg._view
    assert isinstance(v1, Typography)
    assert v1.object == "### Seq"
    assert isinstance(v2, Column)
    assert len(v2.objects) == 1
    assert v2.objects[0].object == "A done"
    assert isinstance(v3, Column)
    assert len(v3.objects) == 1
    assert v3.objects[0].object == "B done"

async def test_taskgroup_continue_on_error():
    order = []

    tg = TaskGroup(AError(order=order), B(order=order), title="Error", abort_on_error=False)

    outs, ctx = await tg.execute()

    assert order == ["A", "B"]

    assert tg.status == "error"
    assert tg[0].status == "error"
    assert tg[1].status == "success"
    assert len(outs) == 2
    out1, out2 = outs
    assert isinstance(out1, Typography)
    assert out1.object == "### Error"
    assert isinstance(out2, Markdown) and out2.object == "B done"

    assert len(tg._view) == 2
    v1, v2 = tg._view
    assert isinstance(v1, Column)
    assert len(v1) == 0
    assert isinstance(v2, Column)
    assert len(v2) == 1
    assert v2[0].object == "B done"

async def test_taskgroup_abort_on_error():
    order = []

    tg = TaskGroup(AError(order=order), B(order=order), title="Error", abort_on_error=True)

    outs, ctx = await tg.execute()

    assert order == ["A"]

    assert tg.status == "error"
    assert tg[0].status == "error"
    assert tg[1].status == "idle"
    assert len(outs) == 1
    assert isinstance(outs[0], Typography)
    assert outs[0].object == "### Error"

    assert len(tg._view) == 2
    v1, v2 = tg._view
    assert isinstance(v1, Column)
    assert len(v1) == 0
    assert isinstance(v2, Column)
    assert len(v2) == 0

async def test_taskgroup_context_invalidate():

    tg = TaskGroup(ASchema(), BSchema())

    await tg.execute()

    tg[0].out_context = {"a": "B"}

    await asyncio.sleep(0.5)

    assert tg.status == "success"
    assert tg[0].status == "success"
    assert tg[1].status == "success"

    assert tg.out_context["a"] == "B"
    assert tg.out_context["b"] == "BB"
    assert len(tg.views) == 1
    assert isinstance(tg.views[0], Markdown)
    assert tg.views[0].object == "B"

async def test_taskgroup_invalidate():

    tg = TaskGroup(ASchema(), BSchema())

    outs, ctx = await tg.execute()

    assert ctx["a"] == "A"
    assert ctx["b"] == "AA"
    assert len(outs) == 1
    assert isinstance(outs[0], Markdown)
    assert outs[0].object == "A"

    is_invalidated, keys = tg.invalidate(["input"])

    assert tg.status == "idle"
    assert tg[0].status == "idle"
    assert tg[1].status == "idle"

    assert is_invalidated
    assert keys == {"input", "a", "b"}
    assert len(tg.views) == 0

    outs, ctx = await tg.execute({"input": "B"})

    assert ctx["a"] == "B"
    assert ctx["b"] == "BB"
    assert len(outs) == 1
    assert isinstance(outs[0], Markdown)
    assert outs[0].object == "B"

async def test_taskgroup_invalidate_partial():

    class BInputs(ContextModel):

        a: str
        m: int

    class BOutputs(ContextModel):

        b: str

    class B(Action):

        input_schema = BInputs
        output_schema = BOutputs

        async def _execute(self, context, **kwargs):
            a = context["a"]
            return [Markdown(a)], {"b": a*context["m"]}

    tg = TaskGroup(ASchema(), B())

    outs, ctx = await tg.execute({"m": 2})

    assert ctx["a"] == "A"
    assert ctx["b"] == "AA"
    assert len(outs) == 1
    assert isinstance(outs[0], Markdown)
    assert outs[0].object == "A"

    tg.invalidate(["m"])

    assert tg.status == "idle"
    assert tg[0].status == "success"
    assert tg[1].status == "idle"

    outs, ctx = await tg.execute({"m": 3})

    assert ctx["a"] == "A"
    assert ctx["b"] == "AAA"
    assert len(outs) == 1
    assert isinstance(outs[0], Markdown)
    assert outs[0].object == "A"

async def test_report_to_notebook():
    report = Report(
        Section(
            HelloAction()
        ),
        title='Hello Report'
    )

    with pytest.raises(RuntimeError):
        report.to_notebook()

    await report.execute()

    assert len(report.views) == 2

    nb_string = report.to_notebook()

    nb = json.loads(nb_string)
    assert len(nb["cells"]) == 3
    cell1, cell2, cell3 = nb["cells"]

    assert cell1['cell_type'] == 'code'
    assert cell1['source'] == [
        'import yaml\n',
        '\n',
        'import lumen as lm\n',
        'import panel as pn\n',
        '\n',
        "pn.extension('tabulator')"
    ]

    assert cell2['cell_type'] == 'markdown'
    assert cell2['source'] == ["# Hello Report"]

    assert cell3['cell_type'] == 'markdown'
    assert cell3['source'] == ["**Hello**"]


async def test_report_to_html():
    report = Report(
        Section(
            HelloAction()
        ),
        title='Hello Report'
    )

    with pytest.raises(RuntimeError):
        report.to_html()

    await report.execute()

    assert len(report.views) == 2

    html_string = report.to_html()

    assert isinstance(html_string, str)
    assert "<html" in html_string.lower()
    assert "Hello Report" in html_string
    assert "Hello" in html_string


async def test_section_include_in_export_default():
    section = Section(HelloAction())
    assert section.include_in_export is True


async def test_report_to_notebook_excludes_deselected_section():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='Multi Report',
    )
    await report.execute()

    # By default every section is included in the export.
    all_sources = "".join(
        "".join(cell["source"]) for cell in json.loads(report.to_notebook())["cells"]
    )
    assert "A done" in all_sources
    assert "B done" in all_sources

    # Deselecting a section drops it (and its header) from the notebook.
    report[1].include_in_export = False
    sources = "".join(
        "".join(cell["source"]) for cell in json.loads(report.to_notebook())["cells"]
    )
    assert "A done" in sources
    assert "B done" not in sources
    assert "Section B" not in sources
    assert "Multi Report" in sources


async def test_report_to_html_excludes_deselected_section():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='Multi Report',
    )
    await report.execute()

    report[1].include_in_export = False
    html = report.to_html()

    assert "A done" in html
    assert "B done" not in html
    # Exporting a subset must not disturb the live report view.
    assert len(report._view) == 2


async def test_report_section_header_has_bound_export_checkbox():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='Multi Report',
    )
    await report.execute()

    headers = report._view._headers
    assert len(headers) == len(report) == 2
    for section, header in zip(report, headers):
        assert isinstance(header, Row)
        checkboxes = [obj for obj in header.objects if isinstance(obj, Checkbox)]
        assert len(checkboxes) == 1
        checkbox = checkboxes[0]
        # Checked by default and bound to include_in_export in both directions.
        assert checkbox.value is True
        section.include_in_export = False
        assert checkbox.value is False
        checkbox.value = True
        assert section.include_in_export is True


async def test_report_export_empty_selection():
    report = Report(
        Section(A(order=[]), title='Section A'),
        Section(B(order=[]), title='Section B'),
        title='Multi Report',
    )
    await report.execute()

    report[0].include_in_export = False
    report[1].include_in_export = False

    cells = json.loads(report.to_notebook())["cells"]
    sources = "".join("".join(cell["source"]) for cell in cells)
    assert "A done" not in sources
    assert "B done" not in sources
    # Only the import preamble survives when nothing is selected.
    assert len(cells) == 1

    # HTML export of an empty selection should not raise.
    html = report.to_html()
    assert "A done" not in html
    assert "B done" not in html


def test_report_from_views_assembles_exportable_report():
    views = [Markdown("# My Title"), Markdown("Some body text")]

    report = Report.from_views(views, title="Prebuilt Report")

    # No tasks were run, but the report is in an exportable state.
    assert len(report) == 0
    assert report.status == "success"
    assert report.views == views

    nb = json.loads(report.to_notebook())
    sources = ["".join(cell["source"]) for cell in nb["cells"]]
    assert any("# My Title" in source for source in sources)
    assert any("Some body text" in source for source in sources)

    html_string = report.to_html()
    assert "<html" in html_string.lower()
    assert "Some body text" in html_string


def test_report_from_views_without_title():
    report = Report.from_views([Markdown("Body")])

    assert report.status == "success"
    assert "<html" in report.to_html().lower()


class BlockingAction(Action):
    """Action that waits on a signal before returning, so tests can deterministically cancel mid-execution."""

    def __init__(self, started, release, **params):
        super().__init__(**params)
        self._started = started
        self._release = release

    async def _execute(self, context, **kwargs):
        self._started.set()
        await self._release.wait()
        return [Markdown("should not render")], {"text": "never"}


async def test_task_execute_marks_cancelled():
    started = asyncio.Event()
    release = asyncio.Event()
    action = BlockingAction(started, release, title="Blocks")

    task = asyncio.create_task(action.execute())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert action.status == "cancelled"
    assert action.running is False


async def test_taskgroup_cancel_propagates():
    started = asyncio.Event()
    release = asyncio.Event()
    tg = TaskGroup(BlockingAction(started, release, title="Blocks"), B(), title="Group")

    task = asyncio.create_task(tg.execute())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tg.status == "cancelled"
    assert tg[0].status == "cancelled"
    assert tg[1].status == "idle"


async def test_report_cancel_mid_execution():
    started = asyncio.Event()
    release = asyncio.Event()
    report = Report(
        Section(BlockingAction(started, release, title="Blocks"), title="S1"),
        Section(B(), title="S2"),
        title="Cancellable",
    )

    await report._execute_event()
    await started.wait()
    assert report._active_task is not None and not report._active_task.done()

    report._handle_cancel()
    # Wait for the task to finish reacting to the cancel
    while report._active_task is not None:
        await asyncio.sleep(0.01)

    assert report.status == "cancelled"
    assert report[0].status == "cancelled"
    assert report[1].status == "idle"


async def test_report_handle_cancel_idempotent():
    report = Report(Section(HelloAction(), title="S1"), title="NoOp")
    # No task running: _handle_cancel must not raise
    report._handle_cancel()
    assert report._active_task is None


async def test_report_completed_sections_preserved_on_cancel():
    started = asyncio.Event()
    release = asyncio.Event()
    report = Report(
        Section(HelloAction(), title="Done"),
        Section(BlockingAction(started, release, title="Blocks"), title="S2"),
        title="Partial",
    )

    await report._execute_event()
    await started.wait()
    report._handle_cancel()
    while report._active_task is not None:
        await asyncio.sleep(0.01)

    assert report[0].status == "success"
    assert report[1].status == "cancelled"
    assert report.status == "cancelled"


# ---------------------------------------------------------------------------
# to_notebook: header buffering, str/ChatMessage handling, Viewable skipping
# ---------------------------------------------------------------------------

class StrAction(Action):
    """Action that returns a plain string output."""
    async def _execute(self, context, **kwargs):
        return ["Plain text summary"], {}


class ChatMessageAction(Action):
    """Action that returns a ChatMessage with string content."""
    async def _execute(self, context, **kwargs):
        msg = ChatMessage(object="Chat response text", user="Agent")
        return [msg], {}


class ChatMessageMarkdownAction(Action):
    """Action that returns a ChatMessage wrapping a Markdown pane."""
    async def _execute(self, context, **kwargs):
        md = Markdown("Markdown inside ChatMessage")
        msg = ChatMessage(object=md, user="Agent")
        return [msg], {}


class ViewableAction(Action):
    """Action that returns a non-LumenEditor Viewable (should be skipped)."""
    async def _execute(self, context, **kwargs):
        from panel.widgets import TextInput
        return [TextInput(value="skip me")], {}


class EmptyAction(Action):
    """Action that returns no outputs (header-only after rendering)."""
    async def _execute(self, context, **kwargs):
        return [], {}


async def test_to_notebook_string_output():
    """Plain string views are exported as markdown cells."""
    report = Report(
        Section(StrAction(title="Summary"), title="S1"),
        title="String Report",
    )
    await report.execute()
    nb = json.loads(report.to_notebook())

    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    md_sources = ["".join(c["source"]) for c in md_cells]
    assert any("Plain text summary" in s for s in md_sources)


@pytest.mark.skipif(ChatMessage is None, reason="panel_material_ui.ChatMessage not available")
async def test_to_notebook_chatmessage_string():
    """ChatMessage with string content is exported as markdown."""
    report = Report(
        Section(ChatMessageAction(title="Chat"), title="S1"),
        title="ChatMsg Report",
    )
    await report.execute()
    nb = json.loads(report.to_notebook())

    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    md_sources = ["".join(c["source"]) for c in md_cells]
    assert any("Chat response text" in s for s in md_sources)


@pytest.mark.skipif(ChatMessage is None, reason="panel_material_ui.ChatMessage not available")
async def test_to_notebook_chatmessage_markdown():
    """ChatMessage wrapping a Markdown pane is exported as markdown."""
    report = Report(
        Section(ChatMessageMarkdownAction(title="Chat"), title="S1"),
        title="ChatMdMsg Report",
    )
    await report.execute()
    nb = json.loads(report.to_notebook())

    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    md_sources = ["".join(c["source"]) for c in md_cells]
    assert any("Markdown inside ChatMessage" in s for s in md_sources)


async def test_to_notebook_viewable_skipped():
    """Non-LumenEditor Viewables produce no cells (not even error stubs)."""
    report = Report(
        Section(ViewableAction(title="Widget"), title="S1"),
        title="Viewable Report",
    )
    await report.execute()
    nb = json.loads(report.to_notebook())

    # Only preamble + report title; no code cells beyond preamble
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 1  # just preamble
    # No "Cannot export" comments
    for c in nb["cells"]:
        source = "".join(c["source"])
        assert "Cannot export" not in source


async def test_to_notebook_orphaned_header_dropped():
    """A Typography header followed by no content is not emitted."""
    report = Report(
        Section(EmptyAction(title="Orphan"), title="S1"),
        title="Orphan Report",
    )
    await report.execute()
    nb = json.loads(report.to_notebook())

    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    md_sources = ["".join(c["source"]) for c in md_cells]
    # The task-level header "Orphan" should NOT appear
    assert not any("Orphan" in s for s in md_sources)


async def test_to_notebook_header_emitted_before_content():
    """A Typography header IS emitted when followed by real content."""
    report = Report(
        Section(HelloAction(title="KeepMe"), title="S1"),
        title="Header Report",
    )
    await report.execute()
    nb = json.loads(report.to_notebook())

    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    md_sources = ["".join(c["source"]) for c in md_cells]
    # The section title appears (report title = "# Header Report")
    assert any("Header Report" in s for s in md_sources)
    # The Markdown content is present
    assert any("**Hello**" in s for s in md_sources)


async def test_to_notebook_consecutive_orphaned_headers():
    """Multiple consecutive orphaned headers produce zero cells."""
    report = Report(
        Section(
            EmptyAction(title="First"),
            EmptyAction(title="Second"),
            title="S1",
        ),
        title="Multi Orphan",
    )
    await report.execute()
    nb = json.loads(report.to_notebook())

    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    md_sources = ["".join(c["source"]) for c in md_cells]
    assert not any("First" in s for s in md_sources)
    assert not any("Second" in s for s in md_sources)
