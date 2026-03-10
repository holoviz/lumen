import asyncio
import json

import panel as pn
import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.layout import Column
from panel.pane import Markdown
from typing_extensions import NotRequired

from lumen.ai.actor import ContextModel
from lumen.ai.report import (
    Action, Report, Section, TaskGroup, Typography,
)


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


async def test_section_incremental_execution():
    order = []
    a = A(order=order, title="Action A")
    b = B(order=order, title="Action B")
    section = Section(a, b, title="Section 1")
    report = Report(section, title="Selector Report")

    # Initial run with both A and B
    await report.execute()
    assert order == ["A", "B"]
    assert a.status == "success"
    assert b.status == "success"
    assert len(a.views) == 2  # Title + Markdown
    assert len(b.views) == 2

    # Deselect B and run again
    section._export_selector.value = ["Action A"]
    await report.execute()
    assert order == ["A", "B"]  # A is cached, B is skipped (and reset)
    assert b.status == "idle"
    assert len(b.views) == 0
    assert len(section._view) == 2 # Placeholder + Action A (Action B is filtered out)

    # Select B again and run again
    section._export_selector.value = ["Action A", "Action B"]
    await report.execute()
    assert order == ["A", "B", "B"] # A is cached, B is re-run
    assert b.status == "success"
    assert len(b.views) == 2


async def test_section_reset_filters():
    section = Section(A(title="Action A"), B(title="Action B"), title="Section 1")
    report = Report(section, title="Selector Report")

    section._export_selector.value = ["Action A"]
    assert section._export_selector.value == ["Action A"]

    report.reset_filters()
    assert section._export_selector.value == ["Action A", "Action B"]


async def test_section_selector_filters_ui():
    a = A(title="Action A")
    b = B(title="Action B")
    section = Section(a, b, title="Section 1")

    # Initial state: both selected and rendered.
    assert section._export_selector.value == ["Action A", "Action B"]
    assert len(section._view) == 3

    # Remove B from selector
    section._export_selector.value = ["Action A"]
    assert section._export_selector.value == ["Action A"]
    assert len(section._view) == 2

    # Add B back
    section._export_selector.value = ["Action A", "Action B"]
    assert section._export_selector.value == ["Action A", "Action B"]
    assert len(section._view) == 3

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


async def test_export_selector_filters_play_execution():
    order = []

    section = Section(
        A(order=order, title="Action A"),
        B(order=order, title="Action B"),
        title="Section 1",
    )
    report = Report(section, title="Selector Report")

    section._export_selector.value = ["Action B"]

    await report.execute()

    assert order == ["B"]
    assert len(section.views) == 3
    assert isinstance(section.views[0], Typography)
    assert section.views[0].object == "## Section 1"
    assert isinstance(section.views[1], Typography)
    assert section.views[1].object == "### Action B"
    assert isinstance(section.views[2], Markdown)
    assert section.views[2].object == "B done"


async def test_sync_context_skipped_while_root_running():
    """_sync_context must not trigger re-execution while the root Report is running.

    Regression test: deferred async param-watcher callbacks for out_context
    could fire after a Section finished but while the Report was still running,
    causing spurious invalidation cascades and duplicate task execution.
    """
    order = []

    section = Section(
        ASchema(title="Action A"),
        BSchema(title="Action B"),
        title="Section 1",
    )
    report = Report(section, title="Test Report")

    await report.execute()

    assert section[0].out_context == {"a": "A"}
    assert section[1].out_context == {"b": "AA"}

    # Simulate a deferred _sync_context callback arriving while the root
    # is running.  Before the fix, this would trigger invalidation and
    # re-execution.
    report.running = True
    try:
        section[0].out_context = {"a": "CHANGED"}
        await asyncio.sleep(0.3)

        # B should NOT have been invalidated or re-run because the root
        # was running — the _sync_context callback should have bailed out.
        assert section[1].out_context == {"b": "AA"}
        assert section[1].status == "success"
    finally:
        report.running = False


async def test_selector_subset_no_reexecution_loop():
    """Deselecting a task and running should not cause tasks to execute more than once.

    Regression test: when a task was deselected via _export_selector, the
    reset of excluded tasks would trigger _sync_context callbacks that
    re-ran the entire report in an infinite loop.
    """
    order = []

    section = Section(
        A(order=order, title="Action A"),
        B(order=order, title="Action B"),
        title="Section 1",
    )
    section_2 = Section(
        A(order=order, title="Action C"),
        title="Section 2",
    )
    report = Report(section, section_2, title="Test Report")

    # Deselect Action B
    section._export_selector.value = ["Action A"]

    await report.execute()

    # Allow any deferred callbacks to settle
    await asyncio.sleep(0.5)

    # Each selected action should have run exactly once
    assert order.count("A") == 2  # Action A in Section 1 + Action C (class A) in Section 2
    assert order.count("B") == 0  # Action B was deselected


async def test_populate_view_no_selector_retrigger():
    """Section._populate_view should not re-trigger _sync_placeholder_to_selector
    when the selector value hasn't changed."""
    section = Section(
        A(title="Action A"),
        B(title="Action B"),
        title="Section 1",
    )

    sync_calls = []
    original = section._sync_placeholder_to_selector

    def tracking_sync(*args, **kwargs):
        sync_calls.append(1)
        return original(*args, **kwargs)

    section._sync_placeholder_to_selector = tracking_sync

    # Calling _populate_view when value hasn't changed should NOT trigger the watcher
    sync_calls.clear()
    section._populate_view()
    assert len(sync_calls) == 0


async def test_export_selector_filters_notebook_export():
    order = []

    section = Section(
        A(order=order, title="Action A"),
        B(order=order, title="Action B"),
        title="Section 1",
    )
    report = Report(section, title="Selector Report")

    await report.execute()
    assert order == ["A", "B"]

    section._export_selector.value = ["Action A"]

    nb_string = report.to_notebook()
    nb = json.loads(nb_string)

    markdown_cells = [
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown"
    ]

    assert markdown_cells == [
        "# Selector Report",
        "## Section 1",
        "### Action A",
        "A done",
    ]
