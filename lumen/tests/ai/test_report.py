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
from lumen.ai.models import SqlQuery
from lumen.ai.report import (
    Action, ActorTask, Report, Section, TaskGroup, Typography,
)
from lumen.ai.views import SQLOutput


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

    await asyncio.sleep(0.1)

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

async def test_sql_query_action(tiny_source):
    action = SqlQuery(source=tiny_source, sql_expr="SELECT category FROM tiny", table="foo", generate_caption=False)

    outs, ctx = await action.execute()

    assert len(outs) == 1
    out = outs[0]
    assert isinstance(out, SQLOutput)
    assert out.spec == action.sql_expr
    assert out.component.data.columns == ['category']
    assert list(out.component.data.category) == ['A', 'B', 'A']

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
