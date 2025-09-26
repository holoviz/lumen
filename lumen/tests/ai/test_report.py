import asyncio

import panel as pn
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.layout import Column
from panel.pane import Markdown

from lumen.ai.report import (
    Action, Report, Section, SQLQuery, TaskGroup, Typography,
)
from lumen.ai.views import SQLOutput

try:
    from lumen.sources.duckdb import DuckDBSource
except ImportError:
    pass

duckdb_available = pytest.mark.skipif(DuckDBSource is None, reason="Duckdb is not installed")


class HelloAction(Action):

    async def _execute(self, **kwargs):
        return [Markdown("**Hello**")]


async def test_action_execute_renders_outputs():
    action = HelloAction(title="A1")
    outs = await action.execute()

    assert action.outputs == outs
    assert len(outs) == 1
    out = outs[0]
    assert isinstance(out, Markdown)

    assert len(action._view) == 1
    assert action._view[0] is out

async def test_action_reset():
    action = HelloAction(title="A1")
    outs = await action.execute()

    action.reset()

    assert len(action._view) == 0
    assert len(action.outputs) == 0

async def test_taskgroup_sequences_actions():
    order = []

    class A(Action):
        async def _execute(self, **kwargs):
            order.append("A")
            return [Markdown("A done")]

    class B(Action):
        async def _execute(self, **kwargs):
            order.append("B")
            return [Markdown("B done")]

    tg = TaskGroup(A(), B(), title="Seq")
    outs = await tg.execute()

    assert order == ["A", "B"]

    assert len(tg.outputs) == 3
    out1, out2, out3 = tg.outputs
    assert isinstance(out1, Typography)
    assert out1.object == "### Seq"
    assert isinstance(out2, Markdown) and out2.object == "A done"
    assert isinstance(out3, Markdown) and out3.object == "B done"
    assert outs == tg.outputs

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

async def test_taskgroup_append_action():
    order = []

    class A(Action):
        async def _execute(self, **kwargs):
            order.append("A")
            return [Markdown("A done")]

    class B(Action):
        async def _execute(self, **kwargs):
            order.append("B")
            return [Markdown("B done")]

    tg = TaskGroup(A(), title="Seq")

    await tg.execute()

    assert order == ["A"]

    tg.append(B())

    outs = await tg.execute()

    assert order == ["A", "B"]

    assert len(tg.outputs) == 3
    out1, out2, out3 = tg.outputs
    assert isinstance(out1, Typography)
    assert out1.object == "### Seq"
    assert isinstance(out2, Markdown) and out2.object == "A done"
    assert isinstance(out3, Markdown) and out3.object == "B done"
    assert outs == tg.outputs

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

    class A(Action):
        async def _execute(self, **kwargs):
            order.append("A")
            raise Exception()

    class B(Action):
        async def _execute(self, **kwargs):
            order.append("B")
            return [Markdown("B done")]

    tg = TaskGroup(A(), B(), title="Error", abort_on_error=False)

    outs = await tg.execute()

    assert order == ["A", "B"]

    assert len(outs) == 2
    out1, out2 = outs
    assert isinstance(out1, Typography)
    assert out1.object == "### Error"
    assert isinstance(out2, Markdown) and out2.object == "B done"

    assert len(tg._view) == 3
    v1, v2, v3 = tg._view
    assert isinstance(v1, Typography)
    assert v1.object == "### Error"
    assert isinstance(v2, Column)
    assert len(v2.objects) == 0
    assert isinstance(v3, Column)
    assert len(v3.objects) == 1
    assert v3.objects[0].object == "B done"

async def test_taskgroup_abort_on_error():
    order = []

    class A(Action):
        async def _execute(self, **kwargs):
            order.append("A")
            raise Exception()

    class B(Action):
        async def _execute(self, **kwargs):
            order.append("B")
            return [Markdown("B done")]

    tg = TaskGroup(A(), B(), title="Error", abort_on_error=True)

    outs = await tg.execute()

    assert order == ["A"]

    assert len(outs) == 1
    out = outs[0]
    assert isinstance(out, Typography)
    assert out.object == "### Error"

    assert len(tg._view) == 1
    assert outs == list(tg._view)

@duckdb_available
async def test_sql_query_action():
    source = DuckDBSource(tables={
    'tiny': """
        SELECT * FROM (
          VALUES (1,'A',10.5),
                 (2,'B',20.0),
                 (3,'A', 7.5)
        ) AS t(id, category, value)
    """
    })

    action = SQLQuery(source=source, sql_expr="SELECT category FROM tiny", table="foo", generate_caption=False)

    outs = await action.execute()

    assert len(outs) == 1
    out = outs[0]
    assert isinstance(out, SQLOutput)
    assert out.spec == action.sql_expr
    assert out.component.data.columns == ['category']
    assert list(out.component.data.category) == ['A', 'B', 'A']
