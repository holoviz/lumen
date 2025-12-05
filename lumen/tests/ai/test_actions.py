from lumen.ai.actions import SQLQuery
from lumen.ai.views import SQLOutput


async def test_sql_query_action(tiny_source):
    action = SQLQuery(source=tiny_source, sql_expr="SELECT category FROM tiny", table="foo", generate_caption=False)

    outs, ctx = await action.execute()

    assert len(outs) == 1
    out = outs[0]
    assert isinstance(out, SQLOutput)
    assert out.spec == action.sql_expr
    assert out.component.data.columns == ['category']
    assert list(out.component.data.category) == ['A', 'B', 'A']
