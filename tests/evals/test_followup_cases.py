import pytest
import yaml

from panel_flowdash.agent import DashboardOps
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_evals import Dataset

from lumen.agent.evals import FlowDashSubject
from lumen.agent.sessions import rerun
from lumen.ai.evals.harness import (
    CheckResult, Expected, Inputs, Output, evaluate,
)
from lumen.sources.duckdb import DuckDBSource

from .cases import DATASET, source

PAID = (
    "SELECT c.customer, c.region, o.amount FROM orders o JOIN customers c "
    "ON o.customer_id = c.customer_id WHERE o.status = 'paid'"
)
TOTALS = "SELECT region, SUM(amount) AS total_amount FROM paid_orders GROUP BY region ORDER BY region"


def scripted(*responses):
    def respond(messages, info):
        count = sum(isinstance(message, ModelResponse) for message in messages)
        return ModelResponse(parts=list(responses[count]))

    return FunctionModel(respond, model_name="gpt-6-luna")


def aggregate_case():
    return next(case for case in DATASET.cases if case.name == "followup_aggregate")


async def scored(model):
    subject = FlowDashSubject(model=model)
    dataset = Dataset[Inputs, Output, Expected](name="followups", cases=[aggregate_case()], evaluators=[CheckResult()])
    report = await evaluate(subject.build_model("gpt-6-luna"), dataset, source, subject=subject)
    (case,) = report.cases
    return {name: result.value for name, result in case.assertions.items()}


FIRST = [
    [ToolCallPart("add_sql", {"source": "data", "name": "paid_orders", "sql": PAID})],
    [TextPart("Four paid orders.")],
]


async def test_a_follow_up_on_the_earlier_node_reads_it_and_keeps_it():
    model = scripted(
        *FIRST,
        [ToolCallPart("add_sql", {"source": "paid_orders", "name": "totals", "sql": TOTALS})],
        [TextPart("East 65, west 40.")],
    )
    assert await scored(model) == {
        "all_turns_completed": True, "status": True, "output": True, "rows": True,
        "turn_rows": True, "turn_reads": True, "stable_turns": True,
    }


async def test_rewriting_the_earlier_node_reads_nothing_and_changes_its_answer():
    rewrite = "SELECT c.region, SUM(o.amount) AS total_amount FROM orders o JOIN customers c " \
        "ON o.customer_id = c.customer_id WHERE o.status = 'paid' GROUP BY c.region"
    model = scripted(
        *FIRST,
        [ToolCallPart("set_config", {"name": "paid_orders", "config": {"sql": rewrite}})],
        [TextPart("East 65, west 40.")],
    )
    checks = await scored(model)
    assert checks["rows"] is True
    assert checks["turn_reads"] is False and checks["stable_turns"] is False


async def test_a_follow_up_chain_reruns_on_more_data_without_a_model():
    model = scripted(
        *FIRST,
        [ToolCallPart("add_sql", {"source": "paid_orders", "name": "totals", "sql": TOTALS})],
        [TextPart("East 65, west 40.")],
    )
    case = aggregate_case()
    subject = FlowDashSubject(model=model)
    output = await subject.run(case.inputs, subject.build_model("x"), source(case.inputs))
    assert [turn.reads for turn in output.turns] == [[], [0]]

    more = source(case.inputs).tables | {
        "orders": source(case.inputs).tables["orders"].replace("(105, 3, 15, 'paid')", "(105, 3, 15, 'paid'), (106, 2, 25, 'paid')")
    }
    editor = rerun(output.artifacts[-1].graph, DuckDBSource(uri=":memory:", tables=more))
    try:
        totals = DashboardOps(editor).node_output("totals.data")
    finally:
        editor.dispose()
    assert totals.values.tolist() == [["east", 65], ["west", 65]]
    assert yaml.safe_load(output.artifacts[-1].graph)["edges"][-1] == {"from": "paid_orders.result", "to": "totals.source"}


@pytest.mark.parametrize("name", ["followup_aggregate", "followup_threshold"])
def test_follow_up_cases_check_reads_and_stability(name):
    expected = next(case for case in DATASET.cases if case.name == name).metadata
    assert expected.turn_reads[1] == 0 and expected.stable_turns


def test_narrowing_an_answer_is_not_scored_as_a_follow_up():
    expected = next(case for case in DATASET.cases if case.name == "commerce_followup_filter").metadata
    assert expected.turn_reads is None and not expected.stable_turns
