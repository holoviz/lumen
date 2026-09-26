import json
import sqlite3

import httpx
import pytest

from openai import RateLimitError
from panel_material_ui import ChatMessage
from pydantic_evals import Case, Dataset

from lumen.ai.agents.sql import SQLAgent, make_sql_model
from lumen.ai.coordinator import Plan
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.evals.bird import bird_source, database_path, execute_read_only
from lumen.ai.evals.harness import (
    CheckResult, Expected, Inputs, MeteredOpenAI, Output, Turn, _snapshot,
    evaluate, run_case,
)
from lumen.ai.report import ActorTask
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource
from lumen.tests.ai.conftest import MockLLM
from lumen.views import VegaLiteView

from .cases import DATASET, bird_dataset, source


@pytest.mark.asyncio
async def test_run_case_captures_executed_sql(monkeypatch):
    """A submitted turn captures its executed SQL, data and status."""
    llm = MockLLM()

    async def respond(self, messages, context):
        query = make_sql_model([(context["source"].name, "sales")])
        llm.set_responses([query(query="SELECT SUM(amount) AS total FROM sales", table_slug="sales_total", tables=["sales"])])
        return Plan(ActorTask(SQLAgent(llm=llm), title="Total sales"), title="Total sales", context=context, history=[{"role": "user", "content": messages}], llm=llm)

    monkeypatch.setattr("lumen.ai.coordinator.planner.Planner.respond", respond)
    result = await run_case(Inputs(["What is the total amount overall?"]), llm, source(Inputs([])))
    turn = result.turns[0]
    assert turn.status == "success"
    assert turn.actors == ["SQLAgent"]
    assert turn.task_statuses == ["success"]
    assert turn.rows == [[35]]
    assert turn.row_count == 1
    assert turn.columns == ["total"]
    assert "SUM(amount)" in turn.sql


@pytest.mark.asyncio
async def test_run_case_captures_followup_separately(monkeypatch):
    """Each turn records only its new SQL and resulting data."""
    llm = MockLLM()

    async def respond(self, messages, context):
        query = make_sql_model([(context["source"].name, "sales")])
        if "category" in messages:
            sql = "SELECT category, SUM(amount) AS total FROM sales GROUP BY category ORDER BY category"
        else:
            sql = "SELECT SUM(amount) AS total FROM sales"
        llm.set_responses([query(query=sql, table_slug="sales_result", tables=["sales"])])
        return Plan(ActorTask(SQLAgent(llm=llm), title="Sales"), title="Sales", context=context, history=[{"role": "user", "content": messages}], llm=llm)

    monkeypatch.setattr("lumen.ai.coordinator.planner.Planner.respond", respond)
    result = await run_case(Inputs(["Total by category", "Total overall"]), llm, source(Inputs([])))
    assert [turn.status for turn in result.turns] == ["success", "success"]
    assert result.turns[0].rows == [["A", 15], ["B", 20]]
    assert result.turns[1].rows == [[35]]
    assert result.turns[1].actors == ["SQLAgent"]


def test_snapshot_does_not_reuse_previous_turn_answer():
    """A follow-up with no reply must not inherit the prior task's answer."""
    previous = ActorTask(SQLAgent(), status="success", out_context={"chat": "Old reply"})
    current = ActorTask(SQLAgent(), status="success", out_context={"sql": "SELECT 1"})
    plan = Plan(previous, current, status="success")
    turn = _snapshot("follow-up", plan, [ChatMessage("follow-up", user="User")], (previous,))
    assert turn.answer is None
    assert turn.actors == ["SQLAgent"]
    assert turn.sql == "SELECT 1"
    assert turn.rows is None


def test_plan_merge_retains_followup_planner_selection():
    """Merged follow-ups expose their own route rather than their parent's route."""
    parent = Plan(ActorTask(SQLAgent(), title="Query"), planner_actors=["SQLAgent"])
    followup = Plan(ActorTask(SQLAgent(), title="Chart"), planner_actors=["VegaLiteAgent"], follow_up_type="direct")
    parent.merge(followup)
    assert parent.planner_actors == ["VegaLiteAgent"]
    assert parent.follow_up_type == "direct"


@pytest.mark.asyncio
async def test_snapshot_captures_chart_mark():
    """Direct-plan chart checks use the rendered Vega-Lite mark, not view type alone."""
    fixture_source = DuckDBSource(uri=":memory:", tables={"sales": "SELECT 1 AS amount"})
    view = VegaLiteEditor(component=VegaLiteView(pipeline=Pipeline(source=fixture_source, table="sales"), spec={"mark": {"type": "bar"}, "encoding": {}}))
    task = ActorTask(SQLAgent(), status="success", views=[view], out_context={})
    turn = _snapshot("Plot bars", Plan(task, status="success"), [])
    assert turn.chart_marks == ["bar"]


@pytest.mark.asyncio
async def test_evaluate_runs_dataset_and_saves_result(tmp_path, monkeypatch):
    """The eval runner persists case outputs and deterministic assertions."""
    async def respond(self, messages, context):
        query = make_sql_model([(context["source"].name, "sales")])
        llm.set_responses([query(query="SELECT SUM(amount) AS total FROM sales", table_slug="sales_total", tables=["sales"])])
        return Plan(ActorTask(SQLAgent(llm=llm), title="Total sales"), title="Total sales", context=context, history=[{"role": "user", "content": messages}], llm=llm)

    llm = MockLLM()
    monkeypatch.setattr("lumen.ai.coordinator.planner.Planner.respond", respond)
    path = tmp_path / "result.json"
    report = await evaluate(llm, DATASET, source, path, "sales_total")
    assert not report.failures
    assert all(result.value for result in report.cases[0].assertions.values())
    result = json.loads(path.read_text())
    assert result["cases"][0]["turns"][0]["rows"] == [[35]]
    assert result["cases"][0]["assertions"]["rows"] is True


@pytest.mark.asyncio
async def test_evaluate_persists_setup_failure_as_case(tmp_path):
    """A failed source setup remains a scored case instead of disappearing from the report."""
    dataset = Dataset[Inputs, Output, Expected](
        name="broken_fixture", cases=[Case(name="broken", inputs=Inputs(["Query missing table"]))],
        evaluators=[CheckResult()],
    )

    def missing_source(inputs):
        raise ValueError("fixture unavailable")

    output = tmp_path / "failed.json"
    report = await evaluate(MockLLM(), dataset, missing_source, output)
    saved = json.loads(output.read_text())
    assert not report.failures
    assert saved["cases"][0]["name"] == "broken"
    assert saved["cases"][0]["turns"][0]["status"] == "error"
    assert saved["cases"][0]["assertions"]["status"] is False


@pytest.mark.asyncio
async def test_direct_sql_plan_runs_without_planner(monkeypatch):
    """The SQL case constructs and executes an ActorTask directly."""
    async def fail_planner(*args, **kwargs):
        raise AssertionError("Planner should not run")

    class SQLMockLLM(MockLLM):
        async def invoke(self, *args, **kwargs):
            if kwargs.get("response_model"):
                query = make_sql_model([("sales", "sales")])
                return query(query="SELECT SUM(amount) AS total FROM sales", table_slug="sales_total", tables=["sales"])
            return await super().invoke(*args, **kwargs)

    monkeypatch.setattr("lumen.ai.coordinator.planner.Planner.respond", fail_planner)
    result = await run_case(Inputs(["Overall sales total"], agents=("SQLAgent",)), SQLMockLLM(), source(Inputs([])))
    assert result.turns[0].status == "success"
    assert result.turns[0].rows == [[35]]
    assert result.turns[0].actors == ["SQLAgent"]


@pytest.mark.asyncio
async def test_direct_commerce_plan_uses_current_followup(monkeypatch):
    """Direct plans use the current prompt and preserve the previous context."""
    queries = [
        "SELECT COUNT(*) AS total_count FROM customers WHERE region = 'east'",
        "SELECT COUNT(*) AS total_count FROM orders WHERE status = 'paid'",
    ]

    class SQLMockLLM(MockLLM):
        async def invoke(self, *args, **kwargs):
            if kwargs.get("response_model"):
                query = make_sql_model([("commerce", "customers"), ("commerce", "orders")])
                sql = queries.pop(0)
                return query(query=sql, table_slug="count_result", tables=["orders" if "orders" in sql else "customers"])
            return await super().invoke(*args, **kwargs)

    inputs = Inputs(["Count east customers", "Now count paid orders"], fixture="commerce", agents=("SQLAgent",))
    result = await run_case(inputs, SQLMockLLM(), source(inputs))
    assert [turn.status for turn in result.turns] == ["success", "success"]
    assert [turn.rows for turn in result.turns] == [[[2]], [[4]]]
    assert "customers" in result.turns[0].sql.lower()
    assert "orders" in result.turns[1].sql.lower()


def test_commerce_fixture_produces_expected_totals():
    """Joined fixture totals exclude the cancelled order."""
    commerce = source(Inputs([], fixture="commerce"))
    result = commerce.get("orders")
    assert result.loc[result.status == "paid", "amount"].sum() == 105
    assert len(commerce.get("customers")) == 4
    assert set(commerce.get("customers").customer) - set(commerce.get("orders").merge(commerce.get("customers"), on="customer_id").customer) == {"Dee"}


@pytest.mark.asyncio
async def test_metered_client_records_completed_tool_calls(monkeypatch):
    """Only tool calls that produced a tool result count toward an eval."""
    llm = MeteredOpenAI(api="chat_completions", model_kwargs={"default": {"model": "qwen/qwen3.8-flash"}})
    from lumen.ai.tools.base import FunctionTool

    def lookup() -> str:
        return "found rows"

    called = FunctionTool(lookup)
    messages = await llm._run_tool_calls(
        {"lookup": called}, [{"id": "1", "function": {"name": "lookup", "arguments": "{}"}}], {}, []
    )
    assert messages[0]["content"] == "found rows"
    assert [trace.name for trace in llm.tool_calls] == ["lookup"]
    assert llm.tool_calls[0].arguments == {}
    assert llm.tool_calls[0].result == "found rows"


@pytest.mark.asyncio
async def test_direct_table_list_captures_listing():
    """A non-LLM agent exposes its listing through plan context."""
    result = await run_case(Inputs(["List tables"], agents=("TableListAgent",)), MockLLM(), source(Inputs([])))
    assert result.turns[0].status == "success"
    assert result.turns[0].listing == "Displayed 1 table(s)"
    assert result.turns[0].actors == ["TableListAgent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_metered_client_retries_rate_limit_before_output(monkeypatch, stream):
    """A pre-output 429 retries without recording duplicate token usage."""
    from types import SimpleNamespace

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    error = RateLimitError("rate limited", response=httpx.Response(429, request=request, headers={"retry-after": "0"}), body=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        if not stream:
            return SimpleNamespace(usage=usage, model="qwen/qwen3.8-flash")

        async def chunks():
            yield SimpleNamespace(usage=usage, model="qwen/qwen3.8-flash")

        return chunks()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("lumen.ai.llm.OpenAI._create_base_client", lambda self, **kwargs: client)
    llm = MeteredOpenAI(api="chat_completions", model_kwargs={"default": {"model": "qwen/qwen3.8-flash"}})
    wrapped = llm._create_base_client().chat.completions.create
    response = await wrapped(stream=stream)
    if stream:
        assert len([chunk async for chunk in response]) == 1
    assert len(calls) == 2
    assert len(llm.usage) == 1
    assert llm.usage[0].input_tokens == 10




@pytest.mark.asyncio
async def test_run_case_reports_no_plan(monkeypatch):
    """A planner that returns no plan is an explicit eval outcome."""
    async def respond(self, messages, context):
        return None

    monkeypatch.setattr("lumen.ai.coordinator.planner.Planner.respond", respond)
    result = await run_case(Inputs(["What data is available?"]), MockLLM(), source(Inputs([])))
    assert result.turns[0].status == "no_plan"


@pytest.mark.asyncio
async def test_evaluator_rejects_wrong_result():
    """A successful task cannot pass when its captured data is wrong."""
    dataset = Dataset[Inputs, Output, Expected](
        name="check_wrong_rows",
        cases=[Case(name="wrong_rows", inputs=Inputs(["total"]), metadata=Expected(rows=[[35]]))],
        evaluators=[CheckResult()],
    )
    report = await dataset.evaluate(lambda inputs: Output([Turn(inputs.prompts[0], "success", None, [], [], None, ["total"], [[34]], 1, [])]))
    assert report.cases[0].assertions["rows"].value is False


@pytest.mark.asyncio
async def test_evaluator_accepts_unordered_rows():
    """SQL results without ORDER BY are compared as row sets."""
    dataset = Dataset[Inputs, Output, Expected](
        name="check_unordered_rows",
        cases=[Case(name="unordered", inputs=Inputs(["by category"]), metadata=Expected(rows=[["A", 15], ["B", 20]]))],
        evaluators=[CheckResult()],
    )
    report = await dataset.evaluate(lambda inputs: Output([Turn(inputs.prompts[0], "success", None, [], [], None, ["category", "total"], [["B", 20], ["A", 15]], 2, [])]))
    assert report.cases[0].assertions["rows"].value is True


@pytest.mark.asyncio
async def test_evaluator_rejects_missing_turn():
    """A multi-turn eval cannot pass after stopping early."""
    dataset = Dataset[Inputs, Output, Expected](
        name="check_missing_turn",
        cases=[Case(name="missing_turn", inputs=Inputs(["first", "second"]))],
        evaluators=[CheckResult()],
    )
    report = await dataset.evaluate(lambda inputs: Output([Turn(inputs.prompts[0], "success", None, [], [], None, None, None, None, [])]))
    assert report.cases[0].assertions["all_turns_completed"].value is False


@pytest.mark.asyncio
async def test_evaluator_rejects_success_without_output():
    """A nominally successful plan must produce something user-visible."""
    dataset = Dataset[Inputs, Output, Expected](
        name="check_missing_output",
        cases=[Case(name="missing_output", inputs=Inputs(["answer"]))],
        evaluators=[CheckResult()],
    )
    report = await dataset.evaluate(lambda inputs: Output([Turn(inputs.prompts[0], "success", None, [], [], None, None, None, None, [])]))
    assert report.cases[0].assertions["output"].value is False


@pytest.mark.asyncio
async def test_evaluator_rejects_task_error_with_rows():
    """Rows do not hide an error recorded by a plan task."""
    dataset = Dataset[Inputs, Output, Expected](
        name="check_task_error",
        cases=[Case(name="task_error", inputs=Inputs(["answer"]), metadata=Expected(rows=[[1]]))],
        evaluators=[CheckResult()],
    )
    turn = Turn("answer", "success", None, ["SQLAgent"], ["success"], "SELECT 1", ["value"], [[1]], 1, [], task_errors=["downstream failed"])
    report = await dataset.evaluate(lambda inputs: Output([turn]))
    assert report.cases[0].assertions["output"].value is False


def test_bird_execution_accuracy_and_read_only_source(tmp_path):
    """Selected BIRD predictions execute on the original SQLite database in read-only mode."""
    db_dir = tmp_path / "dev_databases" / "student_club"
    db_dir.mkdir(parents=True)
    path = db_dir / "student_club.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE budget (category TEXT, spent INTEGER)")
        connection.executemany("INSERT INTO budget VALUES (?, ?)", [("Food", 10), ("Food", 20), ("Other", 5)])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{
        "question_id": 1380, "db_id": "student_club", "question": "Total spent for food?",
        "evidence": "Food means category = 'Food'", "SQL": "SELECT SUM(spent) FROM budget WHERE category = 'Food'",
        "difficulty": "simple",
    }]))
    dataset = bird_dataset(questions, tmp_path, (1380,))
    assert database_path(tmp_path, "student_club") == path
    assert bird_source(path).get("budget").spent.sum() == 35
    assert execute_read_only(path, "SELECT SUM(spent) FROM budget WHERE category = 'Food'") == {(30,)}
    with pytest.raises(sqlite3.OperationalError):
        execute_read_only(path, "DELETE FROM budget")

    class SQLMockLLM(MockLLM):
        async def invoke(self, *args, **kwargs):
            if kwargs.get("response_model"):
                query = make_sql_model([("student_club", "budget")])
                return query(query="SELECT SUM(spent) FROM budget WHERE category = 'Food'", table_slug="bird_result", tables=["budget"])
            return await super().invoke(*args, **kwargs)

    import asyncio

    output = asyncio.run(run_case(dataset.cases[0].inputs, SQLMockLLM(), bird_source(path)))
    assert output.turns[0].status == "success"
    assert output.turns[0].rows == [[30]]

    async def evaluate_prediction(sql):
        case = dataset.cases[0]
        turn = Turn(case.inputs.prompts[0], "success", None, ["SQLAgent"], ["success"], sql, ["total"], [[30]], 1, [])
        report = await dataset.evaluate(lambda inputs: Output([turn]))
        return report.cases[0].assertions["execution_accuracy"].value

    assert asyncio.run(evaluate_prediction("SELECT SUM(spent) FROM budget WHERE category = 'Food'"))
    assert not asyncio.run(evaluate_prediction("SELECT SUM(spent) FROM budget"))
