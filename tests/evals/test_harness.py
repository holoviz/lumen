import asyncio
import importlib
import json
import multiprocessing
import sqlite3
import time

from pathlib import Path

import httpx
import pytest

from openai import RateLimitError
from panel_material_ui import ChatMessage
from pydantic_evals import Case, Dataset

from lumen.ai.agents.sql import SQLAgent, SQLCleanup, make_sql_model
from lumen.ai.coordinator import Plan
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.evals.bird import bird_source, database_path, execute_read_only
from lumen.ai.evals.harness import (
    APPLICABILITY, ARCHITECTURE_CHECKS, NOT_APPLICABLE, OUTCOME_CHECKS,
    SUBJECTS, CheckResult, EvalOpenAI, Expected, Inputs, LumenSubject, Output,
    Subject, Turn, TurnArtifacts, _snapshot, artifacts_path, case_fingerprint,
    evaluate, register_subject, resolve_subject, run_case, run_settings,
    subject_spec,
)
from lumen.ai.report import ActorTask
from lumen.ai.tool_trace import ToolCall
from lumen.ai.usage import Usage as LlmUsage, record_usage
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource
from lumen.tests.ai.conftest import MockLLM
from lumen.views import VegaLiteView

from .bird_runner import simulated_bird_case_process
from .cases import (
    BIRD_STRATIFIED_IDS, DATASET, SUITE_INSTRUCTIONS, bird_dataset, source,
)


def test_suite_instructions_apply_to_system_prompt_and_fingerprint():
    instructions, version = SUITE_INSTRUCTIONS["bird"]
    llm = EvalOpenAI(suite_instructions=instructions, model_kwargs={"default": {"model": "gpt-6-luna"}})
    messages, _ = llm._add_system_message([{"role": "user", "content": "Count sales"}], "SQL agent rules", {})
    assert instructions in messages[0]["content"]
    assert messages[0]["content"].startswith("SQL agent rules")
    case = Case(name="test", inputs=Inputs(["Count sales"]))
    baseline = case_fingerprint([case])
    revised = case_fingerprint([case], instructions=instructions, instruction_version=version)
    assert revised != baseline
    assert revised != case_fingerprint([case], instructions=instructions + " More rules", instruction_version=version)
    assert revised != case_fingerprint([case], instructions=instructions, instruction_version=version + "-no-sql-cleanup-v1")
    assert case_fingerprint([case], settings={}) == baseline
    assert case_fingerprint([case], settings={"option": 1}) not in (baseline, case_fingerprint([case], settings={"option": 2}))


def test_bird_stratified_cases_cover_all_databases():
    questions = Path(__file__).parent / "results" / "bird-mini-dev-sqlite.json"
    databases = Path(__file__).parent / "results"
    if not questions.exists() or not databases.exists():
        pytest.skip("BIRD Mini-Dev fixtures not installed")
    ids = tuple(question_id for group in BIRD_STRATIFIED_IDS.values() for question_id in group)
    dataset = bird_dataset(questions, databases, ids)
    assert len(ids) == len(set(ids)) == len(dataset.cases) == 33
    assert len({case.inputs.fixture for case in dataset.cases}) == 11


def test_local_file_commerce_fixture_uses_read_functions():
    fixture = source(Inputs([], fixture="file_commerce"))
    assert "read_csv(" in fixture.tables["orders_csv"]
    assert "read_json_auto(" in fixture.tables["refunds_json"]
    assert fixture.get("orders_csv").shape[0] == 5
    assert fixture.get("refunds_json").shape[0] == 2


def test_local_read_function_fixture_exposes_expressions():
    fixture = source(Inputs([], fixture="read_functions"))
    assert "READ_CSV(" in fixture.get_sql_expr("orders_csv")
    assert "READ_JSON_AUTO(" in fixture.get_sql_expr("refunds_json")
    assert len(fixture.get("orders_csv")) == 5
    assert len(fixture.get("refunds_json")) == 2


@pytest.mark.parametrize("name", [
    "commerce_select_orders", "file_orders_csv", "file_refunds_json",
    "file_join_paid_orders", "file_join_refunds", "read_csv_function",
    "join_read_functions",
])
def test_local_sql_cases_check_rows_and_tables(name):
    case = next(case for case in DATASET.cases if case.name == name)
    fixture = source(case.inputs)
    expected = case.metadata
    assert expected.rows
    assert sorted(fixture.execute({
        "commerce_select_orders": "SELECT COUNT(*) AS paid_count FROM orders WHERE status = 'paid'",
        "file_orders_csv": "SELECT order_id, amount FROM orders_csv WHERE status = 'paid' ORDER BY order_id",
        "file_refunds_json": "SELECT order_id, refund_amount FROM refunds_json ORDER BY order_id",
        "file_join_paid_orders": "SELECT c.customer, COALESCE(SUM(o.amount), 0) AS paid_total FROM customers c LEFT JOIN orders_csv o ON c.customer_id = o.customer_id AND o.status = 'paid' GROUP BY c.customer ORDER BY c.customer",
        "file_join_refunds": "SELECT o.order_id, r.refund_amount FROM orders_csv o JOIN refunds_json r ON o.order_id = r.order_id WHERE o.status = 'paid' ORDER BY o.order_id",
        "read_csv_function": "SELECT COUNT(*) AS paid_count FROM read_csv('tests/evals/fixtures/orders.csv') WHERE status = 'paid'",
        "join_read_functions": "SELECT o.order_id, r.refund_amount FROM read_csv('tests/evals/fixtures/orders.csv') o JOIN read_json_auto('tests/evals/fixtures/refunds.json') r ON o.order_id = r.order_id WHERE o.status = 'paid'",
    }[name]).values.tolist(), key=str) == sorted(expected.rows, key=str)


@pytest.mark.asyncio
async def test_file_join_eval_checks_query_table_choice():
    case = next(case for case in DATASET.cases if case.name == "file_join_refunds")
    llm = MockLLM()
    query = make_sql_model([("file_commerce", table) for table in ("orders_csv", "refunds_json")])
    llm.set_responses([query(
        query="SELECT o.order_id, r.refund_amount FROM orders_csv o JOIN refunds_json r ON o.order_id = r.order_id WHERE o.status = 'paid'",
        table_slug="paid_refunds", tables=["orders_csv", "refunds_json"],
    )])

    result = await run_case(case.inputs, llm, source(case.inputs))
    assert result.turns[0].status == "success"
    report = await Dataset(name="file_join", cases=[case], evaluators=[CheckResult()]).evaluate(lambda inputs: result)
    assert all(assertion.value for assertion in report.cases[0].assertions.values())

    result.turns[0].sql = "SELECT o.order_id, r.refund_amount FROM orders_csv o JOIN other_refunds r ON o.order_id = r.order_id"
    report = await Dataset(name="wrong_join", cases=[case], evaluators=[CheckResult()]).evaluate(lambda inputs: result)
    assert report.cases[0].assertions["sql_contains"].value is False


@pytest.mark.parametrize("name", ["read_csv_function", "join_read_functions"])
@pytest.mark.asyncio
async def test_read_function_eval_accepts_valid_sql(name):
    case = next(case for case in DATASET.cases if case.name == name)
    fixture = source(case.inputs)
    files = (Path(__file__).parent / "fixtures").resolve()
    orders = f"read_csv('{files / 'orders.csv'}')"
    refunds = f"read_json_auto('{files / 'refunds.json'}')"
    if name == "read_csv_function":
        sql = f"SELECT COUNT(*) AS paid_count FROM {orders} WHERE status = 'paid'"
        tables = ["orders_csv"]
    else:
        sql = (f"SELECT o.order_id, r.refund_amount FROM {orders} o JOIN {refunds} r "
               "ON o.order_id = r.order_id WHERE o.status = 'paid'")
        tables = ["orders_csv", "refunds_json"]
    llm = MockLLM()
    model = make_sql_model([(fixture.name, table) for table in fixture.get_tables()])
    llm.set_responses([model(query=sql, table_slug="read_result", tables=tables)])

    result = await run_case(case.inputs, llm, fixture)
    report = await Dataset(name="read_functions", cases=[case], evaluators=[CheckResult()]).evaluate(lambda inputs: result)

    assert result.turns[0].status == "success"
    assert all(assertion.value for assertion in report.cases[0].assertions.values())


def test_parallel_bird_runner_checkpoints_in_order_and_times_out(tmp_path):
    from tests.evals.__main__ import run_all_bird

    cases = [Case(name=f"case_{index}", inputs=Inputs([str(index)])) for index in range(1, 4)]
    dataset = Dataset(name="bird_test", cases=cases)
    questions = tmp_path / "questions.json"
    questions.write_text("[]")
    output = tmp_path / "result.json"
    started = time.monotonic()
    # The limit includes worker startup, and importing lumen in a spawned
    # worker takes several seconds on a loaded machine.
    summary = run_all_bird(dataset, output, questions, tmp_path, "test", "openai", "test-key", False, 2,
                           worker=simulated_bird_case_process, case_timeout=30)
    saved = json.loads(output.read_text())
    assert time.monotonic() - started < 120
    assert [case["name"] for case in saved["cases"]] == [case.name for case in cases]
    assert saved["cases"][1]["assertions"] == {}
    assert "wall-clock limit" in saved["cases"][1]["error"]
    assert summary["completed"] == 3
    assert summary["unscorable"] == 1
    assert run_all_bird(dataset, output, questions, tmp_path, "test", "openai", "test-key", True, 2,
                        worker=simulated_bird_case_process, case_timeout=30) == summary


def test_parallel_bird_runner_rejects_resume_with_other_settings(tmp_path):
    from tests.evals.__main__ import run_all_bird

    dataset = Dataset(name="bird_test", cases=[Case(name="case_1", inputs=Inputs(["1"]))])
    questions = tmp_path / "questions.json"
    questions.write_text("[]")
    output = tmp_path / "result.json"
    run_all_bird(dataset, output, questions, tmp_path, "test", "openai", "test-key", False, 1,
                 worker=simulated_bird_case_process, case_timeout=60, settings={"option": 1})
    assert json.loads(output.read_text())["run"]["settings"] == {"option": 1}
    with pytest.raises(ValueError, match="different case set, settings"):
        run_all_bird(dataset, output, questions, tmp_path, "test", "openai", "test-key", True, 1,
                     worker=simulated_bird_case_process, case_timeout=60, settings={"option": 2})


def test_parallel_bird_runner_reads_result_sent_before_worker_exit(tmp_path, monkeypatch):
    """A worker that sends and exits while the runner is not waiting still counts."""
    import tests.evals.__main__ as runner

    # Never report readiness, so every result arrives outside wait().
    monkeypatch.setattr(runner, "wait", lambda *args, **kwargs: time.sleep(0.2) or [])
    cases = [Case(name="case_1", inputs=Inputs(["1"]))]
    questions = tmp_path / "questions.json"
    questions.write_text("[]")
    summary = runner.run_all_bird(Dataset(name="bird_test", cases=cases), tmp_path / "result.json", questions,
                                  tmp_path, "test", "openai", "test-key", False, 1,
                                  worker=simulated_bird_case_process, case_timeout=60)
    assert summary["correct"] == 1
    assert summary["unscorable"] == 0


def _nested_bird_score(path, sender):
    try:
        sender.send(execute_read_only(path, "SELECT 1"))
    finally:
        sender.close()


def test_bird_scorer_spawns_inside_case_worker(tmp_path):
    path = tmp_path / "bird.sqlite"
    with sqlite3.connect(path):
        pass
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_nested_bird_score, args=(path, sender))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(60)
        assert receiver.recv() == {(1,)}
    finally:
        receiver.close()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join()
    assert worker.exitcode == 0


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
            if kwargs.get("response_model") is SQLCleanup:
                return SQLCleanup(chain_of_thought="No edit needed.", query=self.last_sql)
            if kwargs.get("response_model"):
                query = make_sql_model([("commerce", "customers"), ("commerce", "orders")])
                sql = self.last_sql = queries.pop(0)
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
async def test_eval_client_records_completed_tool_calls():
    """Only tool calls that produced a tool result count toward an eval."""
    llm = EvalOpenAI(api="chat_completions", model_kwargs={"default": {"model": "qwen/qwen3.8-flash"}})
    from lumen.ai.tools.base import FunctionTool

    def lookup() -> str:
        return "found rows"

    called = FunctionTool(lookup)
    with llm.trace() as traces:
        messages = await llm._run_tool_calls(
            {"lookup": called}, [{"id": "1", "function": {"name": "lookup", "arguments": "{}"}}], {}, [],
        )
    assert messages[0]["content"] == "found rows"
    tool_traces = [trace for trace in traces if isinstance(trace, ToolCall)]
    assert [trace.name for trace in tool_traces] == ["lookup"]
    assert tool_traces[0].arguments == {}
    assert tool_traces[0].result == "found rows"
    assert not hasattr(llm, "tool_calls")


@pytest.mark.asyncio
async def test_direct_table_list_captures_listing():
    """A non-LLM agent exposes its listing through plan context."""
    result = await run_case(Inputs(["List tables"], agents=("TableListAgent",)), MockLLM(), source(Inputs([])))
    assert result.turns[0].status == "success"
    assert result.turns[0].listing == "Displayed 1 table(s)"
    assert result.turns[0].actors == ["TableListAgent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_eval_client_retries_rate_limit_before_output(monkeypatch, stream):
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
    monkeypatch.setattr("lumen.ai.llm.OpenAI._instantiate_client", lambda self, **kwargs: client)
    llm = EvalOpenAI(api="chat_completions", model_kwargs={"default": {"model": "qwen/qwen3.8-flash"}})
    wrapped = llm._create_base_client().chat.completions.create
    with llm.capture_usage() as scope:
        response = await wrapped(model="qwen/qwen3.8-flash", stream=stream)
        if stream:
            assert len([chunk async for chunk in response]) == 1
    assert len(calls) == 2
    assert len(scope.records) == 1
    assert llm.usage.records[0].input_tokens == 10


@pytest.mark.asyncio
async def test_run_case_scopes_usage_to_each_turn(monkeypatch):
    llm = MockLLM()

    async def respond(self, messages, context):
        record_usage(llm, llm.usage, LlmUsage("test", 10, 2, 3, 0.0001))

    monkeypatch.setattr("lumen.ai.coordinator.planner.Planner.respond", respond)
    result = await run_case(Inputs(["First", "Second"]), llm, source(Inputs([])))

    assert [turn.usage.input_tokens for turn in result.turns] == [10, 10]
    assert [turn.usage.cached_tokens for turn in result.turns] == [3, 3]
    assert result.usage.input_tokens == 20
    assert result.usage.cost_usd == pytest.approx(0.0002)




@pytest.mark.asyncio
async def test_run_case_passes_constructor_params(monkeypatch):
    """Configure the UI and directly run agents without changing the harness."""
    from lumen.ai.evals import harness

    created = []

    def capture(cls):
        return lambda **params: created.append((cls.__name__, params)) or cls(**params)

    async def respond(self, messages, context):
        return None

    monkeypatch.setattr("lumen.ai.coordinator.planner.Planner.respond", respond)
    monkeypatch.setattr(harness, "ExplorerUI", capture(harness.ExplorerUI))
    monkeypatch.setattr(harness, "TableListAgent", capture(harness.TableListAgent))
    await run_case(Inputs(["What data is available?"]), MockLLM(), source(Inputs([])), ui_params={"log_level": "INFO"})
    await run_case(Inputs(["List tables"], agents=("TableListAgent",)), MockLLM(), source(Inputs([])),
                   agent_params={"debug": True})

    assert created[0][0] == "ExplorerUI" and created[0][1]["log_level"] == "INFO"
    assert created[1][0] == "TableListAgent" and created[1][1]["debug"] is True


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


def test_bird_gold_failure_is_unscorable(tmp_path):
    """A gold query that cannot run is recorded as an error, not as a model miss."""
    db_dir = tmp_path / "dev_databases" / "student_club"
    db_dir.mkdir(parents=True)
    path = db_dir / "student_club.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE budget (spent INTEGER)")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{
        "question_id": 1, "db_id": "student_club", "question": "Total spent?", "evidence": "",
        "SQL": "SELECT SUM(missing) FROM budget", "difficulty": "simple",
    }]))
    dataset = bird_dataset(questions, tmp_path, (1,))

    class SQLMockLLM(MockLLM):
        async def invoke(self, *args, **kwargs):
            if kwargs.get("response_model") is SQLCleanup:
                return SQLCleanup(chain_of_thought="No edit needed.", query="SELECT SUM(spent) FROM budget")
            if kwargs.get("response_model"):
                query = make_sql_model([("student_club", "budget")])
                return query(query="SELECT SUM(spent) FROM budget", table_slug="total", tables=["budget"])
            return await super().invoke(*args, **kwargs)

    import asyncio

    output = tmp_path / "result.json"
    asyncio.run(evaluate(SQLMockLLM(), dataset, lambda inputs: bird_source(path), output))
    case = json.loads(output.read_text())["cases"][0]

    assert "execution_accuracy" not in case["assertions"]
    assert "Gold SQL failed" in case["error"]

    from tests.evals.dashboard import summarize

    stats = summarize(json.loads(output.read_text()))
    assert (stats["passed"], stats["cases"], stats["unscorable"]) == (0, 0, 1)


def test_dashboard_keeps_partial_run_out_of_complete_group():
    from tests.evals.dashboard import case_table, group_runs

    def report(names, timestamp):
        return {"run": {"dataset": "bird", "model": "m", "commit": "c", "api": "responses", "provider": "openai",
                        "case_fingerprint": "full-suite", "timestamp": timestamp},
                "cases": [{"name": name, "turns": [], "usage": None, "assertions": {"execution_accuracy": True},
                           "duration": 1, "error": None} for name in names],
                "failures": []}

    groups = group_runs({"complete": report(["a", "b"], "2"), "partial": report(["a"], "1")}, "bird")

    assert len(groups) == 2
    assert sorted(case_table(groups)["Case"]) == ["a", "a", "b"]


def test_bird_scorer_times_out_inside_the_child(tmp_path):
    path = tmp_path / "slow.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE t (x INTEGER)")
    slow = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n"
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        execute_read_only(path, slow, timeout=1)
    # The child enforces the deadline, so the startup grace is not spent waiting.
    assert time.monotonic() - started < 30


@register_subject("fake")
class FakeSubject:
    """A non-Lumen subject: a pydantic-ai agent whose answer becomes the turn's single row."""

    supports_architecture = False
    requires_api_key = False
    sql = "SELECT SUM(amount) AS total FROM sales"

    def __init__(self, surface: str = "v1"):
        self.surface = surface

    def build_model(self, model, *, provider="openai", api_key=None, api=None, instructions=""):
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("35")]), model_name=model)

    def fingerprint(self):
        return f"fake-{self.surface}"

    async def run(self, inputs, model, source, documents=None):
        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        turns, artifacts = [], []
        for prompt in inputs.prompts:
            result = await Agent(model).run(prompt)
            turns.append(Turn(prompt, "success", f"The total is {result.output}.", [], [],
                              self.sql, ["total"], [[int(result.output)]], 1, []))
            artifacts.append(TurnArtifacts(graph="nodes: []\n",
                                           messages=ModelMessagesTypeAdapter.dump_python(result.all_messages(), mode="json")))
        return Output(turns, artifacts=artifacts)


@register_subject("fake-bird")
class FakeBirdSubject(FakeSubject):
    sql = "SELECT SUM(spent) FROM budget WHERE category = 'Food'"


class HangingSubject(FakeSubject):
    async def run(self, inputs, model, source, documents=None):
        await asyncio.sleep(60)


FAKE_DATASET = Dataset[Inputs, Output, Expected](name="fake_subject", cases=[
    Case(name="outcome", inputs=Inputs(["What is the total amount overall?"]),
         metadata=Expected(rows=[[35]], sql_contains=["sum"], answer_contains="35")),
    Case(name="mixed", inputs=Inputs(["What is the total amount overall?"]),
         metadata=Expected(actors=["SQLAgent"], tool_calls=["run_exploration_sql"], rows=[[35]])),
    Case(name="architecture_only", inputs=Inputs(["What is the total amount overall?"]),
         metadata=Expected(actors=["SQLAgent"], planned_actors=["SQLAgent"])),
    Case(name="wrong_rows", inputs=Inputs(["What is the total amount overall?"]),
         metadata=Expected(rows=[[34]])),
], evaluators=[CheckResult()])


def _student_club(tmp_path):
    db_dir = tmp_path / "dev_databases" / "student_club"
    db_dir.mkdir(parents=True)
    path = db_dir / "student_club.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE budget (category TEXT, spent INTEGER)")
        connection.executemany("INSERT INTO budget VALUES (?, ?)", [("Food", 10), ("Food", 20), ("Other", 5)])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{
        "question_id": 1380, "db_id": "student_club", "question": "Total spent for food?",
        "evidence": "", "SQL": "SELECT SUM(spent) FROM budget WHERE category = 'Food'", "difficulty": "simple",
    }]))
    return questions, bird_dataset(questions, tmp_path, (1380,))


def test_every_expectation_is_tagged():
    from dataclasses import fields

    assert {item.metadata["kind"] for item in fields(Expected)} == {"outcome", "architecture"}
    assert all(item.metadata["check"] for item in fields(Expected))
    assert not ARCHITECTURE_CHECKS & OUTCOME_CHECKS
    assert Expected().expectation_kinds() == set()
    assert Expected(actors=["SQLAgent"], view_type="VegaLiteEditor").expectation_kinds() == {"architecture"}
    assert Expected(actors=["SQLAgent"], rows=[[1]]).expectation_kinds() == {"architecture", "outcome"}
    assert Expected(status="error").expectation_kinds() == {"outcome"}


@pytest.mark.asyncio
async def test_every_emitted_check_is_classified():
    expected = Expected(
        actors=["SQLAgent"], rows=[[1]], view_type="VegaLiteEditor", columns=["x"], answer_contains="a", listing_contains="l",
        chart_mark="bar", chart_fields={"nominal": "x"}, tool_calls=["t"], tool_arguments={"t": {}}, tool_result_contains={"t": []},
        turn_rows=[[[1]]], turn_tables=["t"], sql_contains=["select"], turn_planner_actors=[["SQLAgent"]],
        forbidden_actors=["ChatAgent"], follow_up_types=["new"], planned_actors=["SQLAgent"], document_contains=["d"],
        validation_correct=True, error_contains="e",
    )
    dataset = Dataset[Inputs, Output, Expected](name="all", cases=[Case(name="all", inputs=Inputs(["x"]), metadata=expected)],
                                                evaluators=[CheckResult()])
    report = await dataset.evaluate(lambda inputs: Output([Turn("x", "success", "a", [], [], "select", ["x"], [[1]], 1, [])]))
    emitted = set(report.cases[0].assertions)
    assert emitted - {"execution_accuracy"} <= ARCHITECTURE_CHECKS | OUTCOME_CHECKS
    assert {item for item in ARCHITECTURE_CHECKS | OUTCOME_CHECKS} - emitted == {"execution_accuracy"}

    class NewCheck(CheckResult):
        def _checks(self, ctx, expected):
            return {"unclassified": True}

    report = await Dataset(name="new", cases=[Case(name="new", inputs=Inputs(["x"]))], evaluators=[NewCheck()]).evaluate(
        lambda inputs: Output([]))
    assert "Unclassified checks: unclassified" in report.cases[0].evaluator_failures[0].error_message


@pytest.mark.asyncio
async def test_chart_marks_count_as_output():
    dataset = Dataset[Inputs, Output, Expected](
        name="chart", cases=[Case(name="chart", inputs=Inputs(["Plot"]), metadata=Expected(chart_mark="bar"))],
        evaluators=[CheckResult(scoring="outcome", supports_architecture=False)],
    )
    turn = Turn("Plot", "success", None, [], [], None, None, None, None, [], chart_marks=["bar"])
    report = await dataset.evaluate(lambda inputs: Output([turn]))
    assert all(result.value for result in report.cases[0].assertions.values())
    assert report.cases[0].assertions["output"].value is True


def test_local_cases_not_applicable_under_outcome_scoring():
    not_applicable = {case.name for case in DATASET.cases if case.metadata.expectation_kinds() == {"architecture"}}
    assert not_applicable == {"validation_incomplete"}


def test_subjects_resolve_by_name_and_import_path():
    assert isinstance(resolve_subject(), LumenSubject)
    assert isinstance(resolve_subject("lumen", ui_params={"log_level": "INFO"}), Subject)
    assert isinstance(resolve_subject("fake"), FakeSubject)
    spec = resolve_subject("lumen.ai.evals.harness:LumenSubject")
    assert spec.name == "lumen" and subject_spec(spec) == "lumen.ai.evals.harness:LumenSubject"
    assert run_settings(subject=spec.name) == run_settings(subject=resolve_subject("lumen").name) == {}
    assert resolve_subject("tests.evals.test_harness:FakeSubject", surface="v2").fingerprint() == "fake-v2"


def test_resolve_subject_errors():
    with pytest.raises(ValueError, match="Unknown eval subject"):
        resolve_subject("missing")
    with pytest.raises(ModuleNotFoundError):
        resolve_subject("tests.evals.no_such_module:Subject")
    with pytest.raises(TypeError, match="missing: supports_architecture, build_model, fingerprint, run"):
        resolve_subject("tests.evals.test_harness:NotASubject")
    assert "tests.evals.test_harness:NotASubject" not in SUBJECTS
    assert not hasattr(importlib.import_module("tests.evals.test_harness").NotASubject, "name")
    with pytest.raises(TypeError, match="does not support the option\\(s\\) sql_cleanup"):
        resolve_subject("fake", sql_cleanup=False)


def test_unnamed_spec_subject_takes_spec_as_name():
    subject = resolve_subject("tests.evals.test_harness:UnnamedSubject")
    assert subject.name == "tests.evals.test_harness:UnnamedSubject"
    assert SUBJECTS[subject.name] is type(subject)
    assert not hasattr(type(subject), "name")


class NotASubject:
    pass


class UnnamedSubject:
    supports_architecture = False

    def build_model(self, model, **kwargs):
        return model

    def fingerprint(self):
        return None

    async def run(self, inputs, model, source, documents=None):
        return Output([])


def test_subject_fingerprint_is_optional_component():
    case = Case(name="test", inputs=Inputs(["Count sales"]))
    baseline = case_fingerprint([case])
    assert case_fingerprint([case], subject_fingerprint=None) == baseline
    assert case_fingerprint([case], subject_fingerprint="fake-v1") not in (baseline, case_fingerprint([case], subject_fingerprint="fake-v2"))


@pytest.mark.asyncio
async def test_fake_subject_scored_end_to_end_on_outcomes(tmp_path):
    """A non-Lumen subject with a pydantic-ai FunctionModel is scored offline on outcome checks only."""
    subject = resolve_subject("fake")
    model = subject.build_model("canned")
    output = tmp_path / "fake.json"
    report = await evaluate(model, FAKE_DATASET, source, output, subject=subject, scoring="outcome", instructions="Be exact.")
    cases = {case.name: case for case in report.cases}

    assert not report.failures
    assert {name: result.value for name, result in cases["outcome"].assertions.items()} == {
        "all_turns_completed": True, "status": True, "output": True, "rows": True, "answer": True, "sql_contains": True,
    }
    assert set(cases["mixed"].assertions) == {"all_turns_completed", "status", "output", "rows"}
    assert not cases["mixed"].labels
    assert not cases["architecture_only"].assertions
    assert cases["architecture_only"].labels[APPLICABILITY].value == NOT_APPLICABLE
    assert cases["wrong_rows"].assertions["rows"].value is False

    saved = json.loads(output.read_text())
    settings = {"subject": "fake", "scoring": "outcome"}
    assert saved["run"]["subject"] == "fake"
    assert saved["run"]["scoring"] == "outcome"
    assert saved["run"]["settings"] == settings
    assert saved["run"]["model"] == "canned"
    assert saved["run"]["suite_instructions"] == "Be exact."
    assert saved["run"]["subject_fingerprint"] == "fake-v1"
    assert saved["run"]["case_fingerprint"] == case_fingerprint(
        FAKE_DATASET.cases, instructions="Be exact.", settings=settings, subject_fingerprint="fake-v1")
    by_name = {case["name"]: case for case in saved["cases"]}
    assert by_name["architecture_only"]["applicable"] is False
    assert by_name["architecture_only"]["assertions"] == {}
    assert by_name["outcome"]["applicable"] is True
    assert by_name["outcome"]["turns"][0]["rows"] == [[35]]
    assert "artifacts" not in by_name["outcome"]
    sidecar = [json.loads(line) for line in artifacts_path(output).read_text().splitlines()]
    assert [line["name"] for line in sidecar] == [case.name for case in FAKE_DATASET.cases]
    assert sidecar[0]["artifacts"][0]["graph"] == "nodes: []\n"
    assert sidecar[0]["artifacts"][0]["messages"][-1]["parts"][0]["content"] == "35"

    from tests.evals.dashboard import case_table, group_runs, summarize

    stats = summarize(saved)
    assert (stats["passed"], stats["cases"], stats["not_applicable"]) == (2, 3, 1)
    table = case_table(group_runs({"fake": saved}, "fake_subject"))
    assert table.set_index("Case").loc["architecture_only", "Checks"] == "n/a"


@pytest.mark.asyncio
async def test_subject_without_architecture_records_outcome_scoring(tmp_path):
    subject = resolve_subject("fake")
    output = tmp_path / "fake.json"
    report = await evaluate(subject.build_model("canned"), FAKE_DATASET, source, output, subject=subject, scoring="all")
    saved = json.loads(output.read_text())["run"]
    assert (saved["scoring"], saved["settings"]) == ("outcome", {"subject": "fake", "scoring": "outcome"})
    assert set(next(case for case in report.cases if case.name == "mixed").assertions) == {
        "all_turns_completed", "status", "output", "rows"}


@pytest.mark.asyncio
async def test_hanging_subject_records_timeout(tmp_path):
    output = tmp_path / "hang.json"
    report = await evaluate(None, FAKE_DATASET, source, output, "outcome", subject=HangingSubject(), prompt_timeout=0.1)
    saved = json.loads(output.read_text())
    assert saved["cases"][0]["turns"][0]["status"] == "timeout"
    assert report.cases[0].assertions["status"].value is False


@pytest.mark.asyncio
async def test_evaluate_rejects_ui_params_for_other_subjects():
    with pytest.raises(ValueError, match="default subject only"):
        await evaluate(None, FAKE_DATASET, source, subject=FakeSubject(), ui_params={"log_level": "INFO"})


@pytest.mark.asyncio
async def test_lumen_subject_sql_cleanup_option(monkeypatch):
    from lumen.ai.evals import harness

    created = []

    def capture(**params):
        created.append(params)
        return harness.TableListAgent(**{key: value for key, value in params.items() if key != "clean_data"})

    monkeypatch.setattr(harness, "SQLAgent", capture)
    subject = resolve_subject("lumen", sql_cleanup=False)
    await subject.run(Inputs(["Count"], agents=("SQLAgent",)), MockLLM(), source(Inputs([])))
    await LumenSubject().run(Inputs(["Count"], agents=("SQLAgent",)), MockLLM(), source(Inputs([])))
    assert created[0]["clean_data"] is False
    assert "clean_data" not in created[1]


@pytest.mark.asyncio
async def test_lumen_subject_reports_architecture_checks_as_diagnostics():
    """Under outcome scoring a subject that supports architecture checks keeps them as labels."""
    dataset = Dataset[Inputs, Output, Expected](
        name="diagnostics", cases=[
            Case(name="mixed", inputs=Inputs(["Total"]), metadata=Expected(actors=["SQLAgent"], rows=[[35]])),
            Case(name="architecture_only", inputs=Inputs(["Total"]), metadata=Expected(actors=["SQLAgent"])),
        ],
        evaluators=[CheckResult(scoring="outcome")],
    )
    turn = Turn("Total", "success", None, ["ChatAgent"], ["success"], None, ["total"], [[35]], 1, [])
    report = await dataset.evaluate(lambda inputs: Output([turn]))
    cases = {case.name: case for case in report.cases}

    assert set(cases["mixed"].assertions) == {"all_turns_completed", "status", "output", "rows"}
    assert cases["mixed"].labels["diagnostic:actors"].value == "fail"
    assert cases["mixed"].labels["diagnostic:validation"].value == "pass"
    assert not cases["architecture_only"].assertions
    assert cases["architecture_only"].labels[APPLICABILITY].value == NOT_APPLICABLE
    assert cases["architecture_only"].labels["diagnostic:actors"].value == "fail"


@pytest.mark.asyncio
async def test_default_scoring_keeps_architecture_assertions():
    dataset = Dataset[Inputs, Output, Expected](
        name="default", cases=[Case(name="architecture_only", inputs=Inputs(["Total"]), metadata=Expected(actors=["SQLAgent"]))],
        evaluators=[CheckResult()],
    )
    turn = Turn("Total", "success", None, ["SQLAgent"], ["success"], None, ["total"], [[35]], 1, [])
    report = await dataset.evaluate(lambda inputs: Output([turn]))
    case = report.cases[0]
    assert set(case.assertions) == {"all_turns_completed", "status", "output", "validation", "actors"}
    assert not case.labels


def test_dashboard_reads_results_without_subject_fields():
    from tests.evals.dashboard import group_runs, run_table, summarize

    def report(subject=None, scoring=None, applicable=None):
        run = {"dataset": "local", "model": "m", "commit": "c", "api": "responses", "provider": "openai",
               "case_fingerprint": "f", "timestamp": "1"}
        if subject:
            run.update(subject=subject, scoring=scoring)
        case = {"name": "a", "turns": [], "usage": None, "assertions": {"rows": True}, "duration": 1, "error": None}
        if applicable is not None:
            case |= {"applicable": applicable, "diagnostics": {}}
        return {"run": run, "cases": [case], "failures": []}

    old, new = report(), report("lumen", "all", True)
    assert summarize(old) == summarize(new)
    assert len(group_runs({"old": old, "new": new}, "local")) == 1
    groups = group_runs({"old": old, "fake": report("fake", "outcome", False)}, "local")
    assert len(groups) == 2
    table = run_table(groups).set_index("Run")
    assert table.loc["fake", "Subject"] == "fake"
    assert (table.loc["fake", "Cases"], table.loc["fake", "N/A"]) == (0, 1)
    assert (table.loc["old", "Subject"], table.loc["old", "Scoring"]) == ("lumen", "all")


def test_parallel_bird_runner_records_subject_and_scoring(tmp_path):
    from tests.evals.__main__ import run_all_bird

    dataset = Dataset(name="bird_test", cases=[Case(name="case_1", inputs=Inputs(["1"]))])
    questions = tmp_path / "questions.json"
    questions.write_text("[]")
    default, fake = tmp_path / "default.json", tmp_path / "fake.json"
    for output, settings in ((default, None), (fake, {"subject": "fake"})):
        run_all_bird(dataset, output, questions, tmp_path, "test", "openai", "test-key", False, 1,
                     worker=simulated_bird_case_process, case_timeout=60, settings=settings)
    default, fake = json.loads(default.read_text())["run"], json.loads(fake.read_text())["run"]
    assert (default["subject"], default["scoring"], default["subject_fingerprint"]) == ("lumen", "all", None)
    assert (fake["subject"], fake["scoring"], fake["subject_fingerprint"]) == ("fake", "outcome", "fake-v1")
    assert fake["settings"] == {"subject": "fake", "scoring": "outcome"}
    assert default["case_fingerprint"] != fake["case_fingerprint"]


def test_bird_worker_resolves_subject_by_spec(tmp_path):
    from tests.evals.bird_runner import run_bird_case

    _, dataset = _student_club(tmp_path)
    subject = resolve_subject("fake-bird")
    settings = run_settings(subject=subject.name, scoring="outcome")
    args = (1, dataset.cases[0], tmp_path, "canned", "openai", None, tmp_path / "bird.json", "Exact rows.", "v1",
            False, settings, None, subject_spec(subject))
    result = run_bird_case(args)
    case = result["cases"][0]
    assert result["run"]["subject"] == "fake-bird"
    assert result["run"]["model"] == "canned"
    assert result["run"]["suite_instructions"] == "Exact rows."
    assert result["run"]["settings"] == {"subject": "fake-bird", "scoring": "outcome"}
    assert case["assertions"] == {"all_turns_completed": True, "status": True, "output": True, "execution_accuracy": True}
    assert result["artifacts"][0]["name"] == case["name"]
    assert not list(tmp_path.glob("bird.1.current*"))

    with pytest.raises(TypeError, match="sql_cleanup"):
        run_bird_case(args[:9] + (True,) + args[10:])


def _run_cli(monkeypatch, *argv):
    from tests.evals.__main__ import main

    monkeypatch.setattr("sys.argv", ["tests.evals", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def test_cli_rejects_invalid_subject_options(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _run_cli(monkeypatch, "--subject", "missing") == 2
    assert "Unknown eval subject 'missing'" in capsys.readouterr().err
    assert _run_cli(monkeypatch, "--subject", "fake", "--no-sql-cleanup") == 2
    assert "does not support the option(s) sql_cleanup" in capsys.readouterr().err
    assert _run_cli(monkeypatch, "--case", "sales_total") == 2
    assert "API key is required" in capsys.readouterr().err


def test_cli_runs_offline_subject_without_key(monkeypatch, tmp_path):
    from tests.evals.__main__ import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "cli.json"
    monkeypatch.setattr("sys.argv", ["tests.evals", "--subject", "fake", "--case", "sales_total", "--output", str(output)])
    main()
    saved = json.loads(output.read_text())
    assert (saved["run"]["subject"], saved["run"]["scoring"]) == ("fake", "outcome")
    assert saved["cases"][0]["assertions"]["rows"] is True
    assert "actors" not in saved["cases"][0]["assertions"]


@pytest.mark.parametrize(("error", "message"), [
    (TimeoutError(), "Case exceeded the 240s evaluation limit"),
    (OSError("disk full"), "Harness failed: OSError: disk full"),
])
def test_bird_case_labels_harness_errors(tmp_path, monkeypatch, error, message):
    from tests.evals import bird_runner

    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(bird_runner, "evaluate", fail)
    case = Case(name="q1", inputs=None)
    result = bird_runner.run_bird_case((1, case, tmp_path, "m", "openai", "test-key", tmp_path / "out.json",
                                        "", "", False, {}, None, "fake"))

    assert result["cases"][0]["error"] == message
    assert result["cases"][0]["assertions"] == {}


def test_evals_without_extra_names_it(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "pydantic_evals", None)
    monkeypatch.delitem(sys.modules, "lumen.ai.evals")
    with pytest.raises(ImportError, match=r"pip install \"lumen\[evals\]\""):
        importlib.import_module("lumen.ai.evals")
