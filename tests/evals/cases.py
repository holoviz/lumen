"""Local behavior and opt-in BIRD Mini-Dev evaluation cases."""

import json

from pathlib import Path

from pydantic_evals import Case, Dataset

from lumen.ai.evals.bird import BirdExecution, database_path, execute_read_only
from lumen.ai.evals.harness import (
    CheckResult, Expected, Inputs, Output,
)
from lumen.ai.schemas import DocumentChunk
from lumen.sources.duckdb import DuckDBSource


def source(inputs: Inputs) -> DuckDBSource:
    fixture = inputs.fixture
    if fixture == "sales":
        tables = {"sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20), ('A', 5)) AS t(category, amount)"}
    elif fixture == "commerce":
        tables = {
            "customers": "SELECT * FROM (VALUES (1, 'Ada', 'east'), (2, 'Bo', 'west'), (3, 'Cy', 'east'), (4, 'Dee', 'north')) AS t(customer_id, customer, region)",
            "orders": "SELECT * FROM (VALUES (101, 1, 30, 'paid'), (102, 1, 20, 'paid'), (103, 2, 40, 'paid'), (104, 2, 10, 'cancelled'), (105, 3, 15, 'paid')) AS t(order_id, customer_id, amount, status)",
        }
    elif fixture == "documents":
        tables = {"document_index": "SELECT 1 AS indexed"}
    else:
        raise ValueError(f"Unknown eval fixture: {fixture}")
    return DuckDBSource(uri=":memory:", tables=tables)


def documents(inputs: Inputs) -> list[DocumentChunk] | None:
    if inputs.fixture != "documents":
        return None
    return [
        DocumentChunk("returns.md", "Customers may return unused items within 30 days with a receipt.", 0.99),
        DocumentChunk("shipping.md", "Standard shipping takes 3-5 business days.", 0.95),
    ]


DATASET = Dataset[Inputs, Output, Expected](name="lumen_ai_behavior_v3", cases=[
    Case(name="sales_total", inputs=Inputs(["What is the total amount overall?"]),
         metadata=Expected(actors=["SQLAgent"], rows=[[35]])),
    Case(name="sales_by_category", inputs=Inputs(["What is the total amount by category?"]),
         metadata=Expected(actors=["SQLAgent"], rows=[["A", 15], ["B", 20]])),
    Case(name="sales_followup", inputs=Inputs(["What is the total amount by category?", "What is the total amount overall?"]),
         metadata=Expected(answer_contains="35")),
    Case(name="sales_data", inputs=Inputs(["What data is available?"]),
         metadata=Expected(answer_contains="sales")),
    Case(name="sales_chart", inputs=Inputs(["Plot the total amount by category as a bar chart."]),
         metadata=Expected(view_type="VegaLiteEditor")),
    Case(name="agent_sql_total", inputs=Inputs(["Calculate the overall sum of amount in sales."], agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[[35]])),
    Case(name="agent_sql_filter", inputs=Inputs(["Sum amount for category A in sales."], agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[[15]])),
    Case(name="agent_sql_group", inputs=Inputs(["Sum amount by category in sales."], agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[["A", 15], ["B", 20]])),
    Case(name="agent_chat", inputs=Inputs(["Which table is available? Answer with its name."], agents=("ChatAgent",)),
         metadata=Expected(actors=["ChatAgent"], answer_contains="sales")),
    Case(name="agent_table_list", inputs=Inputs(["List available tables."], agents=("TableListAgent",)),
         metadata=Expected(actors=["TableListAgent"], listing_contains="1 table")),
    Case(name="agent_sql_chat", inputs=Inputs(["Calculate the sum of amount by category, then explain the result."], agents=("SQLAgent", "ChatAgent")),
         metadata=Expected(actors=["SQLAgent", "ChatAgent"], rows=[["A", 15], ["B", 20]], answer_contains="20")),
    Case(name="agent_vega_bar", inputs=Inputs(["Plot the total amount by category as a bar chart."], agents=("VegaLiteAgent",),
                                       query="SELECT category, SUM(amount) AS total FROM sales GROUP BY category"),
         metadata=Expected(actors=["VegaLiteAgent"], view_type="VegaLiteEditor", chart_mark="bar",
                           chart_fields={"nominal": "category", "quantitative": "total"})),
    Case(name="commerce_cte_paid", inputs=Inputs([
        "Use a WITH CTE to total paid orders per customer. Join customers and orders, include customers with no paid orders, and return customer and total_amount ordered by customer."
    ], fixture="commerce", agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[["Ada", 50], ["Bo", 40], ["Cy", 15], ["Dee", 0]], sql_contains=["with", "join"])),
    Case(name="commerce_cte_rank", inputs=Inputs([
        "Use a WITH CTE to sum paid order amounts per region across customers and orders. Return the region with the highest paid total and its total_amount."
    ], fixture="commerce", agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[["east", 65]], sql_contains=["with", "join"])),
    Case(name="commerce_explore", inputs=Inputs([
        "Before writing the final query, call run_exploration_sql to inspect the orders status values. Then return the count of paid orders as paid_count."
    ], fixture="commerce", agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[[4]], tool_calls=["run_exploration_sql"],
                           tool_result_contains={"run_exploration_sql": ["paid", "cancelled"]})),
    Case(name="commerce_schemas", inputs=Inputs([
        "Call load_table_schemas for customers and orders before writing the final query. Then return every region and the count of paid orders as paid_count, including regions with zero paid orders."
    ], fixture="commerce", agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[["east", 3], ["north", 0], ["west", 1]], tool_calls=["load_table_schemas"],
                           tool_result_contains={"load_table_schemas": ["customer_id", "status"]})),
    Case(name="commerce_switch_table", inputs=Inputs([
        "Count customers in the east region from customers, call the result total_count.",
        "Now switch to orders: count paid orders, call the result total_count."
    ], fixture="commerce", agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[[4]], turn_rows=[[[2]], [[4]]], turn_tables=["customers", "orders"])),
    Case(name="commerce_followup_filter", inputs=Inputs([
        "Sum paid order amounts by region using customers and orders, return region and total_amount.",
        "Only show the west region, returning region and total_amount."
    ], fixture="commerce", agents=("SQLAgent",)),
         metadata=Expected(actors=["SQLAgent"], rows=[["west", 40]], turn_rows=[[["east", 65], ["west", 40]], [["west", 40]]])),
    Case(name="planner_commerce_join", inputs=Inputs([
        "For every customer, show their name and total paid order amount, including customers with no paid orders."
    ], fixture="commerce"),
         metadata=Expected(actors=["SQLAgent"], rows=[["Ada", 50], ["Bo", 40], ["Cy", 15], ["Dee", 0]], planned_actors=["SQLAgent"])),
    Case(name="planner_expand_scope", inputs=Inputs([
        "Show the highest paid order amount and its order ID.", "Now show every paid order, with order ID and amount."
    ], fixture="commerce"),
         metadata=Expected(actors=["SQLAgent"], turn_rows=[[[103, 40]], [[101, 30], [102, 20], [103, 40], [105, 15]]],
                           follow_up_types=["new", "new"], planned_actors=["SQLAgent"])),
    Case(name="planner_reuse_for_chart", inputs=Inputs([
        "Show paid order totals by customer.", "Plot those totals as a bar chart without changing the data."
    ], fixture="commerce"),
         metadata=Expected(actors=["VegaLiteAgent"], view_type="VegaLiteEditor", chart_mark="bar",
                           forbidden_actors=["SQLAgent"], follow_up_types=["new", "direct"], planned_actors=["VegaLiteAgent"])),
    Case(name="document_list", inputs=Inputs(["List the available documents."], fixture="documents", agents=("DocumentListAgent",)),
         metadata=Expected(actors=["DocumentListAgent"], listing_contains="2 documents")),
    Case(name="document_summary", inputs=Inputs(["Summarize the return window in returns.md."], fixture="documents",
                                                agents=("DocumentSummarizerAgent",)),
         metadata=Expected(actors=["DocumentSummarizerAgent"], view_type="DocumentEditor", document_contains=["30 days"])),
    Case(name="validation_incomplete", inputs=Inputs(["Report paid totals by region and explain the west result."],
                                                    fixture="commerce", agents=("ValidationAgent",), seed_chat="East has a paid total of 65."),
         metadata=Expected(actors=["ValidationAgent"], validation_correct=False)),
    Case(name="commerce_no_results", inputs=Inputs(["Show order ID and amount for paid orders with amount greater than 1000."],
                                                   fixture="commerce", agents=("SQLAgent",)),
         metadata=Expected(answer_contains="no paid orders")),
], evaluators=[CheckResult()])


# BIRD Mini-Dev (CC BY-SA 4.0), https://huggingface.co/datasets/birdsql/bird_mini_dev
# Source revision: f65faf4ae3b638c1fa6df1d3370c8d92c8366301
QUESTION_IDS = (1361, 1378, 1352, 1340, 1375, 1457)
REVISION = "f65faf4ae3b638c1fa6df1d3370c8d92c8366301"
QUESTION_URL = (
    f"https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/{REVISION}/"
    "data/mini_dev_sqlite-00000-of-00001.json"
)


def bird_dataset(questions: Path, databases: Path, question_ids: tuple[int, ...] = QUESTION_IDS) -> Dataset:
    records = json.loads(questions.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected the Mini-Dev SQLite JSON array")
    by_id = {item["question_id"]: item for item in records}
    missing = set(question_ids) - by_id.keys()
    if missing:
        raise ValueError(f"Missing Mini-Dev question IDs: {sorted(missing)}")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("BIRD question IDs must be unique")
    cases = []
    for question_id in question_ids:
        item = by_id[question_id]
        db = database_path(databases, item["db_id"])
        execute_read_only(db, item["SQL"])
        question = item["question"]
        if item["evidence"]:
            question += f"\nDatabase evidence: {item['evidence']}"
        cases.append(Case(
            name=f"bird_{question_id}_{item['difficulty']}",
            inputs=Inputs([question], fixture=f"bird:{item['db_id']}", agents=("SQLAgent",)),
            metadata=Expected(actors=["SQLAgent"], gold_sql=item["SQL"]),
        ))
    return Dataset(name=f"bird_minidev_{REVISION[:8]}_{len(cases)}", cases=cases,
                   evaluators=[CheckResult(), BirdExecution(databases)])
