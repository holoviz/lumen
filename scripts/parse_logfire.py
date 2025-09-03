import asyncio
import json
import os

from collections import defaultdict
from io import StringIO

import pandas as pd

from logfire.query_client import AsyncLogfireQueryClient

READ_TOKEN = os.getenv("LOGFIRE_READ_TOKEN")

TRACE_ID_QUERY = """
SELECT trace_id
FROM RECORDS
ORDER BY created_at DESC
LIMIT 1
"""

DATA_QUERY = """
SELECT *
FROM RECORDS
WHERE trace_id = '{trace_id}'
"""

PRICING = {"gpt-4.1-mini": {"input": 0.4, "output": 1.6}}


async def parse_logfire_data(trace_id: str | None = None):
    """
    Parse observability data to extract aggregated span information.
    """
    if trace_id is None:
        async with AsyncLogfireQueryClient(read_token=READ_TOKEN) as client:
            trace_id = (await client.query_json(sql=TRACE_ID_QUERY))["columns"][0]["values"][0]

    async with AsyncLogfireQueryClient(read_token=READ_TOKEN) as client:
        df = pd.read_csv(StringIO(await client.query_csv(sql=DATA_QUERY.format(trace_id=trace_id))))

    # Build parent-child relationships
    parent_to_children = defaultdict(list)
    span_data = {}

    # Collect span data and build hierarchy
    for _, row in df.iterrows():
        span_id = row["span_id"]
        parent_id = row["parent_span_id"]
        model = row.get("_lf_attributes/gen_ai.request.model") or row.get("_lf_attributes/gen_ai.response.model")

        input_tokens = row["_lf_attributes/gen_ai.usage.input_tokens"] if pd.notna(row["_lf_attributes/gen_ai.usage.input_tokens"]) else 0
        output_tokens = row["_lf_attributes/gen_ai.usage.output_tokens"] if pd.notna(row["_lf_attributes/gen_ai.usage.output_tokens"]) else 0

        if model is not None and model in PRICING:
            input_cost = PRICING[model]["input"] * input_tokens / 1e6
            output_cost = PRICING[model]["output"] * output_tokens / 1e6
            total_cost = input_cost + output_cost
        else:
            input_cost = output_cost = total_cost = 0

        attributes = json.loads(row["attributes_reduced"])
        request_data = attributes.get("request_data", [])
        messages = attributes.get("messages", attributes.get("contents", []))

        span_data[span_id] = {
            "trace_id": row["trace_id"],
            "span_id": span_id,
            "name": row["span_name"],
            "duration": row["duration"],
            "start_timestamp": row["start_timestamp"],
            "parent_id": parent_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "request_data": request_data,
            "messages": messages,
        }

        if pd.notna(parent_id):
            parent_to_children[parent_id].append(span_id)

    # Calculate hierarchy depth
    def get_depth(span_id, visited=None):
        if visited is None:
            visited = set()
        if span_id in visited:
            return 0
        visited = visited | {span_id}
        parent_id = span_data[span_id]["parent_id"]
        return 0 if pd.isna(parent_id) else 1 + get_depth(parent_id, visited)

    # Get all descendants recursively
    def get_descendants(span_id):
        descendants = set()
        for child_id in parent_to_children.get(span_id, []):
            descendants.add(child_id)
            descendants.update(get_descendants(child_id))
        return descendants

    # Build result
    result_data = []
    for span_id, data in span_data.items():
        descendants = get_descendants(span_id)

        # Aggregate tokens from self and all descendants
        total_input = data["input_tokens"] + sum(span_data[d]["input_tokens"] for d in descendants)
        total_output = data["output_tokens"] + sum(span_data[d]["output_tokens"] for d in descendants)
        total_tokens = total_input + total_output if total_input + total_output > 0 else None

        total_input_cost = data["input_cost"] + sum(span_data[d]["input_cost"] for d in descendants)
        total_output_cost = data["output_cost"] + sum(span_data[d]["output_cost"] for d in descendants)
        total_cost = total_input_cost + total_output_cost if total_input_cost is not None and total_output_cost is not None else None

        result_data.append(
            {
                "trace_id": data["trace_id"],
                "span_id": data["span_id"],
                "start_timestamp": pd.to_datetime(data["start_timestamp"]),
                "parent_id": data["parent_id"],
                "name": data["name"],
                "model": data["model"],
                "duration": data["duration"],
                "input": total_input if total_input > 0 else None,
                "output": total_output if total_output > 0 else None,
                "total_tokens": total_tokens,
                "input_cost": total_input_cost,
                "output_cost": total_output_cost,
                "total_cost": total_cost,
                "num_spans": len(descendants),
                "levels_deep": get_depth(span_id),
                "request_data": data["request_data"],
                "messages": data["messages"],
            }
        )

    result_df = pd.DataFrame(result_data).sort_values("start_timestamp")
    result_df.index = range(len(result_df))
    return result_df


if __name__ == "__main__":
    df = asyncio.run(parse_logfire_data("0199117540d7bb42d334dfb4e9ba4848"))
    df.to_csv("logfire_output.csv", index=False)
    print(df)
