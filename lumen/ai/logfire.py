import asyncio
import json
import os

from collections import defaultdict
from io import StringIO

import pandas as pd

from logfire.query_client import AsyncLogfireQueryClient

TRACE_ID_QUERY = """
SELECT trace_id
FROM RECORDS
WHERE tags @> {tag_array}
ORDER BY created_at DESC
LIMIT 1
"""

DATA_QUERY = """
SELECT *
FROM RECORDS
WHERE trace_id = '{trace_id}'
"""

PRICING = {"gpt-4.1-mini": {"input": 0.4, "output": 1.6}}


def sanitize_tag(tag: str) -> str:
    """
    Sanitize tags to prevent SQL injection, but very basic
    """
    # Remove/escape single quotes and limit length
    return str(tag).replace("'", "''")[:100]


async def read_trace(trace_id: str | None = None, read_token: str | None = None, tags: list[str] | None = None):
    """
    Parse observability data to extract aggregated span information.
    """
    read_token = read_token or os.getenv("LOGFIRE_READ_TOKEN")

    # Use empty array when tags is None - matches all records
    if tags is None:
        tags = []

    sanitized_tags = [sanitize_tag(tag) for tag in tags if tag and str(tag).strip()]
    tag_array = "ARRAY[" + ", ".join(f"'{tag}'" for tag in sanitized_tags) + "]"

    async with AsyncLogfireQueryClient(read_token=read_token) as client:
        if trace_id is None:
            trace_df = (await client.query_json(sql=TRACE_ID_QUERY.format(tag_array=tag_array)))
            if len(trace_df) == 0:
                return None
            trace_id = trace_df["columns"][0]["values"][0]
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
        request_data = attributes.get("request_data", {})
        response_data = attributes.get("response_data", {})
        messages = attributes.get("messages", attributes.get("contents", request_data.get("messages", [])))

        span_data[span_id] = {
            "trace_id": row["trace_id"],
            "span_id": span_id,
            "name": row["span_name"],
            "duration": row["duration"],
            "start_timestamp": row["start_timestamp"],
            "parent_id": parent_id,
            "model": model,
            "input_tokens": input_tokens,  # TODO: include cached tokens from response_data
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "messages": messages,
            "request_data": request_data,
            "response_data": response_data,
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
                "input_tokens": total_input if total_input > 0 else None,
                "output_tokens": total_output if total_output > 0 else None,
                "total_tokens": total_tokens,
                "input_cost": total_input_cost,
                "output_cost": total_output_cost,
                "total_cost": total_cost,
                "num_spans": len(descendants),
                "levels_deep": get_depth(span_id),
                "messages": data["messages"],
                "request_data": data["request_data"],
                "response_data": data["response_data"],
            }
        )

    result_df = pd.DataFrame(result_data).sort_values("start_timestamp")
    result_df.index = range(len(result_df))
    return result_df


if __name__ == "__main__":
    df = asyncio.run(read_trace("0199117540d7bb42d334dfb4e9ba4848"))
    df.to_csv("logfire_output.csv", index=False)
