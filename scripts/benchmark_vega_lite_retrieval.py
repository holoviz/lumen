"""Run from a Lumen checkout with PYTHONPATH=. and the AI dependencies installed."""

import argparse
import asyncio
import json

from statistics import median
from time import perf_counter

from lumen.ai.agents.vega_lite import VegaLiteAgent

PROMPTS = [
    "Plot monthly revenue as a line chart.",
    "Show monthly website traffic as an area chart.",
    "Make a density plot of delivery times.",
    "Show sales by region in a bar chart with a threshold line and tooltips.",
    "Make a scatter plot with a brush selection and tooltip.",
    "Create a histogram with an interactive filter.",
    "Create a faceted heatmap with sorted categories.",
]


async def benchmark(store: str, repeats: int) -> None:
    agent = VegaLiteAgent(vector_store_path=store)
    await agent._get_doc_pages(PROMPTS[0])
    for prompt in PROMPTS:
        times = []
        for _ in range(repeats):
            start = perf_counter()
            pages = await agent._get_doc_pages(prompt)
            times.append((perf_counter() - start) * 1000)
        print(json.dumps({
            "prompt": prompt,
            "median_ms": round(median(times), 2),
            "sections": [
                [page["metadata"].get("page_title"), page["metadata"].get("section_title")]
                for page in pages
            ],
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", help="Path to the Vega-Lite NumPy documentation store")
    parser.add_argument("--repeats", type=int, choices=range(1, 101), default=5)
    args = parser.parse_args()
    asyncio.run(benchmark(args.store, args.repeats))
