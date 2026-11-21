"""Create a richer demo DuckDB database for lumen-mcp demos and recordings.

~5,000 orders across 2025 with regions, product categories, customer segments, prices, discounts,
seasonality + slight growth, a few bulk-order outliers, and a small fraction of missing segments.
Deterministic (seeded) so a recorded walkthrough is reproducible.

Usage: python examples/make_demo_db.py [path] [n_rows]
"""

from __future__ import annotations

import math
import os
import random
import sys
from datetime import date, timedelta

import duckdb

# region -> demand multiplier; category -> base unit price
_REGIONS = {"West": 1.4, "East": 1.2, "North": 0.9, "South": 0.8, "Central": 1.0}
_CATEGORIES = {"Widgets": 40, "Gadgets": 65, "Gizmos": 25, "Doohickeys": 90, "Sprockets": 15}
_SEGMENTS = ["Consumer", "Enterprise", "SMB"]


def _row(order_id: int, start: date) -> tuple:
    day = start + timedelta(days=random.randint(0, 364))
    month = day.month
    # seasonality (holiday peak) + gentle growth through the year
    season = 1 + 0.35 * math.sin((month - 3) / 12 * 2 * math.pi) + (0.25 if month in (11, 12) else 0)
    growth = 1 + 0.02 * (month - 1)

    region = random.choices(list(_REGIONS), weights=list(_REGIONS.values()))[0]
    category = random.choice(list(_CATEGORIES))
    unit_price = round(_CATEGORIES[category] * random.uniform(0.8, 1.4), 2)
    quantity = max(1, int(random.gauss(6, 3) * _REGIONS[region] * season * growth))
    if random.random() < 0.01:  # occasional bulk-order outlier
        quantity *= random.randint(5, 12)
    discount = random.choice([0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15])
    segment = None if random.random() < 0.02 else random.choice(_SEGMENTS)  # a few missing segments
    revenue = round(quantity * unit_price * (1 - discount), 2)
    return (order_id, day, region, category, segment, quantity, unit_price, discount, revenue)


def main(path: str = "demo.db", n_rows: int = 5000, seed: int = 42) -> str:
    random.seed(seed)
    if os.path.exists(path):
        os.remove(path)
    start = date(2025, 1, 1)
    rows = [_row(i, start) for i in range(1, n_rows + 1)]

    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE sales (
            order_id INTEGER, date DATE, region VARCHAR, category VARCHAR, segment VARCHAR,
            quantity INTEGER, unit_price DOUBLE, discount DOUBLE, revenue DOUBLE
        )
        """
    )
    con.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()
    return os.path.abspath(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo.db"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    print(main(out, count))
