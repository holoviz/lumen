"""Create a small sample DuckDB database for lumen-mcp demos.

Usage: python examples/make_sample_db.py [path]
"""

from __future__ import annotations

import os
import sys

import duckdb


def main(path: str = "sample.db") -> str:
    if os.path.exists(path):
        os.remove(path)
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE sales AS
        SELECT * FROM (VALUES
            ('West',  120.0, DATE '2026-01-03'),
            ('West',  200.0, DATE '2026-02-11'),
            ('East',   90.0, DATE '2026-01-05'),
            ('East',  150.0, DATE '2026-03-02'),
            ('North',  60.0, DATE '2026-01-20'),
            ('South',  30.0, DATE '2026-02-01'),
            ('South',  45.0, DATE '2026-03-15')
        ) AS t(region, amount, date)
        """
    )
    con.close()
    return os.path.abspath(path)


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "sample.db"))
