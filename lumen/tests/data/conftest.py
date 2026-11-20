import sys

import pytest

from lumen.data.stats import TableStatsStore, set_stats_store


@pytest.fixture(autouse=True)
def table_stats_store():
    """A fresh statistics store per test that never touches the user cache."""
    store = TableStatsStore(cache_dir=None)
    set_stats_store(store)
    yield store
    set_stats_store(None)


def pytest_collection_modifyitems(config, items):
    if not sys.platform.startswith("win"):
        return

    for item in items:
        if "duckdb" in item.nodeid.lower() or any("duckdb" in fn for fn in item.fixturenames):
            item.add_marker(pytest.mark.xdist_group("duckdb"))
