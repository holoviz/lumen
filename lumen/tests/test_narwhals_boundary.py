"""Records every place a non-pandas dataframe is currently rejected.

Each test is marked ``xfail(strict=True)`` and names the phase that removes the
lock-in.  ``xfail_strict`` is enabled, so the moment a phase makes one of these
pass the suite fails until its marker is deleted -- the marker is the checklist.

pyarrow is used as the probe backend because it is already installed wherever
Lumen's AI extra is; polars joins the matrix once it is declared as a test
dependency.
"""
import datetime as dt

import pandas as pd
import param
import pytest

from lumen.pipeline import DataFrame as PipelineDataFrame
from lumen.sources.base import InMemorySource
from lumen.transforms.base import Filter as FilterTransform
from lumen.util import get_dataframe_schema

pa = pytest.importorskip("pyarrow")


class ParamHolder(param.Parameterized):
    """Isolates the Pipeline.data parameter from Pipeline's own construction."""

    data = PipelineDataFrame()


@pytest.fixture
def arrow_table():
    return pa.table({"A": [0, 1, 2], "C": ["foo1", "foo2", "foo3"]})


@pytest.mark.xfail(strict=True, reason="phase5: param.DataFrame gates on isinstance(pd.DataFrame)")
def test_pipeline_data_accepts_arrow(arrow_table):
    ParamHolder().data = arrow_table


def test_schema_matches_across_backends(constructor):
    data = {
        "i": [0, 1, 2],
        "f": [0.5, 1.5, 2.5],
        "s": ["foo", "bar", "foo"],
        "b": [True, False, True],
    }
    schema = get_dataframe_schema(constructor(data))["items"]["properties"]
    assert schema["i"] == {"type": "integer", "inclusiveMinimum": 0, "inclusiveMaximum": 2}
    assert schema["f"] == {"type": "number", "inclusiveMinimum": 0.5, "inclusiveMaximum": 2.5}
    assert schema["s"] == {"type": "string", "enum": ["foo", "bar"]}
    assert schema["b"] == {"type": "boolean"}


def test_schema_datetime_across_backends(constructor):
    data = {"d": [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 3)]}
    schema = get_dataframe_schema(constructor(data))["items"]["properties"]
    assert schema["d"] == {
        "type": "string",
        "inclusiveMinimum": "2020-01-01T00:00:00",
        "inclusiveMaximum": "2020-01-03T00:00:00",
        "format": "datetime",
    }


def test_schema_column_subset_across_backends(constructor):
    frame = constructor({"i": [0, 1], "s": ["a", "b"]})
    assert set(get_dataframe_schema(frame, columns=["s"])["items"]["properties"]) == {"s"}


def rows(frame):
    """Row count for a frame from any of the three backends."""
    return frame.num_rows if hasattr(frame, "num_rows") else len(frame)


@pytest.mark.parametrize("conditions, expected", [
    ([("i", 1)], 1),
    ([("s", ["a", "b"])], 3),
    ([("i", (1, 2))], 2),
    ([("i", [(0, 0), (3, 3)])], 2),
    ([("i", (1, 2)), ("s", ["a"])], 1),
    ([("missing", 1)], 4),
    ([("s", [])], 4),
])
def test_filter_matches_across_backends(constructor, conditions, expected):
    frame = constructor({"i": [0, 1, 2, 3], "s": ["a", "b", "a", "c"]})
    assert rows(FilterTransform.apply_to(frame, conditions=conditions)) == expected


def test_filter_with_nulls_matches_pandas(constructor):
    data = {"v": [1.0, 2.0, None, 4.0]}
    conditions = [("v", [(1.0, 2.0), (4.0, 5.0)])]
    reference = FilterTransform.apply_to(pd.DataFrame(data), conditions=conditions)
    assert rows(FilterTransform.apply_to(constructor(data), conditions=conditions)) == len(reference)


def test_filter_preserves_backend(constructor):
    frame = constructor({"i": [0, 1, 2]})
    filtered = FilterTransform.apply_to(frame, conditions=[("i", (0, 1))])
    assert type(filtered) is type(frame)


@pytest.mark.xfail(strict=True, reason="phase5: _set_cache calls to_parquet, arrow has no such method")
def test_disk_cache_roundtrip_arrow(tmp_path, arrow_table):
    source = InMemorySource(tables={}, cache_dir=str(tmp_path))
    source._set_cache(arrow_table, "t")
    assert (tmp_path / "t.parq").is_file()
