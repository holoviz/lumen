"""Covers frames from dataframe libraries other than pandas.

The ``constructor`` fixture builds the same data in pandas, polars and pyarrow,
so each test here runs once per backend and skips the ones that are not
installed. Every assertion is either that the three backends agree, or that a
pandas caller gets back exactly what it got before.
"""
import datetime as dt

import pandas as pd
import param
import pytest

from lumen.filters.base import ConstantFilter
from lumen.pipeline import DataFrame as PipelineDataFrame, Pipeline
from lumen.sources.base import InMemorySource
from lumen.transforms.base import (
    Aggregate, Columns, Filter as FilterTransform, Iloc, Query, Sample, Sort,
)
from lumen.util import as_narwhals, as_pandas, get_dataframe_schema
from lumen.views.base import Table


class ParamHolder(param.Parameterized):
    """Isolates the Pipeline.data parameter from Pipeline's own construction."""

    data = PipelineDataFrame()


def rows(frame):
    """Row count for a frame from any of the three backends."""
    return frame.num_rows if hasattr(frame, "num_rows") else len(frame)


def test_pipeline_data_accepts_backend(constructor):
    ParamHolder().data = constructor({"i": [0, 1]})


@pytest.mark.parametrize("value", [{"i": [0]}, [0, 1], "frame", 3])
def test_pipeline_data_still_rejects_non_frames(value):
    with pytest.raises(ValueError, match="expects a pandas DataFrame"):
        ParamHolder().data = value


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


def test_sort_matches_across_backends(constructor):
    frame = constructor({"i": [2, 0, 1], "s": ["c", "a", "b"]})
    sorted_frame = Sort.apply_to(frame, by=["i"])
    assert as_pandas(sorted_frame)["s"].tolist() == ["a", "b", "c"]


def test_columns_matches_across_backends(constructor):
    frame = constructor({"i": [0, 1], "s": ["a", "b"]})
    selected = Columns.apply_to(frame, columns=["s"])
    assert as_narwhals(selected).collect_schema().names() == ["s"]


def test_iloc_matches_across_backends(constructor):
    frame = constructor({"i": [0, 1, 2, 3]})
    assert as_pandas(Iloc.apply_to(frame, start=1, end=3))["i"].tolist() == [1, 2]


def test_sample_matches_across_backends(constructor):
    frame = constructor({"i": [0, 1, 2, 3]})
    assert rows(Sample.apply_to(frame, n=2)) == 2


def test_aggregate_matches_pandas_across_backends(constructor):
    data = {"g": ["b", "a", "b", "a"], "v": [1.0, 2.0, 3.0, 4.0]}
    reference = Aggregate.apply_to(
        pd.DataFrame(data), by=["g"], with_index=False, method="sum"
    )
    result = as_pandas(
        Aggregate.apply_to(constructor(data), by=["g"], with_index=False, method="sum")
    )
    assert result["g"].tolist() == reference["g"].tolist()
    assert result["v"].tolist() == reference["v"].tolist()


def test_pandas_only_transform_materializes(constructor):
    frame = constructor({"i": [0, 1, 2]})
    result = Query.apply_to(frame, query="i > 0")
    assert isinstance(result, pd.DataFrame)
    assert result["i"].tolist() == [1, 2]


def test_disk_cache_roundtrip(tmp_path, constructor):
    source = InMemorySource(tables={}, cache_dir=str(tmp_path))
    source._set_cache(constructor({"i": [0, 1]}), "t")
    assert (tmp_path / "t.parq").is_file()
    cached, _ = source._get_cache("t")
    assert as_pandas(cached)["i"].tolist() == [0, 1]


def test_pipeline_end_to_end_across_backends(constructor):
    frame = constructor({"g": ["b", "a", "b", "a"], "v": [1.0, 2.0, 3.0, 4.0]})
    source = InMemorySource(tables={"t": frame})
    pipeline = Pipeline(
        source=source, table="t",
        filters=[ConstantFilter(field="g", value="a")],
        transforms=[Sort(by=["v"])],
    )
    assert as_pandas(pipeline.data)["v"].tolist() == [2.0, 4.0]


def test_view_materializes_to_pandas(constructor):
    source = InMemorySource(tables={"t": constructor({"i": [0, 1, 2]})})
    view = Table(pipeline=Pipeline(source=source, table="t"))
    assert isinstance(view.get_data(), pd.DataFrame)
