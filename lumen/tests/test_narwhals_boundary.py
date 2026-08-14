"""Covers frames from dataframe libraries other than pandas.

The ``constructor`` fixture builds the same data in pandas, polars and pyarrow,
so each test here runs once per backend and skips the ones that are not
installed. Every assertion is either that the three backends agree, or that a
pandas caller gets back exactly what it got before.
"""
import datetime as dt

from decimal import Decimal

import pandas as pd
import param
import pytest

from lumen.filters.base import ConstantFilter
from lumen.pipeline import DataFrame as PipelineDataFrame, Pipeline
from lumen.sources.base import DerivedSource, InMemorySource
from lumen.transforms.base import (
    Aggregate, Columns, Filter as FilterTransform, Iloc, Query, Sample, Sort,
)
from lumen.util import (
    as_narwhals, as_pandas, get_dataframe_schema, is_lazyframe,
)
from lumen.views.base import Table


class ParamHolder(param.Parameterized):
    """Isolates the Pipeline.data parameter from Pipeline's own construction."""

    data = PipelineDataFrame()


def rows(frame):
    """Row count for a frame from any of the three backends."""
    return len(as_pandas(frame))


def test_pipeline_data_accepts_backend(constructor):
    frame = constructor({"i": [0, 1]})
    holder = ParamHolder()
    holder.data = frame
    # Read the stored value directly: the parameter's __get__ is Pipeline's
    # resolve-on-access hook, which ParamHolder cannot satisfy.
    assert holder._param__private.values["data"] is frame


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


def test_schema_all_null_datetime_matches_pandas(constructor):
    data = {"d": pd.Series([None, None], dtype="datetime64[ns]")}
    reference = get_dataframe_schema(pd.DataFrame(data))["items"]["properties"]
    schema = get_dataframe_schema(constructor(data))["items"]["properties"]
    assert schema["d"] == reference["d"]


def test_schema_empty_frame_matches_pandas(constructor):
    data = {"i": pd.Series([], dtype="int64"), "s": pd.Series([], dtype="object")}
    reference = get_dataframe_schema(pd.DataFrame(data))["items"]["properties"]
    schema = get_dataframe_schema(constructor(data))["items"]["properties"]
    # A None type is what auto_filters keys off, so it has to survive.
    assert schema["i"]["type"] is reference["i"]["type"] is None


def test_schema_decimal_is_not_truncated_to_integer():
    pa = pytest.importorskip("pyarrow")
    frame = pa.table({"d": pa.array([Decimal("1.5"), Decimal("2.5")], pa.decimal128(5, 2))})
    assert get_dataframe_schema(frame)["items"]["properties"]["d"] == {
        "type": "number", "inclusiveMinimum": 1.5, "inclusiveMaximum": 2.5
    }


def test_schema_omits_dtypes_no_filter_understands():
    pa = pytest.importorskip("pyarrow")
    frame = pa.table({"d": pa.array([1, 2], pa.duration("s")), "i": [0, 1]})
    properties = get_dataframe_schema(frame)["items"]["properties"]
    assert set(properties) == {"i"}


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
    data = {"i": [0, 1, 2, 3], "s": ["a", "b", "a", "c"]}
    reference = FilterTransform.apply_to(pd.DataFrame(data), conditions=conditions)
    result = as_pandas(FilterTransform.apply_to(constructor(data), conditions=conditions))
    assert len(reference) == expected
    assert result["i"].tolist() == reference["i"].tolist()


@pytest.mark.parametrize("value, expected", [
    (["a", None, "b"], ["a", None, "b"]),
    ([None], [None]),
    (["a"], ["a"]),
])
def test_filter_enum_containing_null(constructor, value, expected):
    """The schema emits None inside the enum of a nullable column, and that
    enum is fed straight back as a filter value by the auto-generated widgets.
    """
    data = {"s": ["a", None, "b"]}
    reference = FilterTransform.apply_to(pd.DataFrame(data), conditions=[("s", value)])
    result = as_pandas(FilterTransform.apply_to(constructor(data), conditions=[("s", value)]))
    kept = [None if pd.isna(v) else v for v in result["s"]]
    assert kept == [None if pd.isna(v) else v for v in reference["s"]] == expected


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


def test_sort_puts_nulls_last_like_pandas(constructor):
    data = {"v": [3.0, None, 1.0, 2.0], "s": ["c", "n", "a", "b"]}
    reference = Sort.apply_to(pd.DataFrame(data), by=["v"])
    result = as_pandas(Sort.apply_to(constructor(data), by=["v"]))
    assert result["s"].tolist() == reference["s"].tolist() == ["a", "b", "c", "n"]


def test_sample_matches_across_backends(constructor):
    frame = constructor({"i": [0, 1, 2, 3]})
    sampled = as_pandas(Sample.apply_to(frame, n=2))
    assert len(sampled) == 2
    assert sampled["i"].is_unique
    assert set(sampled["i"]) <= {0, 1, 2, 3}


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


def test_aggregate_with_index_matches_pandas(constructor):
    """with_index moves the keys into an index narwhals does not have, so the
    frame has to be materialized rather than silently reshaped.
    """
    data = {"g": ["b", "a", "b", "a"], "v": [1.0, 2.0, 3.0, 4.0]}
    reference = Aggregate.apply_to(pd.DataFrame(data), by=["g"], method="sum")
    result = Aggregate.apply_to(constructor(data), by=["g"], method="sum")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(reference.columns) == ["v"]
    assert result.index.tolist() == reference.index.tolist() == ["a", "b"]


def test_aggregate_drops_null_groups_like_pandas(constructor):
    data = {"g": ["a", None, "b", "a"], "v": [1.0, 3.0, 1.0, 5.0]}
    reference = Aggregate.apply_to(
        pd.DataFrame(data), by=["g"], with_index=False, method="sum"
    )
    result = as_pandas(
        Aggregate.apply_to(constructor(data), by=["g"], with_index=False, method="sum")
    )
    assert result["g"].tolist() == reference["g"].tolist() == ["a", "b"]
    assert result["v"].tolist() == reference["v"].tolist()


def test_pandas_only_transform_materializes(constructor, caplog):
    frame = constructor({"i": [0, 1, 2]})
    result = Query.apply_to(frame, query="i > 0")
    assert isinstance(result, pd.DataFrame)
    assert result["i"].tolist() == [1, 2]
    if isinstance(frame, pd.DataFrame):
        assert "no narwhals implementation" not in caplog.text
    else:
        assert "Query has no narwhals implementation" in caplog.text


def test_derived_source_coerces_for_pandas_only_transform(constructor):
    """DerivedSource runs transforms in its own loop, not through apply_to."""
    source = InMemorySource(tables={"t": constructor({"i": [0, 1, 2]})})
    derived = DerivedSource(source=source, transforms=[Query(query="i > 0")])
    assert as_pandas(derived.get("t"))["i"].tolist() == [1, 2]


def test_disk_cache_roundtrip(tmp_path, constructor):
    frame = constructor({"i": [0, 1]})
    source = InMemorySource(tables={}, cache_dir=str(tmp_path))
    source._set_cache(frame, "t")
    assert (tmp_path / "t.parq").is_file()
    cached, _ = source._get_cache("t")
    # Served from memory the frame keeps its backend; only a cold read from
    # disk comes back as pandas, because _get_cache uses pd.read_parquet.
    assert type(cached) is type(frame)
    assert as_pandas(cached)["i"].tolist() == [0, 1]
    source._cache.clear()
    from_disk, _ = source._get_cache("t")
    assert isinstance(from_disk, pd.DataFrame)
    assert from_disk["i"].tolist() == [0, 1]


def test_set_cache_failure_keeps_other_cached_tables(tmp_path):
    """A failed write must delete its own file, not the whole cache directory."""
    source = InMemorySource(tables={}, cache_dir=str(tmp_path))
    source._set_cache(pd.DataFrame({"i": [0, 1]}), "keep")
    source._set_cache(object(), "broken")
    assert (tmp_path / "keep.parq").is_file()
    assert not (tmp_path / "broken.parq").exists()


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


def test_pipeline_renders_itself(constructor):
    """pn.panel(pipeline) drives Tabulator directly, not through View."""
    source = InMemorySource(tables={"t": constructor({"i": [0, 1, 2]})})
    Pipeline(source=source, table="t").__panel__()


def test_lazy_source_does_not_leak_a_lazyframe():
    pl = pytest.importorskip("polars")
    source = InMemorySource(tables={"t": pl.LazyFrame({"i": [0, 1, 2]})})
    assert not is_lazyframe(as_narwhals(source.get("t")))
    pipeline = Pipeline(source=source, table="t")
    assert len(as_pandas(pipeline.data)) == 3


def test_aggregate_drops_nan_group_keys_like_pandas(constructor):
    data = {"g": [1.0, float("nan"), 2.0, 1.0], "v": [1.0, 3.0, 1.0, 5.0]}
    reference = Aggregate.apply_to(
        pd.DataFrame(data), by=["g"], with_index=False, method="sum"
    )
    result = as_pandas(
        Aggregate.apply_to(constructor(data), by=["g"], with_index=False, method="sum")
    )
    assert result["g"].tolist() == reference["g"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("method", ["nunique", "prod", "sum"])
def test_aggregate_falls_back_for_unsupported_methods(constructor, method):
    """method is documented as a pandas name and narwhals has fewer of them."""
    data = {"g": ["a", "a", "b"], "v": [2.0, 3.0, 4.0]}
    reference = Aggregate.apply_to(
        pd.DataFrame(data), by=["g"], with_index=False, method=method
    )
    result = as_pandas(
        Aggregate.apply_to(constructor(data), by=["g"], with_index=False, method=method)
    )
    assert result["v"].tolist() == reference["v"].tolist()


def test_aggregate_falls_back_for_non_numeric_columns(constructor):
    data = {"g": ["a", "a", "b"], "s": ["x", "z", "y"]}
    reference = Aggregate.apply_to(
        pd.DataFrame(data), by=["g"], columns=["s"], with_index=False, method="sum"
    )
    result = as_pandas(Aggregate.apply_to(
        constructor(data), by=["g"], columns=["s"], with_index=False, method="sum"
    ))
    assert result["s"].tolist() == reference["s"].tolist()


def test_filter_numeric_enum_with_null_matches_pandas(constructor):
    """pandas isin([None]) does not match NaN on a numeric column."""
    data = {"v": [1.0, None, 3.0]}
    reference = FilterTransform.apply_to(pd.DataFrame(data), conditions=[("v", [1.0, None])])
    result = as_pandas(FilterTransform.apply_to(constructor(data), conditions=[("v", [1.0, None])]))
    assert result["v"].tolist() == reference["v"].tolist() == [1.0]
