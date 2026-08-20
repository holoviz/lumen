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
from lumen.sources.base import DerivedSource, InMemorySource, Source
from lumen.transforms.base import (
    Aggregate, Columns, DropNA, Filter as FilterTransform, Iloc, Melt, Query,
    Rename, Sample, Sort,
)
from lumen.util import (
    _NULLABLE_DTYPES, as_narwhals, as_pandas, get_dataframe_schema,
    is_lazyframe,
)
from lumen.views.base import Table, hvPlotView


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
    assert type(cached) is type(frame)
    assert as_pandas(cached)["i"].tolist() == [0, 1]


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


def test_melt_matches_pandas(constructor):
    data = {"g": ["a", "b"], "x": [1.0, 2.0], "y": [3.0, 4.0]}
    reference = Melt.apply_to(pd.DataFrame(data), id_vars=["g"], value_vars=["x", "y"])
    result = as_pandas(Melt.apply_to(constructor(data), id_vars=["g"], value_vars=["x", "y"]))
    assert sorted(result.columns) == sorted(reference.columns)
    assert sorted(map(tuple, result[reference.columns].values.tolist())) == \
        sorted(map(tuple, reference.values.tolist()))


def test_rename_columns_matches_pandas(constructor):
    data = {"a": [1, 2], "b": [3, 4]}
    reference = Rename.apply_to(pd.DataFrame(data), columns={"a": "z"})
    result = as_pandas(Rename.apply_to(constructor(data), columns={"a": "z"}))
    assert list(result.columns) == list(reference.columns) == ["z", "b"]


def test_dropna_matches_pandas(constructor):
    data = {"a": [1.0, None, 3.0], "b": [1.0, 2.0, None]}
    reference = DropNA.apply_to(pd.DataFrame(data))
    result = as_pandas(DropNA.apply_to(constructor(data)))
    assert result["a"].tolist() == reference["a"].tolist() == [1.0]


def test_every_arrow_integer_type_has_a_nullable_pandas_dtype():
    """An unmapped type would convert to whatever pandas defaults to.

    The reroute in ``_to_pandas`` fires on any integer narwhals reports, so a
    gap here is a column that silently keeps the widened dtype.
    """
    pa = pytest.importorskip("pyarrow")
    arrow_integers = {
        getattr(pa, f"{prefix}int{width}")()
        for prefix in ("", "u") for width in (8, 16, 32, 64)
    }
    assert arrow_integers <= set(_NULLABLE_DTYPES)
    assert pa.bool_() in _NULLABLE_DTYPES


def test_a_dtype_pandas_cannot_hold_raises_rather_than_corrupting():
    """polars Int128 has no pandas or arrow counterpart.

    It already fails in ``to_pandas`` before the reroute, and failing is the
    correct outcome: the alternative is a column of quietly wrong numbers.
    """
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"i": pl.Series([1, None], dtype=pl.Int128)})
    with pytest.raises(Exception, match="i128"):
        as_pandas(frame)


@pytest.mark.parametrize("index_type", ["uint8", "uint16", "uint32", "uint64"])
def test_as_pandas_converts_an_unsigned_dictionary_column(index_type):
    """polars writes uint32 indices for a Categorical, and pyarrow before 23
    cannot convert an unsigned dictionary index to pandas at all."""
    pa = pytest.importorskip("pyarrow")
    table = pa.table({
        "c": pa.DictionaryArray.from_arrays(
            pa.array([0, 1, None], getattr(pa, index_type)()), pa.array(["x", "y"])
        ),
        "n": [1.0, 2.0, 3.0],
    })
    result = as_pandas(table)
    assert isinstance(result["c"].dtype, pd.CategoricalDtype)
    assert result["c"].tolist()[:2] == ["x", "y"]
    assert result["c"].isna().tolist() == [False, False, True]


def test_as_pandas_keeps_a_dictionary_column_ordered():
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"c": pa.DictionaryArray.from_arrays(
        pa.array([1, 0], pa.uint8()), pa.array(["lo", "hi"]), ordered=True
    )})
    assert as_pandas(table)["c"].dtype.ordered


def test_as_pandas_keeps_a_polars_categorical_through_pyarrow():
    """The end of the road for the issue: a polars Categorical read as arrow."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    table = pl.DataFrame({"c": pl.Series(["x", "y", "x"], dtype=pl.Categorical)}).to_arrow()
    assert as_pandas(table)["c"].tolist() == ["x", "y", "x"]


def test_as_pandas_keeps_nullable_integer_and_boolean(constructor):
    """Converting must not widen a nullable column, or an id renders as 3.0."""
    data = {
        "i": pd.Series([1, None, 3], dtype="Int64"),
        "u": pd.Series([1, None, 3], dtype="UInt8"),
        "b": pd.Series([True, None, False], dtype="boolean"),
    }
    result = as_pandas(constructor(data))
    assert [str(dtype) for dtype in result.dtypes] == ["Int64", "UInt8", "boolean"]
    assert result["i"].tolist() == [1, pd.NA, 3]


def test_as_pandas_leaves_a_column_without_nulls_alone(constructor):
    """Only the columns the conversion would widen may change."""
    result = as_pandas(constructor({"i": [1, 2, 3], "b": [True, False, True]}))
    assert [str(dtype) for dtype in result.dtypes] == ["int64", "bool"]


def test_as_pandas_keeps_an_integer_beyond_float_precision(constructor):
    """Above 2**53 the float detour returns a different number, silently."""
    big = 2**62 + 1
    result = as_pandas(constructor({"i": pd.Series([big, None], dtype="Int64")}))
    # Compared as an int: a float64 comparison rounds both sides and passes on
    # a value that has already been corrupted.
    assert int(result["i"][0]) == big


def test_fallback_and_native_paths_agree_on_dtype(constructor):
    """Two configurations of one transform must not disagree on the schema.

    Both configurations here leave the null in ``id``: one drops on ``v``
    natively, the other converts to pandas because ``how='all'`` has no
    narwhals counterpart. The dtype must not depend on which one ran.
    """
    data = {"id": pd.Series([1, 2, None, 4], dtype="Int64"), "v": [1.0, None, 3.0, 4.0]}
    native = as_pandas(DropNA.apply_to(constructor(data), subset=["v"]))
    fallback = as_pandas(DropNA.apply_to(constructor(data), how="all"))
    assert native["id"].isna().any() and fallback["id"].isna().any()
    assert str(fallback["id"].dtype) == str(native["id"].dtype) == "Int64"


def test_hvplot_view_gets_numpy_dtypes(constructor):
    """datashader raises on a pandas extension dtype, so plots take numpy.

    Rasterizing is the case that breaks, but every kind can reach datashader
    through an operation, so the whole hvPlot path widens.
    """
    data = {"id": pd.Series([1, 2, None, 4], dtype="Int64"), "v": [1.0, None, 3.0, 4.0]}
    pipeline = Pipeline(
        source=InMemorySource(tables={"t": constructor(data)}), table="t",
        transforms=[DropNA(how="all")],
    )
    plotted = hvPlotView(pipeline=pipeline, kind="scatter", x="v", y="id").get_data()
    assert str(plotted["id"].dtype) == "float64"
    # The table is not a datashader consumer and keeps the id an id.
    assert str(Table(pipeline=pipeline).get_data()["id"].dtype) == "Int64"


def test_lazy_transforms_stay_lazy():
    pl = pytest.importorskip("polars")
    frame = pl.LazyFrame({"i": [0, 1, 2, 3]})
    for transform in (Iloc(end=2), Sample(n=2), Sort(by=["i"]), Columns(columns=["i"])):
        result = transform.apply(frame)
        assert isinstance(result, pl.LazyFrame), type(transform).__name__


def test_disk_cache_keeps_its_backend(tmp_path, constructor):
    """A cold read must return the same kind of frame the write produced."""
    frame = constructor({"i": [0, 1]})
    source = InMemorySource(tables={}, cache_dir=str(tmp_path))
    source._set_cache(frame, "t")
    source._cache.clear()
    from_disk, _ = source._get_cache("t")
    assert type(from_disk) is type(frame)
    assert as_pandas(from_disk)["i"].tolist() == [0, 1]


def test_disk_cache_survives_an_uninstallable_backend(tmp_path):
    pl = pytest.importorskip("polars")
    source = InMemorySource(tables={}, cache_dir=str(tmp_path))
    source._set_cache(pl.DataFrame({"i": [0, 1]}), "t")
    (tmp_path / "t.backend").write_text("a_library_nobody_has")
    source._cache.clear()
    from_disk, _ = source._get_cache("t")
    assert isinstance(from_disk, pd.DataFrame)
    assert from_disk["i"].tolist() == [0, 1]


def test_filter_numeric_enum_with_null_matches_pandas(constructor):
    """pandas isin([None]) does not match NaN on a numeric column."""
    data = {"v": [1.0, None, 3.0]}
    reference = FilterTransform.apply_to(pd.DataFrame(data), conditions=[("v", [1.0, None])])
    result = as_pandas(FilterTransform.apply_to(constructor(data), conditions=[("v", [1.0, None])]))
    assert result["v"].tolist() == reference["v"].tolist() == [1.0]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_pipeline_dataframe_backend(constructor, backend):
    """Pipeline.data can be pinned to a library regardless of the Source."""
    pytest.importorskip(backend)
    source = InMemorySource(tables={"t": constructor({"i": [0, 1, 2]})})
    pipeline = Pipeline(source=source, table="t", dataframe_backend=backend)
    assert type(pipeline.data).__module__.split(".")[0] == backend
    assert as_pandas(pipeline.data)["i"].tolist() == [0, 1, 2]


def test_pipeline_dataframe_backend_defaults_to_the_source(constructor):
    frame = constructor({"i": [0, 1, 2]})
    pipeline = Pipeline(source=InMemorySource(tables={"t": frame}), table="t")
    assert type(pipeline.data) is type(frame)


def test_pipeline_dataframe_backend_rejects_unknown():
    source = InMemorySource(tables={"t": pd.DataFrame({"i": [0]})})
    with pytest.raises(ValueError):
        Pipeline(source=source, table="t", dataframe_backend="nonesuch")


def test_describe_data_accepts_any_backend(constructor):
    """describe_data converts internally so callers do not have to."""
    pytest.importorskip("pydantic", reason="lumen.ai needs the ai extra")
    from lumen.ai.utils import describe_data_sync
    summary = describe_data_sync(constructor({"i": [0, 1, 2], "s": ["a", "b", "c"]}))
    assert "data_shape" in summary


def test_uncached_source_may_return_a_lazy_frame():
    """A Source that skips the cache can hand back a frame it has not read."""
    pl = pytest.importorskip("polars")

    class ScanSource(Source):

        def get_tables(self):
            return ["t"]

        def get(self, table, **query):
            return pl.LazyFrame({"n": [0, 1, 2]})

    pipeline = Pipeline(source=ScanSource(), table="t")
    assert isinstance(pipeline.data, pl.DataFrame)
    assert pipeline.data["n"].to_list() == [0, 1, 2]


def test_pipeline_collects_a_lazy_frame_for_its_own_backend():
    """dataframe_backend returns early when it already matches, so the collect
    cannot be left to it.
    """
    pl = pytest.importorskip("polars")

    class ScanSource(Source):

        def get_tables(self):
            return ["t"]

        def get(self, table, **query):
            return pl.LazyFrame({"n": [0, 1]})

    pipeline = Pipeline(source=ScanSource(), table="t", dataframe_backend="polars")
    assert isinstance(pipeline.data, pl.DataFrame)
