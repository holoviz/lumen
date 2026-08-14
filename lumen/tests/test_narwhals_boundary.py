"""Records every place a non-pandas dataframe is currently rejected.

Each test is marked ``xfail(strict=True)`` and names the phase that removes the
lock-in.  ``xfail_strict`` is enabled, so the moment a phase makes one of these
pass the suite fails until its marker is deleted -- the marker is the checklist.

pyarrow is used as the probe backend because it is already installed wherever
Lumen's AI extra is; polars joins the matrix once it is declared as a test
dependency.
"""
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


@pytest.mark.xfail(strict=True, reason="phase2: get_dataframe_schema dispatches on numpy dtype.kind")
def test_schema_on_arrow(arrow_table):
    schema = get_dataframe_schema(arrow_table)
    assert set(schema["items"]["properties"]) == {"A", "C"}


@pytest.mark.xfail(strict=True, reason="phase3: Filter builds pandas boolean masks")
def test_filter_on_arrow(arrow_table):
    filtered = FilterTransform.apply_to(arrow_table, conditions=[("A", (0, 1))])
    assert len(filtered) == 2


@pytest.mark.xfail(strict=True, reason="phase5: _set_cache calls to_parquet, arrow has no such method")
def test_disk_cache_roundtrip_arrow(tmp_path, arrow_table):
    source = InMemorySource(tables={}, cache_dir=str(tmp_path))
    source._set_cache(arrow_table, "t")
    assert (tmp_path / "t.parq").is_file()
