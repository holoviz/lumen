import inspect
import sys
import types

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import xarray as xr
    import xarray_sql  # noqa
except ModuleNotFoundError:
    pytest.skip(
        "xarray / xarray-sql not installed, skipping Arraylake control tests.",
        allow_module_level=True,
    )

from lumen.ai.controls import ArraylakeSourceControls
from lumen.ai.controls.ingest.result import SourceResult
from lumen.sources.xarray_sql import XArraySQLSource


def _goes_like():
    """A GOES-like dataset: (y, x) bands plus a 0-dim scalar metadata var."""
    return xr.Dataset(
        {
            "CMI_C13": (("y", "x"), np.random.rand(4, 5)),
            "DQF_C13": (("y", "x"), np.random.randint(0, 4, (4, 5))),
            "goes_imager_projection": ((), 0),
        },
        coords={"y": np.arange(4), "x": np.arange(5)},
    )


@pytest.fixture
def arraylake_controls(context, source_catalog):
    return ArraylakeSourceControls(context=context, source_catalog=source_catalog)


@pytest.fixture
def fake_arraylake(monkeypatch):
    """Inject a stub ``arraylake`` module so the lazy import succeeds."""
    module = types.ModuleType("arraylake")
    module.Client = MagicMock()
    monkeypatch.setitem(sys.modules, "arraylake", module)
    return module


@pytest.mark.asyncio
class TestArraylakeSourceControls:

    async def test_fetch_data_builds_xarray_source(
        self, arraylake_controls, fake_arraylake, monkeypatch
    ):
        monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: _goes_like())
        result = await arraylake_controls._fetch_data(
            "ArraylakeSource",
            repo="earthmover-public/goes-16",
            branch="main",
            group="",
        )
        assert isinstance(result, SourceResult)
        assert len(result.sources) == 1
        source = result.sources[0]
        assert isinstance(source, XArraySQLSource)
        assert set(source.get_tables()) == {"CMI_C13", "DQF_C13"}
        assert result.table == "CMI_C13"

    async def test_scalar_variable_dropped(
        self, arraylake_controls, fake_arraylake, monkeypatch
    ):
        monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: _goes_like())
        result = await arraylake_controls._fetch_data("ArraylakeSource", repo="o/r")
        assert "goes_imager_projection" not in result.sources[0].get_tables()

    async def test_explicit_variables_override(
        self, arraylake_controls, fake_arraylake, monkeypatch
    ):
        monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: _goes_like())
        arraylake_controls.variables = ["CMI_C13"]
        result = await arraylake_controls._fetch_data("ArraylakeSource", repo="o/r")
        assert result.sources[0].get_tables() == ["CMI_C13"]

    async def test_missing_repo_returns_empty(self, arraylake_controls):
        result = await arraylake_controls._fetch_data("ArraylakeSource", repo="")
        assert result.sources == []
        assert "repo" in result.message.lower()

    async def test_missing_arraylake_returns_empty(
        self, arraylake_controls, monkeypatch
    ):
        # Setting the module to None makes ``import arraylake`` raise ImportError.
        monkeypatch.setitem(sys.modules, "arraylake", None)
        result = await arraylake_controls._fetch_data("ArraylakeSource", repo="o/r")
        assert result.sources == []
        assert "lumen[arraylake]" in result.message

    async def test_open_error_returns_empty(
        self, arraylake_controls, fake_arraylake, monkeypatch
    ):
        def boom(*a, **k):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(xr, "open_zarr", boom)
        result = await arraylake_controls._fetch_data("ArraylakeSource", repo="o/r")
        assert result.sources == []
        assert "connection refused" in result.message


def test_as_tools_exposes_typed_action(arraylake_controls):
    tools = arraylake_controls.as_tools()
    assert len(tools) == 1
    name, func = tools[0]
    assert name == "ArraylakeSource"
    assert list(inspect.signature(func).parameters) == ["repo", "branch", "group"]
