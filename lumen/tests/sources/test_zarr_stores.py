import gc

import numpy as np
import pandas as pd
import pytest

zarr = pytest.importorskip("zarr")

import duckdb

import lumen

from lumen.sources.data import _zarr_kind


def _read(data):
    try:
        return lumen.read(data)
    except duckdb.Error as e:
        pytest.skip(f"DuckDB's zarr extension could not be installed: {e}")


def _anndata(sparse=False):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    X = rng.random((40, 6)).astype("float32")
    if sparse:
        scipy_sparse = pytest.importorskip("scipy.sparse")
        X[X < 0.7] = 0
        X = scipy_sparse.csr_matrix(X)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(["B", "T", "NK"], 40)), "n_genes": rng.integers(100, 900, 40)},
        index=[f"cell{i}" for i in range(40)],
    )
    var = pd.DataFrame({"gene_symbol": [f"G{i}" for i in range(6)]}, index=[f"ENSG{i}" for i in range(6)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = rng.random((40, 2))
    return adata


def _mean_by_type(adata, gene):
    values = adata[:, gene].X
    values = np.asarray(values.todense() if hasattr(values, "todense") else values).ravel()
    return pd.Series(values, index=adata.obs.cell_type).groupby(level=0, observed=True).mean().to_dict()


@pytest.mark.parametrize("sparse", [False, True])
def test_an_anndata_store_reads_as_obs_var_and_x(tmp_path, sparse):
    adata = _anndata(sparse)
    adata.write_zarr(tmp_path / "pbmc.zarr")
    assert _zarr_kind(str(tmp_path / "pbmc.zarr")) == "anndata"
    source = _read(str(tmp_path / "pbmc.zarr"))
    assert {"obs", "var", "obs_var", "obsm_x_umap"} <= set(source.get_tables())
    counts = source.execute("SELECT cell_type, COUNT(*) AS n FROM obs GROUP BY 1")
    assert dict(zip(counts.cell_type.astype(str), counts.n, strict=True)) == adata.obs.cell_type.value_counts().to_dict()
    # A sparse X counts its zeros too, so AVG over cells is the true mean
    means = source.execute(
        "SELECT o.cell_type, AVG(x.X) AS m FROM obs_var x JOIN obs o USING (obs) JOIN var v USING (var) "
        "WHERE v.gene_symbol = 'G2' GROUP BY 1"
    )
    assert dict(zip(means.cell_type.astype(str), means.m, strict=True)) == pytest.approx(_mean_by_type(adata, "ENSG2"))
    assert len(source.execute("SELECT * FROM obs_var")) == 40 * 6
    assert "one row per cell (obs) and gene (var)" in source.get_metadata("obs_var")["description"]
    if sparse:
        assert len(source.execute("SELECT * FROM obs_var_nonzero")) == adata.X.nnz


def test_an_anndata_object_reads_through_a_store_kept_with_its_source():
    adata = _anndata()
    source = _read({"pbmc": adata})
    assert {"pbmc_obs", "pbmc_var", "pbmc_obs_var"} <= set(source.get_tables())
    gc.collect()
    assert len(source.execute("SELECT * FROM pbmc_obs")) == 40


def test_an_ome_zarr_image_reads_as_a_table_of_pixels_per_level(tmp_path):
    path = tmp_path / "cells.ome.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=2)
    image = (np.arange(2 * 4 * 6).reshape(2, 4, 6) % 7).astype("uint16")
    group.create_array("0", data=image, chunks=(1, 4, 6))
    group.create_array("1", data=image[:, ::2, ::2], chunks=(1, 2, 3))
    axes = [{"name": "c", "type": "channel"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    group.attrs["multiscales"] = [{"version": "0.4", "axes": axes, "datasets": [{"path": "0"}, {"path": "1"}]}]
    source = _read(str(path))
    assert sorted(source.get_tables()) == ["image", "image_level_1"]
    means = source.execute("SELECT c, AVG(value) AS m, COUNT(*) AS n FROM image GROUP BY c ORDER BY c")
    assert means.m.tolist() == pytest.approx(image.reshape(2, -1).mean(axis=1).tolist()) and means.n.tolist() == [24, 24]
    assert len(source.execute("SELECT * FROM image_level_1")) == 2 * 2 * 3
    assert "(c, y, x)" in source.get_metadata("image")["description"]


def test_a_zarr_store_of_neither_kind_is_left_to_xarray(tmp_path):
    group = zarr.open_group(tmp_path / "grid.zarr", mode="w")
    group.attrs["title"] = "a grid"
    assert _zarr_kind(str(tmp_path / "grid.zarr")) is None
