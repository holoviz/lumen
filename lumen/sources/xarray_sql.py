"""
XArraySQLSource -- SQL-backed xarray source for Lumen.

Uses xarray-sql (Apache DataFusion) to provide SQL query capabilities
over N-dimensional xarray datasets. Each data variable in the dataset
is exposed as a separate SQL table with coordinate columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import param

from ..transforms.sql import SQLFilter
from .base import BaseSQLSource, cached

try:
    import xarray as xr

    from xarray_sql import XarrayContext
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# Map file extensions to xarray engines
XARRAY_ENGINES = {
    ".nc": "netcdf4",
    ".nc4": "netcdf4",
    ".netcdf": "netcdf4",
    ".h5": "h5netcdf",
    ".hdf5": "h5netcdf",
    ".he5": "h5netcdf",
    ".zarr": "zarr",
    ".grib": "cfgrib",
    ".grib2": "cfgrib",
    ".grb": "cfgrib",
    ".grb2": "cfgrib",
}


def _detect_engine(path: str) -> str | None:
    """Detect xarray engine from file extension."""
    suffix = Path(path).suffix.lower()
    return XARRAY_ENGINES.get(suffix)


class XArraySQLSource(BaseSQLSource):
    """
    SQL-queryable xarray data source for Lumen.

    Wraps xarray datasets with Apache DataFusion (via xarray-sql) to
    provide full SQL query support. Each data variable in the dataset
    is exposed as a separate SQL table with coordinate columns.

    To use this source, install the optional xarray dependencies::

        pip install lumen[xarray]

    Parameters
    ----------
    uri : str
        Path or URL to a NetCDF/Zarr/HDF5/GRIB file.
    engine : str or None
        xarray engine to use. Auto-detected from file extension if None.
    chunks : dict or str or None
        Chunking specification for dask-backed lazy loading.
        Use 'auto' for automatic chunking, or a dict like {'time': 100}.
    open_kwargs : dict
        Additional keyword arguments passed to xr.open_dataset().
    variables : list or None
        Subset of data variables to expose. None means all variables.

    Examples
    --------
    >>> source = XArraySQLSource(uri="air_temperature.nc")
    >>> source.get_tables()
    ['air']
    >>> df = source.execute("SELECT lat, lon, AVG(air) FROM air GROUP BY lat, lon")
    >>> schema = source.get_schema("air")
    """

    source_type: ClassVar[str | None] = "xarray"

    dialect = "postgres"

    uri = param.String(default=None, allow_None=True, doc="""
        Path or URL to the xarray-compatible data file.
        Supports NetCDF (.nc), Zarr (.zarr), HDF5 (.h5), GRIB (.grib).""")

    engine = param.String(default=None, allow_None=True, doc="""
        xarray backend engine. Auto-detected from file extension if None.
        Options: netcdf4, h5netcdf, zarr, cfgrib.""")

    chunks = param.Parameter(default="auto", doc="""
        Dask chunk specification for lazy loading.
        Use 'auto', a dict like {'time': 100}, or None for eager loading.""")

    open_kwargs = param.Dict(default={}, doc="""
        Additional keyword arguments for xr.open_dataset().""")

    variables = param.List(default=None, allow_None=True, doc="""
        Subset of data variables to expose as tables.
        None means all variables are exposed.""")

    tables = param.ClassSelector(class_=(list, dict), default=None, allow_None=True, doc="""
        List or dict of tables. Populated automatically from dataset variables.
        When a dict, maps logical table names to SQL expressions.""")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The SQL expression template for table queries.""")

    def __init__(self, _dataset=None, _ctx=None, **params):
        if not XARRAY_AVAILABLE:
            raise ImportError(
                "xarray and xarray-sql are required for XArraySQLSource. "
                "Install them with: pip install lumen[xarray]"
            )
        super().__init__(**params)
        self._dataset = _dataset
        self._ctx = _ctx
        self._registered_tables = set()
        if self._ctx is None:
            self._initialize()
        else:
            if self._dataset is not None:
                self._registered_tables = set(self._dataset.data_vars)
            if self.tables is None:
                self.tables = list(self._registered_tables)

    def _initialize(self):
        """Load the xarray dataset and register variables with DataFusion."""
        self._dataset = self._load_dataset(
            dataset=self._dataset,
            uri=self.uri,
            engine=self.engine,
            chunks=self.chunks,
            open_kwargs=self.open_kwargs,
            variables=self.variables,
        )

        self._ctx = XarrayContext()
        chunk_spec = self._resolve_chunks(self._dataset, self.chunks)

        for var_name in self._dataset.data_vars:
            var_ds = self._dataset[[var_name]]
            self._ctx.from_dataset(var_name, var_ds, chunks=chunk_spec)
            self._registered_tables.add(var_name)

        if self.tables is None:
            self.tables = list(self._registered_tables)

    @staticmethod
    def _resolve_chunks(dataset, chunks):
        """Resolve chunk specification for DataFusion registration."""
        if isinstance(chunks, dict):
            return chunks
        ds_chunks = getattr(dataset, "chunks", None)
        if ds_chunks:
            return {dim: sizes[0] for dim, sizes in ds_chunks.items()}
        return dict(dataset.sizes)

    @staticmethod
    def _load_dataset(dataset, uri, engine, chunks, open_kwargs, variables):
        """Load an xarray dataset from URI or use a provided one."""
        if dataset is not None:
            ds = dataset
        elif uri is not None:
            resolved_engine = engine or _detect_engine(uri)
            kw = dict(open_kwargs)
            if resolved_engine:
                kw["engine"] = resolved_engine
            if chunks is not None:
                kw["chunks"] = chunks
            ds = xr.open_dataset(uri, **kw)
        else:
            raise ValueError("Either 'uri' or '_dataset' must be provided.")

        if variables:
            available = list(ds.data_vars)
            missing = [v for v in variables if v not in available]
            if missing:
                raise ValueError(
                    f"Variables {missing} not found in dataset. "
                    f"Available: {available}"
                )
            ds = ds[variables]
        return ds

    @property
    def dataset(self):
        """Access the underlying xarray Dataset."""
        return self._dataset

    def get_schema(self, table=None, limit=None, shuffle=False):
        # DataFusion doesn't support TABLESAMPLE; force shuffle=False
        # so the parent uses SQLLimit instead of SQLSample.
        return super().get_schema(table, limit, shuffle=False)

    # ---- BaseSQLSource required overrides ----

    def get_tables(self) -> list[str]:
        if isinstance(self.tables, (dict, list)):
            return sorted(self.tables)
        return sorted(self._registered_tables)

    def normalize_table(self, table: str) -> str:
        stripped = table.strip('"').strip("'").strip("`")
        all_known = set(self._registered_tables)
        if isinstance(self.tables, dict):
            all_known.update(self.tables.keys())
        elif isinstance(self.tables, (list, tuple)):
            all_known.update(self.tables)
        if stripped in all_known:
            return stripped
        lower_map = {t.lower(): t for t in all_known}
        if stripped.lower() in lower_map:
            return lower_map[stripped.lower()]
        return stripped

    def execute(self, sql_query, params=None, *args, **kwargs):
        """Execute a SQL query against the DataFusion context."""
        result = self._ctx.sql(sql_query)
        return result.to_pandas()

    @cached
    def get(self, table, **query):
        query.pop('__dask', None)
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = [
            (col, (val.start, val.stop) if isinstance(val, slice) else val)
            for col, val in query.items()
            if not col.startswith("__")
        ]
        sql_transforms = [SQLFilter(conditions=conditions, read=self.dialect)] + sql_transforms
        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)
        return self.execute(sql_expr)

    def create_sql_expr_source(
        self,
        tables: dict[str, str],
        params: dict[str, list | dict] | None = None,
        **kwargs,
    ):
        """Create a new XArraySQLSource sharing the same DataFusion context."""
        all_tables = {}
        if isinstance(self.tables, dict):
            all_tables.update(self.tables)
        elif isinstance(self.tables, (list, tuple)):
            for t in self.tables:
                all_tables[t] = self.sql_expr.format(table=t)
        all_tables.update(tables)

        all_params = dict(self.table_params)
        if params:
            all_params.update(params)

        return XArraySQLSource(
            _dataset=self._dataset,
            _ctx=self._ctx,
            uri=self.uri,
            engine=self.engine,
            chunks=self.chunks,
            variables=self.variables,
            open_kwargs=self.open_kwargs,
            tables=all_tables,
            table_params=all_params,
        )

    # ---- XArray-specific methods ----

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """Build rich metadata from xarray attributes."""
        ds = self._dataset
        metadata = {}

        for table in tables:
            if table not in ds.data_vars:
                continue

            var = ds[table]
            attrs = dict(var.attrs)
            columns = {}

            for coord_name in var.dims:
                coord = ds.coords[coord_name]
                coord_attrs = dict(coord.attrs)
                col_meta = {
                    "data_type": str(coord.dtype),
                    "is_coordinate": True,
                    "is_dimension": True,
                }
                if "units" in coord_attrs:
                    col_meta["units"] = coord_attrs["units"]
                if "long_name" in coord_attrs:
                    col_meta["description"] = coord_attrs["long_name"]
                if coord.dtype.kind in "fiuM":
                    col_meta["min"] = str(coord.values.min())
                    col_meta["max"] = str(coord.values.max())
                    col_meta["size"] = int(coord.size)
                columns[coord_name] = col_meta

            col_meta = {"data_type": str(var.dtype), "is_coordinate": False}
            if "units" in attrs:
                col_meta["units"] = attrs["units"]
            if "long_name" in attrs:
                col_meta["description"] = attrs["long_name"]
            columns[table] = col_meta

            dims_str = " x ".join(f"{d}({ds.sizes[d]})" for d in var.dims)
            description = attrs.get("long_name", table)
            if "units" in attrs:
                description += f" [{attrs['units']}]"

            metadata[table] = {
                "description": f"{description} -- dimensions: {dims_str}",
                "columns": columns,
                "dimensions": list(var.dims),
                "shape": list(var.shape),
                "dataset_attrs": dict(ds.attrs),
            }

        return metadata

    def get_dimension_info(self, table: str | None = None) -> dict[str, Any]:
        """Return dimension information for UI controls and AI context."""
        ds = self._dataset
        tables = [table] if table else list(ds.data_vars)
        info = {}

        for tbl in tables:
            if tbl not in ds.data_vars:
                continue
            var = ds[tbl]
            dim_info = {}
            for dim in var.dims:
                coord = ds.coords[dim]
                vals = coord.values
                d = {"size": int(coord.size), "dtype": str(coord.dtype)}
                if coord.dtype.kind == "M":
                    d["min"] = str(pd.Timestamp(vals.min()))
                    d["max"] = str(pd.Timestamp(vals.max()))
                    d["type"] = "datetime"
                elif coord.dtype.kind in "fiu":
                    d["min"] = float(vals.min())
                    d["max"] = float(vals.max())
                    d["step"] = float(np.diff(vals[:2])[0]) if len(vals) > 1 else None
                    d["type"] = "numeric"
                else:
                    d["values"] = vals.tolist()
                    d["type"] = "categorical"
                dim_info[dim] = d
            info[tbl] = dim_info

        return info if table is None else info.get(table, {})

    def close(self):
        """Clean up resources."""
        if self._dataset is not None:
            self._dataset.close()
        self._ctx = None
        self._registered_tables.clear()

    def __repr__(self):
        tables = self.get_tables()
        dims = dict(self._dataset.sizes) if self._dataset else {}
        return (
            f"XArraySQLSource(uri={self.uri!r}, tables={tables}, "
            f"dims={dims}, dialect='{self.dialect}')"
        )
