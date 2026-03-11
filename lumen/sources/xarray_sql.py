"""
XArraySQLSource — SQL-backed xarray source for Lumen.

Uses xarray-sql (Apache DataFusion) to provide SQL query capabilities
over N-dimensional xarray datasets. Each data variable in the dataset
is exposed as a separate SQL table with coordinate columns.

Architecture:
    NetCDF/Zarr file
        -> xarray.open_dataset() (lazy, dask-chunked)
        -> xarray-sql: XarrayContext registers dataset in DataFusion
        -> XArraySQLSource.execute() runs SQL via DataFusion
        -> Lumen Pipeline / AI agents consume DataFrames
"""

from __future__ import annotations

import asyncio
import logging

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import param

from .base import BaseSQLSource, Source, cached, cached_schema

logger = logging.getLogger(__name__)

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

XARRAY_EXTENSIONS = tuple(XARRAY_ENGINES.keys())


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

        pip install xarray xarray-sql netCDF4

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

    # DataFusion is PostgreSQL-compatible for sqlglot dialect translation
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
        List or dict of tables. Populated automatically from dataset variables.""")

    _supports_sql: ClassVar[bool] = True

    def __init__(self, _dataset=None, _ctx=None, **params):
        if not XARRAY_AVAILABLE:
            raise ImportError(
                "xarray and xarray-sql are required for XArraySQLSource. "
                "Install them with: pip install xarray xarray-sql netCDF4"
            )
        super().__init__(**params)
        self._dataset = _dataset
        self._ctx = _ctx
        self._registered_tables = set()
        self._df_cache = {}
        if self._ctx is None:
            self._initialize()
        else:
            # Context was shared — just populate registered tables
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

        # Create DataFusion context and register each variable as a table
        self._ctx = XarrayContext()

        # xarray-sql requires chunks — resolve 'auto' and None to a dict
        if isinstance(self.chunks, dict):
            chunk_spec = self.chunks
        else:
            chunk_spec = dict(self._dataset.sizes)

        for var_name in self._dataset.data_vars:
            var_ds = self._dataset[[var_name]]
            self._ctx.from_dataset(var_name, var_ds, chunks=chunk_spec)
            self._registered_tables.add(var_name)

        if self.tables is None:
            self.tables = list(self._registered_tables)

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

    def get_tables(self) -> list[str]:
        """Return list of available table names (one per data variable)."""
        return sorted(self._registered_tables)

    def normalize_table(self, table: str) -> str:
        """
        Normalize table names for fuzzy matching.

        Strips quotes, lowercases, and matches against registered tables
        to handle minor variations from Lumen AI agents.
        """
        stripped = table.strip('"').strip("'").strip("`")
        if stripped in self._registered_tables:
            return stripped
        lower_map = {t.lower(): t for t in self._registered_tables}
        if stripped.lower() in lower_map:
            return lower_map[stripped.lower()]
        return stripped

    def get_sql_expr(self, table: str | dict) -> str:
        """
        Return a SQL SELECT expression for the given table.
        """
        if isinstance(table, dict):
            table = list(table.values())[0]
        if isinstance(self.tables, dict):
            table = self.tables.get(table, table)
        table = self.normalize_table(table)
        return f"SELECT * FROM {table}"

    @cached_schema
    def get_schema(
        self, table: str | None = None, limit: int | None = None, shuffle: bool = False
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        Return JSON schema for the given table(s).

        Overrides BaseSQLSource.get_schema to query DataFusion directly
        instead of going through sqlglot SQL transforms, since DataFusion
        has its own SQL dialect nuances.
        """
        from ..util import get_dataframe_schema

        if table is None:
            tables = self.get_tables()
        else:
            tables = [table]

        schemas = {}
        for entry in tables:
            sample = self.execute(f"SELECT * FROM {entry} LIMIT 1")
            schema = get_dataframe_schema(sample)["items"]["properties"]

            count_df = self.execute(f"SELECT COUNT(*) as count FROM {entry}")
            count = int(count_df["count"].iloc[0])
            schema["__len__"] = count

            if limit:
                continue

            enums, min_maxes = [], []
            for name, col_schema in schema.items():
                if name == "__len__":
                    continue
                if "enum" in col_schema:
                    enums.append(name)
                elif "inclusiveMinimum" in col_schema:
                    min_maxes.append(name)

            for col in enums:
                distinct = self.execute(
                    f"SELECT DISTINCT {col} FROM {entry} ORDER BY {col}"
                )
                schema[col]["enum"] = distinct[col].tolist()

            if min_maxes:
                min_max_exprs = ", ".join(
                    f"MIN({c}) as {c}_min, MAX({c}) as {c}_max"
                    for c in min_maxes
                )
                mm = self.execute(f"SELECT {min_max_exprs} FROM {entry}")
                for col in min_maxes:
                    kind = sample[col].dtype.kind
                    if kind in "iu":
                        cast = int
                    elif kind == "f":
                        cast = float
                    elif kind == "M":
                        cast = str
                    else:
                        cast = lambda v: v
                    min_val = mm[f"{col}_min"].iloc[0]
                    max_val = mm[f"{col}_max"].iloc[0]
                    schema[col]["inclusiveMinimum"] = min_val if pd.isna(min_val) else cast(min_val)
                    schema[col]["inclusiveMaximum"] = max_val if pd.isna(max_val) else cast(max_val)

            schemas[entry] = schema

        return schemas if table is None else schemas[table]

    @cached
    def get(self, table: str, **query) -> pd.DataFrame:
        """
        Return a table as a DataFrame, optionally filtered by query params.

        Parameters
        ----------
        table : str
            Name of the table (data variable).
        query : dict
            Column-value pairs for filtering. Special keys:
            - sql_transforms: list of SQLTransform objects to apply
        """
        sql_transforms = query.pop("sql_transforms", [])
        table = self.normalize_table(table)
        sql = f"SELECT * FROM {table}"

        conditions = []
        # NOTE: DataFusion does not support parameterized queries, so values
        # are inlined. This is safe because DataFusion runs in-process (no
        # network boundary) and query values come from Lumen's internal
        # pipeline, not directly from untrusted user input.
        for col, val in query.items():
            if col.startswith("__"):
                continue
            if isinstance(val, list):
                vals_str = ", ".join(repr(v) for v in val)
                conditions.append(f"{col} IN ({vals_str})")
            elif isinstance(val, slice):
                if val.start is not None:
                    conditions.append(f"{col} >= {val.start!r}")
                if val.stop is not None:
                    conditions.append(f"{col} <= {val.stop!r}")
            else:
                conditions.append(f"{col} = {val!r}")

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        for transform in sql_transforms:
            try:
                sql = transform.apply(sql)
            except Exception:
                logger.warning(
                    "SQL transform %s failed, skipping", type(transform).__name__
                )

        return self.execute(sql)

    def execute(
        self,
        sql_query: str,
        params: list | dict | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Execute a SQL query against the registered xarray data via DataFusion.

        Parameters
        ----------
        sql_query : str
            SQL query string. Tables correspond to dataset variable names.
            DataFusion supports PostgreSQL-compatible SQL syntax.
        params : list | dict | None
            Not natively supported by DataFusion — basic inlining is used
            for API compatibility with BaseSQLSource.

        Returns
        -------
        pd.DataFrame
            Query results as a pandas DataFrame.
        """
        if params:
            if isinstance(params, list):
                for p in params:
                    sql_query = sql_query.replace("?", repr(p), 1)
            elif isinstance(params, dict):
                for k, v in params.items():
                    sql_query = sql_query.replace(f":{k}", repr(v))

        result = self._ctx.sql(sql_query)
        return result.to_pandas()

    async def execute_async(
        self,
        sql_query: str,
        params: list | dict | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Execute a SQL query asynchronously.

        Runs the synchronous execute() in a thread pool to avoid
        blocking the event loop. Used by Lumen AI agents.
        """
        return await asyncio.to_thread(self.execute, sql_query, params, *args, **kwargs)

    async def get_async(self, table: str, **query) -> pd.DataFrame:
        """
        Return a table asynchronously; optionally filtered.
        """
        sql = self.get_sql_expr(table)
        conditions = []
        for col, val in query.items():
            if col.startswith("__") or col == "sql_transforms":
                continue
            if isinstance(val, list):
                vals_str = ", ".join(repr(v) for v in val)
                conditions.append(f"{col} IN ({vals_str})")
            elif isinstance(val, slice):
                if val.start is not None:
                    conditions.append(f"{col} >= {val.start!r}")
                if val.stop is not None:
                    conditions.append(f"{col} <= {val.stop!r}")
            else:
                conditions.append(f"{col} = {val!r}")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return await self.execute_async(sql)

    def estimate_size(self, table: str | None = None) -> dict[str, Any]:
        """
        Estimate the memory size of a table for large dataset safety.

        Returns a dict with row count, estimated bytes, and whether
        it exceeds a safety threshold.
        """
        tables = [table] if table else self.get_tables()
        result = {}
        for tbl in tables:
            count_df = self.execute(f"SELECT COUNT(*) as count FROM {tbl}")
            n_rows = int(count_df["count"].iloc[0])
            var = self._dataset[tbl]
            n_cols = len(var.dims) + 1
            est_bytes = n_rows * n_cols * 8
            result[tbl] = {
                "rows": n_rows,
                "columns": n_cols,
                "estimated_bytes": est_bytes,
                "estimated_mb": round(est_bytes / 1024**2, 1),
                "exceeds_warning": n_rows > 10_000_000,
            }
        return result if table is None else result[tbl]

    def create_sql_expr_source(
        self,
        tables: dict[str, str],
        params: dict[str, list | dict] | None = None,
        **kwargs,
    ):
        """
        Create a new XArraySQLSource with custom SQL expressions as tables.

        Shares the same underlying dataset and DataFusion context to avoid
        expensive re-registration of variables.
        """
        return XArraySQLSource(
            _dataset=self._dataset,
            _ctx=self._ctx,
            uri=self.uri,
            engine=self.engine,
            chunks=self.chunks,
            variables=self.variables,
            open_kwargs=self.open_kwargs,
            tables=tables,
        )

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """
        Build rich metadata for each table from xarray attributes.

        Extracts coordinate ranges, units, long_name, dimensions, and
        dataset-level attributes.
        """
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
                "description": f"{description} — dimensions: {dims_str}",
                "columns": columns,
                "dimensions": list(var.dims),
                "shape": list(var.shape),
                "dataset_attrs": dict(ds.attrs),
            }

        return metadata

    def get_dimension_info(self, table: str | None = None) -> dict[str, Any]:
        """
        Return detailed dimension information for UI controls and AI context.

        Provides coordinate values, ranges, and types that the AI agent
        or UI needs to construct meaningful queries.
        """
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

    def clear_cache(self, *events):
        """Clear cached data and schema."""
        super().clear_cache(*events)
        self._df_cache.clear()

    def close(self):
        """Clean up resources."""
        if self._dataset is not None:
            self._dataset.close()
        self._ctx = None
        self._registered_tables.clear()
        self._df_cache.clear()

    def to_spec(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Serialize this source to a Lumen spec dict."""
        spec = {"type": self.source_type}
        if self.uri:
            spec["uri"] = str(self.uri)
        if self.engine:
            spec["engine"] = self.engine
        if self.chunks:
            spec["chunks"] = self.chunks
        if self.variables:
            spec["variables"] = self.variables
        if self.open_kwargs:
            spec["open_kwargs"] = self.open_kwargs
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any], **kwargs):
        """Create an XArraySQLSource from a Lumen spec dict."""
        params = {k: v for k, v in spec.items() if k != "type"}
        params.update(kwargs)
        return cls(**params)

    def __repr__(self):
        tables = self.get_tables()
        dims = dict(self._dataset.sizes) if self._dataset else {}
        return (
            f"XArraySQLSource(uri={self.uri!r}, tables={tables}, "
            f"dims={dims}, dialect='{self.dialect}')"
        )
