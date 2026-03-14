"""
XArraySQLSource -- SQL-backed xarray source for Lumen.

Uses xarray-sql (Apache DataFusion) to provide SQL query capabilities
over N-dimensional xarray datasets. Each data variable in the dataset
is exposed as a separate SQL table with coordinate columns.
"""

from __future__ import annotations

import math

from pathlib import Path
from typing import Any, ClassVar

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
        # Case-insensitive matching and quote stripping for AI-generated queries.
        # Not using normalize_table_name() since that normalizes filenames, not
        # xarray variable names which are already valid Python identifiers.
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

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """Build metadata from xarray attributes for AI table discovery."""
        ds = self._dataset
        metadata = {}
        for table in tables:
            if table not in ds.data_vars:
                continue
            var = ds[table]
            attrs = dict(var.attrs)

            description = attrs.get("long_name", table)
            if "units" in attrs:
                description += f" [{attrs['units']}]"

            columns = {}
            for coord_name in var.dims:
                coord_attrs = dict(ds.coords[coord_name].attrs)
                columns[coord_name] = {
                    "data_type": str(ds.coords[coord_name].dtype),
                    "description": coord_attrs.get("long_name", ""),
                }
            columns[table] = {
                "data_type": str(var.dtype),
                "description": attrs.get("long_name", ""),
            }

            metadata[table] = {
                "description": description,
                "columns": columns,
                "rows": math.prod(var.shape),
            }
        return metadata

    def close(self):
        """Clean up resources."""
        if self._dataset is not None:
            self._dataset.close()
        self._ctx = None
        self._registered_tables.clear()
