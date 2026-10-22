"""
Turn the data a user hands Lumen (paths, URLs, frames, datasets,
connections) into Lumen Sources.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd
import pyarrow as pa

from narwhals.dependencies import is_polars_dataframe, is_pyarrow_table

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..sources.xarray_sql import XArraySQLSource
from ..util import as_pandas, normalize_table_name, try_import_xarray
from .config import PROVIDED_SOURCE_NAME
from .controls.ingest.constants import XARRAY_EXTENSIONS
from .controls.ingest.utils import read_geo_file

if TYPE_CHECKING:
    import polars as pl
    import xarray as xr

    DataT = (
        str | Path | Source | Pipeline | pd.DataFrame | pl.DataFrame | pa.Table
        | xr.Dataset | xr.DataArray | duckdb.DuckDBPyConnection | duckdb.DuckDBPyRelation
    )


def _is_frame(obj) -> bool:
    return (
        isinstance(obj, (pd.DataFrame, duckdb.DuckDBPyRelation))
        or is_polars_dataframe(obj) or is_pyarrow_table(obj)
    )


def _as_frame(obj):
    if isinstance(obj, duckdb.DuckDBPyRelation):
        # arrow() gives a one-shot RecordBatchReader on duckdb >= 1.4, and a
        # view over it would be empty on the second query
        return pa.table(obj.arrow())
    return obj


def read(data: DataT | list[DataT] | dict[str, DataT]) -> Source:
    """
    Load data into a single Lumen Source.

    Parameters
    ----------
    data: DataT | list[DataT] | dict[str, DataT]
        A file path or URL, a Source, a Pipeline, a DataFrame (pandas,
        polars, GeoPandas), a pyarrow Table, an xarray Dataset or
        DataArray, a DuckDB connection or relation, or a list or dict
        of these. Frames in a list or dict share one source. A dict key
        names a frame's table, a DataArray's variable or a DuckDB
        connection's source. A DuckDB relation is read into memory.
        Every entry of a mixed dict lands in one DuckDB source, so they can
        be joined: frames (as pandas when mixed with files or pipelines),
        files, pipelines, xarray data (read lazily), and the tables of a
        DuckDB connection, which are copied in so the user's database is
        never written to. On its own, a connection is used as is and xarray
        data keeps its gridded source.

    Returns
    -------
    Source
    """
    sources = resolve_data(data)
    if len(sources) != 1:
        raise ValueError(
            f"The data resolved to {len(sources)} sources but read returns "
            "one. Call read once per source."
        )
    return sources[0]


def resolve_data(data: DataT | list[DataT] | dict[str, DataT] | None) -> list[Source]:
    """
    Resolve the data into a list of sources.

    Parameters
    ----------
    data: DataT | list[DataT] | dict[str, DataT] | None
        The data to resolve.

    Returns
    -------
    list[Source]
        List of sources.
    """
    if data is None:
        return []
    elif isinstance(data, dict):
        data = data.items()
    elif not isinstance(data, list):
        data = [data]

    sources = []
    remote = False
    mirrors, tables, frames, unnamed, grids, connections = {}, {}, {}, [], {}, []

    for src in data:
        name = None
        if isinstance(src, tuple):
            name, src = src
        if isinstance(src, Source):
            sources.append(src)
        elif isinstance(src, Pipeline):
            mirrors[name or src.name] = src
        elif _is_frame(src):
            if name:
                frames[name] = _as_frame(src)
            else:
                unnamed.append(_as_frame(src))
        elif isinstance(src, duckdb.DuckDBPyConnection):
            connections.append((name, src))
        elif (xr := try_import_xarray()) and isinstance(src, (xr.Dataset, xr.DataArray)):
            if isinstance(src, xr.DataArray):
                # The variable becomes the table, so the dict key names it
                src = src.to_dataset(name=name or src.name or "data")
            grids[name or (f"grid_{len(grids)}" if grids else "grid")] = src
        elif isinstance(src, (str, Path)):
            src = str(src)
            name = name or src

            # Handle "no_data" as a special case for starting without data
            if src == 'no_data':
                continue

            # Handle explicit database connection strings first (before checking file extensions)
            if src.startswith('duckdb://'):
                db_path = src.replace('duckdb:///', '')
                db_path = str(Path(db_path).absolute())
                source = DuckDBSource(uri=db_path)
                sources.append(source)
                continue
            elif src.startswith(('sqlite://', 'postgresql://', 'mysql://', 'mssql://', 'oracle://')):
                try:
                    from ..sources.sqlalchemy import SQLAlchemySource
                except ImportError as e:
                    raise ImportError(
                        "SQLAlchemy is required for database connection strings. "
                        "Install it with: pip install sqlalchemy"
                    ) from e
                source = SQLAlchemySource(url=src)
                sources.append(source)
                continue

            # Handle .db files - need to detect if SQLite or DuckDB
            if src.endswith('.db'):
                db_path = Path(src).absolute()
                if not db_path.exists():
                    raise FileNotFoundError(f"Database file not found: {src}")

                with open(db_path, 'rb') as f:
                    header = f.read(16)

                # DuckDB files start with specific magic bytes
                # The header contains "DUCK" in the first 16 bytes
                if b'DUCK' in header:
                    source = DuckDBSource(uri=str(db_path))
                    sources.append(source)
                else:
                    try:
                        from ..sources.sqlalchemy import SQLAlchemySource
                    except ImportError as e:
                        raise ImportError(
                            "SQLAlchemy is required to read .db files. "
                            "Install it with: pip install sqlalchemy"
                        ) from e
                    source = SQLAlchemySource(url=f'sqlite:///{db_path}')
                    sources.append(source)
                continue

            # Handle xarray files (NetCDF, Zarr, HDF5, GRIB)
            if Path(src).suffix.lstrip('.').lower() in XARRAY_EXTENSIONS:
                source = XArraySQLSource(uri=str(Path(src).absolute()))
                sources.append(source)
                continue

            # Handle local geospatial files (GeoJSON, WKT, zipped shapefile)
            # via the same read_geo_file -> DuckDB spatial path as file upload
            geo_ext = Path(src).suffix.lstrip('.').lower()
            if geo_ext in ('geojson', 'wkt', 'zip') and '://' not in src:
                sources.append(_resolve_geo_source(src, geo_ext))
                continue

            if src.startswith('http'):
                remote = True
            if src.endswith(('.parq', '.parquet', '.csv', '.json', '.tsv', '.jsonl', '.ndjson')):
                table = src
            else:
                raise ValueError(
                    f"Could not determine how to load {src} file."
                )
            tables[name] = table
        else:
            raise TypeError(
                f"Cannot load data of type {type(src).__name__}. Pass a file "
                "path or URL, a Source, a Pipeline, a pandas, polars or "
                "GeoPandas DataFrame, a pyarrow Table, an xarray Dataset or "
                "DataArray, or a DuckDB connection or relation."
            )

    entries = len(frames) + len(unnamed) + len(tables) + len(mirrors) + len(grids) + len(connections)
    if entries > 1:
        # Mixed with other entries, a connection's tables are copied into the
        # shared source, so Lumen never writes into the user's database
        for _, conn in connections:
            for (table,) in conn.execute("SHOW TABLES").fetchall():
                frames[table] = pa.table(conn.table(table).arrow())
    else:
        # On its own a connection is used as is, and a grid keeps the
        # xarray source that the gridded charts rely on
        sources.extend(DuckDBSource(_connection=conn, **({"name": name} if name else {})) for name, conn in connections)
        sources.extend(XArraySQLSource.from_dataset(ds) for ds in grids.values())
        grids = {}

    if len(unnamed) == 1:
        frames["data"] = unnamed[0]
    else:
        frames.update({f"data_{i}": df for i, df in enumerate(unnamed)})
    if frames and (tables or mirrors):
        # Mixed with files or pipelines, frames join the same connection as
        # mirrors, which DuckDBSource ingests and serializes as pandas
        mirrors.update({name: as_pandas(df) for name, df in frames.items()})
    elif frames or grids:
        sources.append(DuckDBSource.from_df(frames))
    if tables or mirrors:
        initializers = ["INSTALL httpfs;", "LOAD httpfs;"] if remote else []
        source = DuckDBSource(
            initializers=initializers,
            mirrors=mirrors,
            name=PROVIDED_SOURCE_NAME,
            # A mirror is only a view until it is listed as a table too
            tables={**tables, **{table: f'SELECT * FROM "{table}"' for table in mirrors}},
            uri=':memory:'
        )
        sources.append(source)
    if grids:
        _add_grids(sources[-1], grids)

    return sources


def _add_grids(source: DuckDBSource, grids: dict) -> None:
    """Expose xarray Datasets as lazy tables on a DuckDB source, so they join with its other tables."""
    try:
        # Deferred: xarray-sql is optional, and only the joined case needs it
        from xarray_sql import arrow_dataset
    except ImportError as e:
        raise ImportError(
            "Joining xarray data with other tables needs a version of xarray-sql "
            "that provides arrow_dataset. Upgrade xarray-sql, or read the xarray "
            "data on its own."
        ) from e
    for name, ds in grids.items():
        # Read lazily: DuckDB pulls record batches from the Dataset per query
        source._connection.from_arrow(arrow_dataset(ds)).to_view(name)
        source.tables[name] = f'SELECT * FROM "{name}"'

def _resolve_geo_source(src: str, extension: str) -> Source:
    """Load a geospatial file into an in-memory DuckDBSource.

    Routes GeoJSON/WKT/zipped-shapefile through the same read_geo_file ->
    DuckDB spatial (ST_GeomFromWKB) path the interactive upload uses, so
    startup and uploaded geospatial data behave identically (gh-1900).
    """
    path = Path(src).absolute()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {src}")

    alias = normalize_table_name(path.stem)
    with open(path, 'rb') as f:
        result = read_geo_file(f, extension, alias)

    source = DuckDBSource(
        initializers=result.source_params.get('initializers', []),
        # WKB carries no CRS, so re apply the one read_geo_file captured
        # from the file after the roundtrip
        geometry_crs=result.source_params.get('geometry_crs'),
        tables={},
        uri=':memory:',
    )
    conn = source._connection
    for tbl_name, df in result.tables.items():
        df_rel = conn.from_df(df)
        if tbl_name in result.conversions:
            conn.register(f"{tbl_name}_temp", df_rel)
            # read_geo_file emits a CREATE TEMP TABLE, but DuckDBSource runs
            # its queries on cursors that cannot see another connection's temp
            # tables, so materialize a persistent (in-memory) table instead.
            conn.execute(result.conversions[tbl_name].replace("TEMP TABLE", "TABLE", 1))
            conn.unregister(f"{tbl_name}_temp")
        else:
            df_rel.to_view(tbl_name)
        source.tables[tbl_name] = f"SELECT * FROM {tbl_name}"
    return source
