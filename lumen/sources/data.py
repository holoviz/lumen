"""
Turn the data a user hands Lumen (paths, URLs, frames, datasets,
connections) into Lumen Sources.
"""
from __future__ import annotations

import importlib.util
import re
import warnings
import weakref

from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd
import pyarrow as pa

from narwhals.dependencies import is_polars_dataframe, is_pyarrow_table

from ..pipeline import Pipeline
from ..util import as_pandas, normalize_table_name, try_import_xarray
from .base import Source
from .duckdb import DuckDBSource
from .xarray_sql import XARRAY_EXTENSIONS, XArraySQLSource

if TYPE_CHECKING:
    import polars as pl
    import xarray as xr

    DataT = (
        str | Path | Source | Pipeline | pd.DataFrame | pl.DataFrame | pa.Table
        | xr.Dataset | xr.DataArray | duckdb.DuckDBPyConnection | duckdb.DuckDBPyRelation
    )

#: Name of the source that holds the files and pipelines handed to Lumen
PROVIDED_SOURCE_NAME = "ProvidedSource00000"


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


#: Remote locations read without credentials, public buckets only
_REMOTE = ("http://", "https://", "s3://", "gs://", "gcs://")

#: A Zarr store, by name; ARCO-ERA5 adds the format version
_ZARR = re.compile(r"\.zarr(-v[23])?/?$")

#: An S3 bucket named after its region, as Overture's is (overturemaps-us-west-2)
_S3_REGION = re.compile(r"^s3://[^/]*?-((?:us|eu|ap|ca|sa|me|af)-[a-z]+-\d)/")


def _https(url: str) -> str:
    """A public gs:// or s3:// location as its HTTPS address, so no cloud SDK is needed to read it."""
    if url.startswith(("gs://", "gcs://")):
        return "https://storage.googleapis.com/" + url.split("://", 1)[1]
    if url.startswith("s3://"):
        bucket, _, key = url[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


#: What reading a remote Zarr store needs: the gridded source, zarr, HTTP access and lazy chunks
_ZARR_MODULES = ("xarray", "xarray_sql", "zarr", "fsspec", "aiohttp", "dask")


def _remote_grid(url: str, variables: list[str] | None, sel: dict | None) -> XArraySQLSource:
    """A public Zarr store as a gridded source, cut to the variables and coordinates asked for before SQL sees it."""
    if missing := [name for name in _ZARR_MODULES if importlib.util.find_spec(name) is None]:
        raise ImportError(
            f"Reading a remote Zarr store needs {', '.join(missing)}: "
            "pip install 'lumen[xarray]' fsspec aiohttp dask"
        )
    if not variables and not sel:
        warnings.warn(
            f"{url} is read whole: a remote store can hold hundreds of variables and terabytes, "
            "and nothing bounds a query over it. Pass variables= and sel= to read only what you need.",
            stacklevel=3,
        )
    xr = try_import_xarray()
    # Lazy: only the metadata is read until a query needs values
    ds = xr.open_zarr(_https(url), chunks={})
    if variables:
        ds = ds[variables]
    if sel:
        ds = ds.sel(sel)
    return XArraySQLSource.from_dataset(ds)


def _remote_box(name: str, url: str, bbox: tuple[float, float, float, float]) -> DuckDBSource:
    """
    The features of a (Geo)Parquet file or glob inside a box, copied into
    memory once, so questions about them do not read the remote files again.
    """
    # Numbers only, as they go into the SQL
    west, south, east, north = (float(v) for v in bbox)
    initializers = ["INSTALL httpfs;", "LOAD httpfs;", "INSTALL spatial;", "LOAD spatial;"]
    if region := _S3_REGION.match(url):
        initializers.append(f"SET s3_region='{region.group(1)}';")
    table = name.replace('"', '""')
    source = DuckDBSource(
        initializers=initializers, name=PROVIDED_SOURCE_NAME, tables={name: f'SELECT * FROM "{table}"'}, uri=":memory:",
    )
    # One connection for the look and the copy, so the files are scanned once.
    # The copy is not an initializer, so a spec or an export does not download it again.
    con = source._connection
    quoted = url.replace("'", "''")
    scan = f"read_parquet('{quoted}', hive_partitioning = true, union_by_name = true)"
    columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()}
    if "bbox" in columns:
        # Overture's bbox column: compared in the scan, so row groups outside the box are skipped
        inside = f"bbox.xmin <= {east} AND bbox.xmax >= {west} AND bbox.ymin <= {north} AND bbox.ymax >= {south}"
    elif "geometry" in columns:
        inside = f"ST_Intersects(geometry, ST_MakeEnvelope({west}, {south}, {east}, {north}))"
    else:
        source.close()
        raise ValueError(f"{url} has neither a bbox nor a geometry column to cut to a box.")
    con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM {scan} WHERE {inside}')
    return source


def _remote_name(url: str) -> str:
    """A table name for a remote location: Overture's type=place gives place, else the file or store name."""
    if match := re.search(r"type=([^/]+)", url):
        return match.group(1)
    stem = url.rstrip("/").rsplit("/", 1)[-1]
    return normalize_table_name(re.sub(r"(\.zarr(-v[23])?|\.parquet|\*.*)$", "", stem) or "data")


def read(
    data: DataT | list[DataT] | dict[str, DataT], *, variables: list[str] | None = None,
    sel: dict[str, Any] | None = None, bbox: tuple[float, float, float, float] | None = None,
) -> Source:
    """
    Load data into a single Lumen Source.

    Parameters
    ----------
    data: DataT | list[DataT] | dict[str, DataT]
        A file path or URL, a Source, a Pipeline, a DataFrame (pandas,
        polars, GeoPandas), a pyarrow Table, an xarray Dataset or
        DataArray, a DuckDB connection or relation, or a list or dict
        of these. Frames in a list or dict share one source. A dict key
        names a frame's, file's or pipeline's table, a DataArray's variable
        or a DuckDB connection's source; it is ignored for a Source and an
        xarray Dataset. A DuckDB relation is read into memory.
        Apart from those named below, every entry of a mixed dict lands in
        one DuckDB source, so they can be joined: frames (as pandas when
        mixed with files or pipelines), files, pipelines, xarray data (read lazily), and the tables of a
        DuckDB connection, which are copied in under their own names (the
        connection's dict key goes unused) so the user's database is never
        written to. On its own, a connection is used as is and left
        open by ``source.close()``, a source built on one cannot be rebuilt
        from its spec, and xarray data keeps its gridded source. A Source,
        and a geo, xarray or database file or a database URI, always stays
        its own source.

    variables: list[str] | None
        For a Zarr store, the variables to read; a store such as ARCO-ERA5
        has hundreds.
    sel: dict | None
        For a Zarr store, coordinates to cut it to before SQL sees it, as
        for xarray's ``Dataset.sel``, e.g. ``{"time": "2020-07-01T12:00"}``.
    bbox: tuple | None
        For (Geo)Parquet, the box (west, south, east, north) in longitude and
        latitude to keep; the features inside are read once into memory.
        Without it a remote (Geo)Parquet file is queried in place, which
        cannot skip row groups by Overture's bbox column and may download
        whole files for every question.

    Public remote data is read without credentials: ``https://``, and
    ``s3://`` or ``gs://`` buckets that allow anonymous reads, e.g. Overture
    (``s3://overturemaps-us-west-2/release/...``), Source Cooperative
    (``https://data.source.coop/...``) and ARCO-ERA5
    (``gs://gcp-public-data-arco-era5/ar/...zarr-v2``).

    Returns
    -------
    Source
    """
    if variables or sel or bbox:
        name, url = next(iter(data.items())) if isinstance(data, dict) and len(data) == 1 else (None, data)
        if not isinstance(url, (str, Path)):
            raise ValueError("variables, sel and bbox apply to one file, glob or URL, e.g. lumen.read(url, bbox=...).")
        url = str(url)
        if _ZARR.search(url):
            if bbox:
                raise ValueError("bbox cuts (Geo)Parquet; cut a Zarr store with sel, e.g. sel={'latitude': slice(60, 40)}.")
            return _remote_grid(url, variables, sel)
        if variables or sel:
            raise ValueError("variables and sel cut a Zarr store; cut (Geo)Parquet with bbox.")
        return _remote_box(name or _remote_name(url), url, bbox)
    sources = resolve_data(data)
    if len(sources) != 1:
        raise ValueError(
            f"The data resolved to {len(sources)} sources but read returns one: "
            "Sources, geo, xarray and database files, and database URIs each stay "
            "their own source. Call read once per source, or pass the list to "
            "ExplorerUI, which takes several."
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

            if src.startswith(_REMOTE) and _ZARR.search(src):
                sources.append(_remote_grid(src, None, None))
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

            if src.startswith(_REMOTE):
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

    # Named first, so a connection table called data collides with them too
    if len(unnamed) == 1:
        frames["data"] = unnamed[0]
    else:
        frames.update({f"data_{i}": df for i, df in enumerate(unnamed)})
    entries = len(frames) + len(tables) + len(mirrors) + len(grids) + len(connections)
    if entries > 1:
        # Mixed with other entries, a connection's tables are copied into the
        # shared source, so Lumen never writes into the user's database
        for _, conn in connections:
            for (table,) in conn.execute("SHOW TABLES").fetchall():
                if table in frames or table in tables or table in mirrors or table in grids:
                    raise ValueError(
                        f"The DuckDB connection's table {table!r} has the same name as "
                        "another entry. Rename one of them."
                    )
                frames[table] = pa.table(conn.table(table).arrow())
    else:
        # On its own a connection is used as is, and a grid keeps the
        # xarray source that the gridded charts rely on
        sources.extend(DuckDBSource(_connection=conn, **({"name": name} if name else {})) for name, conn in connections)
        sources.extend(XArraySQLSource.from_dataset(ds) for ds in grids.values())
        grids = {}

    if frames and (tables or mirrors):
        # Mixed with files or pipelines, frames join the same connection as
        # mirrors, which DuckDBSource ingests and serializes as pandas
        mirrors.update({name: as_pandas(df) for name, df in frames.items()})
    elif frames or grids:
        sources.append(DuckDBSource.from_df(frames))
    if tables or mirrors:
        initializers = ["INSTALL httpfs;", "LOAD httpfs;"] if remote else []
        if remote and any(re.search(r"\.parq(uet)?$", t) for t in tables.values()):
            # Remote GeoParquet arrives as GEOMETRY, which the ST_ functions
            # need spatial for; a plain CSV does without the download
            initializers += ["INSTALL spatial;", "LOAD spatial;"]
        regions = {m.group(1) for table in tables.values() if (m := _S3_REGION.match(table))}
        if len(regions) > 1:
            # DuckDB has one s3_region per connection
            raise ValueError(
                f"These buckets are in different regions ({', '.join(sorted(regions))}); read each with its own lumen.read."
            )
        initializers += [f"SET s3_region='{region}';" for region in regions]
        source = DuckDBSource(
            initializers=initializers,
            mirrors=mirrors,
            name=PROVIDED_SOURCE_NAME,
            # A mirror is ingested as a table of its own name, which get_tables lists only from tables
            tables={**tables, **{name: f'SELECT * FROM "{name}"' for name in mirrors}},
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
        # So the SQL prompt knows the table is a grid, as with XArraySQLSource
        source.metadata = {**source.metadata, name: {**source.metadata.get(name, {}), "gridded": True}}
    JOINED_GRIDS.add(source)


#: Sources with xarray data joined in. Those tables live only on the
#: connection, so the source cannot be rebuilt from its spec.
JOINED_GRIDS: weakref.WeakSet[Source] = weakref.WeakSet()


def joins_grids(source: Source | None) -> bool:
    """Whether source has xarray data joined in, itself or through a source derived from it."""
    # A derived source (a follow-up's SQL result) shares the connection
    return isinstance(source, DuckDBSource) and any(
        source._connection is joined._connection for joined in JOINED_GRIDS
    )

def _resolve_geo_source(src: str, extension: str) -> Source:
    """Load a geospatial file into an in-memory DuckDBSource.

    Routes GeoJSON/WKT/zipped-shapefile through the same read_geo_file ->
    DuckDB spatial (ST_GeomFromWKB) path the interactive upload uses, so
    startup and uploaded geospatial data behave identically (gh-1900).
    """
    path = Path(src).absolute()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {src}")

    # Deferred: the upload helpers live in lumen.ai, which only geo files need
    from ..ai.controls.ingest.utils import read_geo_file

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
