"""
STACSource -- a lazy STAC catalog source for Lumen.

Points Lumen at a SpatioTemporal Asset Catalog (STAC) API. Every cataloged
collection is exposed as a selectable dataset via ``get_tables()`` without
fetching any data. When a collection is queried (by the AI agent or directly),
it is resolved to an xarray dataset through xpystac and handed to an
``XArraySQLSource`` under the hood, with no further configuration.

To use this source, install the optional STAC dependencies::

    pip install 'lumen[stac]'

Notes
-----
Catalog discovery (listing collections and their metadata) works against any
public STAC API with no credentials. Resolving a collection to data requires
its assets to be readable: either genuinely public, or accessible via a
``patch_url`` signer / ``open_kwargs`` storage options. For Planetary Computer
endpoints the ``planetary_computer.sign`` signer is auto-wired when that
package is installed; note that some PC Zarr/kerchunk assets additionally
require Azure storage credentials that xpystac does not propagate, so those
specific datasets may not resolve without extra storage configuration.
"""

from __future__ import annotations

import re
import warnings

from typing import Any, ClassVar

import param
import sqlglot

from ..transforms.sql import SQLLimit
from .base import BaseSQLSource
from .xarray_sql import XArraySQLSource

try:
    import pystac_client
    import xarray as xr
    import xpystac  # noqa: F401  (registers the ``stac`` xarray engine)

    from pystac.extensions.datacube import DatacubeExtension
except ImportError as e:
    raise ImportError(
        "STACSource requires the 'pystac-client', 'xpystac' and 'xarray' "
        "packages. Install them with: pip install 'lumen[stac]'"
    ) from e

# Optional: only used to auto-sign Planetary Computer assets. Suppress its
# import-time pydantic deprecation so it does not surface under -W error.
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import planetary_computer
except ImportError:
    planetary_computer = None


class STACSource(BaseSQLSource):
    """
    Lazy STAC catalog source.

    ``get_tables()`` lists STAC collection IDs. The first time a collection is
    queried it is resolved to an xarray dataset (via xpystac) and wrapped in an
    ``XArraySQLSource``; subsequent queries reuse the cached delegate. A table
    is referenced as ``collection`` (the collection's first data variable) or
    ``collection:variable`` for a specific variable.

    Examples
    --------
    >>> source = STACSource(url="https://planetarycomputer.microsoft.com/api/stac/v1")
    >>> source.get_tables()
    ['daymet-daily-hi', 'era5-pds', ...]
    >>> df = source.get("era5-pds:air_temperature_at_2_metres")
    """

    source_type: ClassVar[str | None] = "stac"

    # Asset media-type / role hints that indicate an xarray-openable asset.
    _xarray_media_tokens: ClassVar[tuple[str, ...]] = (
        "zarr", "netcdf", "x-hdf", "kerchunk", "reference",
    )
    _xarray_roles: ClassVar[frozenset[str]] = frozenset(
        {"data", "zarr", "references", "index"}
    )

    url = param.String(default=None, allow_None=True, doc="""
        Root URL of a STAC API endpoint (e.g. a STAC API landing page).""")

    collections = param.List(default=None, allow_None=True, doc="""
        Optional subset of collection IDs to expose. None exposes all.""")

    chunks = param.Parameter(default="auto", doc="""
        Dask chunk specification forwarded to xr.open_dataset and the
        delegated XArraySQLSource.""")

    open_kwargs = param.Dict(default={}, doc="""
        Additional keyword arguments forwarded to xr.open_dataset().""")

    patch_url = param.Callable(default=None, allow_None=True, doc="""
        Optional callable that rewrites/signs asset URLs before reading
        (forwarded to xpystac as ``patch_url``). For Planetary Computer this
        defaults to planetary_computer.sign when that package is installed.""")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The SQL expression template for table queries.""")

    tables = param.ClassSelector(class_=(list, dict), default=None, allow_None=True, doc="""
        Optional static table list. STACSource discovers tables dynamically
        via get_tables() (one per STAC collection), so this defaults to None;
        the parameter exists for compatibility with consumers (e.g.
        SQLAgent._execute_query) that inspect source.tables directly.""")

    default_limit = param.Integer(default=100_000, allow_None=True, doc="""
        Row cap injected as a SQLLimit transform on get() so a naive call
        against a TB-scale STAC collection does not materialize the whole
        xarray grid. Set to None to disable, or pass sql_transforms=[...]
        explicitly to override (the caller's transforms pass through
        unchanged).""")

    def __init__(self, **params):
        super().__init__(**params)
        if not self.url:
            raise ValueError(
                "STACSource requires a 'url' pointing to a STAC API endpoint."
            )
        self._client = pystac_client.Client.open(self.url)
        self._collection_cache: dict[str, Any] = {}
        self._delegates: dict[str, XArraySQLSource] = {}
        # Maps a resolved delegate's variable table name to its collection ID,
        # so raw SQL in execute() can be routed to the right delegate.
        self._var_index: dict[str, str] = {}
        # Best-effort variable name -> collection id index built from the
        # datacube extension during get_tables(). Lets execute() auto-resolve
        # the owning collection when a caller's SQL references a variable
        # name without first calling _resolve().
        self._var_pre_index: dict[str, str] = {}

    # ---- Catalog listing ----

    def get_tables(self) -> list[str]:
        ids = []
        for collection in self._client.get_collections():
            # Cache the full object: metadata for every collection then costs
            # one get_collections() request, not one request per collection.
            self._collection_cache[collection.id] = collection
            ids.append(collection.id)
            # Pre-index this collection's declared variables so execute()
            # can auto-resolve from a bare variable name. Best-effort: a
            # collection without the datacube extension just contributes
            # nothing here.
            for var_name in self._extract_columns(collection):
                self._var_pre_index.setdefault(var_name, collection.id)
        if self.collections:
            allowed = set(self.collections)
            ids = [cid for cid in ids if cid in allowed]
        return sorted(ids)

    def _get_collection(self, collection_id: str):
        if collection_id not in self._collection_cache:
            self._collection_cache[collection_id] = self._client.get_collection(
                collection_id
            )
        return self._collection_cache[collection_id]

    # ---- Resolution ----

    def _is_xarray_asset(self, asset: Any) -> bool:
        media_type = (asset.media_type or "").lower()
        if media_type:
            # When media_type is declared, trust it: roles like 'data' are
            # generic enough that they alone would whitelist COG/JPEG/etc.
            # assets (e.g. NAIP scenes) that xpystac cannot open.
            return any(token in media_type for token in self._xarray_media_tokens)
        # No media_type: fall back to STAC role hints.
        roles = {r.lower() for r in (asset.roles or [])}
        return bool(roles & self._xarray_roles)

    @staticmethod
    def _asset_accessibility(asset: Any) -> int:
        """
        Rank an asset by how readable it is without extra credentials.

        Public http(s) assets are preferred over cloud-protocol assets
        (abfs/s3/gs/az/adl) which need an extra filesystem and credentials.
        Lower is better.
        """
        href = (asset.href or "").lower()
        if href.startswith(("http://", "https://")):
            return 0
        if href.startswith(("abfs://", "az://", "adl://", "s3://", "gs://", "gcs://")):
            return 2
        return 1

    def _openable_candidates(self, collection: Any) -> list:
        """
        Ranked list of pystac objects xpystac may open for ``collection``.

        Collection-level xarray-openable assets first (most publicly
        accessible first), then the collection's first item as a fallback.
        xpystac supports Asset and Item objects but not Collections directly.
        """
        candidates = sorted(
            (a for a in collection.assets.values() if self._is_xarray_asset(a)),
            key=self._asset_accessibility,
        )
        # Fast path: a well-formed collection (Daymet, ERA5, ...) carries
        # xarray-openable assets at the collection level; skip the per-item
        # search to avoid an extra STAC API round-trip on every resolve.
        if candidates:
            return candidates
        result = self._client.search(collections=[collection.id], max_items=1)
        first_item = next(result.items(), None)
        first_item_assets = (
            list(first_item.assets.values()) if first_item is not None else []
        )
        item_has_xr_asset = any(self._is_xarray_asset(a) for a in first_item_assets)
        if first_item is not None and item_has_xr_asset:
            candidates.append(first_item)
        if not candidates:
            observed = sorted({
                (a.media_type or "(unset)")
                for a in (*collection.assets.values(), *first_item_assets)
            })
            raise ValueError(
                f"STAC collection {collection.id!r} has no xarray-openable "
                f"asset (saw media_types: {observed}); for COG or other image "
                "collections use stackstac or odc-stac instead."
            )
        return candidates

    def _object_signer(self):
        """
        Signer applied to the pystac object before xpystac reads it. For
        Planetary Computer endpoints this is planetary_computer.sign when
        that optional package is installed (PC gates its storage and xpystac
        does not propagate a URL patch_url to the Zarr metadata fetch, so the
        object itself must be signed).
        """
        if planetary_computer is None:
            return None
        if "planetarycomputer.microsoft.com" not in (self.url or ""):
            return None
        return planetary_computer.sign

    def _open_dataset(self, obj: Any) -> xr.Dataset:
        """Open a pystac object as an xarray dataset via the xpystac engine."""
        kwargs = dict(self.open_kwargs)
        if self.patch_url is not None:
            kwargs["patch_url"] = self.patch_url
        return xr.open_dataset(
            obj, engine="stac", chunks=self.chunks, **kwargs
        )

    def _resolve(self, collection_id: str) -> XArraySQLSource:
        if collection_id in self._delegates:
            return self._delegates[collection_id]
        collection = self._get_collection(collection_id)
        candidates = self._openable_candidates(collection)
        signer = self._object_signer()
        dataset, last_error = None, None
        for obj in candidates:
            try:
                signed = signer(obj) if signer is not None else obj
                ds = self._open_dataset(signed)
            except Exception as e:
                last_error = e
                continue
            if ds is not None and len(ds.data_vars):
                dataset = ds
                break
            last_error = last_error or ValueError("no data variables resolved")
        if dataset is None:
            raise ValueError(
                f"STAC collection {collection_id!r} could not be opened as an "
                f"xarray dataset via xpystac ({type(last_error).__name__}: "
                f"{last_error}). Its assets may need a reader (e.g. "
                "stackstac/odc-stac for COGs) or storage credentials/extra "
                "filesystem packages (e.g. adlfs) not configured here."
            ) from last_error
        # CF "bounds" variables (a small vertices dim like nv/bnds) are not
        # queryable tables and break tabular chunk handling; drop them.
        bounds_dims = [
            d for d in dataset.dims if d in ("nv", "bnds", "bounds", "vertices")
        ]
        if bounds_dims:
            dataset = dataset.drop_dims(bounds_dims, errors="ignore")
        # Drop 0-dim grid-mapping/CRS scalar vars (e.g. lambert_conformal_conic,
        # spatial_ref): not queryable tables and break per-variable chunking.
        scalar_vars = [v for v in dataset.data_vars if dataset[v].ndim == 0]
        if scalar_vars:
            dataset = dataset.drop_vars(scalar_vars)
        # Cloud Zarr stores often have ragged/inconsistent dask chunks;
        # XArraySQLSource (DataFusion) requires uniform chunks.
        dataset = dataset.unify_chunks()
        delegate = XArraySQLSource.from_dataset(dataset, chunks=self.chunks)
        self._delegates[collection_id] = delegate
        for table in delegate.get_tables():
            self._var_index.setdefault(table, collection_id)
        return delegate

    def _split(self, table: str) -> tuple[str, str | None]:
        table = self.normalize_table(table)
        if ":" in table:
            collection_id, variable = table.split(":", 1)
            return collection_id, variable
        return table, None

    def _delegate_table(self, table: str) -> tuple[XArraySQLSource, str]:
        collection_id, variable = self._split(table)
        delegate = self._resolve(collection_id)
        if variable is None:
            variable = delegate.get_tables()[0]
        return delegate, variable

    # ---- BaseSQLSource overrides (fully delegated) ----

    def normalize_table(self, table: str) -> str:
        return table.strip('"').strip("'").strip("`")

    def get_sql_expr(self, table: str | dict):
        # Cheap string op only: the AI's catalog gathering calls this for
        # every result, so it must never trigger resolution/network.
        return self.sql_expr.format(table=self.normalize_table(table))

    def get_schema(self, table=None, limit=None, shuffle=False):
        if table is None:
            # Stay lazy: do not resolve every collection just to list them.
            return {cid: {} for cid in self.get_tables()}
        # Delegate to the resolved XArraySQLSource, which already builds the
        # schema from a bounded sample with an O(1) metadata row count.
        # Best-effort: a collection that is not xarray-openable must not
        # abort catalog enumeration; data access via get() still fails loudly.
        try:
            delegate, variable = self._delegate_table(table)
            return delegate.get_schema(variable, limit, shuffle)
        except Exception:
            return {}

    def get(self, table: str, **query):
        delegate, variable = self._delegate_table(table)
        if self.default_limit is not None and "sql_transforms" not in query:
            query["sql_transforms"] = [SQLLimit(limit=self.default_limit)]
        return delegate.get(variable, **query)

    def execute(self, sql_query: str, params=None, *args, **kwargs):
        # Parse the SQL and auto-resolve any referenced variable whose owning
        # collection is known from the datacube pre-index. This frees callers
        # from having to invoke _resolve() (private) before issuing raw SQL.
        parsed = None
        try:
            parsed = sqlglot.parse_one(sql_query, dialect="duckdb")
            referenced = {t.name for t in parsed.find_all(sqlglot.exp.Table)}
        except sqlglot.errors.ParseError:
            referenced = set()
        if referenced and not self._var_pre_index:
            # Lazy one-shot catalog enumeration to populate the pre-index.
            self.get_tables()
        # Map collection-id references in the SQL to the delegate's first
        # variable so callers can write "SELECT * FROM <collection_id>"
        # without knowing which xarray variable backs the collection.
        rewrites: dict[str, str] = {}
        for ref in referenced - set(self._var_index):
            owner = self._var_pre_index.get(ref)
            if owner is not None and owner not in self._delegates:
                self._resolve(owner)
            elif ref in self._collection_cache:
                delegate = (
                    self._delegates[ref] if ref in self._delegates
                    else self._resolve(ref)
                )
                first_var = delegate.get_tables()[0]
                rewrites[ref] = first_var
        if rewrites and parsed is not None:
            for tbl in parsed.find_all(sqlglot.exp.Table):
                if tbl.name in rewrites:
                    tbl.set(
                        "this",
                        sqlglot.exp.to_identifier(rewrites[tbl.name], quoted=True),
                    )
            sql_query = parsed.sql(dialect="duckdb")

        matched = {
            self._var_index[name]
            for name in self._var_index
            if re.search(rf"\b{re.escape(name)}\b", sql_query)
        }
        if len(matched) == 1:
            collection_id = next(iter(matched))
        elif not matched and len(self._delegates) == 1:
            collection_id = next(iter(self._delegates))
        elif not self._delegates:
            known_collections = sorted(self._collection_cache)
            known_variables = sorted(self._var_pre_index)
            referenced_msg = sorted(referenced) if referenced else "(none parsed)"
            raise ValueError(
                f"Cannot run SQL: referenced tables {referenced_msg} are not "
                "a known STAC collection or variable in this catalog. "
                f"Known collections: {known_collections}. "
                f"Known variables (from the datacube extension): "
                f"{known_variables}. "
                "Resolve a collection by calling source.get(collection_id) or "
                "source.get_schema(collection_id) first."
            )
        else:
            raise ValueError(
                "Could not route SQL to a single STAC collection; query one "
                "resolved collection at a time (cross-collection SQL is "
                "unsupported)."
            )
        return self._delegates[collection_id].execute(
            sql_query, params, *args, **kwargs
        )

    def create_sql_expr_source(self, tables: dict[str, str], params=None, **kwargs):
        # Resolve the referenced collection(s) and remap the SQL onto the
        # delegate's variable table so the derived source is queryable. This
        # backs the "Select Data to Explore" flow (explorer.create_sql_output).
        remapped: dict[str, str] = {}
        delegate = None
        for new_table, sql_expr in tables.items():
            match = re.search(r'FROM\s+"?([^"\s]+)"?', sql_expr, re.IGNORECASE)
            ref = match.group(1) if match else new_table
            collection_id, variable = self._split(ref)
            resolved = self._resolve(collection_id)
            if variable is None:
                variable = resolved.get_tables()[0]
            if delegate is None:
                delegate = resolved
            elif resolved is not delegate:
                raise ValueError(
                    "Cannot derive a SQL source spanning multiple STAC "
                    "collections; explore one collection at a time."
                )
            remapped[new_table] = re.sub(
                r'"?' + re.escape(ref) + r'"?', f'"{variable}"', sql_expr, count=1
            )
        if delegate is None:
            raise ValueError("No tables provided to create_sql_expr_source.")
        return delegate.create_sql_expr_source(remapped, params, **kwargs)

    @staticmethod
    def _extract_columns(collection: Any) -> dict[str, dict[str, str]]:
        """
        Best-effort variable columns from the STAC datacube extension.

        Reads ``cube:variables`` (no data fetched) so the AI sees that a
        collection is queryable and routes to the SQL agent rather than
        trying to fetch it externally. A malformed extension on one
        collection must never abort catalog metadata, hence the guard.
        """
        try:
            if not DatacubeExtension.has_extension(collection):
                return {}
            columns = {}
            for name, var in DatacubeExtension.ext(collection).variables.items():
                description = var.description or ""
                if var.unit:
                    description = (
                        f"{description} [{var.unit}]" if description
                        else f"[{var.unit}]"
                    )
                columns[name] = {
                    "description": description,
                    "data_type": var.var_type or "",
                }
            return columns
        except Exception:
            return {}

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """
        Build AI-discovery metadata from STAC collection attributes.

        Stays lazy: only collection-level STAC metadata is read (title,
        description, extent, license, keywords, and datacube variable
        names). Row counts are left None until the collection is resolved.
        """
        metadata = {}
        for collection_id in tables:
            collection = self._get_collection(collection_id)
            parts: list[str] = []
            if collection.title:
                parts.append(collection.title)
            if collection.description:
                parts.append(collection.description)

            extent = collection.extent
            if extent.spatial.bboxes:
                parts.append(f"Spatial extent (bbox): {extent.spatial.bboxes[0]}")
            if extent.temporal.intervals:
                start, end = extent.temporal.intervals[0]
                parts.append(f"Temporal extent: {start} to {end or 'present'}")

            if collection.license:
                parts.append(f"License: {collection.license}")
            if collection.keywords:
                parts.append(f"Keywords: {', '.join(collection.keywords)}")

            metadata[collection_id] = {
                "description": " | ".join(str(part) for part in parts),
                "columns": self._extract_columns(collection),
                "rows": None,
            }
        return metadata

    def close(self):
        for delegate in self._delegates.values():
            delegate.close()
        self._delegates.clear()
        self._collection_cache.clear()
        self._var_index.clear()
