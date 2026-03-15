from __future__ import annotations

import datetime as dt
import hashlib

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import param

from ..util import get_dataframe_schema
from ..validation import ValidationError, match_suggestion_message
from .base import Source, cached, cached_schema

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore

if TYPE_CHECKING:
    import xarray as xr


class XarraySource(Source):
    """
    Minimal xarray-backed Source implementation.

    This first draft is intentionally conservative:
    - accepts an in-memory xarray.Dataset
    - exposes Dataset.data_vars as logical tables
    - supports coordinate-based filtering via .sel(...)
    - returns pandas DataFrames for downstream compatibility
    """

    dataset = param.Parameter(doc="""
        An xarray.Dataset providing the data variables exposed by the source.
    """)

    uri = param.String(default=None, doc="""
        Optional path or URI to an xarray-readable dataset. Used when no in-memory
        dataset is supplied.
    """)

    dataset_format = param.Selector(default="auto", objects=["auto", "netcdf", "zarr"], doc="""
        Dataset storage format. By default Lumen will infer the format from
        the URI, local store markers, or load kwargs. Set explicitly for
        suffixless or non-standard stores.
    """)

    load_kwargs = param.Dict(default={}, doc="""
        Additional keyword arguments forwarded to xarray.open_dataset/open_zarr.
    """)

    filterable_coords = param.List(default=None, allow_None=True, doc="""
        Optional list of coordinate names that should be exposed in schema and
        allowed in queries. By default all 1D coordinates on a variable are
        queryable.
    """)

    max_rows = param.Integer(default=0, bounds=(0, None), doc="""
        Optional safety guard on the number of rows returned after flattening.
        A value of 0 disables the guard.
    """)

    source_type: ClassVar[str] = "xarray"

    def __init__(self, **params):
        super().__init__(**params)
        self._loaded_dataset = None
        self._validate_dataset()

    def _validate_dataset(self) -> None:
        if xr is None:
            raise ImportError("XarraySource requires the 'xarray' package to be installed.")
        if self.dataset is None and not self.uri:
            raise ValueError("XarraySource requires either a dataset or a uri.")
        if self.dataset is not None and not isinstance(self.dataset, xr.Dataset):
            raise TypeError("XarraySource 'dataset' must be an xarray.Dataset.")

    def _is_windows_path(self, uri: str) -> bool:
        parsed = urlparse(uri)
        return len(parsed.scheme) == 1 and uri[1:2] == ":"

    def _is_remote_uri(self, uri: str) -> bool:
        parsed = urlparse(uri)
        return bool(
            parsed.scheme
            and parsed.scheme not in ("file",)
            and not self._is_windows_path(uri)
        )

    def _uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            location = f"//{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
            local_path = url2pathname(unquote(location))
            if local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
                local_path = local_path[1:]
            return Path(local_path)
        return Path(uri)

    def _resolve_uri(self) -> str:
        assert self.uri is not None
        if self._is_remote_uri(self.uri):
            return self.uri
        if self._is_windows_path(self.uri):
            return self.uri
        path = self._uri_to_path(self.uri)
        if not path.is_absolute():
            path = self.root / path
        return str(path)

    def _is_local_zarr_store(self, uri: str) -> bool:
        path = Path(uri)
        if not path.exists() or not path.is_dir():
            return False
        return (path / ".zgroup").exists() or (path / "zarr.json").exists()

    def _get_dataset_format(self, uri: str) -> str:
        if self.dataset_format != "auto":
            return self.dataset_format
        if self.load_kwargs.get("engine") == "zarr":
            return "zarr"
        parsed = urlparse(uri)
        if parsed.path.endswith(".zarr"):
            return "zarr"
        if not self._is_remote_uri(uri) and self._is_local_zarr_store(uri):
            return "zarr"
        return "netcdf"

    def _open_dataset(self) -> xr.Dataset:
        uri = self._resolve_uri()
        if self._get_dataset_format(uri) == "zarr":
            return xr.open_zarr(uri, **self.load_kwargs)
        return xr.open_dataset(uri, **self.load_kwargs)

    def _get_dataset(self) -> xr.Dataset:
        if self.dataset is not None:
            return self.dataset
        if self._loaded_dataset is None:
            self._loaded_dataset = self._open_dataset()
        return self._loaded_dataset

    def _get_source_hash(self):
        sha = hashlib.sha256()
        for k, v in self.param.values().items():
            if k in ("root",):
                continue
            sha.update(k.encode("utf-8"))
            if k == "uri" and isinstance(v, str) and not self._is_remote_uri(v):
                path = Path(self._resolve_uri())
                if path.exists():
                    sha.update(str(path.stat().st_mtime_ns).encode("utf-8"))
            else:
                sha.update(str(v).encode("utf-8"))
        return sha.hexdigest()

    def close(self) -> None:
        if self._loaded_dataset is not None and hasattr(self._loaded_dataset, "close"):
            self._loaded_dataset.close()
        self._loaded_dataset = None

    def clear_cache(self, *events: param.parameterized.Event):
        self.close()
        super().clear_cache(*events)

    def get_tables(self) -> list[str]:
        return list(self._get_dataset().data_vars)

    def _get_array(self, table: str) -> xr.DataArray:
        tables = self.get_tables()
        dataset = self._get_dataset()
        if table not in dataset.data_vars:
            raise KeyError(f"Table {table!r} not found. Available tables: {tables}.")
        arr = dataset[table]
        return arr if arr.name == table else arr.rename(table)

    def _iter_queryable_coords(self, arr: xr.DataArray) -> list[tuple[str, xr.DataArray]]:
        coords = []
        allowed = set(self.filterable_coords) if self.filterable_coords is not None else None
        for coord_name, coord in arr.coords.items():
            if coord.ndim != 1:
                continue
            if allowed is not None and coord_name not in allowed:
                continue
            coords.append((coord_name, coord))
        return coords

    def _iter_auxiliary_coords(self, arr: xr.DataArray) -> list[tuple[str, xr.DataArray]]:
        queryable = {name for name, _ in self._iter_queryable_coords(arr)}
        return [(coord_name, coord) for coord_name, coord in arr.coords.items() if coord_name not in queryable]

    def _dtype_to_schema(self, dtype: np.dtype) -> dict[str, Any]:
        if np.issubdtype(dtype, np.bool_):
            return {"type": "boolean"}
        if np.issubdtype(dtype, np.integer):
            return {"type": "integer"}
        if np.issubdtype(dtype, np.floating):
            return {"type": "number"}
        if np.issubdtype(dtype, np.datetime64):
            return {"type": "string", "format": "datetime"}
        return {"type": "string"}

    def _coord_schema(self, coord: xr.DataArray) -> dict[str, Any]:
        schema = self._dtype_to_schema(coord.dtype)
        values = coord.values
        if values.ndim == 1 and len(values) and np.issubdtype(coord.dtype, np.number):
            schema["inclusiveMinimum"] = values.min().item()
            schema["inclusiveMaximum"] = values.max().item()
        return schema

    def _get_sampled_schema(self, arr: xr.DataArray, table: str, limit: int | None, shuffle: bool) -> dict[str, Any]:
        df = arr.to_dataframe(name=table).reset_index()
        ordered_columns = [*(name for name, _ in self._iter_queryable_coords(arr)), table]
        df = df[[col for col in ordered_columns if col in df.columns]]
        if shuffle and not df.empty:
            sample_size = min(limit or len(df), len(df))
            df = df.sample(n=sample_size, random_state=0)
        elif limit is not None:
            df = df.head(limit)
        return get_dataframe_schema(df)["items"]["properties"]

    @cached_schema
    def get_schema(
        self, table: str | None = None, limit: int | None = None, shuffle: bool = False
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        tables = [table] if table is not None else self.get_tables()
        schemas: dict[str, dict[str, Any]] = {}
        names = self.get_tables()

        for name in tables:
            try:
                arr = self._get_array(name)
            except KeyError as e:
                msg = f"{type(self).__name__} does not contain {name!r}."
                msg = match_suggestion_message(name, names, msg)
                raise ValidationError(msg) from e
            if limit is not None or shuffle:
                schema = self._get_sampled_schema(arr, name, limit, shuffle)
            else:
                schema = {}
                for coord_name, coord in self._iter_queryable_coords(arr):
                    schema[coord_name] = self._coord_schema(coord)
                schema[name] = self._dtype_to_schema(arr.dtype)
            schema["__len__"] = self._get_result_length(arr)
            schemas[name] = schema

        return schemas if table is None else schemas[table]

    def _normalize_query_value(self, value: Any) -> Any:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    "Tuple query values must contain exactly two elements for range selection."
                )
            return slice(value[0], value[1])
        return value

    def _normalize_scalar_for_coord(self, coord: xr.DataArray, value: Any) -> Any:
        if np.issubdtype(coord.dtype, np.datetime64):
            if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
                return np.datetime64(value)
            if isinstance(value, str | dt.datetime | np.datetime64):
                return np.datetime64(value)
        return value

    def _normalize_selector(self, coord: xr.DataArray, value: Any) -> Any:
        if isinstance(value, slice):
            return slice(
                self._normalize_scalar_for_coord(coord, value.start),
                self._normalize_scalar_for_coord(coord, value.stop),
                value.step,
            )
        if isinstance(value, list):
            return [self._normalize_scalar_for_coord(coord, v) for v in value]
        return self._normalize_scalar_for_coord(coord, value)

    def _select_range(self, arr: xr.DataArray, key: str, selector: slice) -> xr.DataArray:
        coord = arr.coords[key]
        if coord.ndim != 1:
            raise ValueError(
                f"Filtering on non-1D coordinate {key!r} is not supported; got {coord.ndim}D coordinate."
            )
        values = coord.values
        if len(values) < 2:
            return arr.sel({key: selector})

        start = selector.start
        stop = selector.stop
        if start is None or stop is None:
            return arr.sel({key: selector})

        increasing = bool(values[0] <= values[-1])
        if increasing:
            ordered = slice(start, stop)
        else:
            ordered = slice(stop, start) if start < stop else slice(start, stop)
        return arr.sel({key: ordered})

    def _is_range_list(self, value: Any) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(v, tuple) and len(v) == 2 for v in value)
        )

    def _select_range_list(self, arr: xr.DataArray, key: str, ranges: list[tuple[Any, Any]]) -> xr.DataArray:
        coord = arr.coords[key]
        if coord.ndim != 1:
            raise ValueError(
                f"Range-list filtering is only supported for 1D coordinates, got {coord.ndim}D for {key!r}."
            )
        values = coord.values
        mask = np.zeros(len(values), dtype=bool)
        for start, end in ranges:
            start = self._normalize_scalar_for_coord(coord, start)
            end = self._normalize_scalar_for_coord(coord, end)
            mask |= (values >= start) & (values <= end)
        return arr.sel({key: values[mask]})

    def _apply_query(self, arr: xr.DataArray, query: dict[str, Any]) -> xr.DataArray:
        queryable = dict(self._iter_queryable_coords(arr))
        for key, value in query.items():
            if key.startswith("__"):
                continue
            if key not in queryable:
                if key in arr.coords and arr.coords[key].ndim != 1:
                    raise ValueError(
                        f"Filtering on non-1D coordinate {key!r} is not supported; got {arr.coords[key].ndim}D coordinate."
                    )
                raise ValueError(
                    f"Unsupported query key {key!r} for table {arr.name!r}. "
                    f"Expected one of: {list(queryable)}."
                )
            coord = queryable[key]
            if self._is_range_list(value):
                arr = self._select_range_list(arr, key, value)
                continue
            normalized = self._normalize_query_value(value)
            normalized = self._normalize_selector(coord, normalized)
            try:
                if isinstance(normalized, slice):
                    arr = self._select_range(arr, key, normalized)
                else:
                    arr = arr.sel({key: normalized})
            except Exception as e:
                raise ValueError(
                    f"Could not apply query {key}={value!r} to table {arr.name!r}."
                ) from e
        return arr

    def _get_result_length(self, arr: xr.DataArray) -> int:
        if arr.ndim == 0:
            return 1
        return int(np.prod(arr.shape, dtype=np.int64))

    def _enforce_max_rows(self, arr: xr.DataArray, table: str) -> None:
        if not self.max_rows:
            return
        rows = self._get_result_length(arr)
        if rows > self.max_rows:
            raise ValueError(
                f"Query result for table {table!r} produced {rows} rows, "
                f"which exceeds max_rows={self.max_rows}."
            )

    def _to_dataframe(self, arr: xr.DataArray, table: str) -> pd.DataFrame:
        arr = arr if arr.name == table else arr.rename(table)
        self._enforce_max_rows(arr, table)
        if arr.ndim == 0:
            return pd.DataFrame({table: [arr.item()]})
        df = arr.to_dataframe(name=table).reset_index()
        ordered_columns = [*(name for name, _ in self._iter_queryable_coords(self._get_array(table))), table]
        df = df[[col for col in ordered_columns if col in df.columns]]
        return df

    def _get_table_metadata(self, tables: list[str]) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        for table in tables:
            arr = self._get_array(table)
            columns: dict[str, dict[str, Any]] = {}
            queryable_coords = self._iter_queryable_coords(arr)
            for coord_name, coord in queryable_coords:
                coord_meta = {
                    "description": coord.attrs.get("long_name") or coord.attrs.get("description") or f"{coord_name} coordinate",
                    "data_type": str(coord.dtype),
                }
                if "units" in coord.attrs:
                    coord_meta["unit"] = coord.attrs["units"]
                columns[coord_name] = coord_meta

            value_meta = {
                "description": arr.attrs.get("long_name") or arr.attrs.get("description") or table,
                "data_type": str(arr.dtype),
            }
            if "units" in arr.attrs:
                value_meta["unit"] = arr.attrs["units"]
            columns[table] = value_meta

            metadata[table] = {
                "description": arr.attrs.get("long_name") or arr.attrs.get("description") or table,
                "dims": list(arr.dims),
                "queryable_coords": [coord_name for coord_name, _ in queryable_coords],
                "auxiliary_coords": [coord_name for coord_name, _ in self._iter_auxiliary_coords(arr)],
                "columns": columns,
            }
        return metadata

    @cached
    def get(self, table: str, **query) -> pd.DataFrame:
        arr = self._get_array(table)
        arr = self._apply_query(arr, query)
        return self._to_dataframe(arr, table)
