from __future__ import annotations

import datetime as dt
import importlib
import io
import json
import os
import re
import sys
import unicodedata

from functools import partial, wraps
from logging import getLogger
from pathlib import Path
from subprocess import check_output

import bokeh
import narwhals as narwhals_any
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import panel as pn
import param
import pyarrow as pa
import yaml

from jinja2 import DebugUndefined, Environment, Undefined
from packaging.version import Version
from pandas.core.dtypes.dtypes import CategoricalDtype
from panel.io.state import state

log = getLogger(__name__)

bokeh3 = Version(bokeh.__version__) > Version("3.0")
param2 = Version(param.__version__) > Version("2.0rc1")

disallow_refs = {'allow_refs': False} if param2 else {}

VARIABLE_RE = re.compile(r'\$variables\.([a-zA-Z_]\w*)')


class NumpyDumper(yaml.SafeDumper):
    """A YAML Dumper that converts NumPy scalars to native Python types."""

    def represent_data(self, data):
        # Check for common NumPy scalar types and convert them to Python equivalents
        if isinstance(data, (np.integer,)):
            data = int(data)
        elif isinstance(data, (np.floating,)):
            data = float(data)
        elif isinstance(data, (np.bool_,)):
            data = bool(data)
        elif isinstance(data, (np.complexfloating,)):
            data = complex(data)
        elif isinstance(data, (np.bytes_,)):
            data = bytes(data)
        elif isinstance(data, (np.str_,)):
            data = str(data)
        elif isinstance(data, np.datetime64):
            data = str(data)
        return super().represent_data(data)

def is_narwhals(obj):
    """Return True if obj is a narwhals DataFrame or LazyFrame.

    Tested against the bare namespace on purpose: narwhals.stable.v1 and v2
    both subclass it, so this recognises a frame whichever namespace the
    caller wrapped it with.
    """
    return isinstance(obj, (narwhals_any.DataFrame, narwhals_any.LazyFrame))


def is_lazyframe(obj):
    """Return True if obj is a narwhals LazyFrame, from any narwhals namespace.

    Callers need this before len(), slicing or item access, none of which a
    LazyFrame supports.
    """
    return isinstance(obj, narwhals_any.LazyFrame)


def as_narwhals(df):
    """Return df wrapped as a narwhals frame, or df unchanged if it cannot be.

    Wrapping a pandas DataFrame is lossless and free: to_native() returns the
    caller's original object. Dask frames are deliberately left alone because
    narwhals maps them to a LazyFrame, which would bypass the hasattr(df,
    'compute') branches the dask paths rely on.
    """
    if df is None or is_narwhals(df):
        return df
    dd = try_import('dask.dataframe', load=False)
    if dd is not None and isinstance(df, dd.DataFrame):
        return df
    return nw.from_native(df, pass_through=True)


def collect_lazy(df):
    """Return df with any lazy frame collected, in its own backend.

    A Source must not hand a lazy frame further in: everything downstream
    calls len(), .iloc or .index on it, none of which a LazyFrame answers to,
    and the failure would surface far from its cause.
    """
    narwhals_df = as_narwhals(df)
    if is_lazyframe(narwhals_df):
        return narwhals_df.collect().to_native()
    return df


DATAFRAME_BACKENDS = ['pandas', 'polars', 'pyarrow']

# The pandas dtype that holds each arrow integer and boolean type without
# widening it. numpy has no missing value for either, which is the whole
# problem _to_pandas below exists to solve.
_NULLABLE_DTYPES = {
    getattr(pa, f'{prefix}int{width}')(): pd.api.types.pandas_dtype(
        f'{prefix.upper()}Int{width}'
    )
    for prefix in ('', 'u') for width in (8, 16, 32, 64)
}
_NULLABLE_DTYPES[pa.bool_()] = pd.BooleanDtype()


def _to_pandas(narwhals_df):
    """Convert to pandas without widening an integer or boolean column.

    Every backend but pandas holds nulls in the column's own type, and
    ``to_pandas`` lands those on numpy, which has no missing value for an
    integer or a boolean. An id comes back as 3.0 and, past 2**53, as a
    different number entirely. The columns that would widen come across
    through arrow as the pandas nullable dtype instead.

    Only those columns are rerouted. Sending the whole frame through arrow
    would be shorter and would break more: pyarrow cannot convert the
    dictionary indices behind a polars categorical, which polars itself
    converts fine.
    """
    df = narwhals_df.to_pandas()
    for name, dtype in narwhals_df.collect_schema().items():
        widened = df[name].dtype.kind in 'fO'
        if widened and (dtype.is_integer() or isinstance(dtype, nw.Boolean)):
            df[name] = narwhals_df[name].to_arrow().to_pandas(
                types_mapper=_NULLABLE_DTYPES.get
            )
    return df


def to_backend(df, backend):
    """Return df as a frame of the named dataframe library.

    A backend of None leaves the frame as whatever produced it. Converting
    costs a full copy, so this is only worth asking for at a boundary where
    the consumer needs a specific library.
    """
    if backend is None or df is None:
        return df
    narwhals_df = as_narwhals(df)
    if not is_narwhals(narwhals_df):
        return df
    if is_lazyframe(narwhals_df):
        narwhals_df = narwhals_df.collect()
    if narwhals_df.implementation.name.lower() == backend:
        return df
    if backend == 'pandas':
        return _to_pandas(narwhals_df)
    if backend == 'polars':
        return narwhals_df.to_polars()
    return narwhals_df.to_arrow()


def as_pandas(df):
    """Return df as a pandas DataFrame, collecting it first if it is lazy.

    The boundary for consumers that need real pandas: hvplot's accessor,
    Panel's Tabulator, Perspective and Vega panes, and the LLM schema summary.
    A pandas frame is returned untouched, which also keeps a GeoDataFrame from
    being flattened into a plain DataFrame on the way through.
    """
    if isinstance(df, pd.DataFrame):
        return df
    narwhals_df = as_narwhals(df)
    if not is_narwhals(narwhals_df):
        return df
    if is_lazyframe(narwhals_df):
        narwhals_df = narwhals_df.collect()
    return _to_pandas(narwhals_df)


def _narwhals_dataframe_schema(df, columns=None):
    """Return a JSON schema for a narwhals-wrapped frame.

    Mirrors get_dataframe_schema for the dataframe libraries pandas cannot
    describe. It is a separate function rather than the shared implementation
    because the pandas path covers two things narwhals has no representation
    for: geopandas geometry columns, and pandas categorical categories that
    no row uses.
    """
    schema = {'type': 'array', 'items': {'type': 'object', 'properties': {}}}
    if is_lazyframe(df):
        df = df.collect()
    df_schema = df.collect_schema()
    empty = len(df) == 0
    properties = schema['items']['properties']
    names = df_schema.names() if columns is None else columns

    # One query for every bound rather than two per column: a frame with a
    # hundred numeric columns is otherwise two hundred separate query plans.
    bounds = {}
    if not empty:
        bounded = [
            n for n in names
            if df_schema[n].is_numeric() or isinstance(df_schema[n], (nw.Datetime, nw.Date))
        ]
        if bounded:
            # Aliased positionally because a column may legally be named
            # anything, including whatever we would otherwise build a key from.
            wanted = [(name, agg) for name in bounded for agg in ('min', 'max')]
            row = df.select(*[
                getattr(nw.col(name), agg)().alias(str(i))
                for i, (name, agg) in enumerate(wanted)
            ]).to_dict(as_series=False)
            bounds = {
                pair: nw.to_py_scalar(row[str(i)][0])
                for i, pair in enumerate(wanted)
            }

    for name in names:
        dtype = df_schema[name]
        temporal = isinstance(dtype, (nw.Datetime, nw.Date))
        if temporal:
            if empty:
                vmin = vmax = pd.NaT
            else:
                vmin = bounds[name, 'min']
                vmax = bounds[name, 'max']
            # An all-null column has no min, and pandas renders its NaT as the
            # string 'NaT', so match that rather than emitting null.
            properties[name] = {
                'type': 'string',
                'inclusiveMinimum': 'NaT' if vmin is None else vmin.isoformat(),
                'inclusiveMaximum': 'NaT' if vmax is None else vmax.isoformat(),
                'format': 'datetime',
            }
        elif dtype.is_numeric():
            # pandas leaves the type unset for an empty frame, and auto_filters
            # keys off that, so an empty frame has to look the same here.
            kind, vmin, vmax = None, float('NaN'), float('NaN')
            if not empty:
                cast = int if dtype.is_integer() else float
                kind = 'integer' if dtype.is_integer() else 'number'
                try:
                    vmin = cast(bounds[name, 'min'])
                    vmax = cast(bounds[name, 'max'])
                except Exception:
                    vmin = vmax = float('NaN')
            properties[name] = {
                'type': kind, 'inclusiveMinimum': vmin, 'inclusiveMaximum': vmax
            }
        elif isinstance(dtype, nw.Boolean):
            properties[name] = {'type': 'boolean'}
        elif isinstance(dtype, nw.Enum):
            properties[name] = {'type': 'string', 'enum': list(dtype.categories)}
        elif isinstance(dtype, (nw.String, nw.Categorical, nw.Object)):
            cats = [] if empty else df[name].unique(maintain_order=True).to_list()
            properties[name] = {'type': 'string', 'enum': cats}
        # Anything else (duration, binary, list, struct, unknown) is left out
        # of the schema: a widget built on a dtype no filter understands is
        # worse than none. The pandas path also omits these, except for time
        # columns, which it happens to reach through its object-dtype branch.
    return schema


def get_dataframe_schema(df, columns=None):
    """
    Returns a JSON schema optionally filtered by a subset of the columns.

    Parameters
    ----------
    df : pandas.DataFrame, dask.DataFrame or any frame narwhals supports
        The DataFrame to describe with the schema
    columns: list(str) or None
        List of columns to include in schema

    Returns
    -------
    dict
        The JSON schema describing the DataFrame
    """
    if df is not None and not isinstance(df, pd.DataFrame):
        # pandas frames keep the path below: it describes geometry columns and
        # unused categorical categories, neither of which survives narwhals.
        narwhals_df = as_narwhals(df)
        if is_narwhals(narwhals_df):
            return _narwhals_dataframe_schema(narwhals_df, columns)

    if 'dask.dataframe' in sys.modules:
        import dask.dataframe as dd
        is_dask = isinstance(df, dd.DataFrame)
    else:
        is_dask = False

    schema = {'type': 'array', 'items': {'type': 'object', 'properties': {}}}
    if df is None:
        return schema

    if columns is None:
        columns = list(df.columns)

    geom_cols = set(geometry_columns(df))

    properties = schema['items']['properties']
    for name in columns:
        dtype = df.dtypes[name]
        column = df[name]
        if name in geom_cols:
            geom_type = 'unknown'
            crs = None
            if not (df.empty or is_dask):
                non_null = column.dropna()
                if len(non_null):
                    geom_type = non_null.iloc[0].geom_type
                # crs lets a consumer decide whether a basemap adds context: a
                # geographic (lat/lon) crs suits a map, a projected/absent one a plain plot
                crs = column.array.crs
            properties[name] = {
                'type': 'string', 'format': 'geometry', 'geometry_type': geom_type,
                'crs': str(crs) if crs is not None else None,
                'geographic': bool(crs is not None and crs.is_geographic),
            }
        elif dtype.kind in 'uifM':
            kind = None
            if df.empty:
                if dtype.kind == 'M':
                    vmin, vmax = pd.NaT, pd.NaT
                else:
                    vmin, vmax = float('NaN'), float('NaN')
            else:
                vmin, vmax = column.min(), column.max()
                if is_dask:
                    vmin, vmax = dd.compute(vmin, vmax)
            if dtype.kind == 'M':
                kind = 'string'
                vmin, vmax = vmin.isoformat(), vmax.isoformat()
            elif not df.empty:
                if dtype.kind == 'f':
                    cast = float
                    kind = 'number'
                else:
                    cast = int
                    kind = 'integer'
                try:
                    vmin, vmax = cast(vmin), cast(vmax)
                except Exception:
                    vmin, vmax = float('NaN'), float('NaN')
            properties[name] = {
                'type': kind,
                'inclusiveMinimum': vmin,
                'inclusiveMaximum': vmax
            }
            if dtype.kind == 'M':
                properties[name]['format'] = 'datetime'
        elif dtype.kind == 'b':
            properties[name] = {'type': 'boolean'}
        elif dtype.kind == 'O':
            if isinstance(dtype, CategoricalDtype) and len(dtype.categories):
                cats = list(dtype.categories)
            elif df.empty:
                cats = []
            else:
                try:
                    cats = column.unique()
                    if is_dask:
                        cats = cats.compute()
                except Exception:
                    cats = []
                cats = list(cats)
            properties[name] = {'type': 'string', 'enum': cats}
    return schema

_period_regex = re.compile(r'((?P<weeks>\d+?)w)?((?P<days>\d+?)d)?((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')


def parse_timedelta(time_str):
    parts = _period_regex.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for (name, p) in parts.items():
        if p:
            time_params[name] = int(p)
    return dt.timedelta(**time_params)


def _j_getenv(x):
    if isinstance(x, Undefined):
        x = x._undefined_name
    return os.getenv(x, '')

def _j_getshell(x):
    if isinstance(x, Undefined):
        x = x._undefined_name
    try:
        return check_output(x, shell=True).decode()
    except OSError:
        return ""

def _j_getheaders(x):
    if isinstance(x, Undefined):
        x = x._undefined_name
    return state.headers.get(x, '')

def _j_getcookies(x):
    if isinstance(x, Undefined):
        x = x._undefined_name
    return state.cookies.get(x, '')

def _j_getoauth(x):
    if isinstance(x, Undefined):
        x = x._undefined_name
    if state.user_info is None:
        return ''
    return state.user_info.get(x, '')

def expand_spec(pars, context=None, getenv=True, getshell=True, getheaders=True,
                getcookies=True, getoauth=True):
    """
    Render variables in context into the set of parameters with jinja2.

    For variables that are not strings, nothing happens.

    Parameters
    ----------
    pars: dict
        values are strings containing some jinja2 controls
    context: dict
        values to use while rendering

    Returns
    -------
    dict with the same keys as ``pars``, but updated values
    """
    if context is None:
        context = {}
    if isinstance(pars, dict):
        return {k: expand_spec(
            v, context, getenv, getshell, getheaders, getcookies, getoauth
        ) for k, v in pars.items()}
    elif isinstance(pars, list | tuple | set):
        return type(pars)(expand_spec(
            v, context, getenv, getshell, getheaders, getcookies, getoauth
        ) for v in pars)
    elif isinstance(pars, str):
        jinja = Environment(undefined=DebugUndefined)
        if getenv:
            jinja.globals['env'] = _j_getenv
        if getshell:
            jinja.globals['shell'] = _j_getshell
        if getheaders:
            jinja.globals['header'] = _j_getheaders
        if getcookies:
            jinja.globals['cookie'] = _j_getcookies
        if getoauth:
            jinja.globals['oauth'] = _j_getoauth
        return jinja.from_string(pars).render(context)
    else:
        # no expansion
        return pars


def merge_schemas(schema, old_schema):
    """
    Merges two JSON schemas on a column.
    """
    if old_schema is None:
        return schema
    elif schema['type'] != old_schema['type']:
        return old_schema
    elif 'enum' in schema and 'enum' in old_schema:
        merged_enum = list(old_schema['enum'])
        for enum in schema['enum']:
            if enum not in merged_enum:
                merged_enum.append(enum)
        return dict(old_schema, enum=merged_enum)
    elif 'inclusiveMinimum' in schema and 'inclusiveMinimum' in old_schema:
        merged_min = min(schema['inclusiveMinimum'], old_schema['inclusiveMinimum'])
        merged_max = max(schema['inclusiveMaximum'], old_schema['inclusiveMaximum'])
        return dict(old_schema, inclusiveMinimum=merged_min, inclusiveMaximum=merged_max)


def resolve_module_reference(reference, component_type=None):
    cls_name = component_type.__name__ if component_type else 'Component'
    *modules, ctype = reference.split('.')
    module = '.'.join(modules)
    try:
        module = importlib.import_module(module)
    except Exception as exc:
        raise ValueError(
            f"{cls_name} reference {reference!r} could not be resolved. "
            f"Module {module!r} could not be found."
        ) from exc
    if not hasattr(module, ctype):
        raise ValueError(
            f"{cls_name} reference {reference!r} could not be resolved. "
            f"Module {module!r} has no member {ctype}."
        )
    component = getattr(module, ctype)
    if component_type and not (isinstance(component, component_type) or issubclass(component, component_type)):
        raise ValueError(f"{cls_name} reference {reference!r} did not resolve "
                         f"to a {cls_name!r} subclass.")
    return component

def is_ref(value):
    """
    Whether the value is a reference.
    """
    if not isinstance(value, str):
        return False
    return bool(VARIABLE_RE.findall(value)) or value.startswith('$')

def extract_refs(spec, ref_type=None):
    refs = []
    if isinstance(spec, dict):
        for v in spec.values():
            for ref in extract_refs(v, ref_type):
                if ref not in refs:
                    refs.append(ref)
    elif isinstance(spec, list):
        for v in spec:
            for ref in extract_refs(v, ref_type):
                if ref not in refs:
                    refs.append(ref)
    elif is_ref(spec):
        refs.append(spec)
    if ref_type is None:
        return refs
    filtered = [ref for ref in refs if f'${ref_type}' in ref[1:]]
    return filtered

def cleanup_expr(expr):
    ref_vars = VARIABLE_RE.findall(expr)
    for var in ref_vars:
        re_var = r'\$variables\.' + var
        expr = re.sub(re_var, var, expr)
    return expr

def catch_and_notify(message=None):
    """Catch exception and notify user

    A decorator which catches all the exception of a function.
    When an error occurs a panel notification will be send to the
    dashboard with the message and logged the error and which method
    it arrived from.

    Parameters
    ----------
    message : str | None
        The notification message, by default None.
        None will give this "Error: {e}" where e is the
        exception message.

    """
    # This is to be able to call the decorator
    # like this @catch_and_notify
    function = None
    if callable(message):
        function = message
        message = None

    if message is None:
        message = "Error: {e}"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                from .state import state as session_state
                if session_state.config and session_state.config.on_error:
                    state.execute(partial(state.config.on_error, e))
                if pn.config.notifications:
                    log.error(
                        f"{func.__qualname__!r} raised {type(e).__name__}: {e}"
                    )
                    state.notifications.error(message.format(e=e))
                    if session_state.config and session_state.config.raise_with_notifications:
                        raise e
                else:
                    raise e
        return wrapper

    if function:
        return decorator(function)

    return decorator

def slugify(value, allow_unicode=False) -> str:
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    From: https://docs.djangoproject.com/en/4.0/_modules/django/utils/text/#slugify
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def detect_file_encoding(file_obj: Path | str | io.BytesIO | io.StringIO | bytes, sample_size: int = 8192) -> str:
    """
    Simple, fast file encoding detection.

    Parameters
    ----------
    file_obj : Path | str | io.BytesIO | io.StringIO | bytes
        File path or file-like object to detect encoding
    sample_size : int, default=8192
        Bytes to read for detection

    Returns
    -------
    str
        Detected encoding
    """
    # Get bytes data from different input types
    if isinstance(file_obj, bytes):
        data = file_obj
    elif isinstance(file_obj, (str, Path)):
        # File path
        file_path = Path(file_obj)
        if not file_path.exists():
            data = None
        else:
            with file_path.open("rb") as f:
                data = f.read(sample_size)
    elif isinstance(file_obj, io.BytesIO):
        # BytesIO - preserve position
        pos = file_obj.tell()
        data = file_obj.read(sample_size)
        file_obj.seek(pos)
    elif isinstance(file_obj, io.StringIO):
        # StringIO - read and encode
        pos = file_obj.tell()
        content = file_obj.read(sample_size)
        file_obj.seek(pos)
        data = content.encode('utf-8')[:sample_size]
    else:
        raise ValueError(f"Unsupported file object type: {type(file_obj)}")

    if not data:
        return "utf-8"

    # Check BOM first (instant detection)
    if data.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    elif data.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    elif data.startswith(b'\xfe\xff'):
        return 'utf-16-be'

    # Try UTF-8 (most common)
    try:
        data.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass

    # Use chardet if available, otherwise fallback
    try:
        import chardet
        result = chardet.detect(data)
        encoding = result.get('encoding', 'latin-1')
        # Clean up common names
        if encoding and encoding.lower() in ['iso-8859-1', 'ascii']:
            return 'utf-8' if encoding.lower() == 'ascii' else 'latin-1'
        return encoding.lower() if encoding else 'latin-1'
    except ImportError:
        # Simple fallback without chardet
        return 'latin-1'  # Can decode any byte sequence

def _set_backend_opts(element, cur_opts, compat_opts):
    """Utility to make it possible to serialize hvPlot generated plots"""
    from hvplot.utilities import hvplot_extension
    element = element.opts(**cur_opts, backend='bokeh')
    if hvplot_extension.compatibility and compat_opts:
        element = element.opts(**compat_opts, backend=hvplot_extension.compatibility)
    return element


def normalize_table_name(name: str) -> str:
    """
    Normalize a table name to a valid SQL identifier.

    Replaces all non-word characters (anything except letters, digits,
    and underscores) with underscores, strips leading/trailing underscores,
    and converts to lowercase.

    This matches the behavior of DuckDBSource.normalize_table() to ensure
    consistent table naming across the codebase.

    Parameters
    ----------
    name : str
        The table name to normalize (e.g., filename without extension)

    Returns
    -------
    str
        The normalized table name

    Examples
    --------
    >>> normalize_table_name("customer (orders)")
    'customer_orders'
    >>> normalize_table_name("My-Data File")
    'my_data_file'
    >>> normalize_table_name("table__name")
    'table_name'
    """
    return re.sub(r'\W+', '_', name).strip('_').lower()


def try_import(module_name, load=True):
    """Import and return a module, or None if it is unavailable.

    With load=False the module is returned only if it has already been
    imported, a cheap check that never triggers an import; this lets a hot path
    ask whether an optional dependency is in play without paying to import it.
    With load=True (the default) the module is imported on demand.
    """
    if not load:
        return sys.modules.get(module_name)
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def try_import_xarray(load=True):
    """Import and return xarray, or None if xarray or xarray-sql is unavailable."""
    if try_import("xarray_sql", load=load) is None:
        return None
    return try_import("xarray", load=load)


def is_geodataframe(df):
    """Return True if df is a geopandas GeoDataFrame.

    Uses an already-imported geopandas rather than importing it: a df cannot be
    a GeoDataFrame unless geopandas is loaded, so this stays cheap on the hot
    path and never speculatively imports geopandas.
    """
    gpd = try_import("geopandas", load=False)
    return gpd is not None and isinstance(df, gpd.GeoDataFrame)


def geometry_columns(df):
    """Return the names of geometry-typed columns in df.

    Empty when geopandas is not imported (no geometry column can exist without
    it) or df has no geometry columns; never speculatively imports geopandas.
    """
    gpd = try_import("geopandas", load=False)
    if gpd is None:
        return []
    return [c for c in df.columns if isinstance(df[c].dtype, gpd.array.GeometryDtype)]


def geometry_to_wkt(df):
    """Return a copy of df with geometry columns converted to WKT strings.

    A GeoDataFrame geometry column holds shapely objects that cannot be
    serialized to the browser (e.g. by Bokeh in a Tabulator), so convert them
    to WKT text for tabular display. Returns df unchanged if it has no
    geometry columns.
    """
    geom_cols = geometry_columns(df)
    if not geom_cols:
        return df
    gpd = try_import("geopandas", load=False)
    df = pd.DataFrame(df).copy()
    for col in geom_cols:
        df[col] = gpd.GeoSeries(df[col]).to_wkt()
    return df


def geometry_to_geojson(df):
    """Return a GeoDataFrame as a GeoJSON FeatureCollection dict.

    Shapely geometry cannot be sent to the browser, so views that render
    geometry emit a FeatureCollection instead. ``default=str`` is required
    because to_json defers to json.dumps, which cannot encode the Timestamps
    and Decimals a SQL source yields.
    """
    return json.loads(df.to_json(default=str))
