from __future__ import annotations

import datetime as dt
import importlib
import io
import os
import re
import sys
import unicodedata

from functools import partial, wraps
from logging import getLogger
from pathlib import Path
from subprocess import check_output

import bokeh
import pandas as pd
import panel as pn
import param

from jinja2 import DebugUndefined, Environment, Undefined
from packaging.version import Version
from pandas.core.dtypes.dtypes import CategoricalDtype
from panel.io.state import state

log = getLogger(__name__)

bokeh3 = Version(bokeh.__version__) > Version("3.0")
param2 = Version(param.__version__) > Version("2.0rc1")

disallow_refs = {'allow_refs': False} if param2 else {}

VARIABLE_RE = re.compile(r'\$variables\.([a-zA-Z_]\w*)')

def get_dataframe_schema(df, columns=None):
    """
    Returns a JSON schema optionally filtered by a subset of the columns.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        The DataFrame to describe with the schema
    columns: list(str) or None
        List of columns to include in schema

    Returns
    -------
    dict
        The JSON schema describing the DataFrame
    """
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

    properties = schema['items']['properties']
    for name in columns:
        dtype = df.dtypes[name]
        column = df[name]
        if dtype.kind in 'uifM':
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
        merged_max = min(schema['inclusiveMaximum'], old_schema['inclusiveMaximum'])
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


def detect_file_encoding(file_obj: Path | str | io.BytesIO | io.StringIO, sample_size: int = 8192) -> str:
    """
    Simple, fast file encoding detection.

    Parameters
    ----------
    file_obj : Path | str | io.BytesIO | io.StringIO
        File path or file-like object to detect encoding
    sample_size : int, default=8192
        Bytes to read for detection

    Returns
    -------
    str
        Detected encoding
    """
    # Get bytes data from different input types
    if isinstance(file_obj, (str, Path)):
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
