from __future__ import annotations

import datetime as dt
import importlib
import os
import re
import sys

from functools import wraps
from logging import getLogger
from subprocess import check_output

import pandas as pd
import panel as pn

from jinja2 import DebugUndefined, Environment, Undefined
from pandas.core.dtypes.dtypes import CategoricalDtype
from panel import state

log = getLogger(__name__)


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
                vmin, vmax = cast(vmin), cast(vmax)
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
    except (IOError, OSError):
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

def expand_spec(pars, context={}, getenv=True, getshell=True, getheaders=True,
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
    if isinstance(pars, dict):
        return {k: expand_spec(
            v, context, getenv, getshell, getheaders, getcookies, getoauth
        ) for k, v in pars.items()}
    elif isinstance(pars, (list, tuple, set)):
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


def resolve_module_reference(reference, component_type):
    cls_name = component_type.__name__
    *modules, ctype = reference.split('.')
    module = '.'.join(modules)
    try:
        module = importlib.import_module(module)
    except Exception:
        raise ValueError(f"{cls_name} type '{reference}' module could "
                         f"not be resolved. Ensure explicit {cls_name} "
                         "type references a valid module.")
    if not hasattr(module, ctype):
        raise ValueError(f"Source type '{reference}' could not be "
                         f"resolved. Module '{module}' has no member "
                         f"{ctype}.")
    component = getattr(module, ctype)
    if not issubclass(component, component_type):
        raise ValueError(f"{cls_name} type '{reference}' did not resolve "
                         f"to a {cls_name} subclass.")
    return component

def is_ref(value):
    """
    Whether the value is a reference.
    """
    return isinstance(value, str) and (value.startswith('$') or value.startswith('@'))

def extract_refs(spec, ref_type=None):
    refs = []
    if isinstance(spec, dict):
        for k, v in spec.items():
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
    filtered = [ref for ref in refs if ref[1:].startswith(ref_type)]
    return filtered


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
                if pn.state.notifications:
                    log.error(
                        f"{func.__qualname__!r} raised {type(e).__name__}: {e}"
                    )
                    pn.state.notifications.error(message.format(e=e))
                else:
                    raise e
        return wrapper

    if function:
        return decorator(function)

    return decorator
