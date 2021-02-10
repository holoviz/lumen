import datetime as dt
import re
import os
import sys
import subprocess

import panel as pn

from jinja2 import Environment, Undefined, DebugUndefined
from pandas.core.dtypes.dtypes import CategoricalDtype
from panel import state


LAYOUTS = {
    'accordion': pn.Accordion,
    'column'   : pn.Column,
    'grid'     : pn.GridBox,
    'row'      : pn.Row,
    'tabs'     : pn.Tabs
}


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

    if columns is None:
        columns = list(df.columns)

    schema = {'type': 'array', 'items': {'type': 'object', 'properties': {}}}
    properties = schema['items']['properties']
    for name in columns:
        dtype = df.dtypes[name]
        column = df[name]
        if dtype.kind in 'uifM':
            vmin, vmax = column.min(), column.max()
            if is_dask:
                vmin, vmax = dd.compute(vmin, vmax)
            if dtype.kind == 'M':
                kind = 'string'
                vmin, vmax = vmin.isoformat(), vmax.isoformat()
            else:
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
        elif dtype.kind == 'O':
            if isinstance(dtype, CategoricalDtype) and len(dtype.categories):
                cats = list(dtype.categories)
            else:
                cats = column.unique()
                if is_dask:
                    cats = cats.compute()
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
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return dt.timedelta(**time_params)


def _j_getenv(x):
    if isinstance(x, Undefined):
        x = x._undefined_name
    return os.getenv(x, '')

def _j_getshell(x):
    if isinstance(x, Undefined):
        x = x._undefined_name
    try:
        return subprocess.check_output(x).decode()
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
