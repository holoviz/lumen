import sys

from pandas.core.dtypes.dtypes import CategoricalDtype


def get_dataframe_schema(df, index, variable):
    """
    Returns a JSON schema for a dataframe given a set of indexes and
    the variable.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        The DataFrame to describe with the schema
    index: list(str) or str
        The list of indexes
    variable: str
        The data variable

    Returns
    -------
    dict
        The JSON schema describing the DataFrame
    """
    if not isinstance(index, list):
        index = [index]

    if 'dask.dataframe' in sys.modules:
        import dask.dataframe as dd
        is_dask = isinstance(df, dd.DataFrame)
    else:
        is_dask = False

    schema = {'type': 'array', 'items': {'type': 'object', 'properties': {}}}
    properties = schema['items']['properties']
    for name, dtype in df.dtypes.iteritems():
        if name != variable and name not in index:
            continue
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
