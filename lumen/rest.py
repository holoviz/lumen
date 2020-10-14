"""
The REST module provides a Tornado based implementation of the monitor
REST specification providing the ability to publish different types of
data which can be queried by the monitoring dashboard or some other
application.
"""

from urllib.parse import parse_qs

import pandas as pd
import param

from tornado import web

from .util import get_dataframe_schema

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

_TABLES = {}


class TableEndpoint(param.Parameterized):
    """
    A TableEndpoint is an abstract class that publishes data
    associated with some table. It defines an API for querying and
    publishing a table consisting of multiple columns. In addition to
    publishing the data the endpoint also defines an API for returning
    a JSON schema of the table.
    """

    data = param.Parameter(doc="The data to publish")

    columns = param.List(doc="The list of columns in the table.")

    __abstract = True

    def query(self, **kwargs):
        """
        Filter the data given a set of queries.
        """
        return []

    def schema(self):
        """
        Return a schema for the data being published.
        """
        return {'type': 'array', 'items': {'type': 'object', 'properties': {}}}


class DataFrameEndpoint(TableEndpoint):

    data = param.DataFrame()

    def __init__(self, **params):
        super().__init__(**params)
        if self.columns is None:
            self.columns = self.data.columns
        else:
            not_found = [col for col in self.columns if col not in self.data.columns]
            if not_found:
                raise ValueError(f"Columns {not_found} not found in published data.")

    def query(self, **kwargs):
        query = None
        data = self.data
        columns = [] if self.data is None else list(self.data.columns)
        selected_cols = kwargs.pop('columns', columns)
        for k, v in kwargs.items():
            if k not in columns:
                self.param.warning(f"Query {k}={v} could not be resolved "
                                   "data does not have queried column.")
            column = data[k]
            if isinstance(v, list):
                q = column.isin(v)
            elif isinstance(v, tuple):
                q = column>=v[0] & column<=v[1]
            else:
                q = column == v
            if query is None:
                query = q
            else:
                query &= q
        if selected_cols is not columns:
            data = data[selected_cols]
        data = self.data if query is None else self.data[query]
        return data.to_json(orient='records')

    def schema(self):
        schema = super().schema()
        if self.data is None:
            return schema
        return get_dataframe_schema(self.data, self.columns)


class ParameterEndpoint(TableEndpoint):

    def __init__(self, **params):
        super().__init__(**params)
        not_found = [col for col in self.columns if self.data is not None
                     and col not in self.data.columns]
        if not_found:
            raise ValueError(f"Columns {not_found} not found in published data.")

    def query(self, **kwargs):
        objects = list(self.data)
        for k, v in kwargs.items():
            if k not in self.columns:
                self.param.warning(f"Query {k}={v} could not be resolved "
                                   "data does not have queried column.")
            if isinstance(v, list):
                filter_fn = lambda o: getattr(o, k) in v
            elif isinstance(v, tuple):
                filter_fn = lambda o: v[0]<= getattr(o, k) <= v[1]
            else:
                filter_fn = lambda o: getattr(o, k) == v
            objects = list(filter(objects, filter_fn))
        return [o.param.serialize_parameters(self.columns) for o in self.objects]

    def schema(self):
        schema = super().schema()
        if not self.data:
            return schema
        properties = schema['items']['properties']
        sample = self.data[0]
        for col in self.columns:
            properties[col] = sample.param[col].schema()
        return schema


def publish(name, obj, columns):
    """
    Publishes a table given an object.

    Arguments
    ---------
    name: str
        The name of the table to publish.
    obj: object
        The object containing the table data. Currently DataFrame
        and Parameterized objects are supported.
    columns: str or list(str)
        The name or list of names of the column(s) to publish
    """
    if isinstance(obj, param.Parameterized):
        obj = [obj]

    if isinstance(obj, pd.DataFrame):
        endpoint = DataFrameEndpoint
    elif isinstance(obj, list) and all(isinstance(o, param.Parameterized) for o in obj):
        endpoint = ParameterEndpoint
    else:
        raise ValueError("Object of type not recognized for publishing")

    _TABLES[name] = endpoint(data=obj, columns=columns)


def unpublish(table):
    """
    Unpublishes a table which was previously published.
    """
    del _TABLES[table]

#-----------------------------------------------------------------------------
# Private API
#-----------------------------------------------------------------------------

class TableHandler(web.RequestHandler):

    def get(self):
        args = parse_qs(self.request.query)
        table = args.pop('table', None)
        if not table:
            return
        endpoint = _TABLES.get(table[0])
        if not endpoint:
            return
        json = endpoint.query(**args)
        self.set_header('Content-Type', 'application/json')
        self.write(json)


class SchemaHandler(web.RequestHandler):

    def get(self):
        args = parse_qs(self.request.query)
        table = args.pop('table', None)
        if table is None:
            schema = _TABLES[table].schema()
        else:
            schema = {table: endpoint.schema() for table, endpoint in _TABLES.items()}
        self.set_header('Content-Type', 'application/json')
        self.write(schema)


def lumen_rest_provider(files, endpoint):
    from pane.io.rest import _exec_files
    _exec_files(files)
    if endpoint:
        prefix = r'^/%s/' % endpoint
    else:
        prefix = r'^/'
    return [
        (prefix + 'data', TableHandler),
        (prefix + 'schema', SchemaHandler)
    ]
