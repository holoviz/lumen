"""
The REST module provides a Tornado based implementation of the monitor
REST specification providing the ability to publish different types of
data which can be queried by the monitoring dashboard or some other
application.
"""

import sys

from urllib.parse import parse_qs

import pandas as pd
import param

from tornado import web

from .util import get_dataframe_schema

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

_VARIABLES = {}


class VariableEndpoint(param.Parameterized):
    """
    A VariableEndpoint is an abstract class that publishes data
    associated with some variable. It consumes some data along with a
    variable name a one or more indexes and provides APIs for
    returning the JSON schema of this data and a method to query the
    data.
    """

    data = param.Parameter(doc="The data to publish")

    index = param.List(default=[], doc="List of indexes")

    variable = param.String(doc="Name of the variable to publish")
    
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


class DataFrameEndpoint(VariableEndpoint):

    data = param.DataFrame()

    def __init__(self, **params):
        super().__init__(**params)
        if self.variable not in self.data.columns:
            raise ValueError(f"Variable column {self.variable} not found in published data.")
        not_found = [index for index in self.index if index not in self.data.columns]
        if not_found:
            raise ValueError(f"Indexes {not_found} not found in published data.")

    def query(self, **kwargs):
        query = None
        columns = [] if self.data is None else self.data.columns
        for k, v in kwargs.items():
            if k not in columns:
                self.param.warning(f"Query {k}={v} could not be resolved "
                                   "data does not have queried column.")
            column = self.data[k]
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
        data = self.data if query is None else self.data[query]
        return data.to_json(orient='records')

    def schema(self):
        schema = super().schema()
        if self.data is None:
            return schema
        return get_dataframe_schema(self.data, self.index, self.variable)


class ParameterEndpoint(VariableEndpoint):

    def __init__(self, **params):
        super().__init__(**params)
        if self.data and self.variable not in self.data[0].param:
            raise ValueError(f"Variable column {self.variable} not found in published data.")
        not_found = [index for index in self.index if self.data and index not in self.data[0].param]
        if not_found:
            raise ValueError(f"Indexes {not_found} not found in published data.")

    def query(self, **kwargs):
        columns = [self.variable] + self.index
        objects = list(self.data)
        for k, v in kwargs.items():
            if k not in columns:
                self.param.warning(f"Query {k}={v} could not be resolved "
                                   "data does not have queried column.")
            if isinstance(v, list):
                filter_fn = lambda o: getattr(o, k) in v
            elif isinstance(v, tuple):
                filter_fn = lambda o: v[0]<= getattr(o, k) <= v[1]
            else:
                filter_fn = lambda o: getattr(o, k) == v
            objects = list(filter(objects, filter_fn))
        return [o.param.serialize_parameters(columns) for o in self.objects]

    def schema(self):
        schema = super().schema()
        if not self.data:
            return schema
        properties = schema['items']['properties']
        sample = self.data[0]
        properties[self.variable] = sample.param[self.variable].schema()
        for index in self.index:
            properties[index] = sample.param[index].schema()
        schema = {'type': 'array', 'items': {'type': 'object', 'properties': properties}}


def publish(variable, obj, index=None):
    """
    Publishes a variable with one or more indexes.

    Arguments
    ---------
    variable: str
      The name of the variable to publish
    obj: object
      The object containing the variable data. Currently DataFrame
      and Parameterized objects are supported.
    index: str or list(str)
      An index or list of indexes to publish alongside the variable.
    """
    if isinstance(obj, param.Parameterized):
        obj = [obj]
    if index is not None and not isinstance(index, list):
        index = [index]
    
    if isinstance(obj, pd.DataFrame):
        endpoint = DataFrameEndpoint
    elif isinstance(obj, list) and all(isinstance(o, param.Parameterized) for o in obj):
        endpoint = ParameterEndpoint
    else:
        raise ValueError("Object of type not recognized for publishing")

    _VARIABLES[variable] = endpoint(data=obj, index=index, variable=variable)
    

def unpublish(variable):
    """
    Unpublishes a variable which was previously published.
    """
    del _VARIABLES[variable]

#-----------------------------------------------------------------------------
# Private API
#-----------------------------------------------------------------------------

class VariableHandler(web.RequestHandler):

    def get(self):
        args = parse_qs(self.request.query)
        variable = args.pop('variable', None)
        if not variable:
            return
        endpoint = _VARIABLES.get(variable[0])
        if not endpoint:
            return
        json = endpoint.query(**args)
        self.set_header('Content-Type', 'application/json')
        self.write(json)


class SchemaHandler(web.RequestHandler):

    def get(self):
        args = parse_qs(self.request.query)
        variable = args.pop('variable', None)
        if variable is None:
            schema = _VARIABLES[variable].schema()
        else:
            schema = {variable: endpoint.schema() for variable, endpoint in _VARIABLES.items()}
        self.set_header('Content-Type', 'application/json')
        self.write(schema)


def monitor_rest_provider(files, endpoint):
    from pane.io.rest import _exec_files
    _exec_files(files)
    if endpoint:
        prefix = r'^/%s/' % endpoint
    else:
        prefix = r'^/'
    return [
        (prefix + 'data', VariableHandler),
        (prefix + 'schema', SchemaHandler)
    ]
