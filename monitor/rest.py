from urllib.parse import parse_qs

import pandas as pd
import param

from tornado import web

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

_METRICS = {}

class DataFrameEndpoint(param.Parameterized):

    data = param.DataFrame()

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
        properties = {}
        for name, dtype in self.data.dtypes.iteritems():
            column = self.data[name]
            if dtype.kind in 'uif':
                if dtype.kind == 'f':
                    cast = float
                    kind = 'number'
                else:
                    cast = int
                    kind = 'integer'
                properties[name] = {
                    'type': kind,
                    'inclusiveMinimum': cast(column.min()),
                    'inclusiveMaximum': cast(column.max())
                }
            elif dtype.kind == 'O':
                properties[name] = {'type': 'string', 'enum': list(column.unique())}
        return {'type': 'array', 'items': {'type': 'object', 'properties': properties}}


class ParameterEndpoint(param.Parameterized):

    def query(self, **kwargs):
        pass

    def schema(self):
        pass


def publish(metric, obj):
    if isinstance(obj, pd.DataFrame):
        obj = DataFrameEndpoint(data=obj)

    if isinstance(obj, DataFrameEndpoint):
        _METRICS[metric] = obj
    else:
        raise ValueError("Object of type not recognized for publishing")


def unpublish(metric):
    del _METRICS[metric]

#-----------------------------------------------------------------------------
# Private API
#-----------------------------------------------------------------------------

class MetricHandler(web.RequestHandler):

    def get(self):
        args = parse_qs(self.request.query)
        metric = args.pop('metric')
        endpoint = _METRICS.get(metric[0])
        if not endpoint:
            return
        json = endpoint.query(**args)
        self.set_header('Content-Type', 'application/json')
        self.write(json)


class MetricsHandler(web.RequestHandler):

    def get(self):
        schema = {metric: endpoint.schema() for metric, endpoint in _METRICS.items()}
        self.set_header('Content-Type', 'application/json')
        self.write(schema)


def monitor_rest_provider(files, endpoint):
    if endpoint:
        prefix = r'^/%s/' % endpoint
    else:
        prefix = r'^/'
    return [
        (prefix + 'metric', MetricHandler),
        (prefix + 'metrics', MetricsHandler)
    ]
