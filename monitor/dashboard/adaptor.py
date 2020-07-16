import pandas as pd
import param
import requests


class QueryAdaptor(param.Parameterized):
    """
    Makes some query to get the metrics and their data
    """

    adaptor_type = None

    __abstract = True

    @classmethod
    def get(cls, adaptor_type):
        for adaptor in param.concrete_descendents(cls).values():
            if adaptor.adaptor_type == adaptor_type:
                return adaptor
        return QueryAdaptor


class RESTAdaptor(QueryAdaptor):

    url = param.String()

    adaptor_type = 'rest'

    def get_metrics(self):
        response = requests.get(self.url+'/metrics')
        return response.json()

    def get_metric(self, metric, **query):
        query = dict(metric=metric, **query)
        r = requests.get(self.url+'/metric', params=query)
        return pd.DataFrame(r.json())


class LiveWebsite(QueryAdaptor):

    url = param.String()

    adaptor_type = 'live'

    def get_metrics(self):
        return {
            "live": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "live": {"type": "boolean"},
                        "url": {"type": "string"}
                    }
                }
            }
        }

    def get_metric(self, metric, **query):
        r = requests.get(self.url)
        return [{"live": r.status_code == 200, "url": self.url}]
