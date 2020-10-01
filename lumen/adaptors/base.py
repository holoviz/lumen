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

    def get_metrics(self):
        """
        Should return a JSON schema describing the data returned by
        the QueryAdaptor.

        Returns
        -------
        dict
           JSON schema describing the types of the data for each metric.
        """

    def get_metric(self, metric, **query):
        """
        Should return data for a particular metric given a query.

        Parameters
        ----------
        metric : str
            The name of the metric to query
        query : dict
            A dictionary containing all the query parameters

        Returns
        -------
        list
           A list of records (i.e. dictionaries) containing the values
           for the metric and any indexes associated with them.
        """

    def update(self):
        """
        QueryAdaptors that cache data should refresh the data when
        this method is called.
        """


class RESTAdaptor(QueryAdaptor):
    """
    Queries a REST API which is expected to conform to the monitoring
    REST API specification.
    """

    url = param.String(doc="URL of the REST endpoint to monitor.")

    adaptor_type = 'rest'

    def get_metrics(self):
        response = requests.get(self.url+'/metrics')
        return response.json()

    def get_metric(self, metric, **query):
        query = dict(metric=metric, **query)
        r = requests.get(self.url+'/metric', params=query)
        return pd.DataFrame(r.json())


class LiveWebsite(QueryAdaptor):
    """
    Queries whether a website responds with a 400 status code.
    """

    url = param.String(doc="URL of the website to monitor.")

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
