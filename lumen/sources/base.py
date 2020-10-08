import pandas as pd
import param
import requests


class Source(param.Parameterized):
    """
    A Source provides a set of named variables and associated indexes
    which can be queried. The Source must also be able to return a
    schema describing the types of the variables and indexes in the
    data.
    """

    source_type = None

    __abstract = True

    @classmethod
    def _get_type(cls, source_type):
        try:
            __import__(f'lumen.sources.{source_type}')
        except Exception:
            pass
        for source in param.concrete_descendents(cls).values():
            if source.source_type == source_type:
                return source
        return Source

    @classmethod
    def _filter_dataframe(cls, df, **query):
        """
        Filter the DataFrame.

        Parameters
        ----------
        df : DataFrame
           The DataFrame to filter
        query : dict
            A dictionary containing all the query parameters

        Returns
        -------
        DataFrame
            The filtered DataFrame
        """
        filters = []
        for k, val in query.items():
            if np.isscalar(val):
                filters.append(df[k] == val)
            elif isinstance(val, list):
                filters.append(df[k].isin(val))
            elif isinstance(val, tuple):
                filters.append((df[k]>val[0]) & (df[k]<=val[1]))
        if filters:
            mask = filters[0]
            for f in filters:
                mask &= f
            df = df[mask]
        return df

    def get_schema(self, variable=None):
        """
        Returns JSON schema describing the data returned by the
        Source.

        Parameters
        ----------
        variable : str or None
            The name of the variable to return the schema for. If None
            returns schema for all available variables.

        Returns
        -------
        dict
           JSON schema describing the types of the data.
        """

    def get(self, variable, **query):
        """
        Return data for a particular variable given a query.

        Parameters
        ----------
        variable : str
            The name of the variable to query
        query : dict
            A dictionary containing all the query parameters

        Returns
        -------
        DataFrame
           A DataFrame containing the indexes and data variables
           declared in the schema.
        """

    def clear_cache(self):
        """
        Clears any cached data.
        """


class RESTSource(Source):
    """
    Queries a REST API which is expected to conform to the monitoring
    REST API specification.
    """

    url = param.String(doc="URL of the REST endpoint to monitor.")

    source_type = 'rest'

    def get_schema(self, variable=None):
        query = {} if variable is None else {'variable': variable}
        response = requests.get(self.url+'/schema', params=query)
        return response.json()

    def get(self, variable, **query):
        query = dict(variable=variable, **query)
        r = requests.get(self.url+'/data', params=query)
        return pd.DataFrame(r.json())


class WebsiteSource(Source):
    """
    Queries whether a website responds with a 400 status code.
    """

    url = param.String(doc="URL of the website to monitor.")

    source_type = 'live'

    def get_schema(self, variable=None):
        schema = {
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
        return schema if variable is None else schema[variable]

    def get(self, variable, **query):
        r = requests.get(self.url)
        return [{"live": r.status_code == 200, "url": self.url}]
