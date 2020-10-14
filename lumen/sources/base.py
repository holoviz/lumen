import numpy as np
import pandas as pd
import param
import requests


class Source(param.Parameterized):
    """
    A Source provides a set of tables which declare their available
    fields. The Source must also be able to return a schema describing
    the types of the variables and indexes in each table and allow
    querying the data.
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
        raise ValueError(f"No Source for source_type '{source_type}' could be found.")

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

    def get_schema(self, table=None):
        """
        Returns JSON schema describing the tables returned by the
        Source.

        Parameters
        ----------
        table : str or None
            The name of the table to return the schema for. If None
            returns schema for all available tables.

        Returns
        -------
        dict
           JSON schema(s) for one or all the tables.
        """

    def get(self, table, **query):
        """
        Return a table; optionally filtered by the given query.

        Parameters
        ----------
        table : str
            The name of the table to query
        query : dict
            A dictionary containing all the query parameters

        Returns
        -------
        DataFrame
           A DataFrame containing the queried table.
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

    def get_schema(self, table=None):
        query = {} if table is None else {'table': table}
        response = requests.get(self.url+'/schema', params=query)
        return {table: schema['items']['properties'] for table, schema in
                response.json().items()}

    def get(self, table, **query):
        query = dict(table=table, **query)
        r = requests.get(self.url+'/data', params=query)
        return pd.DataFrame(r.json())


class WebsiteSource(Source):
    """
    Queries whether a website responds with a 400 status code.
    """

    urls = param.List(doc="URLs of the websites to monitor.")

    source_type = 'live'

    def get_schema(self, table=None):
        schema = {
            "status": {
                "url": {"type": "string"},
                "live": {"type": "boolean"}
            }
        }
        return schema if table is None else schema[table]

    def get(self, table, **query):
        data = []
        for url in self.urls:
            try:
                r = requests.get(url)
                live = r.status_code == 200
            except Exception:
                live = False
            data.append({"live": live, "url": self.url})
        return pd.DataFrame(data)


class PanelSessionSource(Source):

    urls = param.List(doc="URL of the websites to monitor.")

    source_type = 'session_info'

    def get_schema(self, table=None):
        schema = {
            "summary": {
                "url": {"type": "string", "enum": self.urls},
                "total": {"type": "int"},
                "live": {"type": "int"},
                "render_duration": {"type": "float"},
                "session_duration": {"type": "float"}
            },
            "sessions": {
                "url": {"type": "string", "enum": self.urls},
                "id": {"type": "string"},
                "started": {"type": "float"},
                "ended": {"type": "float"},
                "rendered": {"type": "float"},
                "render_duration": {"type": "float"},
                "session_duration": {"type": "float"},
                "user_agent": {"type": "string"}
            }
        }
        return schema if table is None else schema[table]

    def get(self, table, **query):
        data = []
        for url in self.urls:
            r = requests.get(url).json()
            session_info = r['session_info']
            sessions = session_info['sessions']
            if table == "summary":
                rendered = [s for s in sessions.values()
                            if s['rendered'] is not None]
                ended = [s for s in sessions.values()
                         if s['ended'] is not None]
                row = {
                    'url': url,
                    'total': session_info['total'],
                    'live': session_info['live'],
                    'render_duration': np.mean([s['rendered']-s['started']
                                                for s in rendered]),
                    'session_duration': np.mean([s['ended']-s['started']
                                                for s in ended])
                }
                data.append(row)
            elif table == "sessions":
                for sid, session in sessions.items():
                    row = dict(url=url, id=sid, **session)
                    if session["rendered"]:
                        row["render_duration"] = session["rendered"]-session["started"]
                    else:
                        row["render_duration"] = float('NaN')
                    if session["ended"]:
                        row["session_duration"] = session["ended"]-session["started"]
                    else:
                        row["session_duration"] = float('NaN')
                    data.push(row)
        return pd.DataFrame(data)
