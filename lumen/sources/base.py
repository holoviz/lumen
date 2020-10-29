from concurrent import futures
from functools import wraps

import numpy as np
import pandas as pd
import param
import requests


def cached(with_query=True):
    """
    Adds caching to a Source.get query.

    Arguments
    ---------
    with_query: boolean
        Whether the Source.get query uses the query parameters.
        Sources that have no ability to pre-filter the data can
        use this option to cache the full query and the decorator
        will apply the filtering after the fact.

    Returns
    -------
    Returns method wrapped in caching functionality.
    """
    def _inner_cached(method):
        @wraps(method)
        def wrapped(self, table, **query):
            cache_query = query if with_query else {}
            df = self._get_cache(table, **cache_query)
            if df is None:
                df = method(self, table, **cache_query)
                self._set_cache(df, table, **cache_query)
            return df if with_query else self._filter_dataframe(df, **query)
        return wrapped
    return _inner_cached


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
            if k not in df.columns:
                continue
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

    def __init__(self, **params):
        super().__init__(**params)
        self._cache = {}

    def _get_key(self, table, **query):
        key = (table,)
        for k, v in sorted(query.items()):
            if isinstance(v, list):
                v = tuple(v)
            key += (k, v)
        return key

    def _get_cache(self, table, **query):
        key = self._get_key(table, **query)
        if key in self._cache:
            return self._cache[key]

    def _set_cache(self, data, table, **query):
        key = self._get_key(table, **query)
        self._cache[key] = data

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
        self._cache = {}


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

    @cached()
    def get(self, table, **query):
        query = dict(table=table, **query)
        r = requests.get(self.url+'/data', params=query)
        df = pd.DataFrame(r.json())
        return df


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

    @cached()
    def get(self, table, **query):
        data = []
        for url in self.urls:
            try:
                r = requests.get(url)
                live = r.status_code == 200
            except Exception:
                live = False
            data.append({"live": live, "url": self.url})
        df = pd.DataFrame(data)
        return df


class PanelSessionSource(Source):

    endpoint = param.String(default="rest/session_info")

    urls = param.List(doc="URL of the websites to monitor.")

    timeout = param.Parameter(default=(0.5, 2))

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

    def _get_session_info(self, table, url):
        r = requests.get(
            url + self.endpoint, verify=False, timeout=self.timeout
        ).json()
        session_info = r['session_info']
        sessions = session_info['sessions']

        data = []
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
        return data

    @cached(with_query=False)
    def get(self, table, **query):
        data = []
        with futures.ThreadPoolExecutor(len(self.urls)) as executor:
            tasks = [executor.submit(self._get_session_info, table, url)
                     for url in self.urls]
            for future in futures.as_completed(tasks):
                try:
                    data.extend(future.result())
                except Exception:
                    continue
        return pd.DataFrame(data, columns=list(self.get_schema(table)))


class JoinedSource(Source):
    """
    A JoinedSource applies a join on two or more sources returning
    new table(s) with data from all sources. It iterates over the
    `tables` specification and merges the specified tables from the
    declared sources on the supplied index.

    In this way multiple tables from multiple sources can be merged.
    Individual tables from sources that should not be joined may also
    be surfaced by declaring a single source and table in the
    specification.

    As a simple example we may have sources A and B, which contain
    tables 'foo' and 'bar' respectively. We now want to merge these
    tables on column 'a' in Table A with column 'b' in Table B:

        {'new_table': [
          {'source': 'A', 'table': 'foo', 'index': 'a'},
          {'source': 'B', 'table': 'bar', 'index': 'b'}
        ]}

    The joined source will now publish the "new_table" with all
    columns from tables "foo" and "bar" except for the index column
    from table "bar", which was merged with the index column "a" from
    table "foo".
    """

    sources = param.Dict(default={}, doc="""
        A dictionary of sources indexed by their assigned name.""")

    tables = param.Dict(default={}, doc="""
        A dictionary with the names of the joined sources as keys
        and a specification of the source, table and index to merge
        on.

        {"new_table": [
            {'source': <source_name>,
             'table': <table_name>,
             'index': <index_name>
            },
            {'source': <source_name>,
             'table': <table_name>,
             'index': <index_name>
            },
            ...
        ]}""")

    source_type = 'join'

    def __init__(self, **params):
        super().__init__(**params)
        self._sources = {}
        for name, source_spec in self.sources.items():
            source_spec = dict(source_spec)
            source_type = source_spec.pop('type')
            self._sources[name] = Source._get_type(source_type)(**source_spec)

    def get_schema(self, table=None):
        schemas = {}
        for name, specs in self.tables.items():
            if table is not None and name != table:
                continue
            schemas[name] = schema = {}
            for spec in specs:
                source, subtable = spec['source'], spec['table']
                table_schema = self._sources[source].get_schema(subtable)
                if not schema:
                    schema.update(table_schema)
                else:
                    for column, col_schema in table_schema.items():
                        prev_schema = schema.get(column)
                        if prev_schema is None:
                            schema[column] = col_schema
                        elif col_schema['type'] != prev_schema['type']:
                            continue
                        elif 'enum' in col_schema and 'enum' in prev_schema:
                            prev_enum = schema[column]['enum']
                            for enum in col_schema['enum']:
                                if enum not in prev_enum:
                                    prev_enum.append(enum)
                        elif 'inclusiveMinimum' in col_schema and 'inclusiveMinimum' in prev_schema:
                            prev_schema['inclusiveMinimum'] = min(
                                col_schema['inclusiveMinimum'],
                                prev_schema['inclusiveMinimum']
                            )
                            prev_schema['inclusiveMaximum'] = max(
                                col_schema['inclusiveMaximum'],
                                prev_schema['inclusiveMaximum']
                            )
        return schemas if table is None else schemas[table]

    @cached()
    def get(self, table, **query):
        df = None
        for spec in self.tables[table]:
            source, subtable = spec['source'], spec['table']
            df_merge = self._sources[source].get(subtable, **query)
            if df is None:
                df = df_merge
                left_key = spec.get('index')
            else:
                df = pd.merge(df, df_merge, left_on=left_key,
                              right_on=spec.get('index'))
        return df

    def clear_cache(self):
        super().clear_cache()
        for source in self._sources.values():
            source.clear_cache()
