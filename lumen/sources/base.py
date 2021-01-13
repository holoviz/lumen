import hashlib
import os
import shutil
import sys

from concurrent import futures
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import param
import requests

from ..util import get_dataframe_schema


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
                if not with_query and hasattr(self, 'dask'):
                    cache_query['dask'] = True
                df = method(self, table, **cache_query)
                cache_query.pop('dask', None)
                self._set_cache(df, table, **cache_query)
            filtered = df if with_query else self._filter_dataframe(df, **query)
            if getattr(self, 'dask', False) or not hasattr(filtered, 'compute'):
                return filtered
            return filtered.compute()
        return wrapped
    return _inner_cached


class Source(param.Parameterized):
    """
    A Source provides a set of tables which declare their available
    fields. The Source must also be able to return a schema describing
    the types of the variables and indexes in each table and allow
    querying the data.
    """

    cache_dir = param.String(default=None, doc="""
        Whether to enable local cache and write file to disk.""")

    root = param.String(default=None, doc="""
        Root directory where the dashboard specification was loaded from.""")

    shared = param.Boolean(default=False, doc="""
        Whether the Source can be shared across all instances of the
        dashboard. If set to `True` the Source will be loaded on
        initial server load.""")

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
            column = df[k]
            if np.isscalar(val):
                mask = column == val
            elif isinstance(val, list):
                if not val:
                    continue
                mask = column.isin(val)
            elif isinstance(val, tuple):
                start, end = val
                if start is None and end is None:
                    continue
                elif start is None:
                    mask = column<=end
                elif end is None:
                    mask = column>=start
                else:
                    mask = (column>=start) & (column<=end)
            filters.append(mask)
        if filters:
            mask = filters[0]
            for f in filters:
                mask &= f
            df = df[mask]
        return df

    @classmethod
    def _resolve_reference(cls, reference, sources={}):
        refs = reference[1:].split('.')
        if len(refs) == 3:
            sourceref, table, field = refs
        elif len(refs) == 2:
            sourceref, table = refs
        elif len(refs) == 1:
            (sourceref,) = refs

        source = cls.from_spec(sourceref, sources)
        if len(refs) == 1:
            return source
        if len(refs) == 2:
            return source.get(table)
        table_schema = source.get_schema(table)
        if field not in table_schema:
            raise ValueError(f"Field '{field}' was not found in "
                             "'{sourceref}' table '{table}'.")
        field_schema = table_schema[field]
        if 'enum' not in field_schema:
            raise ValueError(f"Field '{field}' schema does not "
                             "declare an enum.")
        return field_schema['enum']

    @classmethod
    def from_spec(cls, spec, sources={}, root=None):
        """
        Creates a Source object from a specification. If a Source
        specification references other sources these may be supplied
        in the sources dictionary and be referenced by name.

        Parameters
        ----------
        spec : dict or str
            Specification declared as a dictionary of parameter values
            or a string referencing a source in the sources dictionary.
        sources: dict
            Dictionary of other Source objects
        root: str
            Root directory where dashboard specification was loaded
            from.

        Returns
        -------
        Resolved and instantiated Source object
        """
        from .. import config
        if spec is None:
            raise ValueError('Source specification empty.')
        elif isinstance(spec, str):
            if spec in sources:
                source = sources[spec]
            elif spec.get('shared') and spec in config.sources:
                source = config.sources[spec]
            else:
                raise ValueError(f'Source with name {spec} was not found.')
            return source

        spec = dict(spec)
        source_type = Source._get_type(spec.pop('type'))
        if 'sources' in source_type.param and 'sources' in spec:
            resolved_sources = {
                source: cls.from_spec(source, sources)
                for source in spec['sources']
            }
            spec['sources'] = resolved_sources
        resolved_spec = {}
        for k, v in spec.items():
            if isinstance(v, str) and v.startswith('@'):
                v = cls._resolve_reference(v, sources)
            resolved_spec[k] = v
        resolved_spec['root'] = root
        return source_type(**resolved_spec)

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
        elif self.cache_dir:
            if query:
                sha = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
                filename = f'{sha}_{table}.parq'
            else:
                filename = f'{table}.parq'
            path = os.path.join(self.root, self.cache_dir, filename)
            if os.path.isfile(path):
                if 'dask.dataframe' in sys.modules:
                    import dask.dataframe as dd
                    return dd.read_parquet(path)
                return pd.read_parquet(path)

    def _set_cache(self, data, table, **query):
        key = self._get_key(table, **query)
        self._cache[key] = data
        if self.cache_dir:
            path = os.path.join(self.root, self.cache_dir)
            Path(path).mkdir(parents=True, exist_ok=True)
            if query:
                sha = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
                filename = f'{sha}_{table}.parq'
            else:
                filename = f'{table}.parq'
            data.to_parquet(os.path.join(path, filename))

    def clear_cache(self):
        """
        Clears any cached data.
        """
        self._cache = {}
        if self.cache_dir:
            path = os.path.join(self.root, self.cache_dir)
            if os.path.isdir(path):
                shutil.rmtree(path)

    @property
    def panel(self):
        """
        A Source can return a Panel object which displays information
        about the Source or controls how the Source queries data.
        """
        return None

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


class FileSource(Source):
    """
    Loads CSV, Excel and Parquet using pandas.read_* or dask.read_*
    functions.
    """

    files = param.List(doc="""
        List of files to load. The filenames will become the table
        names.""")

    use_dask = param.Boolean(default=True, doc="""
        Whether to use dask to load files.""")

    _pd_load_fns = {
        'csv': pd.read_csv,
        'xlsx': pd.read_excel,
        'xls': pd.read_excel,
        'parq': pd.read_parquet
    }

    _load_kwargs = {
        'csv': {'parse_dates': True}
    }

    source_type = 'file'

    def _load_fn(self, ext):
        kwargs = self._load_kwargs.get(ext, {})
        if self.use_dask:
            import dask.dataframe as dd
            if ext == 'csv':
                return dd.read_csv, kwargs
            elif ext == 'parq':
                return dd.read_parquet, kwargs
        if ext not in self._pd_load_fns:
            raise ValueError("File type '{ext}' not recognized and cannot be loaded.")
        return self._pd_load_fns[ext], kwargs

    def get_schema(self, table=None):
        schemas = {}
        for file in self.files:
            name = '.'.join(os.path.basename(file).split('.')[:-1])
            if table is not None and name != table:
                continue
            df = self.get(name)
            schemas[name] = get_dataframe_schema(df)['items']['properties']
        return schemas if table is None else schemas[table]

    def _load_table(self, table):
        df = None
        for file in self.files:
            split_filename = os.path.basename(file).split('.')
            name = '.'.join(split_filename[:-1])
            if name != table:
                continue
            load_fn, kwargs = self._load_fn(split_filename[-1])
            df = load_fn(os.path.join(self.root, file), **kwargs)
        if df is None:
            tables = ['.'.join(f.split('.')[:-1]) for f in self.files]
            raise ValueError(f"Table '{table}' not found. Available tables include: {tables}.")
        return df

    @cached()
    def get(self, table, **query):
        df = self._load_table(table)
        df = self._filter_dataframe(df, **query)
        return df.compute() if hasattr(df, 'compute') else df


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

    timeout = param.Parameter(default=5)

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
        )
        data = []
        if r.status_code != 200:
            return data
        r = r.json()
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
        return data

    @cached(with_query=False)
    def get(self, table, **query):
        data = []
        with futures.ThreadPoolExecutor(len(self.urls)) as executor:
            tasks = {executor.submit(self._get_session_info, table, url): url
                     for url in self.urls}
            for future in futures.as_completed(tasks):
                url = tasks[future] + self.endpoint
                try:
                    data.extend(future.result())
                except Exception as e:
                    exception = f"{type(e).__name__}({e})"
                    self.param.warning("Failed to fetch session_info from "
                                       f"{url}, errored with {exception}.")
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

    def get_schema(self, table=None):
        schemas = {}
        for name, specs in self.tables.items():
            if table is not None and name != table:
                continue
            schemas[name] = schema = {}
            for spec in specs:
                source, subtable = spec['source'], spec['table']
                table_schema = self.sources[source].get_schema(subtable)
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
        df, left_key = None, None
        for spec in self.tables[table]:
            source, subtable = spec['source'], spec['table']
            source_query = dict(query)
            right_key = spec.get('index')
            if df is not None and left_key and right_key not in query:
                source_query[right_key] = list(df[left_key].unique())
            df_merge = self.sources[source].get(subtable, **source_query)
            if df is None:
                df = df_merge
                left_key = spec.get('index')
            else:
                df = pd.merge(df, df_merge, left_on=left_key,
                              right_on=right_key, how='outer')
        return df

    @property
    def panel(self):
        column = pn.Column(sizing_mode='stretch_width')
        for name, source in self.sources.items():
            panel = source.panel
            if not panel:
                continue
            header = pn.pane.Markdown(f'#### {name.title()}', margin=(0, 5))
            column.extend([header, *source.panel])
        return column

    def clear_cache(self):
        super().clear_cache()
        for source in self.sources.values():
            source.clear_cache()
