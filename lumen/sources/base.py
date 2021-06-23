import hashlib
import json
import os
import re
import shutil
import sys

from concurrent import futures
from functools import wraps
from itertools import product
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import panel as pn
import param
import requests

from ..filters import Filter
from ..state import state
from ..transforms import Transform
from ..util import (
    get_dataframe_schema, merge_schemas, resolve_module_reference
)


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
            df, no_query = self._get_cache(table, **cache_query)
            if df is None:
                if not with_query and (hasattr(self, 'dask') or hasattr(self, 'use_dask')):
                    cache_query['__dask'] = True
                df = method(self, table, **cache_query)
                cache_query.pop('__dask', None)
                self._set_cache(df, table, **cache_query)
            filtered = df
            if (not with_query or no_query) and query:
                filtered = self._filter_dataframe(df, **query)
            if getattr(self, 'dask', False) or not hasattr(filtered, 'compute'):
                return filtered
            return filtered.compute()
        return wrapped
    return _inner_cached


def cached_schema(method):
    @wraps(method)
    def wrapped(self, table=None):
        schema = self._get_schema_cache()
        if schema is None:
            schema = method(self)
            self._set_schema_cache(schema)
        if table is None:
            return schema
        return schema[table]
    return wrapped



class Source(param.Parameterized):
    """
    A Source provides a set of tables which declare their available
    fields. The Source must also be able to return a schema describing
    the types of the variables and indexes in each table and allow
    querying the data.
    """

    cache_dir = param.String(default=None, doc="""
        Whether to enable local cache and write file to disk.""")

    shared = param.Boolean(default=False, doc="""
        Whether the Source can be shared across all instances of the
        dashboard. If set to `True` the Source will be loaded on
        initial server load.""")

    source_type = None

    __abstract = True

    @classmethod
    def _get_type(cls, source_type):
        if '.' in source_type:
            return resolve_module_reference(source_type, Source)
        try:
            __import__(f'lumen.sources.{source_type}')
        except Exception:
            pass
        for source in param.concrete_descendents(cls).values():
            if source.source_type == source_type:
                return source
        raise ValueError(f"No Source for source_type '{source_type}' could be found.")

    @classmethod
    def _range_filter(cls, column, start, end):
        if start is None and end is None:
            return None
        elif start is None:
            mask = column<=end
        elif end is None:
            mask = column>=start
        else:
            mask = (column>=start) & (column<=end)
        return mask

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
            elif isinstance(val, list) and all(isinstance(v, tuple) and len(v) == 2 for v in val):
                val = [v for v in val if v is not None]
                if not val:
                    continue
                mask = cls._range_filter(column, *val[0])
                for v in val[1:]:
                    mask |= cls._range_filter(column, *v)
                if mask is not None:
                    filters.append(mask)
                continue
            elif isinstance(val, list):
                if not val:
                    continue
                mask = column.isin(val)
            elif isinstance(val, tuple):
                mask = cls._range_filter(column, *val)
            if mask is not None:
                filters.append(mask)
        if filters:
            mask = filters[0]
            for f in filters[1:]:
                mask &= f
            df = df[mask]
        return df

    @classmethod
    def _resolve_reference(cls, reference):
        refs = reference[1:].split('.')
        if len(refs) == 3:
            sourceref, table, field = refs
        elif len(refs) == 2:
            sourceref, table = refs
        elif len(refs) == 1:
            (sourceref,) = refs

        source = cls.from_spec(sourceref)
        if len(refs) == 1:
            return source
        if len(refs) == 2:
            return source.get(table)
        table_schema = source.get_schema(table)
        if field not in table_schema:
            raise ValueError(f"Field '{field}' was not found in "
                             f"'{sourceref}' table '{table}'.")
        field_schema = table_schema[field]
        if 'enum' not in field_schema:
            raise ValueError(f"Field '{field}' schema does not "
                             "declare an enum.")
        return field_schema['enum']

    @classmethod
    def _recursive_resolve(cls, spec, source_type):
        resolved_spec = {}
        if 'sources' in source_type.param and 'sources' in spec:
            resolved_spec['sources'] = {
                source: cls.from_spec(source)
                for source in spec.pop('sources')
            }
        if 'source' in source_type.param and 'source' in spec:
            resolved_spec['source'] = cls.from_spec(spec.pop('source'))
        for k, v in spec.items():
            if isinstance(v, str) and v.startswith('@'):
                v = cls._resolve_reference(v)
            elif isinstance(v, dict):
                v = cls._recursive_resolve(v, source_type)
            if k == 'filters' and 'source' in resolved_spec:
                source_schema = resolved_spec['source'].get_schema()
                v = [Filter.from_spec(fspec, source_schema) for fspec in v]
            if k == 'transforms':
                v = [Transform.from_spec(tspec) for tspec in v]
            resolved_spec[k] = v
        return resolved_spec

    @classmethod
    def from_spec(cls, spec):
        """
        Creates a Source object from a specification. If a Source
        specification references other sources these may be supplied
        in the sources dictionary and be referenced by name.

        Parameters
        ----------
        spec : dict or str
            Specification declared as a dictionary of parameter values
            or a string referencing a source in the sources dictionary.

        Returns
        -------
        Resolved and instantiated Source object
        """
        if spec is None:
            raise ValueError('Source specification empty.')
        elif isinstance(spec, str):
            if spec in state.sources:
                source = state.sources[spec]
            elif spec in state.spec.get('sources', {}):
                source = state.load_source(spec, state.spec['sources'][spec])
            else:
                raise ValueError(f"Source with name '{spec}' was not found.")
            return source

        spec = dict(spec)
        source_type = Source._get_type(spec.pop('type'))
        resolved_spec = cls._recursive_resolve(dict(spec), source_type)
        return source_type(**resolved_spec)

    def __init__(self, **params):
        from ..config import config
        self.root = params.pop('root', config.root)
        super().__init__(**params)
        self._cache = {}
        self._schema_cache = {}

    def _get_key(self, table, **query):
        key = (table,)
        for k, v in sorted(query.items()):
            if isinstance(v, list):
                v = tuple(v)
            key += (k, v)
        return key

    def _get_schema_cache(self):
        if self._schema_cache:
            return self._schema_cache
        elif self.cache_dir:
            path = os.path.join(self.root, self.cache_dir, f'{self.name}.json')
            if os.path.isfile(path):
                with open(path) as f:
                    return json.load(f)
        return None

    def _set_schema_cache(self, schema):
        self._schema_cache = schema
        if self.cache_dir:
            path = Path(os.path.join(self.root, self.cache_dir))
            path.mkdir(parents=True, exist_ok=True)
            with open(path / f'{self.name}.json', 'w') as f:
                json.dump(schema, f)

    def _get_cache(self, table, **query):
        key = self._get_key(table, **query)
        if key in self._cache:
            return self._cache[key], not bool(query)
        elif key[:1] in self._cache:
            return self._cache[key[:1]], True
        elif self.cache_dir:
            if query:
                sha = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
                filename = f'{sha}_{table}.parq'
            else:
                filename = f'{table}.parq'
            path = os.path.join(self.root, self.cache_dir, filename)
            if os.path.isfile(path) or os.path.isdir(path):
                if 'dask.dataframe' in sys.modules:
                    import dask.dataframe as dd
                    return dd.read_parquet(path), not bool(query)
                return pd.read_parquet(path), not bool(query)
        return None, not bool(query)

    def _set_cache(self, data, table, write_to_file=True, **query):
        key = self._get_key(table, **query)
        self._cache[key] = data
        if self.cache_dir and write_to_file:
            path = os.path.join(self.root, self.cache_dir)
            Path(path).mkdir(parents=True, exist_ok=True)
            if query:
                sha = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
                filename = f'{sha}_{table}.parq'
            else:
                filename = f'{table}.parq'
            try:
                data.to_parquet(os.path.join(path, filename))
            except Exception as e:
                self.param.warning(f"Could not cache '{table}' to parquet"
                                   f"file. Error during saving process: {e}")

    def clear_cache(self):
        """
        Clears any cached data.
        """
        self._cache = {}
        self._schema_cache = {}
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

    def get_tables(self):
        """
        Returns the list of tables available on this source.

        Returns
        -------
        list
            The list of available tables on this source.
        """

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
        schemas = {}
        for name in self.get_tables():
            if table is not None and name != table:
                continue
            df = self.get(name, __dask=True)
            schemas[name] = get_dataframe_schema(df)['items']['properties']
        return schemas if table is None else schemas[table]

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
    Loads CSV, Excel, JSON and Parquet files using pandas.read_* or
    dask.read_* functions.
    """

    dask = param.Boolean(default=False, doc="""
        Whether to return a Dask dataframe.""")

    kwargs = param.Dict(doc="""
        Keyword arguments to the pandas/dask loading function.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables to load. If a list is supplied the
        names are computed from the filenames, otherwise the keys are
        the names. The values must filepaths or URLs to the data:

            {
              'local' : '/home/user/local_file.csv',
              'remote': 'https://test.com/test.csv'
            }

        if the filepath does not have a declared extension an extension
        may be provided in a list or tuple, e.g.:

            {'table': ['http://test.com/api', 'json']}
        """)

    use_dask = param.Boolean(default=True, doc="""
        Whether to use dask to load files.""")

    _pd_load_fns = {
        'csv': pd.read_csv,
        'xlsx': pd.read_excel,
        'xls': pd.read_excel,
        'parq': pd.read_parquet,
        'parquet': pd.read_parquet,
        'json': pd.read_json
    }

    _load_kwargs = {
        'csv': {'parse_dates': True}
    }

    source_type = 'file'

    def __init__(self, **params):
        if 'files' in params:
            params['tables'] = params.pop('files')
        super().__init__(**params)
        self._template_re = re.compile('(@\{.*\})')

    def _load_fn(self, ext, dask=True):
        kwargs = dict(self._load_kwargs.get(ext, {}))
        if self.kwargs:
            kwargs.update(self.kwargs)
        if self.use_dask and dask:
            try:
                import dask.dataframe as dd
            except Exception:
                return self._load_fn(ext, dask=False)
            if ext == 'csv':
                return dd.read_csv, kwargs
            elif ext in ('parq', 'parquet'):
                return dd.read_parquet, kwargs
            elif ext == 'json':
                if 'orient' not in kwargs:
                    kwargs['orient'] = None
                return dd.read_json, kwargs
        if ext not in self._pd_load_fns:
            raise ValueError("File type '{ext}' not recognized and cannot be loaded.")
        return self._pd_load_fns[ext], kwargs

    def _set_cache(self, data, table, **query):
        _, ext = self._named_files[table]
        if ext in ('parq', 'parquet'):
            query['write_to_file'] = False
        super()._set_cache(data, table, **query)

    @property
    def _named_files(self):
        if isinstance(self.tables, list):
            tables = {}
            for f in self.tables:
                if f.startswith('http'):
                    name = f
                else:
                    name = '.'.join(os.path.basename(f).split('.')[:-1])
                tables[name] = f
        else:
            tables = self.tables
        files = {}
        for name, table in tables.items():
            ext = None
            if isinstance(table, (list, tuple)):
                table, ext = table
            else:
                basename = os.path.basename(table)
                if '.' in basename:
                    ext = basename.split('.')[-1]
            files[name] = (table, ext)
        return files

    def _resolve_template_vars(self, table):
        for m in self._template_re.findall(table):
            ref = f'@{m[2:-1]}'
            values = self._resolve_reference(ref)
            values = ','.join([v for v in values])
            table = table.replace(m, quote(values))
        return [table]

    def get_tables(self):
        return list(self._named_files)

    def _load_table(self, table, dask=True):
        df = None
        for name, filepath in self._named_files.items():
            filepath, ext = filepath
            if '://' not in filepath:
                filepath = os.path.join(self.root, filepath)
            if name != table:
                continue
            load_fn, kwargs = self._load_fn(ext, dask=dask)
            paths = self._resolve_template_vars(filepath)
            if self.use_dask and ext in ('csv', 'json', 'parquet', 'parq') and dask:
                try:
                    df = load_fn(paths, **kwargs)
                except Exception as e:
                    if dask:
                        return self._load_table(table, dask=False)
                    raise e
            else:
                try:
                    dfs = [load_fn(path, **kwargs) for path in paths]
                except Exception as e:
                    if dask:
                        return self._load_table(table, dask=False)
                    raise e
                if len(dfs) <= 1:
                    df = dfs[0] if dfs else None
                elif self.use_dask and hasattr(dfs[0], 'compute'):
                    import dask.dataframe as dd
                    df = dd.concat(dfs)
                else:
                    df = pd.concat(dfs)
            if hasattr(df, 'persist'):
                df = df.persist()
        if df is None:
            tables = list(self._named_files)
            raise ValueError(f"Table '{table}' not found. Available tables include: {tables}.")
        return df

    @cached()
    def get(self, table, **query):
        dask = query.pop('__dask', self.dask)
        df = self._load_table(table)
        df = self._filter_dataframe(df, **query)
        return df if dask or not hasattr(df, 'compute') else df.compute()


class JSONSource(FileSource):

    chunk_size = param.Integer(default=0, doc="""
        Number of items to load per chunk if a template variable
        is provided.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables to load. If a list is supplied the
        names are computed from the filenames, otherwise the keys are
        the names. The values must filepaths or URLs to the data:

            {
              'local' : '/home/user/local_file.csv',
              'remote': 'https://test.com/test.csv'
            }
    """)

    source_type = 'json'

    def _resolve_template_vars(self, template):
        template_vars = self._template_re.findall(template)
        template_values = []
        for m in template_vars:
            ref = f'@{m[2:-1]}'
            values = self._resolve_reference(ref)
            template_values.append(values)
        tables = []
        cross_product = list(product(*template_values))
        if self.chunk_size and len(cross_product) > self.chunk_size:
            for i in range(len(cross_product)//self.chunk_size):
                start = i*self.chunk_size
                chunk = cross_product[start: start+self.chunk_size]
                tvalues = zip(*chunk)
                table = template
                for m, tvals in zip(template_vars, tvalues):
                    tvals = ','.join([v for v in set(tvals)])
                    table = table.replace(m, quote(tvals))
                tables.append(table)
        else:
            tvalues = list(zip(*cross_product))
            table = template
            for m, tvals in zip(template_vars, tvalues):
                values = ','.join([v for v in set(tvals)])
                table = table.replace(m, quote(values))
            tables.append(table)
        return tables

    def _load_fn(self, ext, dask=True):
        return super()._load_fn('json', dask=dask)

    @cached(with_query=False)
    def get(self, table, **query):
        return super().get(table, **query)



class WebsiteSource(Source):
    """
    Queries whether a website responds with a 400 status code.
    """

    urls = param.List(doc="URLs of the websites to monitor.")

    source_type = 'live'

    def get_schema(self, table=None):
        schema = {
            "status": {
                "url": {"type": "string", 'enum': self.urls},
                "live": {"type": "boolean"}
            }
        }
        return schema if table is None else schema[table]
    
    def get_tables(self):
        return ['status']

    @cached(with_query=False)
    def get(self, table, **query):
        data = []
        for url in self.urls:
            try:
                r = requests.get(url)
                live = r.status_code == 200
            except Exception:
                live = False
            data.append({"live": live, "url": url})
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

    def get_tables(self):
        return ['summary', 'sessions']

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

    def get_tables(self):
        return list(self.tables)

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
                        schema[column] = merge_schemas(col_schema, schema.get(column))
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



class DerivedSource(Source):
    """
    A DerivedSource references tables on other sources and optionally
    allows applying filters and transforms to the returned data which
    is then made available as a new (derived) table.

    The DerivedSource has two modes:

      1) When an explicit `tables` specification is provided full 
         control over the exact tables to filter and transform is
         available. This is referred to as the 'table' mode.
      2) When a `source` is declared all tables on that Source are
         mirrored and filtered and transformed acccording to the
         supplied `filters` and `transforms`. This is referred to as
         'mirror' mode. 

    1. Table Mode
    ~~~~~~~~~~~~~

    In 'table' mode the tables can reference any table on any source
    using the reference syntax and declare filters and transforms to
    apply to that specific table, e.g. a table specification might
    look like this:

      {
        'derived_table': 
        {
          'source': 'original_source',
          'table': 'original_table'
          'filters': [
            ...
          ],
          'transforms': [
            ...
          ]
        }
      }

    2. Mirror
    ~~~~~~~~~

    In mirror mode the DerivedSource may reference an existing source
    directly, e.g.:
    
      {
        'type': 'derived',
        'source': 'original_source',
        'filters': [...],
        'transforms': [...],
      }
    """

    filters = param.List(doc="""
        A list of filters to apply to all tables of this source.""")

    source = param.ClassSelector(class_=Source, doc="""
        A source to mirror the tables on.""")

    tables = param.Dict(default={}, doc="""
        The dictionary of tables and associated filters and transforms.""")

    transforms = param.List(doc="""
        A list of transforms to apply to all tables of this source.""")

    source_type = 'derived'

    def _get_source_table(self, table):
        if self.tables:
            spec = self.tables.get(table)
            if spec is None:
                raise ValueError(f"Table '{table}' was not declared on the"
                                 "DerivedSource. Available tables include "
                                 f"{list(self.tables)}")
            source, table = spec['source'], spec['table']
            filters = spec.get('filters', []) + self.filters
        else:
            source = self.source
            filters = self.filters
        query = dict({filt.field: filt.value for filt in filters})
        return source.get(table, **query)

    @cached(with_query=False)
    def get(self, table, **query):
        df = self._get_source_table(table)
        if self.tables:
            transforms = self.tables[table].get('transforms', []) + self.transforms
        else:
            transforms = self.transforms
        for transform in transforms:
            df = transform.apply(df)
        return self._filter_dataframe(df, **query)

    get.__doc__ = Source.get.__doc__

    def get_tables(self):
        return list(self.tables) if self.tables else self.source.get_tables()

    def clear_cache(self):
        super().clear_cache()
        if self.tables:
            for spec in self.tables.values():
                spec['source'].clear_cache()
        else:
            self.source.clear_cache()
