import hashlib
import json
import re
import shutil
import threading
import warnings
import weakref

from concurrent import futures
from functools import wraps
from itertools import product
from os.path import basename
from pathlib import Path
from urllib.parse import quote, urlparse

import numpy as np
import pandas as pd
import panel as pn
import param
import requests

from panel.io.cache import _generate_hash

from ..base import MultiTypeComponent
from ..filters import Filter
from ..state import state
from ..transforms import Filter as FilterTransform, Transform
from ..util import get_dataframe_schema, is_ref, merge_schemas
from ..validation import ValidationError, match_suggestion_message

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


def cached(method, locks=weakref.WeakKeyDictionary()):
    """
    Adds caching to a Source.get query.

    Returns
    -------
    Returns method wrapped in caching functionality.
    """
    @wraps(method)
    def wrapped(self, table, **query):
        if self._supports_sql and not self.cache_per_query and 'sql_transforms' in query:
            raise RuntimeError(
                'SQLTransforms cannot be used on a Source with cache_per_query '
                'being disabled. Ensure you set `cache_per_query=True`.'
            )
        if self in locks:
            main_lock = locks[self]['main']
        else:
            main_lock = threading.RLock()
            locks[self] = {'main': main_lock}
        with main_lock:
            if table in locks:
                lock = locks[self][table]
            else:
                locks[self][table] = lock = threading.RLock()
        cache_query = query if self.cache_per_query else {}
        with lock:
            df, no_query = self._get_cache(table, **cache_query)
        if df is None:
            if not self.cache_per_query and (hasattr(self, 'dask') or hasattr(self, 'use_dask')):
                cache_query['__dask'] = True
            df = method(self, table, **cache_query)
            with lock:
                self._set_cache(df, table, **cache_query)
        filtered = df
        if (not self.cache_per_query or no_query) and query:
            filtered = FilterTransform.apply_to(
                df, conditions=list(query.items())
            )
        if getattr(self, 'dask', False) or not hasattr(filtered, 'compute'):
            return filtered
        return filtered.compute()
    return wrapped


def cached_schema(method, locks=weakref.WeakKeyDictionary()):
    @wraps(method)
    def wrapped(self, table=None):
        if self in locks:
            main_lock = locks[self]['main']
        else:
            main_lock = threading.RLock()
            locks[self] = {'main': main_lock}
        with main_lock:
            schema = self._get_schema_cache() or {}
        tables = self.get_tables() if table is None else [table]
        if all(table in schema for table in tables):
            return schema if table is None else schema[table]
        for missing in tables:
            if missing in schema:
                continue
            with main_lock:
                if missing in locks[self]:
                    lock = locks[self][missing]
                else:
                    locks[self][missing] = lock = threading.RLock()
            with lock:
                with main_lock:
                    new_schema = self._get_schema_cache() or {}
                if missing in new_schema:
                    schema[missing] = new_schema[missing]
                else:
                    schema[missing] = method(self, missing)
            with main_lock:
                self._set_schema_cache(schema)
        return schema if table is None else schema[table]
    return wrapped


class Source(MultiTypeComponent):
    """
    `Source` components provide allow querying all kinds of data.

    A `Source` can return one or more tables queried using the
    `.get_tables` method, a description of the data returned by each
    table in the form of a JSON schema accessible via the `.get_schema`
    method and lastly a `.get` method that allows filtering the data.

    The Source base class also implements both in-memory and disk
    caching which can be enabled if a `cache_dir` is provided. Data
    cached to disk is stored as parquet files.
    """

    cache_per_query = param.Boolean(default=True, doc="""
        Whether to query the whole dataset or individual queries.""")

    cache_dir = param.String(default=None, doc="""
        Whether to enable local cache and write file to disk.""")

    shared = param.Boolean(default=False, doc="""
        Whether the Source can be shared across all instances of the
        dashboard. If set to `True` the Source will be loaded on
        initial server load.""")

    root = param.ClassSelector(class_=Path, precedence=-1, doc="""
        Root folder of the cache_dir, default is config.root""")

    source_type = None

    __abstract = True

    # Specification configuration
    _internal_params = ['name', 'root']

    # Declare whether source supports SQL transforms
    _supports_sql = False

    @property
    def _reload_params(self):
        "List of parameters that trigger a data reload."
        return list(self.param)

    @classmethod
    def _recursive_resolve(cls, spec, source_type):
        resolved_spec, refs = {}, {}
        if 'sources' in source_type.param and 'sources' in spec:
            resolved_spec['sources'] = {
                source: cls.from_spec(source)
                for source in spec.pop('sources')
            }
        if 'source' in source_type.param and 'source' in spec:
            resolved_spec['source'] = cls.from_spec(spec.pop('source'))
        for k, v in spec.items():
            if is_ref(v):
                refs[k] = v
                v = state.resolve_reference(v)
            elif isinstance(v, dict):
                v, subrefs = cls._recursive_resolve(v, source_type)
                for sk, sv in subrefs.items():
                    refs[f'{k}.{sk}'] = sv
            if k == 'filters' and 'source' in resolved_spec:
                source_schema = resolved_spec['source'].get_schema()
                v = [Filter.from_spec(fspec, source_schema) for fspec in v]
            if k == 'transforms':
                v = [Transform.from_spec(tspec) for tspec in v]
            resolved_spec[k] = v
        return resolved_spec, refs

    @classmethod
    def _validate_filters(cls, filter_specs, spec, context):
        warnings.warn(
            'Providing filters in a Source definition is deprecated, '
            'please declare filters as part of a Pipeline.', DeprecationWarning
        )
        return cls._validate_dict_subtypes('filters', Filter, filter_specs, spec, context)

    @classmethod
    def validate(cls, spec, context=None):
        if isinstance(spec, str):
            if spec not in context['sources']:
                msg = f'Referenced non-existent source {spec!r}.'
                msg = match_suggestion_message(spec, list(context['sources']), msg)
                raise ValidationError(msg, spec, spec)
            return spec
        return super().validate(spec, context)

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
        if isinstance(spec, str):
            if spec in state.sources:
                source = state.sources[spec]
            elif spec in state.spec.get('sources', {}):
                source = state.load_source(spec, state.spec['sources'][spec])
            return source

        spec = spec.copy()
        source_type = Source._get_type(spec.pop('type', None))
        resolved_spec, refs = cls._recursive_resolve(spec, source_type)
        return source_type(refs=refs, **resolved_spec)

    def __init__(self, **params):
        from ..config import config
        params['root'] = Path(params.get('root', config.root))
        super().__init__(**params)
        self.param.watch(self.clear_cache, self._reload_params)
        self._cache = {}
        self._schema_cache = {}

    def _get_key(self, table, **query):
        sha = hashlib.sha256()
        sha.update(table.encode('utf-8'))
        if 'sql_transforms' in query:
            sha.update(_generate_hash([hash(t) for t in query.pop('sql_transforms')]))
        sha.update(_generate_hash(query))
        return sha.hexdigest()

    def _get_schema_cache(self):
        schema = self._schema_cache if self._schema_cache else None
        if self.cache_dir:
            path = self.root / self.cache_dir / f'{self.name}.json'
            if not path.is_file():
                return schema
            with open(path) as f:
                json_schema = json.load(f)
            if schema is None:
                schema = {}
            for table, tschema in json_schema.items():
                if table in schema:
                    continue
                for col, cschema in tschema.items():
                    if cschema.get('type') == 'string' and cschema.get('format') == 'datetime':
                        cschema['inclusiveMinimum'] = pd.to_datetime(
                            cschema['inclusiveMinimum']
                        )
                        cschema['inclusiveMaximum'] = pd.to_datetime(
                            cschema['inclusiveMaximum']
                        )
                schema[table] = tschema
        return schema

    def _set_schema_cache(self, schema):
        self._schema_cache = schema
        if self.cache_dir:
            path = self.root / self.cache_dir
            path.mkdir(parents=True, exist_ok=True)
            try:
                with open(path / f'{self.name}.json', 'w') as f:
                    json.dump(schema, f, default=str)
            except Exception as e:
                self.param.warning(
                    f"Could not cache schema to disk. Error while "
                    f"serializing schema to disk: {e}"
                )

    def _get_cache(self, table, **query):
        query.pop('__dask', None)
        key = self._get_key(table, **query)
        if key in self._cache:
            return self._cache[key], not bool(query)
        elif self.cache_dir:
            if query:
                filename = f'{key}_{table}.parq'
            else:
                filename = f'{table}.parq'
            path = self.root / self.cache_dir / filename
            if path.is_file():
                return pd.read_parquet(path), not bool(query)
            if dd and path.is_dir():
                return dd.read_parquet(path), not bool(query)
            path = path.with_suffix('')
            if dd and path.is_dir():
                return dd.read_parquet(path), not bool(query)
        return None, not bool(query)

    def _set_cache(self, data, table, write_to_file=True, **query):
        query.pop('__dask', None)
        key = self._get_key(table, **query)
        self._cache[key] = data
        if self.cache_dir and write_to_file:
            path = self.root / self.cache_dir
            path.mkdir(parents=True, exist_ok=True)
            if query:
                filename = f'{key}_{table}.parq'
            else:
                filename = f'{table}.parq'
            filepath = path / filename
            if dd:
                if isinstance(data, dd.DataFrame):
                    filepath = filepath.with_suffix('')
            try:
                data.to_parquet(filepath)
            except Exception as e:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                self.param.warning(
                    f"Could not cache '{table}' to parquet file. "
                    f"Error during saving process: {e}"
                )

    def clear_cache(self, *events):
        """
        Clears any cached data.
        """
        self._cache = {}
        self._schema_cache = {}
        if self.cache_dir:
            path = self.root / self.cache_dir
            if path.is_dir():
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

    @cached_schema
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
        names = list(self.get_tables())
        for name in names:
            if table is not None and name != table:
                continue
            df = self.get(name, __dask=True)
            schemas[name] = get_dataframe_schema(df)['items']['properties']

        try:
            return schemas if table is None else schemas[table]
        except KeyError as e:
            msg = f"{type(self).name} does not contain '{table}'"
            msg = match_suggestion_message(table, names, msg)
            raise ValidationError(msg) from e

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
    `RESTSource` allows querying REST endpoints conforming to the Lumen REST specification.

    The `url` must offer two endpoints, the `/data` endpoint must
    return data in a records format while the `/schema` endpoint must
    return a valid Lumen JSON schema.
    """

    url = param.String(doc="URL of the REST endpoint to monitor.")

    source_type = 'rest'

    @cached_schema
    def get_schema(self, table=None):
        query = {} if table is None else {'table': table}
        response = requests.get(self.url+'/schema', params=query)
        return {table: schema['items']['properties'] for table, schema in
                response.json().items()}

    @cached
    def get(self, table, **query):
        query = dict(table=table, **query)
        r = requests.get(self.url+'/data', params=query)
        df = pd.DataFrame(r.json())
        return df


class FileSource(Source):
    """
    `FileSource` loads CSV, Excel and Parquet files using pandas and dask `read_*` functions.

    The `FileSource` can declare a list or dictionary of local or
    remote files which are then loaded using either `pandas.read_*` or
    `dask.dataframe.read_*` functions depending on whether `use_dask`
    is enabled.
    """

    dask = param.Boolean(default=False, doc="""
        Whether to return a Dask dataframe.""")

    kwargs = param.Dict(doc="""
        Keyword arguments to the pandas/dask loading function.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables to load. If a list is supplied the
        names are computed from the filenames, otherwise the keys are
        the names. The values must filepaths or URLs to the data:

        ```
        {
            'local' : '/home/user/local_file.csv',
            'remote': 'https://test.com/test.csv'
        }
        ```

        if the filepath does not have a declared extension an extension
        may be provided in a list or tuple, e.g.:

        ```
        {'table': ['http://test.com/api', 'json']}
        ```
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
        self._template_re = re.compile(r'(@\{.*\})')

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
            raise ValueError(f"File type '{ext}' not recognized and cannot be loaded.")
        return self._pd_load_fns[ext], kwargs

    def _set_cache(self, data, table, **query):
        file, ext = self._named_files[table]
        if ext in ('parq', 'parquet') and Path(file).exists():
            query['write_to_file'] = False
        super()._set_cache(data, table, **query)

    @property
    def _named_files(self):
        if isinstance(self.tables, list):
            tables = {}
            for f in self.tables:
                if isinstance(f, str) and f.startswith('http'):
                    name = f
                else:
                    name = '.'.join(basename(f).split('.')[:-1])
                tables[name] = f
        else:
            tables = self.tables or {}
        files = {}
        for name, table in tables.items():
            if isinstance(table, (list, tuple)):
                table, ext = table
            else:
                if isinstance(table, str) and table.startswith('http'):
                    file = basename(urlparse(table).path)
                else:
                    file = basename(table)
                ext = re.search(r"\.(\w+)$", file)
                if ext:
                    ext = ext.group(1)
            files[name] = (table, ext)
        return files

    def _resolve_template_vars(self, table):
        for m in self._template_re.findall(str(table)):
            values = state.resolve_reference(f'${m[2:-1]}')
            values = ','.join([v for v in values])
            table = table.replace(m, quote(values))
        return [table]

    def get_tables(self):
        return list(self._named_files)

    def _load_table(self, table, dask=True):
        df = None
        for name, (filepath, ext) in self._named_files.items():
            if isinstance(filepath, Path) or '://' not in filepath:
                filepath = self.root / filepath
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

    @cached
    def get(self, table, **query):
        dask = query.pop('__dask', self.dask)
        df = self._load_table(table)
        df = FilterTransform.apply_to(df, conditions=list(query.items()))
        return df if dask or not hasattr(df, 'compute') else df.compute()


class JSONSource(FileSource):
    """
    The JSONSource is very similar to the FileSource but loads json files.

    Both local and remote JSON files can be fetched by declaring them
    as a list or dictionaries of `tables`.
    """

    cache_per_query = param.Boolean(default=False, doc="""
        Whether to query the whole dataset or individual queries.""")

    chunk_size = param.Integer(default=0, doc="""
        Number of items to load per chunk if a template variable
        is provided.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables to load. If a list is supplied the
        names are computed from the filenames, otherwise the keys are
        the names. The values must filepaths or URLs to the data:

        ```
        {
            'local' : '/home/user/local_file.csv',
            'remote': 'https://test.com/test.csv'
        }
        ```
    """)

    source_type = 'json'

    def _resolve_template_vars(self, template):
        template_vars = self._template_re.findall(template)
        template_values = []
        for m in template_vars:
            values = state.resolve_reference(f'${m[2:-1]}')
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

    @cached
    def get(self, table, **query):
        return super().get(table, **query)


class WebsiteSource(Source):
    """
    `WebsiteSource` queries whether a website responds with a 400 status code.
    """

    cache_per_query = param.Boolean(default=False, doc="""
        Whether to query the whole dataset or individual queries.""")

    urls = param.List(doc="URLs of the websites to monitor.")

    source_type = 'live'

    @cached_schema
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

    @cached
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
    """"
    `PanelSessionSource` queries the session_info endpoint of a Panel application.

    Panel applications with --rest-session-info enabled can be queried
    about session statistics. This source makes this data available to
    Lumen for monitoring.
    """

    cache_per_query = param.Boolean(default=False, doc="""
        Whether to query the whole dataset or individual queries.""")

    endpoint = param.String(default="rest/session_info")

    urls = param.List(doc="URL of the websites to monitor.")

    timeout = param.Parameter(default=5)

    source_type = 'session_info'

    @cached_schema
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

    @cached
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
    `JoinedSource` performs a join on tables from one or more sources.

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

    ```
    {'new_table': [
        {'source': 'A', 'table': 'foo', 'index': 'a'},
        {'source': 'B', 'table': 'bar', 'index': 'b'}
    ]}
    ```

    The joined source will now publish the "new_table" with all
    columns from tables "foo" and "bar" except for the index column
    from table "bar", which was merged with the index column "a" from
    table "foo".
    """

    sources = param.ClassSelector(class_=(list, dict), doc="""
        A dictionary of sources indexed by their assigned name.""")

    tables = param.Dict(default={}, doc="""
        A dictionary with the names of the joined sources as keys
        and a specification of the source, table and index to merge
        on.

        ```
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
        ]}
        ```
        """)

    source_type = 'join'

    def get_tables(self):
        return list(self.tables)

    @cached_schema
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

    @cached
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
    `DerivedSource` applies filtering and transforms to tables from other sources.

    A DerivedSource references tables on other sources and optionally
    allows applying filters and transforms to the returned data which
    is then made available as a new (derived) table.

    The DerivedSource has two modes:

    **Table Mode**

    When an explicit `tables` specification is provided full control
    over the exact tables to filter and transform is available. This
    is referred to as the 'table' mode.

    In 'table' mode the tables can reference any table on any source
    using the reference syntax and declare filters and transforms to
    apply to that specific table, e.g. a table specification might
    look like this:

    ```
    {
      'derived_table': {
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
    ```

    **Mirror**

    When a `source` is declared all tables on that Source are mirrored
    and filtered and transformed according to the supplied `filters`
    and `transforms`. This is referred to as 'mirror' mode.

    In mirror mode the DerivedSource may reference an existing source
    directly, e.g.:

    ```
    {
        'type': 'derived',
        'source': 'original_source',
        'filters': [...],
        'transforms': [...],
    }
    ```
    """

    cache_per_query = param.Boolean(default=False, doc="""
        Whether to query the whole dataset or individual queries.""")

    filters = param.List(doc="""
        A list of filters to apply to all tables of this source.""")

    source = param.ClassSelector(class_=Source, doc="""
        A source to mirror the tables on.""")

    tables = param.Dict(default={}, doc="""
        The dictionary of tables and associated filters and transforms.""")

    transforms = param.List(doc="""
        A list of transforms to apply to all tables of this source.""")

    source_type = 'derived'

    @classmethod
    def _validate_filters(cls, *args, **kwargs):
        return cls._validate_list_subtypes('filters', Filter, *args, **kwargs)

    def _get_source_table(self, table):
        if self.tables:
            spec = self.tables.get(table)
            if spec is None:
                raise ValidationError(
                    f"Table '{table}' was not declared on the DerivedSource. "
                    f"Available tables include {list(self.tables)}."
                )
            source, table = spec['source'], spec['table']
            filters = spec.get('filters', []) + self.filters
        else:
            source = self.source
            filters = self.filters
        query = dict({filt.field: filt.value for filt in filters})
        return source.get(table, **query)

    @cached
    def get(self, table, **query):
        df = self._get_source_table(table)
        if self.tables:
            transforms = self.tables[table].get('transforms', []) + self.transforms
        else:
            transforms = self.transforms
        transforms.append(FilterTransform(conditions=list(query.items())))
        for transform in transforms:
            df = transform.apply(df)
        return df

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
