import intake
import param

from ..util import get_dataframe_schema
from .base import Source, cached


class IntakeBaseSource(Source):

    load_schema = param.Boolean(default=True, doc="""
        Whether to load the schema""")

    __abstract = True

    def _read(self, entry, dask=True):
        if self.dask or dask:
            try:
                return entry.to_dask()
            except Exception:
                if self.dask:
                    self.param.warning(
                        f"Could not load {entry.name!r} table with dask."
                    )
        return entry.read()

    def get_tables(self):
        return list(self.cat)

    def get_schema(self, table=None):
        schemas = {}
        for entry in list(self.cat):
            if table is not None and entry != table:
                continue
            data = self.get(entry, __dask=True)
            if self.load_schema:
                schemas[entry] = get_dataframe_schema(data)['items']['properties']
            else:
                schemas[entry] = {}
        return schemas if table is None else schemas[table]

    @cached(with_query=False)
    def get(self, table, **query):
        dask = query.pop('__dask', self.dask)
        try:
            entry = self.cat[table]
        except KeyError:
            raise KeyError(f"'{table}' table could not be found in Intake "
                           "catalog. Available tables include: "
                           f"{list(self.cat)}.")
        df = self._read(entry, dask)
        return df if dask or not hasattr(df, 'compute') else df.compute()


class IntakeSource(IntakeBaseSource):
    """
    An IntakeSource loads data from an Intake catalog.
    """

    catalog = param.Dict(doc="An inlined Catalog specification.")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the catalog file.")

    source_type = 'intake'

    def __init__(self, **params):
        super().__init__(**params)
        if self.uri and self.catalog:
            raise ValueError("Either specify a Catalog uri or an "
                             "inlined catalog, not both.")
        elif self.uri:
            self.cat = intake.open_catalog(self.uri)
        elif self.catalog:
            context = {'root': self.root}
            result = intake.catalog.local.CatalogParser(self.catalog, context=context)
            if result.errors:
                raise intake.catalog.exceptions.ValidationError(
                    "Catalog '{}' has validation errors:\n\n{}"
                    "".format(self.catalog, "\n".join(result.errors)), result.errors)
            cfg = result.data
            del cfg['plugin_sources']
            entries = {entry.name: entry for entry in cfg.pop('data_sources')}
            self.cat = intake.catalog.Catalog(**cfg)
            self.cat._entries = entries


class IntakeBaseSQLSource(IntakeBaseSource):

    # Declare this source supports SQL transforms
    _sql = True

    @cached()
    def get(self, table, **query):
        '''
        Applies SQL Transforms, creating new temp catalog on the fly
        and querying the database.
        '''
        dask = query.pop('__dask', self.dask)
        sql_transforms = query.pop('sql_transforms', [])

        try:
            source = self.cat[table]
        except KeyError:
            raise KeyError(
                f"'{table}' table could not be found in Intake catalog. "
                f"Available tables include: {list(self.cat)}."
            )
        sql_expr = source._sql_expr
        for sql_transform in sql_transforms:
            sql_expr = sql_transform.apply(sql_expr)
        new_source = type(source)(**dict(source._init_args, sql_expr=sql_expr))
        df = self._read(new_source)
        return df if dask or not hasattr(df, 'compute') else df.compute()


class IntakeDremioSource(IntakeBaseSQLSource):

    cert = param.String(default="Path to certificate file")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the catalog file.")

    tls = param.Boolean(default=False, doc="Enable encryption")

    username = param.String(default=None, doc="Dremio username")

    password = param.String(default=None, doc="Dremio password or token")

    source_type = 'intake_dremio'

    def __init__(self, **params):
        from intake_dremio.dremio_cat import DremioCatalog
        super().__init__(**params)
        self.cat = DremioCatalog(
            self.uri, cert=self.cert, tls=self.tls, username=self.username,
            password=self.password
        )


class IntakeSQLSource(IntakeBaseSQLSource, IntakeSource):
    """
    Intake source specifically for SQL sources.
    Allows for sql transformations to be applied prior to querying the source.
    """

    source_type = 'intake_sql'
