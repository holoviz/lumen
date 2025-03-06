from __future__ import annotations

import re

from pathlib import Path

import param
import snowflake.connector

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, load_pem_private_key,
)

from ..transforms.sql import SQLFilter
from .base import BaseSQLSource, cached

# PEM certificates have the pattern:
#   -----BEGIN PKEY-----  # renamed private... to pkey to prevent ruff from flagging
#   <- multiple lines of encoded data->
#   -----END PKEY-----
#
# The regex captures the header and footer into groups 1 and 3, the body into group 2
# group 1: "header" captures series of hyphens followed by anything that is
#           not a hyphen followed by another string of hyphens
# group 2: "body" capture everything upto the next hyphen
# group 3: "footer" duplicates group 1
_SIMPLE_PEM_CERTIFICATE_REGEX = "^(-+[^-]+-+)([^-]+)(-+[^-]+-+)"


class SnowflakeSource(BaseSQLSource):
    """
    SnowflakeSource uses the snowflake-python-connector library to load data
    from Snowflake.
    """

    account = param.String(default=None, doc="""
        The account identifier to connect to.""")

    authenticator = param.Selector(default=None, objects=[
        'externalbrowser', 'oauth', 'snowflake', 'username_password_mfa', 'SNOWFLAKE_JWT'], doc="""
        The authentication approach to use.""", allow_None=True)

    conn_kwargs = param.Dict(default={}, doc="""
        Additional connection parameters to pass to the Snowflake connector.""")

    database = param.String(default=None, doc="""
        The database to connect to.""")

    host = param.String(default=None, doc="""
        The host to authenticate with.""")

    token = param.String(default=None, doc="""
        The OAuth token if authenticator is set to "oauth".""")

    user = param.String(default=None, doc="""
        The user to authenticate as.""")

    password = param.String(default=None, doc="""
        The password to authenticate with (if authenticator is set to "snowflake").""")

    private_key = param.ClassSelector(
        default=None, class_=(str, bytes, Path), doc="""
        The path or contents of the private key file.""")

    private_key_password = param.String(default=None, doc="""
        The password to decrypt the private key file.""")

    schema = param.String(default=None, doc="""
        The database schema to load data from.""")

    warehouse = param.String(default=None, doc="""
        The warehouse to connect to.""")

    filter_in_sql = param.Boolean(default=True, doc="""
        Whether to apply filters in SQL or in-memory.""")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The SQL expression to execute.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables.""")

    dialect = 'snowflake'

    def __init__(self, **params):
        conn = params.pop('conn', None)
        super().__init__(**params)
        conn_kwargs = self.conn_kwargs.copy()
        if self.account is not None:
            conn_kwargs['account'] = self.account
        if self.authenticator is not None:
            conn_kwargs['authenticator'] = self.authenticator
        if self.database is not None:
            conn_kwargs['database'] = self.database
        if self.host is not None:
            conn_kwargs['host'] = self.host
        if self.token is not None:
            conn_kwargs['token'] = self.token
        if self.password is not None:
            conn_kwargs['password'] = self.password
        if self.private_key is not None:
            conn_kwargs['private_key'] = self.resolve_private_key()
        if self.schema is not None:
            conn_kwargs['schema'] = self.schema
        if self.user is not None:
            conn_kwargs['user'] = self.user
        if self.warehouse is not None:
            conn_kwargs['warehouse'] = self.warehouse

        self._conn = conn or snowflake.connector.connect(**conn_kwargs)
        self._cursor = self._conn.cursor()

    @staticmethod
    def _decode_secret(secret: str | bytes) -> bytes | None:
        """
        Decode the provided secret into bytes.

        If the secret is not a string or bytes, or it is whitespace, then return None.

        Parameters
        ----------
        secret : str or bytes
            The value to decode.

        Returns
        -------
        bytes or None
            The decoded secret as bytes.
        """
        if not isinstance(secret, (bytes, str)) or len(secret) == 0 or secret.isspace():
            return None

        return secret if isinstance(secret, bytes) else secret.encode()

    @staticmethod
    def _compose_pem(private_key: bytes) -> bytes:
        """
        Validate structure of PEM certificate.

        The original key passed from Prefect is sometimes malformed.
        This function recomposes the key into a valid key that will
        pass the serialization step when resolving the key to a DER.

        Parameters
        ----------
        private_key : bytes
            A valid PEM format byte encoded string.

        Returns
        -------
        bytes
            Byte encoded certificate.

        Raises
        ------
        InvalidPemFormat
            If private key is an invalid format.
        """
        pem_parts = re.match(_SIMPLE_PEM_CERTIFICATE_REGEX, private_key.decode())
        if pem_parts is None:
            raise ValueError("Invalid PEM format")

        body = "\n".join(re.split(r"\s+", pem_parts[2].strip()))
        # reassemble header+body+footer
        return f"{pem_parts[1]}\n{body}\n{pem_parts[3]}".encode()

    def resolve_private_key(self):
        """
        Converts a PEM encoded private key into a DER binary key.

        Parameters
        ----------

        Returns
        -------
        bytes or None
            DER encoded key if private_key has been provided otherwise returns None.

        Raises
        ------
        InvalidPemFormat
            If private key is not in PEM format.
        """
        if Path(self.private_key).exists():
            private_key = Path(self.private_key).read_bytes()
        else:
            private_key = self._decode_secret(self.private_key)

        if self.private_key_password is not None:
            password = self._decode_secret(self.private_key_password)
        else:
            password = None

        composed_private_key = self._compose_pem(private_key)
        return load_pem_private_key(
            data=composed_private_key,
            password=password,
            backend=default_backend(),
        ).private_bytes(
            encoding=Encoding.DER,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )

    def create_sql_expr_source(self, tables: dict[str, str], **kwargs):
        """
        Creates a new SQL Source given a set of table names and
        corresponding SQL expressions.
        """
        params = dict(self.param.values(), **kwargs)
        params.pop("name", None)
        params['tables'] = tables
        params['conn'] = self._conn
        return SnowflakeSource(**params)

    def execute(self, sql_query: str):
        return self._cursor.execute(sql_query).fetch_pandas_all()

    def get_tables(self) -> list[str]:
        if isinstance(self.tables, dict | list):
            return list(self.tables)
        tables = self.execute(f'SELECT TABLE_NAME, TABLE_SCHEMA FROM {self.database}.INFORMATION_SCHEMA.TABLES;')
        return [f'{self.database}.{row.TABLE_SCHEMA}.{row.TABLE_NAME}' for _, row in tables.iterrows()]

    def get_sql_expr(self, table: str):
        if isinstance(self.tables, dict):
            table = self.tables[table]
        if 'select ' in table.lower():
            sql_expr = table
        else:
            sql_expr = self.sql_expr.format(table=table)
        return sql_expr.rstrip(";")

    @cached
    def get(self, table, **query):
        query.pop('__dask', None)
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = list(query.items())
        if self.filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms
        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)
        return self.execute(sql_expr)

    def _get_metadata(self, table: str) -> dict[str, dict]:
        """
        Generate metadata for a single table in Snowflake.
        """
        table_query = f"""
            SELECT
                TABLE_NAME,
                COMMENT as TABLE_DESCRIPTION
            FROM
                {self.database}.INFORMATION_SCHEMA.TABLES
            WHERE
                TABLE_NAME = '{table}'
        """
        table_metadata = self.execute(table_query)
        if table_metadata.empty:
            return {"description": "", "columns": {}}

        description = table_metadata.iloc[0]['TABLE_DESCRIPTION'] or ""
        column_query = f"""
            SELECT
                COLUMN_NAME,
                COMMENT
            FROM
                {self.database}.INFORMATION_SCHEMA.COLUMNS
            WHERE
                TABLE_NAME = '{table}'
            ORDER BY
                ORDINAL_POSITION
        """
        columns = {}
        columns_info = self.execute(column_query)
        for _, row in columns_info.iterrows():
            columns[row['COLUMN_NAME']] = {"description": row['COMMENT'] or ""}
        return {"description": description, "columns": columns}
