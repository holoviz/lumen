from __future__ import annotations

import decimal
import re

from pathlib import Path
from typing import Any

import pandas as pd
import param
import snowflake.connector

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, load_pem_private_key,
)

from ..transforms.sql import SQLFilter, SQLSelectFrom
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

    excluded_tables = param.List(default=[], doc="""
        List of table names that should be excluded from the results.
        The items can be fully qualified (database.schema.table), partially
        qualified (schema.table), simply table names, or wildcards
        (e.g. database.schema.*).""")

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

    @staticmethod
    def _convert_decimals_to_float(df: pd.DataFrame, sample: int = 100) -> pd.DataFrame:
        """
        Convert decimal.Decimal to float in a pandas DataFrame, as
        most packages do not support decimal.Decimal natively.
        Samples only a subset of the DataFrame to check for decimal.Decimal.

        Arguments
        ---------
        df (pd.DataFrame):
            the DataFrame to convert
        sample (int):
            number of rows to sample to check for decimal.Decimal
        """
        df = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            try:
                if df[col].sample(min(sample, len(df))).apply(lambda x: isinstance(x, decimal.Decimal)).any():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                df[col] = df[col].astype(str)
        return df

    def execute(self, sql_query: str, *args, **kwargs):
        df = self._cursor.execute(sql_query, *args, **kwargs).fetch_pandas_all()
        return self._convert_decimals_to_float(df)

    def get_tables(self) -> list[str]:
        # limited set of tables was provided
        if isinstance(self.tables, dict | list):
            return [t for t in list(self.tables) if not self._is_table_excluded(t)]

        tables_df = self.execute(f'SELECT TABLE_NAME, TABLE_SCHEMA FROM {self.database}.INFORMATION_SCHEMA.TABLES;')
        return [
            f'{self.database}.{row.TABLE_SCHEMA}.{row.TABLE_NAME}'
            for _, row in tables_df.iterrows()
            if not self._is_table_excluded(f'{self.database}.{row.TABLE_SCHEMA}.{row.TABLE_NAME}')
        ]

    def get_sql_expr(self, table: str):
        if isinstance(self.tables, dict):
            table = self.tables[table]
        return SQLSelectFrom(sql_expr=self.sql_expr).apply(table)

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

    def _get_table_metadata(self, table: str | list[str], batched: bool = False) -> dict[str, Any]:
        """
        Generate metadata for all tables or a single table (batched=False) in Snowflake.
        Handles formats: database.schema.table_name, schema.table_name, or table_name.
        Schema can be None to be used as a wildcard.
        """
        null_result = {"description": "", "columns": {}, "rows": 0, "updated_at": None, "created_at": None}
        if batched:
            table_names = table if isinstance(table, list) else [table]
            parsed_tables = []
            for t in table_names:
                parts = t.split(".")
                if len(parts) == 3:
                    parsed_tables.append(parts)  # database.schema.table_name
                elif len(parts) == 2:
                    parsed_tables.append([self.database, parts[0], parts[1]])  # schema.table_name
                elif len(parts) == 1:
                    parsed_tables.append([self.database, self.schema, parts[0]])  # table_name
                else:
                    raise ValueError(f"Invalid table format: {t}")

            table_slugs = pd.Series([".".join(t) for t in parsed_tables]).str.upper()

            table_metadata = self.execute(
                """
                SELECT
                TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, COMMENT as TABLE_DESCRIPTION, ROW_COUNT, LAST_ALTERED, CREATED
                FROM INFORMATION_SCHEMA.TABLES
                """
            )
            table_metadata["TABLE_SLUG"] = table_metadata[
                ["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME"]
            ].agg(lambda x: ".".join(x), axis=1).str.upper()

            # TODO: maybe do this in SQL?
            table_metadata = table_metadata[table_metadata["TABLE_SLUG"].isin(table_slugs)]

            table_metadata = table_metadata.drop(
                columns=["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME"]
            ).set_index("TABLE_SLUG")

            table_columns = self.execute(
                """
                SELECT
                TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, COMMENT AS COLUMN_DESCRIPTION, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                """
            )
            table_columns["TABLE_SLUG"] = table_columns[
                ["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME"]
            ].agg(lambda x: ".".join(x), axis=1).str.upper()

            table_columns = table_columns[table_columns["TABLE_SLUG"].isin(table_slugs)]

            table_columns = table_columns.drop(
                columns=["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME"]
            ).set_index("TABLE_SLUG")

            table_metadata_columns = table_metadata.join(table_columns).reset_index()

            result = {}
            for table_slug, group in table_metadata_columns.groupby("TABLE_SLUG"):
                first_row = group.iloc[0]
                description = first_row["TABLE_DESCRIPTION"] or ""
                rows = first_row["ROW_COUNT"]
                rows = None if pd.isna(rows) else int(rows)
                updated_at = first_row["LAST_ALTERED"].isoformat()
                created_at = first_row["CREATED"].isoformat()
                columns = (
                    group[["COLUMN_NAME", "COLUMN_DESCRIPTION", "DATA_TYPE"]]
                    .rename({"COLUMN_DESCRIPTION": "description", "DATA_TYPE": "data_type"}, axis=1)
                    .set_index("COLUMN_NAME")
                    .transpose()
                    .to_dict()
                )
                result[table_slug] = {
                    "description": description,
                    "columns": columns,
                    "rows": rows,
                    "updated_at": updated_at,
                    "created_at": created_at,
                }
            return result

        # Not batched below; for quick access to a specific
        parts = table.split(".")

        if len(parts) == 3:
            database, schema, table_name = parts
        elif len(parts) == 2:
            database, schema, table_name = self.database, parts[0], parts[1]
        elif len(parts) == 1:
            database, schema, table_name = self.database, self.schema, parts[0]
        else:
            raise ValueError(f"Invalid table format: {table}")
        schema_condition = "" if schema is None else "AND TABLE_SCHEMA = %s"
        params = (database, table_name) if schema is None else (database, schema, table_name)

        table_query = f"""
            SELECT TABLE_NAME, TABLE_SCHEMA, COMMENT, ROW_COUNT, LAST_ALTERED, CREATED
            FROM {database}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_CATALOG = %s {schema_condition} AND TABLE_NAME = %s
        """

        table_metadata = self.execute(table_query, params)
        if table_metadata.empty:
            return null_result

        actual_schema = table_metadata.iloc[0]['TABLE_SCHEMA']
        description = table_metadata.iloc[0]['COMMENT'] or ""
        rows = table_metadata.iloc[0]['ROW_COUNT']
        rows = None if pd.isna(rows) else int(rows)
        updated_at = table_metadata.iloc[0]['LAST_ALTERED'].isoformat()
        created_at = table_metadata.iloc[0]['CREATED'].isoformat()

        column_query = f"""
            SELECT COLUMN_NAME, COMMENT, DATA_TYPE
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_CATALOG = %s AND TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """

        columns_info = self.execute(column_query, (database, actual_schema, table_name))
        columns = columns_info.set_index("COLUMN_NAME")[["COMMENT", "DATA_TYPE"]].fillna("").rename(
            {"COMMENT": "description", "DATA_TYPE": "data_type"}, axis=1
        ).transpose().to_dict()
        return {"description": description, "columns": columns, "rows": rows, "updated_at": updated_at, "created_at": created_at}
