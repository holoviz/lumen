import json
import threading

from typing import Any

import google.auth
import pandas as pd
import param

from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery, exceptions
from google.cloud.bigquery.client import Client
from tqdm import tqdm

from ..transforms.sql import SQLFilter, SQLMinMax
from .base import BaseSQLSource, cached, cached_schema


class BigQuerySource(BaseSQLSource):
    """
    BigQuerySource provides access to a Google BigQuery project.
    """

    datasets = param.List(default=None, doc="List of datasets to include")

    filter_in_sql = param.Boolean(default=True, doc="Whether to apply filters in SQL or in-memory.")

    location = param.String(doc="Location where the project resides.")

    project_id = param.String(doc="The Google Cloud's project ID.")

    tables = param.ClassSelector(class_=(list, dict), doc="""
       A list of tables or a dictionary mapping from table name to a SQL query.""")

    dialect = "bigquery"

    def __init__(self, **params) -> None:
        self._metadata__client: Client | None = None
        self._sql__client: Client | None = None
        self._credentials: Credentials | None = None
        self._cached_tables: list[str] | None = None
        self._metadata_lock = threading.Lock()
        super().__init__(**params)

    def _create_client(
        self,
        project_id: str | None = None,
        location: str | None = None,
    ) -> Client | None:
        client = None
        kwargs = {"credentials": self._credentials}
        if project_id is not None:
            kwargs["project"] = project_id
        if location is not None:
            kwargs["location"] = location
        try:
            client = bigquery.Client(**kwargs)
        except Exception as e:
            msg = "Unable to create a BigQuery client."
            raise exceptions.ClientError(msg) from e

        return client

    @property
    def _sql_client(self):
        if not self._credentials:
            self._authorize()
        self._sql__client = self._create_client(location=self.location)
        return self._sql__client

    @property
    def _metadata_client(self):
        if not self._credentials:
            self._authorize()
        self._metadata__client = self._create_client(project_id=self.project_id, location=self.location)
        return self._metadata__client

    def _filter_dataset(self, dataset: str):
        return False

    def _authorize(self) -> None:
        """Get the default credentials for the current environment.

        To enable application default credentials with the Cloud SDK for the first time, run the
        following manually.

            gcloud init

        To manually enable authentication, run the following.

            gcloud auth application-default login

        This method will attempt to connect to Google Cloud using the SDK after you have run
        `gcloud init`.

        Returns
        -------
        None

        Raises
        ------
        DefaultCredentialsError
            This error is raised if no default credentials can be found. To prevent this in the
            future, you will need to run `gcloud init` at least once.
        """
        try:
            self._credentials, project_id = google.auth.default()  # pyright: ignore [reportAttributeAccessIssue]
        except DefaultCredentialsError:
            msg = (
                "No default credentials can be found. Please run `gcloud init` at least once "
                "to create default credentials to your Google Cloud project."
            )
            raise DefaultCredentialsError(msg)

        if self.project_id == "":
            self.project_id = project_id

    def execute(self, sql_query: str) -> pd.DataFrame:
        return self._sql_client.query_and_wait(sql_query).to_dataframe()

    def get_tables(self) -> list[str]:
        """Get a list of available tables for the project.

        Returns
        -------
        list[str]
            Table names are composed of f"{project_id}.{dataset_id}.{table_id}".
        """
        if self.tables:
            return list(self.tables)
        elif self._cached_tables:
            return self._cached_tables

        table_references = []
        project_id = self._metadata_client.project
        dataset_records = list(self._metadata_client.list_datasets())
        for dataset_record in tqdm(dataset_records):
            dataset_id = dataset_record.dataset_id
            if dataset_id not in self.datasets or self._filter_dataset(dataset_id):
                continue
            table_records = self._metadata_client.list_tables(dataset=dataset_id)
            table_ids = [
                table_record.table_id
                for table_record in table_records
                if table_record.table_id is not None
            ]
            for table_id in table_ids:
                table_references.append(f"{project_id}.{dataset_id}.{table_id}")

        self._cached_tables = table_references
        return table_references

    def get_sql_expr(self, table: str) -> str:
        if isinstance(self.tables, dict):
            if table not in self.tables:
                raise KeyError(f"Table {table} not found in {list(self.tables)}.")
            return self.tables[table]
        return f"SELECT * FROM {table}"

    def create_sql_expr_source(self, tables: dict[str, str], **kwargs):
        params = dict(self.param.values(), **kwargs)
        params.pop("name", None)
        params["tables"] = tables
        source = type(self)(**params)
        return source

    def _get_dataset_metadata(self, datasets: list[str] | None = None) -> dict:
        """Get metadata for all available datasets in the project.

        Parameters
        ----------
        datasets: list[str] | None

        Returns
        -------
        dict
        """
        data = {}
        for record in self._metadata_client.list_datasets():
            dataset_id = record.dataset_id
            if (datasets is not None and dataset_id not in datasets):
                continue
            project_id = record.project
            dataset = self._metadata_client.get_dataset(dataset_ref=f"{project_id}.{dataset_id}")
            data[dataset_id] = {
                "friendly_name": dataset.friendly_name,
                "description": dataset.description,
                "created": str(dataset.created),
                "modified": str(dataset.modified),
            }
        return data

    def _get_tables_metadata(self, project_id: str, dataset_id: str, tables: list[str] | None = None) -> dict:
        """Get metadata for all tables associated with the given dataset.

        Parameters
        ----------
        project_id : str
            The BigQuery project ID.
        dataset_id : str
            The dataset ID within the given project.

        Returns
        -------
        dict
        """
        data = {}
        for record in self._metadata_client.list_tables(dataset=dataset_id):
            table_id = record.table_id
            table_reference = f"{project_id}.{dataset_id}.{table_id}"
            if tables and table_reference not in tables:
                continue
            table = self._metadata_client.get_table(table=table_reference)
            table_clustering_fields = json.dumps(table.clustering_fields)
            if table_clustering_fields == "null":
                table_clustering_fields = None
            table_labels = json.dumps(table.labels)
            if table_labels == "null":
                table_labels = None

            columns = {}
            for column in table.schema:
                column_fields = []
                for column_field in column.fields:
                    field_datum = column_field.__dict__["_properties"]
                    column_fields.append(field_datum)
                if column_fields:
                    column_fields = json.dumps(column_fields)
                else:
                    column_fields = None
                columns[column.name] = {
                    "type": column.field_type,
                    "description": column.description,
                    "fields": column_fields,
                }
            data[table_reference] = {
                "columns": columns,
                "friendly_name": table.friendly_name,
                "description": table.description,
                "created": str(table.created),
                "modified": str(table.modified),
                "num_rows": table.num_rows,
                "clustering_fields": table_clustering_fields,
                "labels": table_labels,
            }
        return data

    def _get_table_metadata(self, tables: list[str]) -> dict:
        if self.datasets is None:
            datasets = [record.dataset_id for record in self._metadata_client.list_datasets()]
        else:
            datasets = self.datasets
        metadata = {}
        for dataset in datasets:
            if self.datasets is None or dataset in self.datasets:
                metadata.update(self._get_tables_metadata(self.project_id, dataset, tables))
        return metadata

    def _get_table(self, table: str):
        with self._metadata_lock:
            return self._metadata_client.get_table(table=table)

    def _get_table_schema(
        self,
        table: str,
        limit: int | None = None,
        shuffle: bool = False,
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        schema = {}
        bq_table = self._get_table(table)
        enums = []
        min_maxes = []
        for column in bq_table.schema:
            column_name = column.name
            if column_name in schema:
                continue
            column_type = column.field_type

            if column_type == "STRING":
                schema[column_name] = {"type": "string", "enum": []}
                enums.append(column_name)
            elif column_type in ["TIMESTAMP", "DATE"]:
                schema[column_name] = {
                    "type": "str",
                    "format": "datetime",
                    "min": "",
                    "max": "",
                }
                min_maxes.append(column_name)
            elif column_type in ["FLOAT", "INTEGER"]:
                schema[column_name] = {
                    "type": "integer" if column_type == "INTEGER" else "number",
                }
                min_maxes.append(column_name)
            elif column_type == "BOOLEAN":
                schema[column_name] = {"type": column_type.lower()}
            elif column_type == "RECORD":
                print(f"{column_name} is of type {column_type}, which we currently ignore.")
                continue

        sql_expr = self.get_sql_expr(table)
        for col in enums:
            distinct_expr = f"SELECT DISTINCT {col} FROM {sql_expr} LIMIT 1000"
            distinct = self.execute(distinct_expr)
            schema[col]['enum'] = distinct[col].tolist()

        if not min_maxes:
            return schema

        minmax_expr = SQLMinMax(columns=min_maxes, read=self.dialect).apply(sql_expr)
        minmax_expr = ' '.join(minmax_expr.splitlines())
        minmax_data = self.execute(minmax_expr)
        for col in min_maxes:
            kind = minmax_data[col].dtype.kind
            if kind in 'iu':
                cast = int
            elif kind == 'f':
                cast = float
            elif kind == 'M':
                cast = str
            else:
                cast = lambda v: v

            # some dialects, like snowflake output column names to UPPERCASE regardless of input case
            min_col = f'{col}_min' if f'{col}_min' in minmax_data else f'{col}_MIN'
            min_data = minmax_data[min_col].iloc[0]
            max_col = f'{col}_max' if f'{col}_max' in minmax_data else f'{col}_MAX'
            max_data = minmax_data[max_col].iloc[0]
            schema[col]['inclusiveMinimum'] = min_data if pd.isna(min_data) else cast(min_data)
            schema[col]['inclusiveMaximum'] = max_data if pd.isna(max_data) else cast(max_data)
        return schema

    @cached_schema
    def get_schema(
        self, table: str | None = None, limit: int | None = None, shuffle: bool = False
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """Determine the schema of the given `table`.

        This method overrides the inherited `get_schema` from the base class `Source`. The reason
        why we override

        Parameters
        ----------
        table : str | None
            The name of the table. Must be in the form: f"{project_id}.{dataset_id}.{table_id}"
            or reference a table in the tables dictionary.
        limit : int | None
            The maximum number of rows to sample from the table.
        shuffle : bool
            Whether to shuffle the rows of the table.

        Returns
        -------
        dict[str, dict[str, Any]] | dict[str, Any]
        """
        tables = [table] if table else self.get_tables()
        schemas = {}
        for t in tables:
            if isinstance(self.tables, dict) or limit:
                # BigQuery sampling is extremely slow, so we don't shuffle
                schemas[t] = super().get_schema(t, limit, False)
            else:
                schemas[t] = self._get_table_schema(t)
        return schemas[table] if table else schemas

    @cached
    def get(self, table, **query):
        query.pop("__dask", None)
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop("sql_transforms", [])
        conditions = list(query.items())
        filter_in_sql = bool(self.filter_in_sql)
        if filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms
        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)
        return self.execute(sql_expr)

    def close(self):
        """
        Close the BigQuery client connections, releasing associated resources.

        This method should be called when the source is no longer needed to prevent
        connection leaks and properly clean up resources.
        """
        # Close SQL client
        if self._sql__client is not None:
            self._sql__client.close()
            self._sql__client = None

        # Close metadata client
        if self._metadata__client is not None:
            self._metadata__client.close()
            self._metadata__client = None

        # Clear credentials
        self._credentials = None
        self._cached_tables = None
