from abc import abstractmethod

import google.auth
import pandas as pd

from google.cloud import bigquery, exceptions
from google.cloud.bigquery.client import Client

from lumen.sources.sqlalchemy import SQLAlchemySource


class BigQuerySource(SQLAlchemySource):
    def __init__(self, project_id: str, location: str) -> None:
        self.driver = "bigquery"
        self.dialect = "bigquery"
        self.project_id = project_id
        self.location = location

    def authorize(self) -> None:
        self.credentials, project_id = google.auth.default()
        if self.project_id is None:
            self.project_id = project_id

    def create_client(self) -> None | Client:
        client = None
        try:
            client = bigquery.Client(self.project_id, location=self.location)
        except Exception as e:
            raise exceptions.ClientError from e
        return client

    def get_all_datasets_metadata(self) -> list[dict[str, str]]:
        data = []
        if self.client is not None:
            for record in self.client.list_datasets():
                dataset = self.client.get_dataset(dataset_ref=f"{record.project}.{record.dataset_id}")
                datum = {
                    "project": dataset.project,
                    "dataset_id": dataset.dataset_id,
                    "dataset_description": dataset.description,
                    "dataset_name": dataset.friendly_name,
                    "dataset_creation": dataset.created.timestamp(),
                    "dataset_modified": dataset.modified.timestamp(),
                }
                data.append(datum)
        return data

    def get_table_metadata(self, project: str, dataset_id: str) -> list[dict[str, str]]:
        data = []
        if self.client is not None:
            for record in self.client.list_tables(dataset=dataset_id):
                table = self.client.get_table(table=f"{project}.{dataset_id}.{record.table_id}")
                datum = {
                    "project": project,
                    "dataset_id": dataset_id,
                    "table_id": table.table_id,
                    "table_description": table.description,
                    "table_name": table.friendly_name,
                    "table_creation": table.created.timestamp(),
                    "table_modified": table.modified.timestamp(),
                    "table_num_rows": table.num_rows,
                }
                data.append(datum)
        return data

    def get_column_metadata(self, project: str, dataset_id: str, table_id: str) -> list[dict[str, str]]:
        data = []
        if self.client is not None:
            table = self.client.get_table(table=f"{project}.{dataset_id}.{table_id}")
            columns = table.schema
            for column in columns:
                datum = {
                    "project": project,
                    "dataset_id": dataset_id,
                    "table_id": table.table_id,
                    "table_name": table.friendly_name,
                    "column_id": column.name,
                    "column_description": column.description,
                }
                data.append(datum)
        return data

    def _get_schema(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("To be created by the inheriting class.")

    @abstractmethod
    def get_schema(self, *args, **kwargs) -> pd.DataFrame:
        return self._get_schema(*args, **kwargs)
