import os

import param


class DbtslMixin(param.Parameterized):

    auth_token = param.String(default=None, doc="""
        The auth token for the dbt semantic layer client;
        if not provided will fetched from DBT_AUTH_TOKEN env var.""")

    environment_id = param.Integer(default=None, doc="""
        The environment ID for the dbt semantic layer client.""")

    host = param.String(default="semantic-layer.cloud.getdbt.com", doc="""
        The host for the dbt semantic layer client.""")

    def __init__(self, environment_id: int, **params):

        super().__init__(environment_id=environment_id, **params)

    def _get_dbtsl_client(self):
        from dbtsl.asyncio import AsyncSemanticLayerClient
        dbtsl_client = AsyncSemanticLayerClient(
            environment_id=self.environment_id,
            auth_token=self.auth_token or os.getenv("DBT_AUTH_TOKEN"),
            host=self.host,
        )
        return dbtsl_client
