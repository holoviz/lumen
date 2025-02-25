import param

from sqlalchemy.engine.create import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.expression import text

from lumen.sources.base import BaseSQLSource


class SQLAlchemySource(BaseSQLSource):
    driver = param.String(default=None, doc="SQL driver.")
    username = param.String(default=None, doc="Username used for authentication.")
    password = param.String(default=None, doc="The password for the given username.")
    host = param.String(default=None, doc="IP address of the database.")
    port = param.String(default=None, doc="Port used to connect to the database.")
    database = param.String(default=None, doc="Database name.")
    query = param.Dict(
        default=None,
        doc=(
            "A dictionary of string keys to string values to be passed to the dialect "
            "and/or the DBAPI upon connect. To specify non-string parameters to a "
            "Python DBAPI directly, use connect_args."
        ),
    )

    def __init__(self) -> None:
        self.url = self.create_url()
        self.engine = create_engine(url=self.url)

    def create_url(self) -> URL:
        url_params = {
            "drivername": self.driver,
            "username": self.username,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "query": self.query,
        }
        return URL.create(**{url_key: url_param for url_key, url_param in url_params.items() if url_param is not None})

    def run_query(self, query: str):
        records = None
        with self.engine.begin() as connection:
            query_ = text(query)
            records = connection.execute(query_)
        return records
