import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

import pandas as pd
import param

from lumen.ai.controls import (
    CodeSourceControls, DownloadSourceControls, SourceCatalog,
    UploadSourceControls, URLSourceControls,
)
from lumen.ai.embeddings import NumpyEmbeddings
from lumen.ai.vector_store import NumpyVectorStore


@pytest.fixture
def vector_store():
    """Create a vector store for testing."""
    return NumpyVectorStore(embeddings=NumpyEmbeddings())


@pytest.fixture
def context():
    """Basic context."""
    return {
        "sources": [],
        "visible_slugs": set(),
        "tables_metadata": {},
    }


@pytest.fixture
def source_catalog(context, vector_store):
    """Create a SourceCatalog instance with vector store."""
    catalog = SourceCatalog(context=context, vector_store=vector_store)
    return catalog


@pytest.fixture
def upload_controls(context, source_catalog):
    """Create UploadSourceControls with reference to catalog."""
    controls = UploadSourceControls(context=context, source_catalog=source_catalog)
    return controls


@pytest.fixture
def download_controls(context, source_catalog):
    """Create DownloadSourceControls with reference to catalog."""
    controls = DownloadSourceControls(context=context, source_catalog=source_catalog)
    return controls


# ─────────────────────────────────────────────────────────────────────────────
# CodeSourceControls fixtures
# ─────────────────────────────────────────────────────────────────────────────


def sync_fetch_data(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Sync function that returns a DataFrame."""
    return pd.DataFrame({"ticker": [ticker] * limit, "price": range(limit)})


async def async_fetch_data(ticker: str, limit: int = 5) -> pd.DataFrame:
    """Async function that returns a DataFrame."""
    return pd.DataFrame({"ticker": [ticker] * limit, "value": range(limit)})


def returns_list_of_dicts(n: int = 3) -> list:
    """Function that returns list of dicts."""
    return [{"id": i, "name": f"item_{i}"} for i in range(n)]


def returns_none() -> None:
    """Function that returns None."""
    return None


def raises_error() -> pd.DataFrame:
    """Function that raises an exception."""
    raise ValueError("Test error")


class MockAPIClient:
    """Mock API client with methods to expose."""

    def get_prices(self, symbol: str, days: int = 7) -> pd.DataFrame:
        return pd.DataFrame({"symbol": [symbol] * days, "price": range(days)})

    def get_details(self, symbol: str) -> dict:
        return {"symbol": symbol, "name": f"Company {symbol}"}


@pytest.fixture
def mock_api_client():
    """Create a mock API client instance."""
    return MockAPIClient()


@pytest.fixture
def code_controls_single_func(context, source_catalog):
    """CodeSourceControls with a single function."""
    return CodeSourceControls(
        functions=sync_fetch_data,
        context=context,
        source_catalog=source_catalog,
    )


@pytest.fixture
def code_controls_dict_funcs(context, source_catalog):
    """CodeSourceControls with dict of functions."""
    return CodeSourceControls(
        functions={
            "Fetch Data": sync_fetch_data,
            "List Items": returns_list_of_dicts,
        },
        context=context,
        source_catalog=source_catalog,
    )


@pytest.fixture
def code_controls_instance_methods(context, source_catalog, mock_api_client):
    """CodeSourceControls wrapping instance methods."""
    return CodeSourceControls(
        instance=mock_api_client,
        methods=["get_prices", "get_details"],
        context=context,
        source_catalog=source_catalog,
    )


@pytest.fixture
def code_controls_async(context, source_catalog):
    """CodeSourceControls with async function."""
    return CodeSourceControls(
        functions={"Async Fetch": async_fetch_data},
        context=context,
        source_catalog=source_catalog,
    )


# ─────────────────────────────────────────────────────────────────────────────
# URLSourceControls fixtures
# ─────────────────────────────────────────────────────────────────────────────


class TestURLControls(URLSourceControls):
    """URLSourceControls subclass for testing."""

    url_template = "https://api.example.com/data?region={region}&year={year}"

    region = param.Selector(default="us", objects=["us", "eu", "apac"])
    year = param.Integer(default=2024, bounds=(2000, 2030))


@pytest.fixture
def url_controls_class():
    """Return the TestURLControls class."""
    return TestURLControls


@pytest.fixture
def url_controls(context, source_catalog):
    """Create a TestURLControls instance."""
    return TestURLControls(context=context, source_catalog=source_catalog)
