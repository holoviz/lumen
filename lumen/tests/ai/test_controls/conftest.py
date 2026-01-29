import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls import DownloadControls, SourceCatalog, UploadControls
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
    """Create UploadControls with reference to catalog."""
    controls = UploadControls(context=context, source_catalog=source_catalog)
    return controls


@pytest.fixture
def download_controls(context, source_catalog):
    """Create DownloadControls with reference to catalog."""
    controls = DownloadControls(context=context, source_catalog=source_catalog)
    return controls
