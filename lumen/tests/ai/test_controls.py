import asyncio
import io

from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls import DownloadControls, SourceCatalog, UploadControls
from lumen.ai.embeddings import NumpyEmbeddings
from lumen.ai.vector_store import NumpyVectorStore
from lumen.sources.duckdb import DuckDBSource


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


@pytest.mark.asyncio
class TestDocumentVectorStoreIntegration:
    """Tests for document storage in vector store WITHOUT table associations in metadata."""

    async def test_add_document_without_tables_in_metadata(self, upload_controls, source_catalog):
        """Test that documents are added to vector store without 'tables' field in metadata."""
        # Create a metadata file
        readme_content = "# Population Data\n\nRetrieved from UN on 2024-07-11"
        readme_file = io.BytesIO(readme_content.encode())
        
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme_content):
            from lumen.ai.controls import UploadedFileRow
            card = UploadedFileRow(
                file_obj=readme_file,
                filename="readme",
                extension="md",
                file_type="metadata"
            )
            
            # Process the metadata file
            result = upload_controls._add_metadata_file(card)
            assert result == 1
            
            # Wait for async upsert to complete
            await asyncio.sleep(0.1)
            
            # Verify it was added to vector store WITHOUT tables field
            docs = source_catalog.vector_store.filter_by({"filename": "readme.md"})
            assert len(docs) == 1
            assert docs[0]["metadata"]["type"] == "document"
            assert docs[0]["metadata"]["filename"] == "readme.md"
            # Key assertion: NO tables field in metadata
            assert "tables" not in docs[0]["metadata"]

    async def test_orphaned_documents_are_queryable(self, upload_controls, source_catalog):
        """Test that documents without any table associations remain queryable."""
        # Add document
        general_content = "# General Guidelines\n\nApplies to all data"
        general_file = io.BytesIO(general_content.encode())
        
        with patch.object(upload_controls, '_extract_metadata_content', return_value=general_content):
            from lumen.ai.controls import UploadedFileRow
            card = UploadedFileRow(
                file_obj=general_file,
                filename="general",
                extension="md",
                file_type="metadata"
            )
            upload_controls._add_metadata_file(card)
        
        # Wait for async upsert to complete
        await asyncio.sleep(0.1)
        
        # Verify it's in vector store
        docs = source_catalog.vector_store.filter_by({"filename": "general.md"})
        assert len(docs) == 1
        
        # Should still be queryable via semantic search
        results = await source_catalog.vector_store.query("General Guidelines")
        assert len(results) > 0
        assert any(r["metadata"]["filename"] == "general.md" for r in results)

    async def test_multiple_documents_stored_independently(self, upload_controls, source_catalog):
        """Test that multiple documents are stored as separate entries."""
        readme_content = "# Population README"
        schema_content = "# Schema Documentation"
        
        with patch.object(upload_controls, '_extract_metadata_content') as mock_extract:
            mock_extract.side_effect = [readme_content, schema_content]
            
            from lumen.ai.controls import UploadedFileRow
            readme_card = UploadedFileRow(
                file_obj=io.BytesIO(readme_content.encode()),
                filename="readme",
                extension="md",
                file_type="metadata"
            )
            schema_card = UploadedFileRow(
                file_obj=io.BytesIO(schema_content.encode()),
                filename="schema",
                extension="md",
                file_type="metadata"
            )
            
            upload_controls._add_metadata_file(readme_card)
            upload_controls._add_metadata_file(schema_card)
        
        # Wait for async upserts to complete
        await asyncio.sleep(0.2)
        
        # Both documents should exist independently
        all_docs = source_catalog.vector_store.filter_by({"type": "document"})
        assert len(all_docs) == 2
        
        filenames = {doc["metadata"]["filename"] for doc in all_docs}
        assert filenames == {"readme.md", "schema.md"}


@pytest.mark.asyncio
class TestSourceCatalogAssociationTracking:
    """Tests for tracking document-table associations in Source.metadata."""

    async def test_association_tracked_in_source_metadata(
        self, upload_controls, source_catalog, context
    ):
        """Test that associations are tracked in Source.metadata, not vector store."""
        # Add a table source
        source = DuckDBSource(
            uri=":memory:",
            name="population_db",
            tables={"population": "SELECT 1 as id, 'USA' as country"}
        )
        context["sources"] = [source]
        context["visible_slugs"] = {"population_db::population"}
        
        # Add metadata file
        readme_content = "# Population Data"
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme_content):
            from lumen.ai.controls import UploadedFileRow
            card = UploadedFileRow(
                file_obj=io.BytesIO(readme_content.encode()),
                filename="readme",
                extension="md",
                file_type="metadata"
            )
            upload_controls._add_metadata_file(card)
        
        # Wait for async upsert to complete
        await asyncio.sleep(0.1)
        
        # Simulate user associating document with table via SourceCatalog
        if source.metadata is None:
            source.metadata = {}
        source.metadata["population"] = {"docs": ["readme.md"]}
        
        # Verify association is in Source.metadata
        assert "docs" in source.metadata["population"]
        assert "readme.md" in source.metadata["population"]["docs"]
        
        # Verify vector store metadata is unchanged (no tables field)
        docs = source_catalog.vector_store.filter_by({"filename": "readme.md"})
        assert len(docs) == 1
        assert "tables" not in docs[0]["metadata"]

    async def test_multiple_associations_same_document(
        self, upload_controls, source_catalog, context
    ):
        """Test that one document can be associated with multiple tables."""
        # Create source with multiple tables
        source = DuckDBSource(
            uri=":memory:",
            name="sports_db",
            tables={
                "athletes": "SELECT 1 as id",
                "events": "SELECT 1 as id"
            }
        )
        context["sources"] = [source]
        
        # Add shared documentation
        shared_content = "# Sports Database Overview"
        with patch.object(upload_controls, '_extract_metadata_content', return_value=shared_content):
            from lumen.ai.controls import UploadedFileRow
            card = UploadedFileRow(
                file_obj=io.BytesIO(shared_content.encode()),
                filename="overview",
                extension="md",
                file_type="metadata"
            )
            upload_controls._add_metadata_file(card)
        
        # Wait for async upsert to complete
        await asyncio.sleep(0.1)
        
        # Associate with both tables
        source.metadata = {
            "athletes": {"docs": ["overview.md"]},
            "events": {"docs": ["overview.md"]}
        }
        
        # Verify associations
        assert "overview.md" in source.metadata["athletes"]["docs"]
        assert "overview.md" in source.metadata["events"]["docs"]
        
        # Vector store still has only one document
        docs = source_catalog.vector_store.filter_by({"filename": "overview.md"})
        assert len(docs) == 1

    async def test_remove_association_from_source_metadata(
        self, upload_controls, source_catalog, context
    ):
        """Test removing a table association from Source.metadata."""
        # Setup
        source = DuckDBSource(
            uri=":memory:",
            name="analytics_db",
            tables={"users": "SELECT 1"}
        )
        context["sources"] = [source]
        
        # Add document and associate
        doc_content = "# Analytics Documentation"
        with patch.object(upload_controls, '_extract_metadata_content', return_value=doc_content):
            from lumen.ai.controls import UploadedFileRow
            card = UploadedFileRow(
                file_obj=io.BytesIO(doc_content.encode()),
                filename="analytics",
                extension="md",
                file_type="metadata"
            )
            upload_controls._add_metadata_file(card)
        
        # Wait for async upsert to complete
        await asyncio.sleep(0.1)
        
        source.metadata = {"users": {"docs": ["analytics.md"]}}
        
        # Remove association
        source.metadata["users"]["docs"].remove("analytics.md")
        
        # Verify removed from Source.metadata
        assert "analytics.md" not in source.metadata["users"]["docs"]
        
        # Document still exists in vector store (orphaned but queryable)
        docs = source_catalog.vector_store.filter_by({"filename": "analytics.md"})
        assert len(docs) == 1


@pytest.mark.asyncio
class TestDocumentQueryFiltering:
    """Tests for filtering documents by associations during queries."""

    async def test_filter_documents_by_association(self, upload_controls, source_catalog, context):
        """Test that we can filter query results based on Source.metadata associations."""
        # Setup: Multiple documents and tables
        source = DuckDBSource(
            uri=":memory:",
            name="test_db",
            tables={
                "table_a": "SELECT 1",
                "table_b": "SELECT 1"
            }
        )
        context["sources"] = [source]
        
        # Add documents
        doc1_content = "# Documentation for Table A"
        doc2_content = "# Documentation for Table B"
        
        with patch.object(upload_controls, '_extract_metadata_content') as mock_extract:
            mock_extract.side_effect = [doc1_content, doc2_content]
            
            from lumen.ai.controls import UploadedFileRow
            for content, name in [(doc1_content, "doc_a"), (doc2_content, "doc_b")]:
                card = UploadedFileRow(
                    file_obj=io.BytesIO(content.encode()),
                    filename=name,
                    extension="md",
                    file_type="metadata"
                )
                upload_controls._add_metadata_file(card)
        
        # Wait for async upserts to complete
        await asyncio.sleep(0.2)
        
        # Associate documents with specific tables
        source.metadata = {
            "table_a": {"docs": ["doc_a.md"]},
            "table_b": {"docs": ["doc_b.md"]}
        }
        
        # Query all documents
        all_docs = await source_catalog.vector_store.query("Documentation", top_k=10, filters={"type": "document"})
        assert len(all_docs) == 2
        
        # Filter by association to table_a
        table_a_docs = [
            doc for doc in all_docs
            if doc["metadata"]["filename"] in source.metadata["table_a"]["docs"]
        ]
        assert len(table_a_docs) == 1
        assert table_a_docs[0]["metadata"]["filename"] == "doc_a.md"
        
        # Filter by association to table_b
        table_b_docs = [
            doc for doc in all_docs
            if doc["metadata"]["filename"] in source.metadata["table_b"]["docs"]
        ]
        assert len(table_b_docs) == 1
        assert table_b_docs[0]["metadata"]["filename"] == "doc_b.md"


@pytest.mark.asyncio
class TestUploadControlsMetadataProcessing:
    """Tests for metadata file processing in UploadControls."""

    async def test_metadata_auto_detection(self, upload_controls):
        """Test that metadata files are auto-detected by extension and filename patterns."""
        from lumen.ai.controls import UploadedFileRow

        # Test extension-based detection
        md_file = UploadedFileRow(
            file_obj=io.BytesIO(b"content"),
            filename="document",
            extension="md"
        )
        assert md_file.file_type == "metadata"
        
        txt_file = UploadedFileRow(
            file_obj=io.BytesIO(b"content"),
            filename="notes",
            extension="txt"
        )
        assert txt_file.file_type == "metadata"
        
        # Test filename pattern detection
        readme_file = UploadedFileRow(
            file_obj=io.BytesIO(b"content"),
            filename="README",
            extension="txt"
        )
        assert readme_file.file_type == "metadata"
        
        schema_file = UploadedFileRow(
            file_obj=io.BytesIO(b"content"),
            filename="schema_info",
            extension="json"
        )
        assert schema_file.file_type == "metadata"
        
        # Data file should not be auto-detected as metadata
        csv_file = UploadedFileRow(
            file_obj=io.BytesIO(b"content"),
            filename="data",
            extension="csv"
        )
        assert csv_file.file_type == "data"

    async def test_process_mixed_files(self, upload_controls, source_catalog, context):
        """Test processing a mix of data and metadata files."""
        # Setup files
        csv_content = b"country,population\nUSA,331000000"
        readme_content = "# Population Data"
        
        files = {
            "population.csv": csv_content,
            "readme.md": readme_content.encode()
        }
        
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme_content):
            upload_controls._generate_file_cards(files)
            
            # Verify file classification
            assert len(upload_controls._file_cards) == 2
            
            csv_card = next(c for c in upload_controls._file_cards if c.extension == "csv")
            md_card = next(c for c in upload_controls._file_cards if c.extension == "md")
            
            assert csv_card.file_type == "data"
            assert md_card.file_type == "metadata"
            
            # Process files
            n_tables, n_docs, n_metadata = upload_controls._process_files()
            
            # Wait for async upserts to complete
            await asyncio.sleep(0.1)
            
            # Sync outputs to context
            if "sources" in upload_controls.outputs:
                context["sources"].extend(upload_controls.outputs["sources"])
            
            # Should have 1 table and 1 metadata file
            assert n_tables == 1
            assert n_metadata == 1
            
            # Verify source was created
            assert len(context["sources"]) == 1
            source = context["sources"][0]
            assert "population" in source.get_tables()
            
            # Verify metadata was stored
            assert len(source_catalog._available_metadata) == 1
            assert source_catalog._available_metadata[0]["filename"] == "readme.md"
            
            # Verify document is in vector store
            docs = source_catalog.vector_store.filter_by({"filename": "readme.md"})
            assert len(docs) == 1
            assert docs[0]["metadata"]["type"] == "document"


@pytest.mark.asyncio
class TestSourceCatalogDocumentToggling:
    """Tests for document toggling functionality in SourceCatalog."""

    async def test_global_toggle_associates_with_all_tables(self, source_catalog, context):
        """Test that checking global doc associates it with ALL tables."""
        # Setup: Add a document to available metadata
        source_catalog._available_metadata = [
            {"filename": "readme.md", "display_name": "readme", "content": "# Test"}
        ]
        
        # Setup: Create sources with tables  
        source1 = DuckDBSource(
            uri=":memory:",
            name="db1",
            tables={"table_a": "SELECT 1", "table_b": "SELECT 1"}
        )
        source2 = DuckDBSource(
            uri=":memory:",
            name="db2",
            tables={"table_c": "SELECT 1"}
        )
        context["sources"] = [source1, source2]
        
        # Sync to build trees
        source_catalog.sync()
        
        # Simulate checking in global tree
        mock_event = Mock()
        mock_event.new = [(0,)]  # First item checked
        
        with patch.object(source_catalog, '_sync_sources_tree_only'):
            source_catalog._on_docs_active_change(mock_event)
        
        # Verify doc was associated with ALL tables
        assert source1.metadata["table_a"]["docs"] == ["readme.md"]
        assert source1.metadata["table_b"]["docs"] == ["readme.md"]
        assert source2.metadata["table_c"]["docs"] == ["readme.md"]

    async def test_global_toggle_removes_from_all_tables(self, source_catalog, context):
        """Test that unchecking global doc removes it from ALL tables."""
        # Setup: Create sources with pre-associated docs
        source = DuckDBSource(
            uri=":memory:",
            name="db1",
            tables={"table_a": "SELECT 1", "table_b": "SELECT 1"}
        )
        source.metadata = {
            "table_a": {"docs": ["readme.md"]},
            "table_b": {"docs": ["readme.md"]}
        }
        context["sources"] = [source]
        
        source_catalog._available_metadata = [
            {"filename": "readme.md", "display_name": "readme", "content": "# Test"}
        ]
        
        # Sync to build trees
        source_catalog.sync()
        
        # Simulate unchecking in global tree
        mock_event = Mock()
        mock_event.new = []  # No active items
        
        with patch.object(source_catalog, '_sync_sources_tree_only'):
            source_catalog._on_docs_active_change(mock_event)
        
        # Verify doc was removed from ALL tables
        assert "readme.md" not in source.metadata["table_a"]["docs"]
        assert "readme.md" not in source.metadata["table_b"]["docs"]

    async def test_individual_table_toggle(self, source_catalog, context):
        """Test that toggling under individual table only affects that table."""
        # Setup
        source = DuckDBSource(
            uri=":memory:",
            name="db1",
            tables={"table_a": "SELECT 1", "table_b": "SELECT 1"}
        )
        source.metadata = {
            "table_a": {"docs": ["readme.md"]},
            "table_b": {"docs": ["readme.md"]}
        }
        context["sources"] = [source]
        
        source_catalog._available_metadata = [
            {"filename": "readme.md", "display_name": "readme", "content": "# Test"}
        ]
        
        # Sync to build trees
        source_catalog.sync()
        
        # Simulate unchecking doc under table_a only
        # Path: (0, 0, 0) = source 0, table 0, metadata 0
        # Active paths should NOT include (0, 0, 0)
        mock_event = Mock()
        mock_event.new = [(0,), (0, 0), (0, 1), (0, 1, 0)]  # table_a and its doc unchecked, but table_b checked
        
        source_catalog._on_sources_active_change(mock_event)
        
        # Verify: removed from table_a, but still in table_b
        assert "readme.md" not in source.metadata["table_a"]["docs"]
        assert "readme.md" in source.metadata["table_b"]["docs"]

    async def test_mixed_global_and_table_associations(self, source_catalog, context):
        """Test complex scenario with global doc and table-specific doc."""
        # Setup
        source = DuckDBSource(
            uri=":memory:",
            name="db1",
            tables={"table_a": "SELECT 1", "table_b": "SELECT 1"}
        )
        context["sources"] = [source]
        
        source_catalog._available_metadata = [
            {"filename": "global.md", "display_name": "global", "content": "# Global"},
            {"filename": "specific.md", "display_name": "specific", "content": "# Specific"}
        ]
        
        # global.md associated with all tables, specific.md only with table_a
        source.metadata = {
            "table_a": {"docs": ["global.md", "specific.md"]},
            "table_b": {"docs": ["global.md"]}
        }
        
        # Sync to build trees
        source_catalog.sync()
        
        # Verify global.md shows as checked in global tree (associated with ALL tables)
        active_paths = source_catalog._compute_docs_active_paths()
        global_idx = 0  # global.md is first
        assert (global_idx,) in active_paths
        
        # specific.md should NOT show as checked (only associated with table_a)
        specific_idx = 1
        assert (specific_idx,) not in active_paths

    async def test_doc_associations_persist_across_syncs(self, source_catalog, context):
        """Test that doc associations are preserved during sync operations."""
        # Setup
        source = DuckDBSource(
            uri=":memory:",
            name="db1",
            tables={"table_a": "SELECT 1"}
        )
        source.metadata = {"table_a": {"docs": ["readme.md"]}}
        context["sources"] = [source]
        
        source_catalog._available_metadata = [
            {"filename": "readme.md", "display_name": "readme", "content": "# Test"}
        ]
        
        # First sync
        source_catalog.sync()
        
        # Associations should be preserved
        assert source.metadata["table_a"]["docs"] == ["readme.md"]
        
        # Add another source
        source2 = DuckDBSource(
            uri=":memory:",
            name="db2",
            tables={"table_b": "SELECT 1"}
        )
        context["sources"].append(source2)
        
        # Second sync
        source_catalog.sync()
        
        # Original associations should still be preserved
        assert source.metadata["table_a"]["docs"] == ["readme.md"]

    async def test_duplicate_filename_handling(self, upload_controls, source_catalog):
        """Test that duplicate filenames are auto-renamed with counter suffix."""
        # Add first readme.md
        readme1_content = "# First README"
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme1_content):
            from lumen.ai.controls import UploadedFileRow
            card1 = UploadedFileRow(
                file_obj=io.BytesIO(readme1_content.encode()),
                filename="readme",
                extension="md",
                file_type="metadata"
            )
            result1 = upload_controls._add_metadata_file(card1)
            assert result1 == 1
        
        # Verify first file was added
        assert len(source_catalog._available_metadata) == 1
        assert source_catalog._available_metadata[0]["filename"] == "readme.md"
        
        # Add second readme.md (duplicate)
        readme2_content = "# Second README"
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme2_content):
            card2 = UploadedFileRow(
                file_obj=io.BytesIO(readme2_content.encode()),
                filename="readme",
                extension="md",
                file_type="metadata"
            )
            result2 = upload_controls._add_metadata_file(card2)
            assert result2 == 1
        
        # Verify second file was renamed
        assert len(source_catalog._available_metadata) == 2
        assert source_catalog._available_metadata[0]["filename"] == "readme.md"
        assert source_catalog._available_metadata[1]["filename"] == "readme_1.md"
        
        # Add third readme.md (another duplicate)
        readme3_content = "# Third README"
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme3_content):
            card3 = UploadedFileRow(
                file_obj=io.BytesIO(readme3_content.encode()),
                filename="readme",
                extension="md",
                file_type="metadata"
            )
            result3 = upload_controls._add_metadata_file(card3)
            assert result3 == 1
        
        # Verify third file was renamed with counter 2
        assert len(source_catalog._available_metadata) == 3
        assert source_catalog._available_metadata[2]["filename"] == "readme_2.md"


@pytest.mark.asyncio
class TestDocumentListAgentIntegration:
    """Tests for DocumentListAgent with metaset."""

    async def test_document_list_agent_with_metaset(self):
        """Test that DocumentListAgent works with metaset.docs."""
        from lumen.ai.agents.document_list import DocumentListAgent
        from lumen.ai.schemas import DocumentChunk, Metaset

        # Create metaset with document chunks
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[
                DocumentChunk(filename="readme.md", text="chunk 1", similarity=0.9),
                DocumentChunk(filename="readme.md", text="chunk 2", similarity=0.8),
                DocumentChunk(filename="schema.md", text="chunk 3", similarity=0.7),
            ]
        )
        
        context = {"metaset": metaset}
        
        # Test applies
        applies = await DocumentListAgent.applies(context)
        assert applies is True  # More than 1 unique document
        
        # Test _get_items
        agent = DocumentListAgent()
        items = agent._get_items(context)
        
        # Should return unique, sorted filenames
        assert items == {"Documents": ["readme.md", "schema.md"]}

    async def test_document_list_agent_no_docs(self):
        """Test that DocumentListAgent doesn't apply when no docs."""
        from lumen.ai.agents.document_list import DocumentListAgent
        from lumen.ai.schemas import Metaset

        # Metaset without docs
        metaset = Metaset(query="test", catalog={}, docs=None)
        context = {"metaset": metaset}
        
        applies = await DocumentListAgent.applies(context)
        assert applies is False

    async def test_document_list_agent_single_doc(self):
        """Test that DocumentListAgent doesn't apply for single doc."""
        from lumen.ai.agents.document_list import DocumentListAgent
        from lumen.ai.schemas import DocumentChunk, Metaset

        # Metaset with only one unique document
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[DocumentChunk(filename="readme.md", text="chunk", similarity=0.9)]
        )
        context = {"metaset": metaset}
        
        applies = await DocumentListAgent.applies(context)
        assert applies is False  # Only 1 document, not worth listing


class TestDownloadControlsFilenameExtraction:
    """Tests for filename extraction from URLs and headers in DownloadControls."""

    def test_extract_filename_simple_url(self, download_controls):
        """Test extracting filename from a simple URL with valid extension."""
        url = "https://example.com/data/population.csv"
        filename = download_controls._extract_filename_from_url(url)
        assert filename == "population.csv"

    def test_extract_filename_url_with_query_params(self, download_controls):
        """Test extracting filename from URL with query parameters."""
        url = "https://example.com/data.csv?version=1&auth=abc"
        filename = download_controls._extract_filename_from_url(url)
        assert filename == "data.csv"

    def test_extract_filename_from_format_query_param(self, download_controls):
        """Test extracting filename using format= query parameter when extension is invalid."""
        # This is the actual URL pattern from the bug report
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?stations=OAK&sts=2025-12-08&ets=2025-12-10&network=CA_ASOS&format=csv"
        filename = download_controls._extract_filename_from_url(url)
        assert filename == "daily.csv"

    def test_extract_filename_format_param_parquet(self, download_controls):
        """Test format= query param with parquet format."""
        url = "https://api.example.com/export.php?format=parquet&table=users"
        filename = download_controls._extract_filename_from_url(url)
        assert filename == "export.parquet"

    def test_extract_filename_format_param_json(self, download_controls):
        """Test format= query param with json format."""
        url = "https://api.example.com/data.aspx?id=123&format=json"
        filename = download_controls._extract_filename_from_url(url)
        assert filename == "data.json"

    def test_extract_filename_format_param_xlsx(self, download_controls):
        """Test format= query param with xlsx format."""
        url = "https://api.example.com/report?format=xlsx"
        filename = download_controls._extract_filename_from_url(url)
        assert filename == "report.xlsx"

    def test_extract_filename_valid_extension_ignores_format_param(self, download_controls):
        """Test that format= param is ignored when URL already has valid extension."""
        url = "https://example.com/data.csv?format=json"
        filename = download_controls._extract_filename_from_url(url)
        # Should keep .csv since it's a valid extension, ignore format=json
        assert filename == "data.csv"

    def test_extract_filename_invalid_format_param(self, download_controls):
        """Test that invalid format= values are ignored, keeping original filename."""
        url = "https://api.example.com/data.php?format=invalid_format"
        filename = download_controls._extract_filename_from_url(url)
        # Should keep original filename since .php is invalid and format is also invalid
        # The file will later be skipped during processing with a warning
        assert filename == "data.php"

    def test_extract_filename_no_extension_no_format(self, download_controls):
        """Test fallback when no extension and no format param."""
        url = "https://api.example.com/getData"
        filename = download_controls._extract_filename_from_url(url)
        # Should use hash-based default with .json extension
        assert filename.startswith("data_")
        assert filename.endswith(".json")

    def test_extract_filename_from_headers_content_disposition(self, download_controls):
        """Test extracting filename from Content-Disposition header."""
        headers = {
            "content-disposition": 'attachment; filename="exported_data.csv"'
        }
        filename = download_controls._extract_filename_from_headers(headers, "fallback.json")
        assert filename == "exported_data.csv"

    def test_extract_filename_from_headers_content_type_csv(self, download_controls):
        """Test extracting extension from Content-Type header for CSV."""
        headers = {
            "content-type": "text/csv; charset=utf-8"
        }
        filename = download_controls._extract_filename_from_headers(headers, "data.php")
        assert filename == "data.csv"

    def test_extract_filename_from_headers_content_type_json(self, download_controls):
        """Test extracting extension from Content-Type header for JSON."""
        headers = {
            "content-type": "application/json"
        }
        filename = download_controls._extract_filename_from_headers(headers, "api_response.aspx")
        assert filename == "api_response.json"

    def test_extract_filename_from_headers_content_type_parquet(self, download_controls):
        """Test extracting extension from Content-Type header for Parquet."""
        headers = {
            "content-type": "application/vnd.apache.parquet"
        }
        filename = download_controls._extract_filename_from_headers(headers, "export.bin")
        assert filename == "export.parquet"

    def test_extract_filename_from_headers_content_type_xlsx(self, download_controls):
        """Test extracting extension from Content-Type header for Excel."""
        headers = {
            "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        filename = download_controls._extract_filename_from_headers(headers, "report.bin")
        assert filename == "report.xlsx"

    def test_extract_filename_from_headers_content_type_geojson(self, download_controls):
        """Test extracting extension from Content-Type header for GeoJSON."""
        headers = {
            "content-type": "application/geo+json"
        }
        filename = download_controls._extract_filename_from_headers(headers, "map.bin")
        assert filename == "map.geojson"

    def test_extract_filename_from_headers_valid_extension_unchanged(self, download_controls):
        """Test that valid extensions are not changed by Content-Type."""
        headers = {
            "content-type": "application/json"  # Different from filename extension
        }
        filename = download_controls._extract_filename_from_headers(headers, "data.csv")
        # Should keep .csv since it's already valid
        assert filename == "data.csv"

    def test_extract_filename_from_headers_content_disposition_priority(self, download_controls):
        """Test that Content-Disposition takes priority over Content-Type."""
        headers = {
            "content-disposition": 'attachment; filename="correct.parquet"',
            "content-type": "text/csv"
        }
        filename = download_controls._extract_filename_from_headers(headers, "default.json")
        # Content-Disposition should take priority
        assert filename == "correct.parquet"

    def test_extract_filename_from_headers_text_plain_to_csv(self, download_controls):
        """Test that text/plain is converted to CSV (common for CSV downloads)."""
        headers = {
            "content-type": "text/plain"
        }
        filename = download_controls._extract_filename_from_headers(headers, "data.php")
        assert filename == "data.csv"


class TestDownloadControlsUnsupportedFiles:
    """Tests for handling unsupported file extensions in DownloadControls."""

    def test_unsupported_extension_shows_warning(self, download_controls):
        """Test that unsupported file extensions show a warning message."""
        from lumen.ai.controls import UploadedFileRow

        # Create a file card with unsupported extension
        card = UploadedFileRow(
            file_obj=io.BytesIO(b"some content"),
            filename="script",
            extension="py",
            file_type="data"
        )
        download_controls._file_cards = [card]

        # Process files
        n_tables, n_docs, n_metadata = download_controls._process_files()

        # Should have processed 0 tables
        assert n_tables == 0
        assert n_metadata == 0

        # Warning should be visible
        assert download_controls._error_placeholder.visible is True
        assert "script.py" in download_controls._error_placeholder.object
        assert "unsupported format" in download_controls._error_placeholder.object

    def test_mixed_supported_unsupported_files(self, download_controls):
        """Test processing mix of supported and unsupported files."""
        from lumen.ai.controls import UploadedFileRow

        # Create cards with mixed extensions
        csv_card = UploadedFileRow(
            file_obj=io.BytesIO(b"col1,col2\nval1,val2"),
            filename="data",
            extension="csv",
            file_type="data"
        )
        py_card = UploadedFileRow(
            file_obj=io.BytesIO(b"print('hello')"),
            filename="script",
            extension="py",
            file_type="data"
        )
        download_controls._file_cards = [csv_card, py_card]

        # Process files
        n_tables, n_docs, n_metadata = download_controls._process_files()

        # Should have processed 1 table (csv), skipped 1 (py)
        assert n_tables == 1
        assert n_metadata == 0

        # Warning should be visible for the skipped file
        assert download_controls._error_placeholder.visible is True
        assert "script.py" in download_controls._error_placeholder.object

    def test_error_placeholder_cleared_on_new_process(self, download_controls):
        """Test that error placeholder is cleared at start of processing."""
        # Set some previous error
        download_controls._error_placeholder.object = "Previous error"
        download_controls._error_placeholder.visible = True

        # Process with no files
        download_controls._file_cards = []
        download_controls._process_files()

        # Error should be cleared
        assert download_controls._error_placeholder.object == ""
        assert download_controls._error_placeholder.visible is False

    def test_multiple_unsupported_files_all_shown(self, download_controls):
        """Test that warnings for multiple unsupported files are all shown."""
        from lumen.ai.controls import UploadedFileRow

        # Create multiple unsupported file cards
        cards = [
            UploadedFileRow(
                file_obj=io.BytesIO(b"content"),
                filename="script",
                extension="py",
                file_type="data"
            ),
            UploadedFileRow(
                file_obj=io.BytesIO(b"content"),
                filename="binary",
                extension="exe",
                file_type="data"
            ),
        ]
        download_controls._file_cards = cards

        # Process files
        download_controls._process_files()

        # Both files should be mentioned in warning
        error_text = download_controls._error_placeholder.object
        assert "script.py" in error_text
        assert "binary.exe" in error_text


@pytest.mark.asyncio
class TestUploadControlsUnsupportedFiles:
    """Tests for handling unsupported file extensions in UploadControls."""

    async def test_unsupported_extension_shows_warning(self, upload_controls):
        """Test that unsupported file extensions show a warning message."""
        from lumen.ai.controls import UploadedFileRow

        # Create a file card with unsupported extension
        card = UploadedFileRow(
            file_obj=io.BytesIO(b"some content"),
            filename="script",
            extension="py",
            file_type="data"
        )
        upload_controls._file_cards = [card]

        # Process files
        n_tables, n_docs, n_metadata = upload_controls._process_files()

        # Should have processed 0 tables
        assert n_tables == 0

        # Warning should be visible
        assert upload_controls._error_placeholder.visible is True
        assert "script.py" in upload_controls._error_placeholder.object
        assert "unsupported format" in upload_controls._error_placeholder.object
