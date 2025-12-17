import asyncio
import io

from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls import SourceCatalog, UploadControls
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
        "disabled_docs": set(),  # Initialize disabled_docs
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

    async def test_global_toggle_updates_disabled_docs(self, source_catalog, context):
        """Test that toggling in global tree updates disabled_docs."""
        # Setup: Add a document to available metadata
        source_catalog._available_metadata = [
            {"filename": "readme.md", "display_name": "readme", "content": "# Test"}
        ]
        
        # Initialize disabled_docs
        context["disabled_docs"] = set()
        
        # Sync to build trees
        source_catalog.sync()
        
        # Simulate unchecking in global tree (empty active list)
        mock_event = Mock()
        mock_event.new = []  # No active items = all unchecked
        
        source_catalog._on_docs_active_change(mock_event)
        
        # Verify doc was added to disabled_docs
        assert "readme.md" in context["disabled_docs"]
        
        # Simulate checking again
        mock_event.new = [(0,)]  # First item checked
        source_catalog._on_docs_active_change(mock_event)
        
        # Verify doc was removed from disabled_docs
        assert "readme.md" not in context["disabled_docs"]

    async def test_global_toggle_associates_with_all_tables(self, source_catalog, context):
        """Test that checking global doc associates it with ALL tables."""
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
        context["disabled_docs"] = set()
        
        source_catalog._available_metadata = [
            {"filename": "readme.md", "display_name": "readme", "content": "# Test"}
        ]
        
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
        context["disabled_docs"] = set()
        
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
        # And added to disabled_docs
        assert "readme.md" in context["disabled_docs"]

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
        context["disabled_docs"] = set()
        
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
        # disabled_docs should NOT be affected (still global)
        assert "readme.md" not in context["disabled_docs"]

    async def test_mixed_global_and_table_associations(self, source_catalog, context):
        """Test complex scenario with both global and table-specific associations."""
        # Setup: Two docs - one global, one table-specific
        source = DuckDBSource(
            uri=":memory:",
            name="db1",
            tables={"table_a": "SELECT 1", "table_b": "SELECT 1"}
        )
        context["sources"] = [source]
        context["disabled_docs"] = {"disabled.md"}  # One doc disabled
        
        source_catalog._available_metadata = [
            {"filename": "global.md", "display_name": "global", "content": "# Global"},
            {"filename": "disabled.md", "display_name": "disabled", "content": "# Disabled"}
        ]
        
        # Associate global.md with all tables (simulating checked in global tree)
        source.metadata = {
            "table_a": {"docs": ["global.md"]},
            "table_b": {"docs": ["global.md"]}
        }
        
        # Verify initial state
        assert "global.md" not in context["disabled_docs"]
        assert "disabled.md" in context["disabled_docs"]
        assert source.metadata["table_a"]["docs"] == ["global.md"]
        assert source.metadata["table_b"]["docs"] == ["global.md"]

    async def test_disabled_docs_persists_across_syncs(self, source_catalog, context):
        """Test that disabled_docs state is preserved during sync operations."""
        # Setup
        context["disabled_docs"] = {"readme.md"}
        source_catalog._available_metadata = [
            {"filename": "readme.md", "display_name": "readme", "content": "# Test"}
        ]
        
        # First sync
        source_catalog.sync()
        
        # disabled_docs should be preserved
        assert "readme.md" in context["disabled_docs"]
        
        # Add a source
        source = DuckDBSource(
            uri=":memory:",
            name="db1",
            tables={"table_a": "SELECT 1"}
        )
        context["sources"] = [source]
        
        # Second sync
        source_catalog.sync()
        
        # disabled_docs should still be preserved
        assert "readme.md" in context["disabled_docs"]
