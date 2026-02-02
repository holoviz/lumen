import asyncio
import io

from unittest.mock import Mock, patch

import pytest

from lumen.ai.controls import UploadedFileRow
from lumen.sources.duckdb import DuckDBSource


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
