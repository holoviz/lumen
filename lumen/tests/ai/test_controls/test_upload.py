import asyncio
import io

from unittest.mock import patch

import pytest

from lumen.ai.controls import UploadedFileRow


@pytest.mark.asyncio
class TestDocumentVectorStoreIntegration:
    """Tests for document storage in vector store WITHOUT table associations in metadata."""

    async def test_add_document_without_tables_in_metadata(self, upload_controls, source_catalog):
        """Test that documents are added to vector store without 'tables' field in metadata."""
        # Create a metadata file
        readme_content = "# Population Data\n\nRetrieved from UN on 2024-07-11"
        readme_file = io.BytesIO(readme_content.encode())
        
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme_content):
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
class TestUploadControlsMetadataProcessing:
    """Tests for metadata file processing in UploadControls."""

    async def test_metadata_auto_detection(self, upload_controls):
        """Test that metadata files are auto-detected by extension and filename patterns."""
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

    async def test_duplicate_filename_handling(self, upload_controls, source_catalog):
        """Test that duplicate filenames are auto-renamed with counter suffix."""
        # Add first readme.md
        readme1_content = "# First README"
        with patch.object(upload_controls, '_extract_metadata_content', return_value=readme1_content):
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
class TestUploadControlsUnsupportedFiles:
    """Tests for handling unsupported file extensions in UploadControls."""

    async def test_unsupported_extension_shows_warning(self, upload_controls):
        """Test that unsupported file extensions show a warning message."""

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
