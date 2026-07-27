import asyncio
import io
import threading

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from lumen.ai.controls import SourceResult, UploadedFileRow
from lumen.ai.controls.ingest.utils import FileReadResult, read_geo_file
from lumen.sources.duckdb import DuckDBSource
from lumen.tests.utils import Polygon, gpd, requires_geopandas


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

class TestUploadControlsSourceDeduplication:
    """Tests for source de-duplication when uploading multiple files."""

    def test_multiple_data_files_add_single_source_reference(self, upload_controls):
        files = {
            "a.csv": b"x,y\n1,2\n",
            "b.csv": b"x,y\n3,4\n",
        }
        upload_controls._generate_file_cards(files)

        n_tables, n_docs, n_metadata = upload_controls._process_files()

        assert n_tables == 2
        assert n_docs == 0
        assert n_metadata == 0
        assert "sources" in upload_controls.outputs
        assert len(upload_controls.outputs["sources"]) == 1
        assert upload_controls.outputs["source"] is upload_controls.outputs["sources"][0]
        assert set(upload_controls.outputs["sources"][0].get_tables()) == {"a", "b"}
class TestUploadControlsSelectionUX:
    """Tests for staged file selection UX in UploadControls."""

    def test_file_selection_shows_guidance_message(self, upload_controls):
        upload_controls._on_file_upload(
            SimpleNamespace(new={"a.csv": b"x,y\n1,2\n", "b.csv": b"x,y\n3,4\n"})
        )
        assert upload_controls._message_placeholder.visible is True
        assert "2 file(s) selected." in upload_controls._message_placeholder.object
        assert "Upload file(s)" in upload_controls._message_placeholder.object

    def test_clear_selection_resets_staged_files(self, upload_controls):
        upload_controls._on_file_upload(
            SimpleNamespace(new={"a.csv": b"x,y\n1,2\n"})
        )
        assert len(upload_controls._file_cards) == 1
        assert upload_controls._upload_cards.visible is True

        upload_controls._on_clear_selection(None)

        assert len(upload_controls._file_cards) == 0
        assert upload_controls._upload_cards.visible is False
        assert upload_controls._file_input.value == {}
        assert upload_controls._message_placeholder.visible is True
        assert upload_controls._message_placeholder.object == "Selection cleared."


class TestUploadControlsOutputContract:
    """Regression tests for source output invariants."""

    def test_multi_table_upload_deduplicates_outputs_sources(self, upload_controls):
        upload_controls._generate_file_cards(
            {
                "first.csv": b"a,b\n1,2\n",
                "second.csv": b"a,b\n3,4\n",
            }
        )

        n_tables, n_docs, n_metadata = upload_controls._process_files()

        assert n_tables == 2
        assert n_docs == 0
        assert n_metadata == 0
        assert "sources" in upload_controls.outputs
        assert len(upload_controls.outputs["sources"]) == 1
        assert upload_controls.outputs["source"] in upload_controls.outputs["sources"]

    def test_handle_success_deduplicates_duplicate_source_entries(self, upload_controls):
        source = DuckDBSource(uri=":memory:", ephemeral=True, name="dedup_test", tables={})
        result = SourceResult(sources=[source, source], table="my_table")

        upload_controls._handle_success(result)

        assert "sources" in upload_controls.outputs
        assert len(upload_controls.outputs["sources"]) == 1
        assert upload_controls.outputs["source"] is upload_controls.outputs["sources"][0]
        assert upload_controls.outputs["table"] == "my_table"


class TestUploadControlsUX:
    """Tests for upload affordance and guidance text."""

    def test_upload_button_label_is_explicit(self, upload_controls):
        """Upload controls should use explicit upload action text."""
        assert upload_controls._add_button.name == "Upload file(s)"


@pytest.mark.asyncio
class TestProcessFilesAsync:
    """Tests for the non-blocking ``_process_files_async`` ingest path.

    The async path exists so slow pure-compute work (custom upload handlers,
    file parsing) runs off the event loop. It must stay behaviourally identical
    to the blocking ``_process_files``, and must not perform the compute on the
    event loop thread.
    """

    async def test_matches_sync_path_for_csv(self, upload_controls, download_controls):
        """Async and blocking paths produce the same counts, tables and source."""
        files = {"a.csv": b"x,y\n1,2\n", "b.csv": b"x,y\n3,4\n"}

        upload_controls._generate_file_cards(files)
        async_result = await upload_controls._process_files_async()

        download_controls._generate_file_cards(files)
        sync_result = download_controls._process_files()

        assert async_result == sync_result == (2, 0, 0)
        assert set(upload_controls.outputs["sources"][0].get_tables()) == {"a", "b"}
        assert len(upload_controls.outputs["sources"]) == 1

    async def test_custom_handler_runs_off_event_loop(self, upload_controls):
        """A registered handler must execute on a worker thread, not the loop.

        This is the whole point of the async path: a slow handler (e.g. reading
        an .h5ad) would otherwise freeze the session.
        """
        loop_thread = threading.get_ident()
        handler_threads = []

        def handler(context, file_obj, alias, filename):
            handler_threads.append(threading.get_ident())
            return DuckDBSource(
                uri=":memory:", ephemeral=True, name="custom", tables={}
            )

        upload_controls.upload_handlers = {"h5ad": handler}
        upload_controls._file_cards = [
            UploadedFileRow(
                file_obj=io.BytesIO(b"binary"),
                filename="cells",
                extension="h5ad",
                file_type="data",
            )
        ]

        await upload_controls._process_files_async()

        assert handler_threads and handler_threads[0] != loop_thread

    async def test_parsing_runs_off_event_loop(self, upload_controls):
        """Standard table parsing is also offloaded to a worker thread."""
        loop_thread = threading.get_ident()
        read_threads = []
        original = upload_controls._read_tables

        def spy(file, card):
            read_threads.append(threading.get_ident())
            return original(file, card)

        upload_controls._generate_file_cards({"a.csv": b"x,y\n1,2\n"})
        with patch.object(upload_controls, "_read_tables", side_effect=spy):
            await upload_controls._process_files_async()

        assert read_threads and read_threads[0] != loop_thread

    async def test_no_cards_returns_zero_counts(self, upload_controls):
        """An empty batch short-circuits without touching the error placeholder."""
        upload_controls._file_cards = []

        assert await upload_controls._process_files_async() == (0, 0, 0)
        assert upload_controls._error_placeholder.visible is False

    async def test_unsupported_extension_shows_warning(self, upload_controls):
        """Unsupported extensions are reported even though errors are deferred
        to the end of the batch on the main thread."""
        upload_controls._file_cards = [
            UploadedFileRow(
                file_obj=io.BytesIO(b"some content"),
                filename="script",
                extension="py",
                file_type="data",
            )
        ]

        n_tables, _, _ = await upload_controls._process_files_async()

        assert n_tables == 0
        assert upload_controls._error_placeholder.visible is True
        assert "script.py" in upload_controls._error_placeholder.object
        assert "unsupported format" in upload_controls._error_placeholder.object

    async def test_unreadable_file_does_not_abort_remaining_cards(self, upload_controls):
        """A card that fails to parse is reported but the batch continues.

        Errors accumulate rather than short-circuiting, so a single bad file
        cannot silently drop the files queued behind it.
        """
        upload_controls._generate_file_cards(
            {"bad.csv": b"x,y\n1,2\n", "good.csv": b"x,y\n3,4\n"}
        )
        bad_card = next(c for c in upload_controls._file_cards if c.filename == "bad")

        def selective_read(file, card):
            if card.filename == "bad":
                return None, "\n⚠️ Error processing 'bad.csv': boom"
            return FileReadResult(tables={card.alias: pd.DataFrame({"x": [1]})}), None

        with patch.object(upload_controls, "_read_tables", side_effect=selective_read):
            n_tables, _, _ = await upload_controls._process_files_async()

        assert bad_card is not None
        assert n_tables == 1
        assert "bad.csv" in upload_controls._error_placeholder.object
        assert upload_controls.outputs["sources"][0].get_tables() == ["good"]


class TestReadTablesPurity:
    """``_read_tables`` must be safe to call from a worker thread."""

    def test_returns_error_without_mutating_placeholder(self, upload_controls):
        """Parse failures are returned, not written to Panel models.

        Writing to ``_error_placeholder`` off-thread would mutate the Bokeh
        document from a worker thread; the string is surfaced by the caller.
        """
        card = UploadedFileRow(
            file_obj=io.BytesIO(b"not really a parquet"),
            filename="broken",
            extension="parquet",
            file_type="data",
        )

        result, error = upload_controls._read_tables(card.file_obj, card)

        assert result is None
        assert "broken.parquet" in error
        assert upload_controls._error_placeholder.object == ""
        assert upload_controls._error_placeholder.visible is False

    def test_returns_result_for_valid_csv(self, upload_controls):
        """A readable file yields a FileReadResult and no error."""
        card = UploadedFileRow(
            file_obj=io.BytesIO(b"x,y\n1,2\n"),
            filename="fine",
            extension="csv",
            file_type="data",
        )

        result, error = upload_controls._read_tables(card.file_obj, card)

        assert error is None
        assert list(result.tables) == ["fine"]


@requires_geopandas
def test_read_geo_file_captures_crs():
    """read_geo_file surfaces the source CRS in source_params so DuckDBSource
    can reapply it after the WKB roundtrip (gh-1904)."""
    gdf = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326"
    )
    buf = io.BytesIO(gdf.to_json().encode())
    result = read_geo_file(buf, "geojson", "geo")
    assert result.source_params["geometry_crs"] == "EPSG:4326"
