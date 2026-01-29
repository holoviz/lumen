import io

import pytest

from lumen.ai.controls import UploadedFileRow


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
