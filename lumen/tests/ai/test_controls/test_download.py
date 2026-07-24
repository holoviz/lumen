import pytest

from lumen.ai.controls import UploadedFileRow
from lumen.ai.controls.ingest.utils import (
    extract_filename_from_headers, extract_filename_from_url,
)


class TestDownloadControlsFilenameExtraction:
    """Tests for filename extraction from URLs and headers in DownloadControls."""

    def test_extract_filename_simple_url(self):
        """Test extracting filename from a simple URL with valid extension."""
        url = "https://example.com/data/population.csv"
        filename, guessed = extract_filename_from_url(url)
        assert filename == "population.csv"
        assert guessed is False

    def test_extract_filename_url_with_query_params(self):
        """Test extracting filename from URL with query parameters."""
        url = "https://example.com/data.csv?version=1&auth=abc"
        filename, guessed = extract_filename_from_url(url)
        assert filename == "data.csv"
        assert guessed is False

    def test_extract_filename_from_format_query_param(self):
        """Test extracting filename using format= query parameter when extension is invalid."""
        # This is the actual URL pattern from the bug report
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?stations=OAK&sts=2025-12-08&ets=2025-12-10&network=CA_ASOS&format=csv"
        filename, guessed = extract_filename_from_url(url)
        assert filename == "daily.csv"
        assert guessed is False

    def test_extract_filename_format_param_parquet(self):
        """Test format= query param with parquet format."""
        url = "https://api.example.com/export.php?format=parquet&table=users"
        filename, guessed = extract_filename_from_url(url)
        assert filename == "export.parquet"
        assert guessed is False

    def test_extract_filename_format_param_json(self):
        """Test format= query param with json format."""
        url = "https://api.example.com/data.aspx?id=123&format=json"
        filename, guessed = extract_filename_from_url(url)
        assert filename == "data.json"
        assert guessed is False

    def test_extract_filename_format_param_xlsx(self):
        """Test format= query param with xlsx format."""
        url = "https://api.example.com/report?format=xlsx"
        filename, guessed = extract_filename_from_url(url)
        assert filename == "report.xlsx"
        assert guessed is False

    def test_extract_filename_valid_extension_ignores_format_param(self):
        """Test that format= param is ignored when URL already has valid extension."""
        url = "https://example.com/data.csv?format=json"
        filename, guessed = extract_filename_from_url(url)
        # Should keep .csv since it's a valid extension, ignore format=json
        assert filename == "data.csv"
        assert guessed is False

    def test_extract_filename_invalid_format_param(self):
        """Test that invalid format= values are ignored, keeping original filename."""
        url = "https://api.example.com/data.php?format=invalid_format"
        filename, guessed = extract_filename_from_url(url)
        # Should keep original filename since .php is invalid and format is also invalid
        # The file will later be skipped during processing with a warning
        assert filename == "data.php"
        assert guessed is False

    def test_extract_filename_no_extension_no_format(self):
        """Test fallback when no extension and no format param."""
        url = "https://api.example.com/getData"
        filename, guessed = extract_filename_from_url(url)
        # Should use hash-based default without extension (extension comes from headers)
        assert filename.startswith("data_")
        assert "." not in filename
        assert guessed is True

    def test_extract_filename_from_headers_content_disposition(self):
        """Test extracting filename from Content-Disposition header."""
        headers = {
            "content-disposition": 'attachment; filename="exported_data.csv"'
        }
        filename = extract_filename_from_headers(headers, "fallback.json")
        assert filename == "exported_data.csv"

    def test_extract_filename_from_headers_content_type_csv(self):
        """Test extracting extension from Content-Type header for CSV."""
        headers = {
            "content-type": "text/csv; charset=utf-8"
        }
        filename = extract_filename_from_headers(headers, "data.php")
        assert filename == "data.csv"

    def test_extract_filename_from_headers_content_type_json(self):
        """Test extracting extension from Content-Type header for JSON."""
        headers = {
            "content-type": "application/json"
        }
        filename = extract_filename_from_headers(headers, "api_response.aspx")
        assert filename == "api_response.json"

    def test_extract_filename_from_headers_content_type_parquet(self):
        """Test extracting extension from Content-Type header for Parquet."""
        headers = {
            "content-type": "application/vnd.apache.parquet"
        }
        filename = extract_filename_from_headers(headers, "export.bin")
        assert filename == "export.parquet"

    def test_extract_filename_from_headers_content_type_xlsx(self):
        """Test extracting extension from Content-Type header for Excel."""
        headers = {
            "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        filename = extract_filename_from_headers(headers, "report.bin")
        assert filename == "report.xlsx"

    def test_extract_filename_from_headers_content_type_geojson(self):
        """Test extracting extension from Content-Type header for GeoJSON."""
        headers = {
            "content-type": "application/geo+json"
        }
        filename = extract_filename_from_headers(headers, "map.bin")
        assert filename == "map.geojson"

    def test_extract_filename_from_headers_valid_extension_unchanged(self):
        """Test that valid extensions are not changed by Content-Type."""
        headers = {
            "content-type": "application/json"  # Different from filename extension
        }
        filename = extract_filename_from_headers(headers, "data.csv")
        # Should keep .csv since it's already valid
        assert filename == "data.csv"

    def test_extract_filename_from_headers_content_disposition_priority(self):
        """Test that Content-Disposition takes priority over Content-Type."""
        headers = {
            "content-disposition": 'attachment; filename="correct.parquet"',
            "content-type": "text/csv"
        }
        filename = extract_filename_from_headers(headers, "default.json")
        # Content-Disposition should take priority
        assert filename == "correct.parquet"

    def test_extract_filename_from_headers_text_plain_to_csv(self):
        """Test that text/plain is converted to CSV (common for CSV downloads)."""
        headers = {
            "content-type": "text/plain"
        }
        filename = extract_filename_from_headers(headers, "data.php")
        assert filename == "data.csv"
