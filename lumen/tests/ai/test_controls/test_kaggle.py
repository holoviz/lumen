"""Tests for Kaggle dataset integration in DownloadSourceControls."""
import os
import tempfile

from unittest.mock import MagicMock, patch

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls.ingest.download import DownloadSourceControls

# ─────────────────────────────────────────────────────────────────────────────
# _parse_kaggle_ref
# ─────────────────────────────────────────────────────────────────────────────


class TestParseKaggleRef:
    """Tests for Kaggle URL parsing."""

    @pytest.mark.parametrize("url,expected", [
        ("https://www.kaggle.com/datasets/owner/dataset", "owner/dataset"),
        ("https://kaggle.com/datasets/owner/dataset", "owner/dataset"),
        ("http://www.kaggle.com/datasets/owner/dataset", "owner/dataset"),
        ("https://www.kaggle.com/datasets/owner/dataset/data", "owner/dataset"),
        ("https://www.kaggle.com/datasets/owner/dataset?select=file.csv", "owner/dataset"),
        ("https://www.kaggle.com/datasets/owner/dataset#overview", "owner/dataset"),
    ])
    def test_valid_kaggle_urls(self, url, expected):
        assert DownloadSourceControls._parse_kaggle_ref(url) == expected

    @pytest.mark.parametrize("url", [
        "https://www.kaggle.com/competitions/some-comp",
        "https://www.kaggle.com/owner/dataset",  # missing /datasets/
        "https://example.com/datasets/owner/dataset",
        "https://www.kaggle.com/datasets/",  # no owner/dataset
        "owner/dataset",  # bare slug, no kaggle.com
        "",
    ])
    def test_non_kaggle_urls(self, url):
        assert DownloadSourceControls._parse_kaggle_ref(url) is None

    def test_whitespace_is_stripped(self):
        assert DownloadSourceControls._parse_kaggle_ref(
            "  https://www.kaggle.com/datasets/owner/dataset  "
        ) == "owner/dataset"

    def test_trailing_slash_is_stripped(self):
        assert DownloadSourceControls._parse_kaggle_ref(
            "https://www.kaggle.com/datasets/owner/dataset/"
        ) == "owner/dataset"


# ─────────────────────────────────────────────────────────────────────────────
# _kagglehub init
# ─────────────────────────────────────────────────────────────────────────────


class TestKagglehubInit:
    """Tests for kagglehub availability detection at init time."""

    def test_kagglehub_attr_exists(self, download_controls):
        """self._kagglehub is set (module or None depending on env)."""
        assert hasattr(download_controls, "_kagglehub")

    def test_kagglehub_none_when_missing(self, context, source_catalog):
        """If kagglehub is not importable, self._kagglehub is None."""
        import builtins
        real_import = builtins.__import__

        def _block_kagglehub(name, *args, **kwargs):
            if name == "kagglehub":
                raise ModuleNotFoundError("mocked")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_block_kagglehub):
            controls = DownloadSourceControls(
                context=context, source_catalog=source_catalog,
            )
        assert controls._kagglehub is None


# ─────────────────────────────────────────────────────────────────────────────
# _download_kaggle_files
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDownloadKaggleFiles:
    """Tests for the Kaggle file download method."""

    async def test_returns_error_when_kagglehub_missing(self, download_controls):
        download_controls._kagglehub = None
        files, error = await download_controls._download_kaggle_files("owner/ds")
        assert files == {}
        assert "pip install kagglehub" in error

    async def test_reads_supported_files(self, download_controls):
        """Mocks kagglehub.dataset_download to return a temp dir with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test files
            csv_path = os.path.join(tmpdir, "data.csv")
            json_path = os.path.join(tmpdir, "extra.json")
            txt_path = os.path.join(tmpdir, "readme.txt")  # not a TABLE_EXTENSIONS
            with open(csv_path, "w") as f:
                f.write("a,b\n1,2\n")
            with open(json_path, "w") as f:
                f.write('[{"x": 1}]')
            with open(txt_path, "w") as f:
                f.write("not a table")

            mock_kagglehub = MagicMock()
            mock_kagglehub.dataset_download.return_value = tmpdir
            download_controls._kagglehub = mock_kagglehub

            files, error = await download_controls._download_kaggle_files("owner/ds")

            assert error is None
            assert "data.csv" in files
            assert "extra.json" in files
            # txt is in METADATA_EXTENSIONS, not TABLE_EXTENSIONS
            assert "readme.txt" not in files
            mock_kagglehub.dataset_download.assert_called_once_with("owner/ds")

    async def test_reads_files_in_subdirectories(self, download_controls):
        """Files nested in subdirs should be found via rglob."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "nested.csv"), "w") as f:
                f.write("x,y\n1,2\n")

            mock_kagglehub = MagicMock()
            mock_kagglehub.dataset_download.return_value = tmpdir
            download_controls._kagglehub = mock_kagglehub

            files, error = await download_controls._download_kaggle_files("owner/ds")
            assert error is None
            assert "nested.csv" in files

    async def test_returns_error_when_no_supported_files(self, download_controls):
        """Empty or unsupported-only dataset should return an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.pkl"), "wb") as f:
                f.write(b"pickle data")

            mock_kagglehub = MagicMock()
            mock_kagglehub.dataset_download.return_value = tmpdir
            download_controls._kagglehub = mock_kagglehub

            files, error = await download_controls._download_kaggle_files("owner/ds")
            assert files == {}
            assert "No supported data files" in error

    async def test_returns_error_on_download_failure(self, download_controls):
        mock_kagglehub = MagicMock()
        mock_kagglehub.dataset_download.side_effect = RuntimeError("auth failed")
        download_controls._kagglehub = mock_kagglehub

        files, error = await download_controls._download_kaggle_files("owner/ds")
        assert files == {}
        assert "auth failed" in error


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_kaggle (agent tool path)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchKaggle:
    """Tests for the agent-facing Kaggle fetch method."""

    async def test_single_csv_produces_one_table(self, download_controls):
        download_controls._kagglehub = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "users.csv"), "w") as f:
                f.write("name,age\nAlice,30\nBob,25\n")
            download_controls._kagglehub.dataset_download.return_value = tmpdir

            result = await download_controls._fetch_kaggle("owner/ds")

        assert len(result.sources) == 1
        source = result.sources[0]
        assert "users" in source.tables
        df = source.get("users")
        assert len(df) == 2
        assert result.table == "users"

    async def test_multiple_files_produce_multiple_tables_in_one_source(self, download_controls):
        download_controls._kagglehub = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "orders.csv"), "w") as f:
                f.write("id,amount\n1,100\n2,200\n")
            with open(os.path.join(tmpdir, "customers.csv"), "w") as f:
                f.write("id,name\n1,Alice\n2,Bob\n")
            download_controls._kagglehub.dataset_download.return_value = tmpdir

            result = await download_controls._fetch_kaggle("owner/ds")

        assert len(result.sources) == 1
        source = result.sources[0]
        tables = set(source.get_tables())
        assert tables == {"customers", "orders"}

    async def test_metadata_includes_kaggle_ref(self, download_controls):
        download_controls._kagglehub = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.csv"), "w") as f:
                f.write("x\n1\n")
            download_controls._kagglehub.dataset_download.return_value = tmpdir

            result = await download_controls._fetch_kaggle("alice/my-dataset")

        source = result.sources[0]
        assert source.metadata["data"]["kaggle_ref"] == "alice/my-dataset"

    async def test_empty_result_when_kagglehub_missing(self, download_controls):
        download_controls._kagglehub = None
        result = await download_controls._fetch_kaggle("owner/ds")
        assert not result.sources
        assert "pip install kagglehub" in result.message


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_url integration (Kaggle URL detection in agent path)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFetchUrlKaggleDetection:
    """Tests that _fetch_url routes Kaggle URLs correctly."""

    async def test_kaggle_url_routed_to_fetch_kaggle(self, download_controls):
        download_controls._kagglehub = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.csv"), "w") as f:
                f.write("col\n1\n")
            download_controls._kagglehub.dataset_download.return_value = tmpdir

            result = await download_controls._fetch_url(
                "https://www.kaggle.com/datasets/owner/ds"
            )

        assert len(result.sources) == 1
        assert "data" in result.sources[0].tables

    async def test_non_kaggle_url_not_routed_to_kaggle(self, download_controls):
        """A normal URL should not go through the Kaggle path."""
        with patch.object(download_controls, "_fetch_kaggle") as mock_kaggle:
            with patch(
                "lumen.ai.controls.ingest.download.download_file",
                return_value=("data.csv", b"x\n1\n", None),
            ):
                await download_controls._fetch_url("https://example.com/data.csv")
            mock_kaggle.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# _download_and_process_urls (UI flow)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDownloadAndProcessUrlsKaggle:
    """Tests that the UI download path handles Kaggle URLs."""

    async def test_kaggle_url_generates_file_cards(self, download_controls):
        download_controls._kagglehub = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "sales.csv"), "w") as f:
                f.write("month,revenue\nJan,100\n")
            download_controls._kagglehub.dataset_download.return_value = tmpdir

            await download_controls._download_and_process_urls(
                ["https://www.kaggle.com/datasets/owner/ds"]
            )

        assert len(download_controls._file_cards) == 1
        assert download_controls._file_cards[0].filename == "sales"

    async def test_kaggle_error_appended_to_errors(self, download_controls):
        download_controls._kagglehub = None
        await download_controls._download_and_process_urls(
            ["https://www.kaggle.com/datasets/owner/ds"]
        )
        assert download_controls._error_placeholder.visible is True
        assert "kagglehub" in download_controls._error_placeholder.object
