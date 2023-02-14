import pathlib

from io import StringIO

import pytest

from lumen.downloads import Download
from lumen.pipeline import Pipeline
from lumen.views import Table


class ExampleDownload(Download):

    download_type = "example"

    def _table_data(self):
        io = StringIO()
        io.write("hello")

        io.seek(0)
        return io


@pytest.mark.parametrize(
    "downloader,download_type",
    [
        (ExampleDownload(format="csv"), "example"),
        (Download(format="csv"), "default"),
    ],
)
def test_custom_download(downloader, download_type, make_filesource):
    root = pathlib.Path(__file__).parent / "sources"
    source = make_filesource(str(root))
    pipeline = Pipeline(source=source, table="test", name="Pipeline")

    table = Table(
        pipeline=pipeline,
        download=downloader,
        title="Test",
    )
    assert table.to_spec()["download"]["type"] == download_type



def test_no_download(make_filesource):
    root = pathlib.Path(__file__).parent / "sources"
    source = make_filesource(str(root))
    pipeline = Pipeline(source=source, table="test", name="Pipeline")

    table = Table(
        pipeline=pipeline,
        title="Test",
    )
    assert "download" not in table.to_spec()
