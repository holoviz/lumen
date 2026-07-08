import io

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported", allow_module_level=True)

from lumen.ai.controls.ingest.utils import (
    read_file_to_dataframes, read_json_to_dataframe,
)


def test_read_csv_non_utf8_encoding():
    """A latin-1 CSV (byte 0xf1 is invalid UTF-8) is parsed via encoding detection."""
    content = "name,county\nJose,Dona Ana\n".replace("Dona", "Do\xf1a").encode("latin-1")
    result = read_file_to_dataframes(io.BytesIO(content), "csv", alias="counties")
    df = result.tables["counties"]
    assert list(df.columns) == ["name", "county"]
    assert df["county"].iloc[0] == "Do\xf1a Ana"


def test_read_json_non_utf8_encoding():
    """A latin-1 JSON payload is decoded via encoding detection."""
    content = '[{"county": "Do\xf1a Ana"}]'.encode("latin-1")
    df = read_json_to_dataframe(content)
    assert df["county"].iloc[0] == "Do\xf1a Ana"
