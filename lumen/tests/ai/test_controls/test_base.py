import io
import json

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)


@pytest.fixture
def make_json_file():
    def _make(data):
        return io.BytesIO(json.dumps(data).encode())
    return _make


class TestReadJsonFile:
    """Tests for _read_json_file nested object flattening."""

    def test_flat_json_array_loads_correctly(self, upload_controls, make_json_file):
        data = [{"id": 1, "name": "foo"}, {"id": 2, "name": "bar"}]
        df = upload_controls._read_json_file(make_json_file(data), "test.json")
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 2

    def test_nested_object_is_flattened(self, upload_controls, make_json_file):
        data = [
            {"id": 1, "meta": {"rate": 3.9, "count": 120}},
            {"id": 2, "meta": {"rate": 4.1, "count": 259}},
        ]
        df = upload_controls._read_json_file(make_json_file(data), "test.json")
        assert "meta" not in df.columns
        assert "meta.rate" in df.columns
        assert "meta.count" in df.columns
        assert df["meta.rate"].tolist() == [3.9, 4.1]
        assert df["meta.count"].tolist() == [120, 259]

    def test_deeply_nested_object_is_flattened(self, upload_controls, make_json_file):
        data = [{"id": 1, "address": {"city": "NYC", "geo": {"lat": 40.7, "lng": -74.0}}}]
        df = upload_controls._read_json_file(make_json_file(data), "test.json")
        assert "address.city" in df.columns
        assert "address.geo.lat" in df.columns
        assert "address.geo.lng" in df.columns

    def test_nested_object_under_wrapper_key_is_flattened(self, upload_controls, make_json_file):
        data = {"results": [
            {"id": 1, "score": {"value": 99, "grade": "A"}},
            {"id": 2, "score": {"value": 75, "grade": "B"}},
        ]}
        df = upload_controls._read_json_file(make_json_file(data), "test.json")
        assert "score" not in df.columns
        assert "score.value" in df.columns
        assert "score.grade" in df.columns

    def test_no_nested_objects_no_regression(self, upload_controls, make_json_file):
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        df = upload_controls._read_json_file(make_json_file(data), "test.json")
        assert list(df.columns) == ["a", "b"]
        assert df["a"].tolist() == [1, 2]
