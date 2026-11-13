import pytest

from lumen.util import normalize_table_name


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("customer (orders)", "customer_orders"),
        ("My-Data File", "my_data_file"),
        ("table_name", "table_name"),
        ("123", "t_123"),
        ("20_load_cells", "t_20_load_cells"),
        ("_123", "t_123"),
        ("", "table"),
        ("!!!", "table"),
        ("@@@###", "table"),
        ("   ", "table"),
    ],
)
def test_normalize_table_name(raw, expected):
    assert normalize_table_name(raw) == expected
