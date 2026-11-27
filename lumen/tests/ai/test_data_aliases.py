import importlib

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)


@pytest.mark.parametrize("old, new", [
    ("lumen.ai.sharing", "lumen.data.sharing"),
    ("lumen.ai.table_stats", "lumen.data.stats"),
    ("lumen.ai.schemas", "lumen.data.catalog"),
    ("lumen.ai.data_quality", "lumen.data.quality"),
])
def test_old_module_path_is_an_alias(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)
