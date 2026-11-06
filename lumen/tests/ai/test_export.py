import os

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.sources.data import read
from lumen.ai.export import format_output, make_preamble
from lumen.pipeline import Pipeline


@pytest.mark.parametrize("linesep", ["\n", "\r\n"])
def test_exported_code_rebuilds_data_on_any_platform(monkeypatch, linesep):
    """Windows ends lines with \\r\\n, and text with tabs makes YAML write escapes."""
    monkeypatch.setattr(os, "linesep", linesep)
    frame = pd.DataFrame({"category": ["A", "tab\there"], "value": [1.5, 2.0]})
    cell, _ = format_output(Pipeline(source=read({"tiny": frame}), table="tiny"))
    (preamble,) = make_preamble("")
    namespace = {}
    exec(f"{preamble.source}\n{cell.source}", namespace)
    assert namespace["pipeline"].data.to_dict("records") == frame.to_dict("records")


def test_exported_code_keeps_data_that_looks_like_code_as_data(capsys):
    frame = pd.DataFrame({"note": ['"+str(print(\'INJECTED\'))+r"', 'say """hi"""', "a|-1 b|2"]})
    cell, _ = format_output(Pipeline(source=read({"tiny": frame}), table="tiny"))
    (preamble,) = make_preamble("")
    namespace = {}
    exec(f"{preamble.source}\n{cell.source}", namespace)
    assert "INJECTED" not in capsys.readouterr().out
    assert namespace["pipeline"].data["note"].tolist() == frame["note"].tolist()
