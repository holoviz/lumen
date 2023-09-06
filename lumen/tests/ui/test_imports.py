from pathlib import Path

import pytest

ui_path = Path(__file__).parents[2] / "ui"
ui_files = sorted(ui_path.glob("[!_]*.py"))


@pytest.mark.parametrize("file", sorted(ui_files), ids=lambda f: f.name)
def test_ui_import(file) -> None:
    module_name = file.with_suffix("").name
    __import__(f"lumen.ui.{module_name}")
