import subprocess
import sys

from unittest.mock import patch

import pytest

IPython = pytest.importorskip("IPython")

from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell

from lumen.ai.notebook import load_ipython_extension


@pytest.fixture
def shell():
    shell = InteractiveShell.instance()
    load_ipython_extension(shell)
    shell.user_ns.update(penguins="the source", year=2008)
    yield shell
    for name in ("penguins", "year", "_lumen", "heaviest"):
        shell.user_ns.pop(name, None)


def test_cell_magic_asks_the_cell_with_variables_filled_in(shell):
    with patch("lumen.ai.notebook.ask", return_value="answer") as ask:
        result = shell.run_cell_magic("lumen", "penguins --as heaviest", "Which species was heaviest in {year}?\n")
    ask.assert_called_once_with("the source", "Which species was heaviest in 2008?", confirm=False)
    assert result == "answer" and shell.user_ns["heaviest"] == "answer"


def test_line_magic_asks_the_rest_of_the_line(shell):
    with patch("lumen.ai.notebook.ask", return_value="answer") as ask:
        # Run as a cell, so IPython expands {year} on the line as it does in a notebook
        shell.run_cell("%lumen penguins --confirm How many penguins in {year}?")
    ask.assert_called_once_with("the source", "How many penguins in 2008?", confirm=True)
    assert shell.user_ns["_lumen"] == "answer"


@pytest.mark.parametrize("line, cell, message", [
    ("nothing", "Rows?", "No variable named 'nothing'"),
    ("penguins", "Rows in {month}?", "names {month}"),
    ("penguins", "   ", "Write the question"),
    ("penguins heaviest", "Rows?", "put the question in the cell"),
    ("penguins --as --confirm", "Rows?", "--as needs a variable name"),
])
def test_magic_errors_say_what_to_fix(shell, line, cell, message):
    with patch("lumen.ai.notebook.ask") as ask, pytest.raises(UsageError, match=message):
        shell.run_cell_magic("lumen", line, cell)
    ask.assert_not_called()


def test_importing_lumen_ai_does_not_register_the_magics():
    code = "import sys, lumen.ai; assert 'lumen.ai.notebook' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_quote_in_the_question_is_kept(shell):
    with patch("lumen.ai.notebook.ask", return_value="answer") as ask:
        shell.run_cell_magic("lumen", "penguins", "What's the \"max\" mass?\n")
        shell.run_cell("%lumen penguins What's the max")
    assert [call.args[1] for call in ask.call_args_list] == ['What\'s the "max" mass?', "What's the max"]


@pytest.mark.parametrize("cell", ["Rows {", "Row {0}", "Row {year.nope}"])
def test_a_stray_brace_is_a_usage_error(shell, cell):
    with pytest.raises(UsageError, match="brace|variable"):
        shell.run_cell_magic("lumen", "penguins", cell)
