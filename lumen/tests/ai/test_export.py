import json
import os

from typing import Literal

import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown

from lumen.ai.export import export_notebook, format_output, make_preamble
from lumen.ai.report import (
    ActorTask, Report, Section, export_outputs,
)
from lumen.ai.tools import FunctionTool, define_tool
from lumen.ai.tools.base import FunctionCall
from lumen.pipeline import Pipeline
from lumen.sources.data import read


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


def load_fees() -> float:
    """Load the taker fee rate."""
    return 0.00045


def fee_schedule(exchange: Literal["hyperliquid", "binance"]) -> dict:
    """Look up the fee rates of an exchange."""
    return {"hyperliquid": {"taker": 0.00045}, "binance": {"taker": 0.001}}[exchange]


def fee_split(exchange: str = "hyperliquid") -> dict:
    """Split the fee rates of an exchange."""
    return {"maker": 0.00015, "taker": 0.00045}


def taker_fees(trades: list) -> float:
    """Total taker fees of the given trades."""
    return sum(trades) * 0.00045


@define_tool(provides=["rate"])
def decorated_rate() -> float:
    """A rate defined outside an importable module."""
    return 0.5


def show_fees() -> Markdown:
    """Render the fees as Markdown."""
    return Markdown("Fees")


def code_cells(nb_string):
    return ["".join(cell["source"]) for cell in json.loads(nb_string)["cells"] if cell["cell_type"] == "code"]


def run_cells(sources):
    namespace = {}
    for source in sources:
        exec(source, namespace)
    return namespace


async def test_report_notebook_repeats_function_tool_call():
    report = Report(Section(ActorTask(FunctionTool(load_fees, provides=["fees"]))))
    await report.execute()

    cells = code_cells(report.to_notebook())

    assert cells[-1] == (
        "from lumen.tests.ai.test_export import load_fees\n\nfees = load_fees()"
    )
    assert run_cells(cells[-1:])["fees"] == 0.00045


async def test_notebook_writes_llm_arguments_as_literals(llm):
    tool = FunctionTool(fee_schedule, llm=llm)
    llm.set_responses([tool._model(exchange="hyperliquid")])
    report = Report(Section(ActorTask(tool)))
    await report.execute()

    cell = code_cells(report.to_notebook())[-1]

    assert cell.endswith("\nfee_schedule(exchange='hyperliquid')")
    run_cells([cell])


async def test_notebook_unpacks_multiple_provided_values(llm):
    tool = FunctionTool(fee_split, provides=["maker", "taker"], llm=llm)
    llm.set_responses([tool._model(exchange="hyperliquid")])
    report = Report(Section(ActorTask(tool)))
    await report.execute()

    cell = code_cells(report.to_notebook())[-1]

    assert "result = fee_split(exchange='hyperliquid')" in cell
    namespace = run_cells([cell])
    assert (namespace["maker"], namespace["taker"]) == (0.00015, 0.00045)


async def test_notebook_passes_context_arguments_by_name():
    tool = FunctionTool(taker_fees, requires=["trades"], provides=["total"])
    report = Report(Section(ActorTask(tool)))
    await report.execute({"trades": [100.0, 200.0]})

    cell = code_cells(report.to_notebook())[-1]

    assert "# Lumen passed in trades; define it before running this cell" in cell
    assert cell.endswith("total = taker_fees(trades=trades)")


async def test_notebook_uses_names_defined_by_earlier_calls():
    report = Report(Section(
        ActorTask(FunctionTool(load_fees, provides=["trades"])),
        ActorTask(FunctionTool(load_fees, provides=["fees"])),
    ))
    await report.execute()

    cells = code_cells(report.to_notebook())

    # The second call reuses the import made for the first
    assert cells[-2].startswith("from lumen.tests.ai.test_export import load_fees")
    assert cells[-1] == "fees = load_fees()"


async def test_notebook_inlines_source_of_unimportable_function(monkeypatch):
    monkeypatch.setattr(decorated_rate, "__module__", "__main__")
    report = Report(Section(ActorTask(FunctionTool(decorated_rate))))
    await report.execute()

    cell = code_cells(report.to_notebook())[-1]

    # The decorator is dropped since the notebook may not have it
    assert cell.startswith("def decorated_rate() -> float:")
    assert run_cells([cell])["rate"] == 0.5


async def test_notebook_inlined_source_brings_the_imports_it_uses(llm, monkeypatch):
    monkeypatch.setattr(fee_schedule, "__module__", "__main__")
    tool = FunctionTool(fee_schedule, llm=llm)
    llm.set_responses([tool._model(exchange="binance")])
    report = Report(Section(ActorTask(tool)))
    await report.execute()

    cell = code_cells(report.to_notebook())[-1]

    # Literal is imported at the top of this file, not inside the function
    assert cell.startswith("from typing import Literal\n\ndef fee_schedule(")
    run_cells([cell])


async def test_notebook_keeps_task_heading_ahead_of_its_call():
    task = ActorTask(FunctionTool(load_fees, provides=["fees"]), title="Load fees")
    report = Report(Section(task))
    await report.execute()

    sources = ["".join(cell["source"]) for cell in json.loads(report.to_notebook())["cells"]]

    heading = next(i for i, s in enumerate(sources) if s.endswith("Load fees"))
    assert sources[heading + 1].endswith("fees = load_fees()")


async def test_tool_returning_a_view_is_not_recorded():
    task = ActorTask(FunctionTool(show_fees))
    await Report(Section(task)).execute()

    assert task.function_calls == []


async def test_exploration_export_includes_function_tool_call():
    section = Section(ActorTask(FunctionTool(load_fees, provides=["fees"])))
    await section.execute()

    nb = export_notebook(export_outputs(section, include_calls=True))

    assert code_cells(nb)[-1].endswith("fees = load_fees()")


async def test_html_export_leaves_out_function_tool_calls():
    report = Report(Section(ActorTask(FunctionTool(load_fees, provides=["fees"]))))
    await report.execute()

    def has_call(views):
        return any(isinstance(view, FunctionCall) for view in views)

    assert not has_call(report._selected_views(report[0], include_header=False))
    assert has_call(report._selected_views(report[0], include_calls=True))
