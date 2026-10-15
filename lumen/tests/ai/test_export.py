import json

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown

from lumen.ai.export import export_notebook, format_function_call
from lumen.ai.report import (
    Action, ActorTask, Report, Section,
)
from lumen.ai.tools import FunctionCall, FunctionTool, define_tool


def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b


def summarize(numbers: list) -> dict:
    """Summarize a list of numbers."""
    return {"total": sum(numbers), "count": len(numbers)}


@define_tool(provides=["numbers"])
def load_numbers() -> list:
    """Load some numbers."""
    return [1, 2, 3]


@define_tool(render_output=True)
def render_greeting() -> str:
    return "Hello from tool"


class Opaque:
    """An object whose repr is not a Python literal."""


class IntroAction(Action):

    async def _execute(self, context, **kwargs):
        return [Markdown("Intro text")], {}


def _source(cell):
    return "".join(cell["source"])


def _cells(notebook):
    return json.loads(notebook)["cells"]


async def test_function_tool_output_carries_call():
    tool = FunctionTool(add, requires=["a", "b"], provides=["c"])
    outputs, _ = await tool.respond([], {"a": 1, "b": 3})

    # The text handed to the LLM is unchanged
    assert outputs[0] == "add(a=1, b=3) returned: 4"
    call = outputs[0].call
    assert isinstance(call, FunctionCall)
    assert call.function is add
    assert call.arguments == {"a": 1, "b": 3}
    assert call.provides == ["c"]
    assert call.unpack is False


async def test_function_tool_output_marks_dict_unpacking():
    tool = FunctionTool(summarize, requires=["numbers"], provides=["total", "count"])
    outputs, out_context = await tool.respond([], {"numbers": [1, 2]})

    assert out_context == {"total": 3, "count": 2}
    assert outputs[0].call.unpack is True


def test_format_function_call_imports_module_function():
    call = FunctionCall(function=add, arguments={"a": 1, "b": 3}, provides=["c"])

    assert format_function_call(call)["source"] == (
        f"from {__name__} import add\n"
        "\n"
        "c = add(a=1, b=3)"
    )


def test_format_function_call_without_provides():
    call = FunctionCall(function=add, arguments={"a": 1, "b": 3})

    assert format_function_call(call)["source"].splitlines()[-1] == "add(a=1, b=3)"


def test_format_function_call_unpacks_provided_keys():
    call = FunctionCall(
        function=summarize, arguments={"numbers": [1, 2]}, provides=["total", "count"], unpack=True
    )

    assert format_function_call(call)["source"].splitlines()[-3:] == [
        "result = summarize(numbers=[1, 2])",
        "total = result['total']",
        "count = result['count']",
    ]


def test_format_function_call_references_non_literal_arguments():
    call = FunctionCall(function=add, arguments={"a": Opaque(), "b": 3})

    lines = format_function_call(call)["source"].splitlines()
    assert lines[-2] == "# Expects a from the Lumen context, e.g. defined by an earlier cell"
    assert lines[-1] == "add(a=a, b=3)"


def test_format_function_call_wraps_long_calls():
    call = FunctionCall(function=add, arguments={"a": "x" * 40, "b": "y" * 40})

    lines = format_function_call(call)["source"].splitlines()
    assert lines[-4:] == ["add(", f"    a={'x' * 40!r},", f"    b={'y' * 40!r},", ")"]


def test_format_function_call_inlines_local_function_source():
    @define_tool(provides=["value"])
    def local_value(x: int) -> int:
        return x * 2

    call = FunctionCall(function=local_value, arguments={"x": 2}, provides=["value"])

    # The decorator is dropped, since it only matters to Lumen
    assert format_function_call(call)["source"] == (
        "def local_value(x: int) -> int:\n"
        "    return x * 2\n"
        "\n"
        "value = local_value(x=2)"
    )


def test_format_function_call_without_source():
    call = FunctionCall(function=lambda: 1, arguments={})

    lines = format_function_call(call)["source"].splitlines()
    assert lines[0].startswith("# <lambda> could not be imported")
    assert lines[-1] == "<lambda>()"


async def test_report_to_notebook_includes_function_tool_calls_in_order():
    report = Report(
        Section(
            IntroAction(title="Intro"),
            ActorTask(FunctionTool(load_numbers), title="Load"),
            ActorTask(
                FunctionTool(summarize, requires=["numbers"], provides=["total", "count"]),
                title="Summarize",
            ),
            title="Analysis",
        ),
        title="Numbers",
    )
    await report.execute()
    assert report.status == "success"

    sources = [_source(cell) for cell in _cells(report.to_notebook())]
    intro = sources.index("Intro text")
    load = next(i for i, s in enumerate(sources) if s.endswith("numbers = load_numbers()"))
    summary = next(i for i, s in enumerate(sources) if "result = summarize(numbers=[1, 2, 3])" in s)
    assert intro < load < summary
    assert sources[load] == (
        f"from {__name__} import load_numbers\n"
        "\n"
        "numbers = load_numbers()"
    )


async def test_function_tool_call_not_rendered_in_html_or_docx():
    pytest.importorskip("docx")
    report = Report(
        Section(ActorTask(FunctionTool(load_numbers), title="Load"), title="S"),
        title="Numbers",
    )
    await report.execute()

    assert any(isinstance(view, FunctionCall) for view in report.views)
    assert "FunctionCall" not in report.to_html()
    assert report.to_docx()


async def test_export_notebook_includes_function_tool_calls():
    # Explorations are exported from the plan's views via export_notebook
    section = Section(ActorTask(FunctionTool(load_numbers), title="Load"), title="S")
    await section.execute({})

    sources = [_source(cell) for cell in _cells(export_notebook(section.views))]
    assert any(s.endswith("numbers = load_numbers()") for s in sources)


async def test_view_returning_function_tool_is_not_exported_as_call():
    task = ActorTask(FunctionTool(render_greeting), title="Greet")
    await task.execute({})

    assert not any(isinstance(view, FunctionCall) for view in task.views)


async def test_function_tool_call_with_dataframe_argument():
    pd = pytest.importorskip("pandas")

    def count_rows(df) -> int:
        return len(df)

    task = ActorTask(
        FunctionTool(count_rows, requires=["df"], provides=["n_rows"]),
        title="Count", context={"df": pd.DataFrame({"x": [1, 2]})},
    )
    await task.execute({})

    calls = [view for view in task.views if isinstance(view, FunctionCall)]
    assert len(calls) == 1
    lines = format_function_call(calls[0])["source"].splitlines()
    assert lines[-1] == "n_rows = count_rows(df=df)"
