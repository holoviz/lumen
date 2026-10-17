import functools
import json

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown

from lumen.ai.export import (
    assigned_names, export_notebook, format_function_call,
)
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


def type_name(data) -> str:
    """Name the type of an object."""
    return type(data).__name__


@define_tool(provides=["data"])
def load_opaque() -> Opaque:
    """Load an object that cannot be written as a literal."""
    return Opaque()


class IntroAction(Action):

    async def _execute(self, context, **kwargs):
        return [Markdown("Intro text")], {}


def _source(cell):
    return "".join(cell["source"])


def _cells(notebook):
    return json.loads(notebook)["cells"]


def _run_code_cells(notebook):
    """Execute the code cells of an exported notebook in a fresh namespace."""
    namespace = {}
    for cell in _cells(notebook):
        if cell["cell_type"] == "code":
            exec(_source(cell), namespace)
    return namespace


async def test_function_tool_returns_call_alongside_text():
    tool = FunctionTool(add, requires=["a", "b"], provides=["c"])
    outputs, _ = await tool.respond([], {"a": 1, "b": 3})

    # The text handed to the LLM is unchanged and stays a plain string
    assert outputs[0] == "add(a=1, b=3) returned: 4"
    assert type(outputs[0]) is str
    call = outputs[1]
    assert isinstance(call, FunctionCall)
    assert call.function is add
    assert call.arguments == {"a": 1, "b": 3}
    assert call.provides == ["c"]
    assert call.unpack is False


async def test_function_tool_call_marks_dict_unpacking():
    tool = FunctionTool(summarize, requires=["numbers"], provides=["total", "count"])
    outputs, out_context = await tool.respond([], {"numbers": [1, 2]})

    assert out_context == {"total": 3, "count": 2}
    assert outputs[1].unpack is True


def test_function_call_repr_leaves_out_argument_values():
    call = FunctionCall(function=add, arguments={"a": "secret-value", "b": 3})

    assert repr(call) == "FunctionCall(add(a, b))"


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


def test_format_function_call_stops_on_undefined_context_arguments():
    call = FunctionCall(function=add, arguments={"a": Opaque(), "b": 3})

    lines = format_function_call(call)["source"].splitlines()
    assert lines[-4:] == [
        "# a came from the Lumen context and is not defined by an earlier cell.",
        'raise NameError("Define a before running this cell")',
        "",
        "add(a=a, b=3)",
    ]


def test_format_function_call_uses_names_defined_by_earlier_cells():
    call = FunctionCall(function=add, arguments={"a": Opaque(), "b": 3})

    assert format_function_call(call, defined={"a"})["source"] == (
        f"from {__name__} import add\n"
        "\n"
        "add(a=a, b=3)"
    )


def test_format_function_call_leaves_out_secret_arguments():
    def fetch(url: str, api_token: str) -> str:
        return url

    call = FunctionCall(
        function=fetch, arguments={"url": "https://example.com", "api_token": "abc123"}
    )

    source = format_function_call(call)["source"]
    assert "abc123" not in source
    assert "# api_token was left out because it may hold a secret." in source
    assert source.splitlines()[-1] == "fetch(url='https://example.com', api_token=api_token)"


def test_format_function_call_wraps_long_calls():
    call = FunctionCall(function=add, arguments={"a": "x" * 40, "b": "y" * 40})

    lines = format_function_call(call)["source"].splitlines()
    assert lines[-4:] == ["add(", f"    a={'x' * 40!r},", f"    b={'y' * 40!r},", ")"]


def test_format_function_call_does_not_copy_local_function_source():
    import math

    @define_tool(provides=["value"])
    def local_value(x: int) -> float:
        return math.sqrt(x)

    call = FunctionCall(function=local_value, arguments={"x": 4}, provides=["value"])

    # Copying the source would drop the math import, so ask for a definition instead
    assert format_function_call(call)["source"] == (
        "# local_value cannot be imported, so it must be defined before this cell.\n"
        'raise NameError("Define local_value before running this cell")\n'
        "\n"
        "value = local_value(x=4)"
    )


def test_format_function_call_skips_call_without_usable_name():
    for function in (functools.partial(add, 1), lambda: 1):
        call = FunctionCall(function=function, arguments={}, provides=["y"])

        source = format_function_call(call)["source"]
        assert source.startswith("# The call to ")
        assert source.endswith("cannot be repeated here, since it has no name to call it by.")
        assert "\n" not in source


def test_assigned_names():
    source = (
        "from lumen.ai import tools\n"
        "import numpy as np\n"
        "result = await fetch()\n"
        "total = result['total']\n"
        "a, (b, c) = 1, (2, 3)\n"
        "def helper():\n"
        "    inner = 1\n"
    )

    assert assigned_names(source) == {"tools", "np", "result", "total", "a", "b", "c", "helper"}


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
    summary = next(i for i, s in enumerate(sources) if "result = summarize(" in s)
    assert intro < load < summary
    assert sources[load] == (
        f"from {__name__} import load_numbers\n"
        "\n"
        "numbers = load_numbers()"
    )
    assert sources[summary].splitlines()[-3] == "result = summarize(numbers=[1, 2, 3])"


async def test_report_notebook_runs_in_fresh_namespace():
    report = Report(
        Section(
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

    namespace = _run_code_cells(report.to_notebook())
    assert namespace["numbers"] == [1, 2, 3]
    assert namespace["total"] == 6
    assert namespace["count"] == 3


async def test_report_notebook_passes_earlier_results_by_name():
    report = Report(
        Section(
            ActorTask(FunctionTool(load_opaque), title="Load"),
            ActorTask(
                FunctionTool(type_name, requires=["data"], provides=["kind"]),
                title="Describe",
            ),
            title="Analysis",
        ),
        title="Objects",
    )
    await report.execute()

    namespace = _run_code_cells(report.to_notebook())
    assert namespace["kind"] == "Opaque"


async def test_report_notebook_stops_clearly_on_undefined_context_argument():
    report = Report(
        Section(
            ActorTask(
                FunctionTool(type_name, requires=["data"], provides=["kind"]),
                title="Describe", context={"data": Opaque()},
            ),
            title="Analysis",
        ),
        title="Objects",
    )
    await report.execute()

    with pytest.raises(NameError, match="Define data before running this cell"):
        _run_code_cells(report.to_notebook())


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
    source = format_function_call(calls[0])["source"]
    assert "# df came from the Lumen context and is not defined by an earlier cell." in source
    assert source.splitlines()[-1] == "n_rows = count_rows(df=df)"
