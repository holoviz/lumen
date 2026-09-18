import asyncio

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from panel.chat import ChatFeed

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip(
        "lumen.ai could not be imported, skipping tests.",
        allow_module_level=True,
    )

from lumen.ai.agents.mosaic import MosaicAgent
from lumen.ai.agents.vega_lite import VegaLiteAgent
from lumen.ai.config import PROMPTS_DIR
from lumen.ai.editors import MosaicEditor
from lumen.ai.ui import ExplorerUI
from lumen.config import dump_yaml
from lumen.pipeline import Pipeline
from lumen.sources.base import InMemorySource
from lumen.views.base import MosaicView

SIMPLE_SPEC = {
    "plot": [
        {"mark": "lineY", "data": {"from": "table"}, "x": "date", "y": "close"}
    ],
    "width": 680,
    "height": 240,
}


def test_mosaic_is_the_default_tabular_chart_agent():
    """Planner guidance makes Mosaic the default for ordinary tabular charts."""
    assert any("default for standard tabular" in condition.lower() for condition in MosaicAgent.conditions)
    assert all("default" not in condition.lower() for condition in VegaLiteAgent.conditions)

    planner_template = (PROMPTS_DIR / "Planner" / "main.jinja2").read_text()
    assert "For standard tabular chart requests, use `MosaicAgent`" in planner_template
    assert ExplorerUI.default_agents.index(MosaicAgent) < ExplorerUI.default_agents.index(VegaLiteAgent)


def test_editor_validate_accepts_plot_containers():
    """A spec with a plot container (plot/vconcat/hconcat) is accepted."""
    for spec in (
        {"plot": [{"mark": "dot", "data": {"from": "t"}, "x": "a", "y": "b"}]},
        {"vconcat": []},
        {"hconcat": []},
    ):
        MosaicEditor.validate_spec(spec)


def test_editor_validate_rejects_empty_or_non_mapping():
    """Only the clearly-broken cases are rejected so the retry loop can fire."""
    with pytest.raises(ValueError):
        MosaicEditor.validate_spec({})
    with pytest.raises(ValueError):
        MosaicEditor.validate_spec("not a mapping")


def test_editor_validate_rejects_common_llm_mistakes():
    """Structural mistakes LLMs make are caught so the retry loop can fix them,
    with a message that names the correct key."""
    # `marks:` instead of `plot:`
    with pytest.raises(ValueError, match="plot"):
        MosaicEditor.validate_spec({"marks": [{"mark": "dot"}]})
    # a spec with no plot container at all (e.g. only `intervals:`/`params:`)
    with pytest.raises(ValueError, match="container"):
        MosaicEditor.validate_spec({"intervals": {"brush": {"type": "brush"}}})


def test_editor_validate_rejects_invalid_interaction_grammar():
    """Selection modes and table components use distinct Mosaic grammar."""
    with pytest.raises(ValueError, match="intervalXY"):
        MosaicEditor.validate_spec({
            "params": {"brush": {"select": "intervalXY"}},
            "plot": [{"mark": "dot", "data": {"from": "t"}, "x": "a", "y": "b"}],
        })
    with pytest.raises(ValueError, match="input: table"):
        MosaicEditor.validate_spec({
            "plot": [{"mark": "table", "data": {"from": "t"}}],
        })
    with pytest.raises(ValueError, match=r"as:.*\$"):
        MosaicEditor.validate_spec({
            "vconcat": [
                {"input": "menu", "from": "t", "column": "category", "as": "filter"},
                {"plot": [{"mark": "dot", "data": {"from": "t"}, "x": "x", "y": "y"}]},
            ],
        })
    with pytest.raises(ValueError, match=r"filterBy:.*\$"):
        MosaicEditor.validate_spec({
            "plot": [{
                "mark": "dot", "data": {"from": "t", "filterBy": "filter"},
                "x": "x", "y": "y",
            }],
        })


def test_editor_validate_accepts_linked_interactive_spec():
    """A brush may link a scatter plot, histogram, and data table."""
    MosaicEditor.validate_spec({
        "params": {"brush": {"select": "intersect"}},
        "hconcat": [
            {"plot": [
                {"mark": "dot", "data": {"from": "t"}, "x": "x", "y": "y"},
                {"select": "intervalXY", "as": "$brush"},
            ]},
            {"vconcat": [
                {"plot": [{
                    "mark": "rectY", "data": {"from": "t", "filterBy": "$brush"},
                    "x": {"bin": "value"}, "y": {"count": None},
                }]},
                {"input": "table", "from": "t", "filterBy": "$brush"},
            ]},
        ],
    })


def test_editor_validate_accepts_linked_slider_spec():
    """An interval slider can create a selection that filters linked views."""
    MosaicEditor.validate_spec({
        "params": {"range": {"select": "intersect"}},
        "vconcat": [
            {
                "input": "slider",
                "select": "interval",
                "as": "$range",
                "from": "t",
                "column": "value",
            },
            {"plot": [{
                "mark": "rectY",
                "data": {"from": "t", "filterBy": "$range"},
                "x": {"bin": "value"},
                "y": {"count": None},
            }]},
            {"input": "table", "from": "t", "filterBy": "$range"},
        ],
    })


def test_editor_validate_accepts_linked_menu_spec():
    """A menu may create a categorical selection that filters linked views."""
    MosaicEditor.validate_spec({
        "params": {"category_filter": {"select": "intersect"}},
        "vconcat": [
            {
                "input": "menu",
                "from": "t",
                "column": "category",
                "as": "$category_filter",
            },
            {"plot": [{
                "mark": "dot",
                "data": {"from": "t", "filterBy": "$category_filter"},
                "x": "x",
                "y": "y",
            }]},
            {"input": "table", "from": "t", "filterBy": "$category_filter"},
        ],
    })


def test_editor_validate_accepts_value_parameter():
    """A regular reactive parameter is not a linked selection."""
    MosaicEditor.validate_spec({
        "params": {"point_size": {"select": "value", "value": 3}},
        "plot": [{"mark": "dot", "data": {"from": "t"}, "x": "x", "y": "y"}],
    })


def test_editor_validate_unwraps_nested_spec_key():
    """`validate_spec` accepts either the bare spec or a {'spec': ...} wrapper."""
    MosaicEditor.validate_spec({"spec": SIMPLE_SPEC})


async def test_extract_spec_parses_yaml_and_sets_sizing(llm):
    """`_extract_spec` parses the YAML and returns responsive view params."""
    agent = MosaicAgent(llm=llm)
    result = await agent._extract_spec({}, {"yaml_spec": dump_yaml(SIMPLE_SPEC)})

    assert result["spec"]["plot"][0]["mark"] == "lineY"
    assert result["sizing_mode"] == "stretch_both"
    assert result["min_height"] == 400
    assert result["responsive"] is True


async def test_extract_spec_drops_top_level_data_block(llm):
    """A model-supplied top-level `data:` block is dropped; Lumen injects the
    table, and marks still reference it by name via `data: {from: <table>}`."""
    agent = MosaicAgent(llm=llm)
    spec = dict(SIMPLE_SPEC, data={"table": {"file": "data/foo.parquet"}})

    result = await agent._extract_spec({}, {"yaml_spec": dump_yaml(spec)})

    assert "data" not in result["spec"]
    assert result["spec"]["plot"][0]["data"] == {"from": "table"}


async def test_extract_spec_rejects_empty_spec(llm):
    """An empty spec raises so `retry_llm_output` can regenerate it."""
    agent = MosaicAgent(llm=llm)
    with pytest.raises(ValueError):
        await agent._extract_spec({}, {"yaml_spec": dump_yaml({})})


async def test_generate_spec_revises_invalid_interaction_before_full_retry(llm):
    """A bad binding is revised and revalidated without surfacing its traceback."""
    agent = MosaicAgent(llm=llm)
    invalid = {
        "params": {"category_filter": {"select": "intersect"}},
        "vconcat": [
            {
                "input": "menu",
                "from": "t",
                "column": "category",
                "as": "category_filter",
            },
            {"plot": [{
                "mark": "dot",
                "data": {"from": "t", "filterBy": "category_filter"},
                "x": "x",
                "y": "y",
            }]},
        ],
    }
    corrected = {
        "params": {"category_filter": {"select": "intersect"}},
        "vconcat": [
            {
                "input": "menu",
                "from": "t",
                "column": "category",
                "as": "$category_filter",
            },
            {"plot": [{
                "mark": "dot",
                "data": {"from": "t", "filterBy": "$category_filter"},
                "x": "x",
                "y": "y",
            }]},
        ],
    }
    feedback = []

    async def fake_stream_prompt(*args, **kwargs):
        yield SimpleNamespace(chain_of_thought="", yaml_spec=dump_yaml(invalid))

    async def fake_revise(instruction, *args, **kwargs):
        feedback.append(instruction)
        return dump_yaml(corrected)

    pipeline = SimpleNamespace(table="t")
    context = {"pipeline": pipeline}
    with patch.object(agent, "_stream_prompt", side_effect=fake_stream_prompt), \
         patch.object(agent, "revise", side_effect=fake_revise):
        result = await agent._generate_yaml_spec(
            [{"role": "user", "content": "filter by category"}],
            context,
            pipeline,
            "Mosaic view",
        )

    assert len(feedback) == 1
    assert "beginning with `$`" in feedback[0]
    assert result["spec"] == corrected


def test_rebind_table_rewrites_every_from_reference():
    """All `from:` references are pointed at the given table, so an LLM-invented
    or drifted table name in the spec cannot break the data binding."""
    spec = {
        "params": {"brush": {"select": "intersect"}},
        "vconcat": [
            {"input": "menu", "from": "wrong_a", "column": "x"},
            {"plot": [
                {"mark": "dot", "data": {"from": "wrong_b", "filterBy": "$brush"}},
                {"mark": "regressionY", "data": {"from": "wrong_c"}},
            ]},
        ],
    }
    MosaicView._rebind_table(spec, "real_table")

    assert spec["vconcat"][0]["from"] == "real_table"
    plot = spec["vconcat"][1]["plot"]
    assert plot[0]["data"]["from"] == "real_table"
    assert plot[0]["data"]["filterBy"] == "$brush"  # non-`from` keys untouched
    assert plot[1]["data"]["from"] == "real_table"


def test_editor_automatically_retries_browser_render_errors():
    """Browser errors enter the normal revision path, with a bounded retry count."""
    pipeline = Pipeline(
        source=InMemorySource(tables={"table": pd.DataFrame({"date": [1], "close": [2.0]})}),
        table="table",
    )
    view = MosaicView(pipeline=pipeline, spec=SIMPLE_SPEC)
    editor = MosaicEditor(component=view)
    retry = SimpleNamespace(instruction="")
    editor._auto_retry_control = retry

    view.error = "Unknown mark"
    assert "Unknown mark" in retry.instruction
    assert "1/2" in retry.instruction

    view.error = "Bad channel"
    assert "Bad channel" in retry.instruction
    assert "2/2" in retry.instruction

    view.error = "Third failure"
    assert "Third failure" not in retry.instruction

    view.ready = True
    assert editor._auto_retry_attempts == 0


async def test_editor_browser_error_runs_agent_revision(llm):
    """The render-error watcher executes a real asynchronous revision cycle."""
    pipeline = Pipeline(
        source=InMemorySource(tables={"table": pd.DataFrame({"date": [1], "close": [2.0]})}),
        table="table",
    )
    view = MosaicView(pipeline=pipeline, spec=SIMPLE_SPEC)
    editor = MosaicEditor(component=view)
    agent = MosaicAgent(llm=llm)
    revised_spec = dict(SIMPLE_SPEC, width=720)
    feedback = []

    async def fake_revise(instruction, *_args, **_kwargs):
        feedback.append(instruction)
        return dump_yaml(revised_spec)

    task = SimpleNamespace(actor=agent, history=[], out_context={})
    with patch.object(agent, "revise", side_effect=fake_revise):
        editor.render_controls(task, ChatFeed())
        view.error = "Unknown Mosaic mark"
        for _ in range(20):
            if feedback and "width: 720" in editor.spec:
                break
            await asyncio.sleep(0.01)

    assert len(feedback) == 1
    assert "Unknown Mosaic mark" in feedback[0]
    assert "width: 720" in editor.spec
