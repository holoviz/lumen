import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip(
        "lumen.ai could not be imported, skipping tests.",
        allow_module_level=True,
    )

from lumen.ai.agents.mosaic import MosaicAgent
from lumen.ai.editors import MosaicEditor
from lumen.config import dump_yaml
from lumen.views.base import MosaicView

SIMPLE_SPEC = {
    "plot": [
        {"mark": "lineY", "data": {"from": "table"}, "x": "date", "y": "close"}
    ],
    "width": 680,
    "height": 240,
}


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


def test_rebind_table_rewrites_every_from_reference():
    """All `from:` references are pointed at the given table, so an LLM-invented
    or drifted table name in the spec cannot break the data binding."""
    spec = {
        "params": {"brush": {"select": "intervalX"}},
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
