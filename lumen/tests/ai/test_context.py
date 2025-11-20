from typing import Annotated, TypedDict

import pytest

from typing_extensions import NotRequired

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.context import (
    AccumulateSpec, ContextError, ContextModel, ValidationIssue, _dedupe,
    _parse_accumulate_meta, isinstance_like, merge_contexts,
    render_issues_tree, schema_fields, types_compatible, validate_task_inputs,
    validate_taskgroup_exclusions,
)

# -----------------------
# Test schemas & tasks
# -----------------------

class Source(TypedDict):
    id: str
    title: str

class SQLMetaset(TypedDict):
    tables: list[str]

class SQLInputs(ContextModel):
    # single source
    source: Source
    # accumulate into sources from 'source'
    sources: Annotated[list[Source], ("accumulate", "source")]
    # another required input
    sql_metaset: SQLMetaset

class SummOutputs(TypedDict, total=False):
    # produced optional summary
    summary: str

class ReportOutputs(TypedDict):
    report: str

class DummyTask:
    def __init__(self, name: str, input_schema: type[TypedDict] | None, output_schema: type[TypedDict] | None, not_with=None):
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.not_with = not_with

# -----------------------
# Tests
# -----------------------

def test_parse_accumulate_meta_tuple_form():
    ann = Annotated[list[Source], ("accumulate", "source")]
    spec = _parse_accumulate_meta(ann)
    assert isinstance(spec, AccumulateSpec)
    assert spec.from_key == "source"
    assert spec.func is list

def test_dedupe_value_and_callable():
    assert _dedupe([1, 2, 2, 3], "value") == [1, 2, 3]
    items = [{"id": 1}, {"id": 2}, {"id": 1}]
    out = _dedupe(items, key=lambda x: x["id"])
    assert out == [{"id": 1}, {"id": 2}]

def test_merge_contexts_accumulates_from_alt_key_and_direct_target():
    class S(TypedDict):
        id: int
    class Inputs(TypedDict, total=False):
        sources: Annotated[list[S], ("accumulate", "source")]
    c1 = {"source": {"id": 1}}
    c2 = {"sources": [{"id": 2}], "source": {"id": 3}}
    merged = merge_contexts(Inputs, [c1, c2])
    assert merged["sources"] == [{"id": 1}, {"id": 2}, {"id": 3}]

def test_schema_fields_required_optional_and_meta():
    fields = schema_fields(SQLInputs)
    assert fields["source"]["required"] is True
    assert fields["sources"]["required"] is True
    assert fields["sql_metaset"]["required"] is True
    assert ("accumulate", "source") in fields["sources"]["meta"] or _accumulate_alt_key(fields["sources"]["meta"]) == "source"

def test_isinstance_like_typed_dict_and_list():
    src = {"id": "1", "title": "Doc"}
    assert isinstance_like(src, Source)
    assert isinstance_like([src], list[Source])
    assert not isinstance_like([{"id": 2}], list[Source])

def test_types_compatible_list_elem_and_typed_dict():
    assert types_compatible(list[Source], list[Source])
    assert types_compatible(Source, Source)
    assert not types_compatible(list[str], list[int])

def test_validate_task_inputs_with_concrete_values_ok():
    task = DummyTask("Summarize", SQLInputs, SummOutputs, None)
    ctx = {
        "source": {"id": "1", "title": "Doc"},
        "sources": [{"id": "2", "title": "Doc2"}],
        "sql_metaset": {"tables": ["t1"]}
    }
    issues = validate_task_inputs(task, ctx, available_types={}, path=("Group", "TaskA"))
    assert issues == []

def test_validate_task_inputs_uses_upstream_types():
    task = DummyTask("Summarize", SQLInputs, SummOutputs, None)
    available_types = {
        "source": Source,
        "sql_metaset": SQLMetaset,
    }
    issues = validate_task_inputs(task, {}, available_types, path=("Group", "TaskA"))
    assert [i for i in issues if i.key == "sources"] == []
    assert issues == []

def test_validate_task_inputs_flags_incompatible_type():
    task = DummyTask("Summarize", SQLInputs, SummOutputs, None)
    class Wrong(TypedDict):
        x: int
    issues = validate_task_inputs(task, {}, available_types={"sql_metaset": Wrong}, path=("G", "Task"))
    assert any("Incompatible upstream type" in i.message and i.key == "sql_metaset" for i in issues)

def test_validate_task_inputs_accumulator_value_single_elem_allowed():
    task = DummyTask("Summarize", SQLInputs, SummOutputs, None)
    ctx = {"source": {"id": "1", "title": "Doc"}, "sql_metaset": {"tables": []}}
    issues = validate_task_inputs(task, ctx, {}, path=("G", "Task"))
    assert [i for i in issues if i.key == "sources"] == []
    assert issues == []

def test_render_issues_tree_and_context_error():
    issues = [
        ValidationIssue(path=("Group", "Step[0]", "Summarize"), key="summary", expected=str, actual=int, message="Type mismatch"),
        ValidationIssue(path=("Group", "Step[0]", "Summarize"), key="sources", expected=list, actual=None, message="Missing required key 'sources'"),
        ValidationIssue(path=("Group", "Step[1]", "Report"), key="report", expected=str, actual=None, message="Missing required key 'report'"),
    ]
    msg = render_issues_tree(issues, title="Context validation failed")
    assert "Context validation failed" in msg
    assert "Group" in msg
    assert "Summarize" in msg and "Report" in msg
    assert "summary  [error]" in msg
    err = ContextError(issues)
    s = str(err)
    assert "Context validation failed" in s
    assert "Report" in s
    assert "Missing required key 'report'" in s
    assert "issue(s)" in repr(err)

def test_validate_taskgroup_exclusions_pairwise_and_nested():
    class A: pass
    class B: pass

    t1 = DummyTask("TaskA", None, None, not_with=["B"])
    t2 = DummyTask("TaskB", None, None, not_with=[A])
    t1.__class__ = A
    t2.__class__ = B

    issues = validate_taskgroup_exclusions([t1, t2], path="Root")
    assert len(issues) == 1
    assert "Mutually exclusive tasks selected" in issues[0].message
    assert "A × B" in issues[0].key

def test_merge_contexts_empty_bucket_absent_by_default():
    class Inputs(TypedDict, total=False):
        sources: Annotated[list[dict], ("accumulate", "source")]
    merged = merge_contexts(Inputs, [{}])
    assert "sources" not in merged

def test_types_compatible_typed_dict_key_compatibility():
    class A(TypedDict):
        a: int
        b: str
    class B(TypedDict):
        a: int
        b: str
        c: float
    assert types_compatible(A, B)

def test_isinstance_like_nested_collections():
    class Node(TypedDict):
        id: int
        tags: list[str]
    value = {"id": 1, "tags": ["x", "y"]}
    assert isinstance_like([value], list[Node])
    assert not isinstance_like([{"id": "1", "tags": [1]}], list[Node])

def test_typed_dict_notrequired_presence_and_type():
    class User(TypedDict):
        name: str
        age: NotRequired[int]

    # Missing optional key is ok
    assert isinstance_like({"name": "Alice"}, User)
    # Present with correct type is ok
    assert isinstance_like({"name": "Alice", "age": 30}, User)
    # Present with wrong type should fail
    assert not isinstance_like({"name": "Alice", "age": "30"}, User)

def test_schema_fields_notrequired_required_and_type_unwrapped():
    class Item(TypedDict):
        id: int
        note: NotRequired[str]

    fields = schema_fields(Item)
    assert fields["id"]["required"] is True
    assert fields["note"]["required"] is False

    # When present, optional field type should still be enforced as str
    # (accept correct)
    assert isinstance_like("hello", fields["note"]["type"])
    # (reject incorrect)
    assert not isinstance_like(123, fields["note"]["type"])

def test_validate_task_inputs_notrequired_type_mismatch():
    class Inp(ContextModel):
        name: str
        notes: NotRequired[str]

    task = DummyTask("T", Inp, None, None)

    # Wrong type for optional field when present should be flagged
    issues = validate_task_inputs(task, {"name": "ok", "notes": 123}, {}, path=("G", "T"))
    assert any(i.key == "notes" and "Type mismatch" in i.message for i in issues)

    # Optional field omitted: no issues
    issues2 = validate_task_inputs(task, {"name": "ok"}, {}, path=("G", "T"))
    assert issues2 == []

def test_isinstance_like_notrequired_vs_optional_none():
    class TD(TypedDict):
        a: NotRequired[int]
        b: int | None

    assert isinstance_like({"b": None}, TD)

    assert not isinstance_like({"a": None, "b": 1}, TD)

    assert isinstance_like({"a": 5, "b": None}, TD)
