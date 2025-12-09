from __future__ import annotations

import sys

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING, Annotated, Any, Literal, NotRequired, Required, TypedDict,
    Union, get_args, get_origin, get_type_hints,
)

if TYPE_CHECKING:
    from .actor import Actor
    from .report import Action

TContext = Mapping[str, Any]


class ContextModel(TypedDict):
    """
    Baseclass for context models, responsible for defining the inputs and outputs
    of an Actor.
    """

@dataclass(frozen=True)
class AccumulateSpec:
    from_key: str
    # how to extend: either "accumulate" a list or a callable that takes a list of all from_key
    # values and may process them in some way.
    func: str = Callable[[list[Any]], Any] | Literal["extend"]
    # provide to remove duplicates; can be "value" or a callable(item)->hashable
    dedupe_by: str | Callable[[Any], Any] | None = None
    # if contexts also supply the accumulator field directly, do we include those too?
    include_target_field: bool = True


def _parse_accumulate_meta(annotation: Any) -> AccumulateSpec | None:
  """
  Parse the accumulate metadata from an Annotated field.

  Parameters
  ----------
  annotation: Annotated[Any, ...]
    The Annotated field to parse.

  Returns
  -------
  AccumulateSpec | None
    The accumulate specification, or None if the field is not an Annotated field.

  """
  if get_origin(annotation) is not Annotated:
      return None
  base, *meta = get_args(annotation)
  # accept either a tuple ("accumulate", "source"), a Callabe or an AccumulateSpec
  for m in meta:
      if isinstance(m, AccumulateSpec):
          return m
      elif isinstance(m, tuple) and len(m) >= 2:
          # ("accumulate", "source", {optional kwargs...})
          if m[0] == "accumulate":
              func = list
          else:
              func = m[0]
          from_key = m[1]
          kwargs = m[2] if len(m) > 2 else {}
          return AccumulateSpec(from_key=from_key, func=func, **kwargs)

def _dedupe(seq: Iterable[Any], key: str | Callable[[Any], Any] | None) -> list[Any]:
    """
    Deduplicate a sequence of items based on a key.

    Parameters
    ----------
    seq: Iterable[Any]
      The sequence to deduplicate.
    key: str | Callable[[Any], Any] | None
      The key to deduplicate by.

    Returns
    -------
    list[Any]
    The deduplicated sequence.

    Examples
    --------
    >>> _dedupe([1, 2, 3, 2, 1], "value")
    [1, 2, 3]
    """
    if key is None:
        new = []
        for v in seq:
            if v not in new:
                new.append(v)
        return new
    key_fn: Callable[[Any], Any]
    if key == "value":
        key_fn = lambda x: x
    elif callable(key):
        key_fn = key  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown dedupe_by: {key!r}")
    seen = set()
    out = []
    for item in seq:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


class LWW(TypedDict):
    """
    A dictionary that overrides the last-write-wins (LWW) behavior.
    """
    __override_lww__: Literal[True]


def merge_contexts(
    schema: type[TypedDict],
    contexts: list[Mapping[str, Any]],
    *,
    lww_keys: Iterable[str] | None = None,   # override keys for last-write-wins
) -> dict[str, Any]:
    """
    Merge a list of context dicts according to the schema's Annotated metadata.
      - Accumulated fields: extend/append from 'from_key' occurrences across contexts
      - Regular fields: last-write-wins (LWW) unless overridden by lww_keys

    Parameters
    ----------
    schema: type[TypedDict]
      The schema to merge the contexts according to.
    contexts: list[Mapping[str, Any]]
      The list of contexts to merge.

    Returns
    -------
    dict[str, Any]
      The merged context.
    """
    annotations: dict[str, Any] = get_type_hints(schema, include_extras=True)
    result: dict[str, Any] = {}

    # Pre-scan accumulators from schema
    accumulators: dict[str, AccumulateSpec] = {}
    for field, ann in annotations.items():
        spec = _parse_accumulate_meta(ann)
        if spec:
            accumulators[field] = spec

    for ctx in contexts:
        for k, v in ctx.items():
            if k in accumulators:
                continue
            if v is not None:
                result[k] = v

    for target_field, spec in accumulators.items():
        payloads = []
        for ctx in contexts:
            if spec.from_key not in ctx or ctx[spec.from_key] is None:
                continue
            payload = ctx[spec.from_key]
            if spec.include_target_field and target_field in ctx and (tgt_value:= ctx[target_field]) is not None:
                if isinstance(tgt_value, list):
                    payloads.extend(tgt_value)
                else:
                    payloads.append(tgt_value)
            payloads.append(payload)

        bucket = spec.func(payloads)
        bucket = _dedupe(bucket, spec.dedupe_by)
        if bucket:
            result[target_field] = bucket
        elif spec.func == "accumulate":
            result.setdefault(target_field, [])

    return result


@dataclass
class ValidationIssue:
    path: str | tuple[str, ...] | list[str]
    key: str
    expected: Any
    actual: Any | None
    message: str
    severity: str = "error"


def resolved_annotations(tp: type) -> dict[str, Any]:
    """Resolve ForwardRefs and preserve Annotated extras."""
    mod = sys.modules[tp.__module__]
    return get_type_hints(tp, globalns=vars(mod), localns=None, include_extras=True)


def unwrap_annotated(tp: Any) -> tuple[Any, list[Any]]:
    """Return (inner_type, metadata[]) for Annotated, else (tp, [])."""
    if get_origin(tp) is Annotated:
        base, *meta = get_args(tp)
        return base, list(meta)
    return tp, []


_NOTREQ_ORIGIN = getattr(NotRequired, "__origin__", NotRequired)  # defensive
_REQ_ORIGIN = getattr(Required, "__origin__", Required)

def unwrap_field_type(tp: Any) -> Any:
    """Unwrap Annotated, NotRequired, and Required to get the inner type."""
    # Unwrap Annotated
    if get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    # Unwrap NotRequired / Required (PEP 655)
    origin = get_origin(tp)
    if origin in (_NOTREQ_ORIGIN, _REQ_ORIGIN):
        tp = get_args(tp)[0]
    return tp

def is_typed_dict(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, dict) and hasattr(tp, "__required_keys__")

def schema_fields(schema: type[TypedDict]) -> dict[str, dict[str, Any]]:
    """
    { key: { 'type': <inner type>, 'required': bool, 'meta': [...] } }
    """
    ann = resolved_annotations(schema)
    req = getattr(schema, "__required_keys__", set())
    out: dict[str, dict[str, Any]] = {}
    for k, tp in ann.items():
        base, meta = unwrap_annotated(tp)
        base = unwrap_field_type(base)
        out[k] = {"type": base, "required": (k in req), "meta": meta}
    return out

def _accumulate_from_key(meta: list[Any]) -> str | None:
    for m in meta:
        if isinstance(m, tuple) and len(m) >= 2 and m[0] == "accumulate":
            return m[1]
    return None

def input_dependency_keys(schema: type[TypedDict]) -> set[str]:
    """
    Keys that, if invalidated, require rerunning a task with this inputs schema.
    Includes the field names themselves PLUS any ('accumulate','<from_key>') sources.
    """
    deps: set[str] = set()
    # include_extras=True keeps Annotated metadata
    ann = get_type_hints(schema, include_extras=True)
    for key, tp in ann.items():
        deps.add(key)
        if get_origin(tp) is Annotated:
            _, *meta = get_args(tp)
            src = _accumulate_from_key(meta)
            if src:
                deps.add(src)
    return deps

def isinstance_like(value: Any, tp: Any) -> bool:
    """A pragmatic structural checker for common typing forms."""
    tp = unwrap_field_type(tp)

    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return True
    if origin is Union:  # includes Optional
        return any(isinstance_like(value, alt) for alt in args)
    if is_typed_dict(tp):
        if not isinstance(value, Mapping):
            return False
        ann = resolved_annotations(tp)
        req = getattr(tp, "__required_keys__", set())
        # required keys present?
        for k in req:
            if k not in value:
                return False
            if not isinstance_like(value[k], ann[k]):
                return False
        # optional keys (if present) must match
        opt = getattr(tp, "__optional_keys__", set())
        for k in (value.keys() & opt):
            if not isinstance_like(value[k], ann[k]):
                return False
        # ignore extra keys
        return True
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            return False
        if not args:
            return True
        (elem_type,) = args if len(args) == 1 else (args[0],)
        return all(isinstance_like(v, elem_type) for v in value)
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            return False
        if len(args) == 2:
            kt, vt = args
            return all(isinstance_like(k, kt) and isinstance_like(v, vt) for k, v in value.items())
        return True
    if isinstance(tp, type):
        return isinstance(value, tp)
    # Fallback: accept
    return True


def collect_task_outputs(task: Action | Actor) -> dict[str, Any]:
    """
    Returns {key: type} for what this task guarantees to add/enrich in context,
    based on task.output_schema (a TypedDict schema).
    """
    out_schema = task.output_schema
    fields = schema_fields(out_schema)
    return {k: f["type"] for k, f in fields.items()}


def _base_type(tp: Any) -> Any:
    tp = unwrap_field_type(tp)
    return get_args(tp)[0] if get_origin(tp) is Annotated else tp


def types_compatible(expected: Any, produced: Any) -> bool:
    """
    Is a value of 'produced' type usable where 'expected' is required?
    (Permissive but practical; handles Annotated, Union/Optional, list/dict/Mapping, TypedDict.)
    """
    expected = unwrap_field_type(_base_type(expected))
    produced = unwrap_field_type(_base_type(produced))

    if expected is Any or produced is Any:
        return True

    if get_origin(expected) is Union:
        return any(types_compatible(alt, produced) for alt in get_args(expected))

    e_origin, p_origin = get_origin(expected), get_origin(produced)
    e_args, p_args = get_args(expected), get_args(produced)

    if e_origin and p_origin:
        if e_origin in (list, tuple) and p_origin in (list, tuple):
            e_elem = e_args[0] if e_args else Any
            p_elem = p_args[0] if p_args else Any
            return types_compatible(e_elem, p_elem)

        if e_origin in (dict, Mapping) and p_origin in (dict, Mapping):
            ek = e_args[0] if len(e_args) == 2 else Any
            ev = e_args[1] if len(e_args) == 2 else Any
            pk = p_args[0] if len(p_args) == 2 else Any
            pv = p_args[1] if len(p_args) == 2 else Any
            return types_compatible(ek, pk) and types_compatible(ev, pv)

        return False

    try:
        if isinstance(expected, type) and issubclass(expected, dict) and hasattr(expected, "__required_keys__"):
            if not (isinstance(produced, type) and issubclass(produced, dict) and hasattr(produced, "__required_keys__")):
                return True
            exp_ann = get_type_hints(expected, include_extras=True)
            prod_ann = get_type_hints(produced, include_extras=True)
            for k in expected.__required_keys__:  # type: ignore[attr-defined]
                if k not in prod_ann or not types_compatible(exp_ann[k], prod_ann[k]):
                    return False
            opt = getattr(expected, "__optional_keys__", set())
            for k in (opt & prod_ann.keys()):
                if k in exp_ann and not types_compatible(exp_ann[k], prod_ann[k]):
                    return False
            return True
    except Exception:
        pass

    if isinstance(expected, type) and isinstance(produced, type):
        return issubclass(produced, expected)

    return True

def _list_elem_type(tp: Any) -> Any | None:
    """If tp is list[X] (or Annotated[list[X], ...]), return X; else None."""
    tp = unwrap_field_type(tp)
    if get_origin(tp) is list:
        args = get_args(tp)
        return args[0] if args else Any
    return None

def _tname(tp: Any) -> str:
    try:
        return getattr(tp, "__name__", str(tp))
    except Exception:
        return str(tp)

def _accumulate_alt_key(meta: list[Any]) -> str | None:
    for m in meta:
        if isinstance(m, tuple) and len(m) >= 2 and m[0] == "accumulate":
            return m[1]
    return None

def validate_task_inputs(
    task: Action | Actor, value_ctx: Mapping[str, Any], available_types: dict[str, Any], path: str
) -> list[ValidationIssue]:
    """
    Validates that task.input_schema are satisfied by value_ctx or prior available_types.
    """
    issues: list[ValidationIssue] = []
    schema = task.input_schema

    if not schema:
        return issues

    fields = schema_fields(schema)
    for key, spec in fields.items():
        expected = spec["type"]
        required = spec["required"]
        meta = spec["meta"]
        alt_key = _accumulate_alt_key(meta)

        # Concrete value present in context
        if key in value_ctx:
            val = value_ctx[key]
            if not isinstance_like(val, expected):
                issues.append(ValidationIssue(
                    path=path, key=key, expected=expected, actual=type(val),
                    message=f"Type mismatch for '{key}'. Expected {_tname(expected)}, got {_tname(type(val))}."
                ))
            continue

        # No value; does upstream promise a *compatible* type
        if key in available_types:
            produced = available_types[key]
            if not types_compatible(expected, produced):
                issues.append(ValidationIssue(
                    path=path, key=key, expected=expected, actual=produced,
                    message=(f"Incompatible upstream type for '{key}': "
                             f"expected {_tname(expected)} but pipeline provides {_tname(produced)}.")
                ))
            else:
                continue  # satisfied by upstream type

        # Does the schema provide an accumulator
        if alt_key:
            elem = _list_elem_type(expected)  # only special-case when expected is list[Elem]

            # 3a) Concrete value under alt_key?
            if alt_key in value_ctx:
                val = value_ctx[alt_key]
                if elem is not None:
                    # Accept single Elem when accumulating into list[Elem]
                    if isinstance_like(val, elem):
                        continue
                # Otherwise fall back to normal check against full expected
                if isinstance_like(val, expected):
                    continue
                issues.append(ValidationIssue(
                    path=path, key=key, expected=expected, actual=type(val),
                    message=(f"Type mismatch via accumulator '{alt_key}' for '{key}': "
                             f"expected {_tname(expected)}, got {_tname(type(val))}.")
                ))
                continue

            # 3b) Upstream type promised under alt_key?
            if alt_key in available_types:
                produced = available_types[alt_key]
                if elem is not None:
                    # Accept Elem (or compatible) when target expects list[Elem]
                    if types_compatible(elem, produced):
                        continue
                # Otherwise require full compatibility with expected
                if types_compatible(expected, produced):
                    continue
                issues.append(ValidationIssue(
                    path=path, key=key, expected=expected, actual=produced,
                    message=(f"Incompatible upstream type via accumulator '{alt_key}' for '{key}': "
                             f"expected {_tname(expected)} but pipeline provides {_tname(produced)}.")
                ))
                continue

        # 4) Still unsatisfied
        if required:
            issues.append(ValidationIssue(
                path=path, key=key, expected=expected, actual=None,
                message=f"Missing required key '{key}'."
            ))
    return issues

def _normalize_path(path: str | tuple[str, ...] | list[str]) -> list[str]:
    """
    Normalize a path to a list of segments.

    Supports:
      - tuple/list of strings: ('TaskGroup', 'Step[0]', 'Summarize')
      - string with ' -> ' delimiters: 'TaskGroup[0] -> Summarize'
    """
    if isinstance(path, (tuple, list)):
        return [str(p) for p in path]
    # String form: split on '->'
    parts = [seg.strip() for seg in str(path).split("->")]
    # Keep '[idx]' attached if present; callers can pass nicer tuples to avoid parsing.
    return parts

def _pretty_type(tp: Any) -> str:
    """Conservative type pretty-printer."""
    try:
        return getattr(tp, "__name__", str(tp))
    except Exception:
        return str(tp)


class Node:
    __slots__ = ("children", "leaf_issues")
    def __init__(self):
        self.children: dict[str, Node] = {}
        self.leaf_issues: list[ValidationIssue] = []


def render_issues_tree(issues: list[ValidationIssue], *, title: str = "Validation errors") -> str:
    """
    Build a tree like:

    Validation errors
    ├─ TaskGroup
    │  └─ Step[0] -> Summarize
    │     ├─ summary  [error] Type mismatch: expected str, got int
    │     └─ sources  [error] Missing required key 'sources'
    └─ TaskGroup[1] -> Report
       └─ report   [error] Missing required key 'report'
    """
    # Build a nested tree: dict[str, node]; each node has children dict + issues list

    root = Node()

    for issue in issues:
        segs = _normalize_path(issue.path)
        node = root
        for seg in segs:
            node = node.children.setdefault(seg, Node())
        node.leaf_issues.append(issue)

    lines: list[str] = []

    def walk(node: Node, prefix: str = "", is_last: bool = True, label: str | None = None):
        branch = "└─ " if is_last else "├─ "
        child_prefix = prefix + ("   " if is_last else "│  ")

        if label is not None:
            lines.append(prefix + branch + label)

        # Collect children sorted for stable output
        items = list(node.children.items())
        for i, (name, child) in enumerate(items):
            walk(child, child_prefix, i == len(items) - 1, name)

        # Leaf issues under this node
        if node.leaf_issues:
            # If this node also had children, we still print issues as leaves
            for j, iss in enumerate(node.leaf_issues):
                leaf_branch = "└─ " if j == len(node.leaf_issues) - 1 else "├─ "
                # one-line summary per issue
                exp = _pretty_type(iss.expected)
                act = _pretty_type(iss.actual) if iss.actual is not None else "∅"
                lines.append(
                    (child_prefix if label is not None else prefix)
                    + leaf_branch
                    + f"{iss.key}  [{iss.severity}] {iss.message}"
                    + (f" (expected {exp}, got {act})" if "expected" not in iss.message.lower() else "")
                )

    top_items = list(root.children.items())
    for i, (name, child) in enumerate(top_items):
        walk(child, "", i == len(top_items) - 1, name)

    if root.leaf_issues:
        lines.append("└─ (root)")
        for j, iss in enumerate(root.leaf_issues):
            leaf_branch = "└─ " if j == len(root.leaf_issues) - 1 else "├─ "
            exp = _pretty_type(iss.expected)
            act = _pretty_type(iss.actual) if iss.actual is not None else "∅"
            lines.append(f"   {leaf_branch}{iss.key}  [{iss.severity}] {iss.message} (expected {exp}, got {act})")
    tree = "\n".join(lines)
    return f"{title}\n```\n{tree}\n```"


class ContextError(RuntimeError):
    """
    Raised when context validation fails.
    Wraps a list of ValidationIssue objects and renders them
    as a readable ASCII tree in str().
    """

    def __init__(self, issues: list[ValidationIssue], *, title: str = "Context validation failed"):
        self.issues = issues
        self.title = title
        # build the rendered message once for efficiency
        self._message = render_issues_tree(issues, title=title)
        super().__init__(self._message)

    def __str__(self) -> str:
        return self._message

    def __repr__(self) -> str:
        return f"<ContextError: {len(self.issues)} issue(s)>"

    def summary(self) -> str:
        """
        Short one-line summary suitable for logs.
        """
        n_errors = sum(1 for i in self.issues if i.severity == "error")
        n_warnings = sum(1 for i in self.issues if i.severity == "warning")
        return f"{self.title}: {n_errors} error(s), {n_warnings} warning(s)"


def _class_name(obj: Any) -> str:
    return obj.__class__.__name__


def _normalize_not_with(obj: Any) -> set[str]:
    """
    Accepts:
      - missing/None -> empty set
      - iterable of strings and/or types -> convert to class-name strings
    """
    raw = getattr(obj, "not_with", None)
    if not raw:
        return set()
    names: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, type):
            names.add(item.__name__)
        else:
            names.add(getattr(item, "__name__", str(item)))
    return names


def validate_taskgroup_exclusions(group, *, path: str = "TaskGroup") -> list[ValidationIssue]:
    """
    Validate that within this TaskGroup there are no tasks that exclude each other.
    A conflict occurs if task_i.not_with contains task_j's CLASS NAME, or vice versa.

    Recurses into nested TaskGroups, but only checks conflicts *within* each group,
    not across siblings or parents.
    """
    issues: list[ValidationIssue] = []

    tasks = list(group)

    table: list[tuple[int, Any, str, set[str]]] = [
        (idx, t, _class_name(t), _normalize_not_with(t))
        for idx, t in enumerate(tasks)
    ]

    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(table)):
        idx_i, task_i, name_i, not_with_i = table[i]
        for j in range(i + 1, len(table)):
            idx_j, task_j, name_j, not_with_j = table[j]

            i_blocks_j = name_j in not_with_i
            j_blocks_i = name_i in not_with_j
            if not (i_blocks_j or j_blocks_i):
                continue

            pair_key = tuple(sorted((name_i, name_j)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            tname_i = getattr(task_i, "name", name_i)
            tname_j = getattr(task_j, "name", name_j)

            reasons = []
            if i_blocks_j:
                reasons.append(f"{name_i}.not_with contains '{name_j}'")
            if j_blocks_i:
                reasons.append(f"{name_j}.not_with contains '{name_i}'")
            reason_txt = " and ".join(reasons)
            key = f"{name_i} × {name_j}"  # noqa: RUF001
            issues.append(ValidationIssue(
                path=path,
                key=key,
                expected=None,
                actual=None,
                message=(
                    f"Mutually exclusive tasks selected: "
                    f"'{tname_i}' (index {idx_i}) and '{tname_j}' (index {idx_j}). "
                    f"Conflict because {reason_txt}. Remove one of them."
                ),
                severity="error",
            ))

    return issues
