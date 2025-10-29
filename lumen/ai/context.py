from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import (
    Annotated, Any, Literal, TypedDict, get_args, get_origin, get_type_hints,
)


@dataclass(frozen=True)
class AccumulateSpec:
    from_key: str
    # how to extend: either "accumulate" a list or a callable that takes a list of all from_key
    # values and may process them in some way.
    func: str = Callable[[list[Any]], Any] | "extend"
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
        return list(seq)
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
