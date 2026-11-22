"""
What the agent needs to edit a dashboard rather than rebuild it: the
dashboard's state at the start of every user turn, partial changes to a
view's spec, a check that node names say what the nodes hold, and the tile
slot of a node removed and added again.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import re

from collections import OrderedDict
from typing import Any

import sqlglot
import yaml

from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ToolReturn, UserPromptPart,
)

from ..flowdash import (
    DECKGL_ID, FILTER_ID, METRIC_ID, SOURCE_ID, SQL_ID, TABLE_ID,
    TABLE_VIEW_ID, VIEW_ID,
)
from . import _graph

# The most nodes the state block lists; a dashboard with more ends it with a count.
STATE_NODES = 60
# Characters of a SQL node's query and of a text node's text in the state block.
STATE_GIST_CHARS = 90
STATE_TEXT_CHARS = 60
# How the state block starts, which also marks a user turn that has one.
STATE_HEADER = "[Dashboard now]"

KINDS = {
    SQL_ID: "SQL", VIEW_ID: "chart", TABLE_VIEW_ID: "table", DECKGL_ID: "map", METRIC_ID: "metric",
    SOURCE_ID: "source", TABLE_ID: "table node", FILTER_ID: "filter", _graph.HEADING_ID: "heading",
    _graph.MARKDOWN_ID: "markdown", _graph.DIVIDER_ID: "divider",
}

# Marks that annotate a chart rather than plot its data, and the encoding channels
# that style a chart's own marks, which such an annotation layered over it must not inherit.
ANNOTATION_MARKS = frozenset({"rule", "text"})
STYLE_CHANNELS = frozenset({
    "color", "fill", "stroke", "opacity", "fillOpacity", "strokeOpacity", "strokeWidth", "strokeDash",
    "size", "shape", "tooltip",
})


def kind(component: str) -> str:
    """A component as the agent's tools name it, e.g. "chart" for a Lumen Vega-Lite view."""
    return KINDS.get(component, component)


def _cut(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _mark(spec: Any) -> str | None:
    mark = spec.get("mark") if isinstance(spec, dict) else None
    mark = mark.get("type") if isinstance(mark, dict) else mark
    return mark if isinstance(mark, str) else None


def _fields(encoding: Any) -> list[str]:
    if not isinstance(encoding, dict):
        return []
    return [f"{channel}={value['field']}" for channel, value in encoding.items()
            if isinstance(value, dict) and isinstance(value.get("field"), str)]


def _title(spec: dict[str, Any]) -> str | None:
    title = spec.get("title")
    title = title.get("text") if isinstance(title, dict) else title
    return _cut(title, STATE_TEXT_CHARS) if isinstance(title, str) and title.strip() else None


def chart_summary(spec: Any) -> str:
    """A Vega-Lite spec as its marks and encoded fields, layer by layer, and its title."""
    spec = spec if isinstance(spec, dict) else {}
    layers = spec.get("layer") if isinstance(spec.get("layer"), list) else None
    shared = _fields(spec.get("encoding"))
    if layers is None:
        text = f"{_mark(spec) or 'composite'} {', '.join(shared) or 'without fields'}"
    else:
        parts = [f"{_mark(layer) or 'composite'} {' '.join(_fields(layer.get('encoding')))}".strip()
                 for layer in layers if isinstance(layer, dict)]
        text = f"layers {' + '.join(parts) or 'none'}" + (f"; shared {', '.join(shared)}" if shared else "")
    if title := _title(spec):
        text += f"; title {title!r}"
    return text


def spec_summary(component: str, spec: Any) -> str:
    """What a Lumen view's spec shows, in a few words."""
    spec = spec if isinstance(spec, dict) else {}
    if component == VIEW_ID:
        return chart_summary(spec)
    if component == DECKGL_ID:
        layers = [str(layer.get("@@type")) for layer in spec.get("layers") or [] if isinstance(layer, dict)]
        return f"layers {', '.join(layers) or 'none'}"
    options = ", ".join(sorted(spec))
    return f"options {options}" if options else "default options"


def _gist(component: str, config: dict[str, Any]) -> str:
    if component == SQL_ID:
        return _cut(config.get("sql") or "", STATE_GIST_CHARS)
    if component in (VIEW_ID, TABLE_VIEW_ID, DECKGL_ID):
        return spec_summary(component, config.get("spec"))
    if component == METRIC_ID:
        picked = [*(config.get("measures") or []), *(f"by {d}" for d in config.get("dimensions") or [])]
        return f"{config.get('metric')}" + (f": {', '.join(map(str, picked))}" if picked else "")
    if component == TABLE_ID:
        return f"table {config.get('table')!r}"
    if component == FILTER_ID:
        return f"on {config.get('field')!r}"
    if component in _graph.TEXT_FIELDS:
        text = config.get(_graph.TEXT_FIELDS[component]) or ""
        return repr(_cut(text, STATE_TEXT_CHARS)) if text else ""
    return ""


def _rows(tiles: list[dict[str, Any]]) -> dict[str, str]:
    """Where each grid tile shown sits: its row, counted as the grid wraps tiles at 100%, and its size."""
    where, row, used = {}, 0, 100.0
    for tile in tiles:
        width = tile.get("width")
        width = float(width) if isinstance(width, (int, float)) else 100.0
        if used + width > 100.001:
            row, used = row + 1, 0.0
        used += width
        size = f"{width:g}%" + (f", {tile['height']:g}px" if isinstance(tile.get("height"), (int, float)) else "")
        where[tile["node"]] = f"grid row {row}, {size}"
    return where


def dashboard_state(ops) -> str | None:
    """
    The dashboard as the agent sees it at the start of a user turn: its
    title and every node in reading order (top bar, grid tiles row by row,
    sidebar, then the nodes without a visible tile), each with its kind,
    what it holds or shows, the node it reads and where its tile is, or
    why it has none. It depends only on the dashboard, so it is the same
    until the dashboard changes. At most ``STATE_NODES`` nodes are listed.
    None when the dashboard's definition is hidden from the user.
    """
    try:
        doc = yaml.safe_load(ops.get_dashboard()) or {}
    except (PermissionError, OperationError):
        return None
    nodes = doc.get("nodes") or {}
    edges = doc.get("edges") or []
    tiles = [tile for tile in (doc.get("layout") or {}).get("tiles") or [] if tile.get("node") in nodes]
    by_area: dict[str, list[dict]] = {"top": [], "grid": [], "sidebar": []}
    hidden = []
    for tile in tiles:
        if tile.get("visible") is False:
            hidden.append(tile)
        else:
            by_area.get(tile.get("area") or "grid", by_area["grid"]).append(tile)
    where = {tile["node"]: f"top bar {i + 1}" for i, tile in enumerate(by_area["top"])}
    where |= _rows(by_area["grid"])
    where |= {tile["node"]: f"sidebar {i + 1}" for i, tile in enumerate(by_area["sidebar"])}
    order = [tile["node"] for tile in [*by_area["top"], *by_area["grid"], *by_area["sidebar"], *hidden]]
    order += [name for name in nodes if name not in order]

    def reads(name):
        return list(dict.fromkeys(e["from"].split(".")[0] for e in edges if e["to"].split(".")[0] == name))

    def readers(name):
        return list(dict.fromkeys(e["to"].split(".")[0] for e in edges if e["from"].split(".")[0] == name))

    lines = []
    for name in order[:STATE_NODES]:
        node = nodes[name] or {}
        component = node.get("component", "")
        head = kind(component)
        if gist := _gist(component, node.get("config") or {}):
            head += f": {gist}"
        if upstream := reads(name):
            head += f"; reads {', '.join(upstream)}"
        if name in where:
            place = where[name]
        elif component in (VIEW_ID, TABLE_VIEW_ID, DECKGL_ID, *_graph.TEXT_FIELDS):
            place = "HIDDEN tile"
        else:
            place = f"no tile, read by {', '.join(followers)}" if (followers := readers(name)) else "no tile, nothing reads it"
        lines.append(f"- {name} ({head}); {place}")
    if len(order) > STATE_NODES:
        lines.append(f"- and {len(order) - STATE_NODES} more nodes; list_tables lists them.")
    title = doc.get("title") or "Untitled"
    return (f"{STATE_HEADER} Title {title!r}; {len(nodes)} node{'s' * (len(nodes) != 1)} in reading order "
            "(top bar, grid rows, sidebar, then nodes without a visible tile):\n" + "\n".join(lines))


def add_dashboard_state(ctx: RunContext[FlowDashDeps], messages: list[ModelMessage]) -> list[ModelMessage]:
    """
    Put the dashboard's state after the user's prompt at the first request
    of each user turn. It stays in the history as it was sent, so
    every later request, and the next turn, keeps the cached prefix.
    """
    if not messages or not isinstance(last := messages[-1], ModelRequest):
        return messages
    prompts = [i for i, part in enumerate(last.parts) if isinstance(part, UserPromptPart)]
    if not prompts or any(isinstance(p.content, str) and p.content.startswith(STATE_HEADER) for p in last.parts
                          if isinstance(p, UserPromptPart)):
        return messages
    if (state := dashboard_state(ctx.deps.ops)) is None:
        return messages
    parts = list(last.parts)
    parts.insert(prompts[-1] + 1, UserPromptPart(state))
    return [*messages[:-1], dataclasses.replace(last, parts=parts)]


_DELETE = object()


def _merge_mark(old: Any, new: Any) -> Any:
    """A mark changed by ``new``: a type given alone keeps the old mark's properties."""
    if isinstance(new, str) and isinstance(old, dict):
        return {**old, "type": new}
    if isinstance(new, dict) and "type" not in new and isinstance(old, str):
        new = {"type": old, **new}
    return merge_patch(old, new)


def merge_patch(base: Any, patch: Any) -> Any:
    """
    ``patch`` merged into ``base`` as a JSON merge patch (RFC 7386): an
    object merges key by key, None removes a key and anything else
    replaces it, except that a mark given as a type keeps its properties
    and the items of a ``layer`` or ``layers`` list merge by position.
    """
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif key == "mark":
            result[key] = _merge_mark(result.get(key), value)
        elif key in ("layer", "layers") and isinstance(value, list) and isinstance(result.get(key), list):
            old = result[key]
            result[key] = [merge_patch(old[i], item) if i < len(old) else copy.deepcopy(item) for i, item in enumerate(value)]
            result[key] += old[len(value):]
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def _layered(spec: dict[str, Any]) -> dict[str, Any]:
    """
    A single-view chart as a layered one with its mark as the first layer.
    The encoding stays shared, so layers added over it keep its x and y,
    except the channels that style its own marks, and a selection param
    goes with the layer that draws it.
    """
    spec = copy.deepcopy(spec)
    layer: dict[str, Any] = {}
    if "mark" in spec:
        layer["mark"] = spec.pop("mark")
    encoding = spec.pop("encoding", None)
    if isinstance(encoding, dict):
        if own := {k: v for k, v in encoding.items() if k in STYLE_CHANNELS}:
            layer["encoding"] = own
        if shared := {k: v for k, v in encoding.items() if k not in STYLE_CHANNELS}:
            spec["encoding"] = shared
    if "params" in spec:
        layer["params"] = spec.pop("params")
    spec["layer"] = [layer]
    return spec


def merge_chart(base: dict[str, Any], changes: dict[str, Any] | None, add_layers: list[dict] | None = None) -> dict[str, Any]:
    """
    A Vega-Lite spec with ``changes`` merged in (see ``merge_patch``) and
    ``add_layers`` drawn over it. A single-view chart counts as one layer,
    so ``changes["layer"]`` merges its first item into the chart's own mark
    and encoding and adds the others. On a layered chart, a ``mark`` given
    at the top changes the mark of each layer that plots data, and an
    encoding channel changes where the chart defines it: in the shared
    encoding, else in each layer that has it.
    """
    changes = copy.deepcopy(changes or {})
    spec = copy.deepcopy(base)
    if ("layer" in changes or add_layers) and not isinstance(spec.get("layer"), list):
        spec = _layered(spec)
    if isinstance(spec.get("layer"), list):
        mark = changes.pop("mark", _DELETE)
        encoding = changes.pop("encoding", None)
        spec = merge_patch(spec, changes)
        layers = [layer for layer in spec["layer"] if isinstance(layer, dict)]
        if mark is not _DELETE:
            for layer in layers:
                if _mark(layer) not in ANNOTATION_MARKS:
                    layer["mark"] = _merge_mark(layer.get("mark"), mark)
        for channel, value in (encoding or {}).items():
            shared = spec.get("encoding") if isinstance(spec.get("encoding"), dict) else None
            holders = [layer["encoding"] for layer in layers if isinstance(layer.get("encoding"), dict) and channel in layer["encoding"]]
            if (shared is not None and channel in shared) or not holders:
                holders = [spec.setdefault("encoding", {})]
            for holder in holders:
                if value is None:
                    holder.pop(channel, None)
                else:
                    holder[channel] = merge_patch(holder.get(channel), value)
    else:
        spec = merge_patch(spec, changes)
    if add_layers:
        spec["layer"] = [*spec["layer"], *(merge_patch({}, layer) for layer in add_layers)]
    return spec


# Words a node name may use without saying what the node holds.
_GENERIC = frozenset("""
the and for with from per all top new old each over into than this that only kpi kpis node nodes sql data result
results query chart charts table tables map maps view views plot list total totals count counts sum avg average
mean max min number num share rate value values summary overview breakdown latest current last first trend trends
""".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
            if len(w) >= 3 and w not in _GENERIC}


def _alike(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a) or a.rstrip("s") == b.rstrip("s")


def unrelated(name: str, *texts: str) -> bool:
    """Whether ``name`` shares no word with ``texts``, e.g. the columns and query of what it names."""
    words, pool = _words(name), _words(" ".join(texts))
    return bool(words) and bool(pool) and not any(_alike(w, p) for w in words for p in pool)


def name_notes(name: str, columns: list[str], sql: str = "", dialect: str | None = None) -> list[str]:
    """
    Notes when the name of a node, or of a column a query computes, says
    something its contents do not: the node's name shares no word with its
    columns or query, or a computed column's alias shares none with the
    columns it is computed from (``SUM("p_cap") AS electricity_output``).
    """
    notes = []
    if unrelated(name, *columns, sql):
        notes.append(f"the name {name!r} shares no word with its columns ({', '.join(columns[:8])}) or query; "
                     "name a node after what it holds, e.g. its measure and grouping.")
    if not sql:
        return notes
    try:
        select = sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.SqlglotError, ValueError):
        return notes
    for alias in select.expressions if isinstance(select, sqlglot.exp.Select) else []:
        if not isinstance(alias, sqlglot.exp.Alias) or isinstance(alias.this, sqlglot.exp.Column):
            continue
        sources = [column.name for column in alias.this.find_all(sqlglot.exp.Column)]
        if unrelated(alias.alias, *sources):
            notes.append(f"column {alias.alias!r} is computed from {', '.join(dict.fromkeys(sources))}; "
                         "name a column after what it holds.")
    return notes


def append(result: Any, text: str) -> Any:
    """A tool's ``result`` with ``text`` added to what the model reads."""
    if isinstance(result, str):
        return result + text
    if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
        return dataclasses.replace(result, return_value=result.return_value + text)
    return result


def similar(a: str, b: str) -> bool:
    """Whether two node names share most of their words, e.g. a node and its replacement ``sales_chart_gw``."""
    wa, wb = set(a.lower().split("_")) - {""}, set(b.lower().split("_")) - {""}
    shared = {w for w in wa if any(_alike(w, v) for v in wb if min(len(w), len(v)) >= 3)}
    return len(shared) * 2 >= min(len(wa), len(wb))


@dataclasses.dataclass
class _Slot:
    component: str
    area: str
    index: int
    width: float | None
    height: float | None

    def describe(self) -> str:
        if self.area != "grid":
            return f"{'top bar' if self.area == 'top' else self.area} position {self.index}"
        return f"grid position {self.index}" + (f", {self.width:g}% wide" if self.width else "")


def _slot(ops, name: str) -> _Slot | None:
    """Where node ``name``'s visible tile is, or None if it has none."""
    try:
        from panel_flowdash.ui import tile_layout
        instance = ops._instance(name)
        item = ops._item(instance)
        area = tile_layout.area_of(ops.editor, item)
        entry = tile_layout.tile_entry(ops.editor, instance)
        if area is None or entry.get("visible") is False:
            return None
        ids = [i["instance_id"] for i in tile_layout.area_items(ops.editor, area)]
        return _Slot(_graph.component_of(ops, name), area, ids.index(instance), entry.get("width"), entry.get("height"))
    except Exception:
        return None


def _undefer(ops, name: str) -> None:
    """
    Keep the editor from placing node ``name``'s tile again when the agent's
    request ends: during a request FlowDash lays out the tiles it adds only
    then, which would move a tile laid out in the meantime among the others.
    """
    instance = ops._instance(name)
    editor = ops.editor
    if (deferred := getattr(editor, "_deferred_tiles", None)) is not None:
        deferred.discard(instance)
    for pending in (getattr(editor, "_deferred_by_run", None) or {}).values():
        pending.discard(instance)


# Runs whose removed tiles are remembered at once; older runs have ended.
RUNS_REMEMBERED = 64


class KeepTileSlots(AbstractCapability[FlowDashDeps]):
    """
    Keeps the tile slot of a node the agent removes for the node of the
    same kind and a similar name it adds in the same run, e.g. a chart
    removed and placed again under a new name, so replacing a node does
    not drop its tile to the bottom of the grid. ``remove_node``'s result
    says so, and that changing the node in place keeps it anyway.

    Parameters
    ----------
    placing
        The tools that place a node named by their ``name`` argument.
    """

    def __init__(self, placing):
        self.placing = frozenset(placing)
        self._removed: OrderedDict[str, dict[str, _Slot]] = OrderedDict()

    def _run(self, ctx) -> dict[str, _Slot]:
        run = ctx.run_id or ""
        if run not in self._removed:
            self._removed[run] = {}
            while len(self._removed) > RUNS_REMEMBERED:
                self._removed.popitem(last=False)
        return self._removed[run]

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        name = args.get("name") if isinstance(args, dict) else None
        if tool_def.name == "remove_node" and isinstance(name, str):
            slot = _slot(ctx.deps.ops, name)
            result = await handler(args)
            if slot is None or name in _graph.nodes(ctx.deps.ops):
                return result
            self._run(ctx)[name] = slot
            return append(result, (
                f"\nNote: its tile ({slot.describe()}) goes to the next {kind(slot.component)} with a similar name "
                "you add in this turn, and is lost otherwise. To change a node, call its add_* tool with its name "
                "or update_chart instead of removing it."
            ))
        if tool_def.name not in self.placing or not isinstance(name, str):
            return await handler(args)
        result = await handler(args)
        text = result.return_value if isinstance(result, ToolReturn) else result
        removed = self._run(ctx)
        if not (isinstance(text, str) and text.startswith("Placed") and removed):
            return result
        ops = ctx.deps.ops
        try:
            component = _graph.component_of(ops, name)
        except StopIteration:
            return result
        old = next((o for o, slot in removed.items() if slot.component == component and similar(o, name)), None)
        if old is None:
            return result
        slot = removed.pop(old)
        try:
            if slot.area == "grid":
                ops.place_tile(name, index=slot.index, width=slot.width, height=slot.height)
            else:
                ops.place_tile(name, area=slot.area, index=slot.index)
            _undefer(ops, name)
        except (OperationError, PermissionError):
            return result
        return append(result, f" It took the tile of the removed {old!r} ({slot.describe()}).")


def view_summary(component: str, spec: Any) -> str:
    """A view's spec for a tool result: its summary and, for a chart, the spec itself when short."""
    summary = spec_summary(component, spec)
    text = json.dumps(spec, separators=(",", ":"))
    return f"{summary}. Spec: {text}" if len(text) <= 600 else f"{summary}."
