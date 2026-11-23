"""
Checks of a scenario turn, run on the dashboard the turn left (its graph
YAML and the live outputs of its nodes) and the turn's model responses,
not on what the assistant says it did. Only the claims scenarios, whose
answer is the reply, also check what it says.

A goal check tests what the turn was asked to do. A guard (``guard=True``)
tests that it did no harm, e.g. left no hidden nodes or notes behind, and
passes for an agent that does nothing, so it never earns a turn credit on
its own. A quality check (``quality=True``) tests design and cost.

Each check constructor returns a ``Check`` whose test gets the
``TurnState`` and returns None when it passes, or why it failed.
"""
from __future__ import annotations

import math
import re

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml

from ...ai.tool_trace import ToolCall
from ...flowdash import (
    DECKGL_ID, FILTER_ID, METRIC_ID, SQL_ID, VIEW_ID,
)
from ...flowdash.table_view import TABLE_VIEW_ID

KPI_ID = "Widgets/KPI"
HEADING_ID, MARKDOWN_ID = "Text/Heading", "Text/Markdown"
DATA_COMPONENTS = (SQL_ID, METRIC_ID, "Lumen/table", FILTER_ID, "Lumen/data")

# Kinds a check names, by the components that count as one.
KINDS: dict[str, tuple[str, ...]] = {
    "chart": (VIEW_ID,),
    "table": (TABLE_VIEW_ID,),
    "map": (DECKGL_ID,),
    "kpi": (KPI_ID, "Views/kpi"),
    "heading": (HEADING_ID,),
    "markdown": (MARKDOWN_ID,),
    "sql": (SQL_ID,),
    "metric": (METRIC_ID,),
    "filter": (FILTER_ID,),
}
VIEW_KINDS = ("chart", "table", "map", "kpi")

# Unit factors a dashboard number may be shown at relative to its reference, e.g. kW as MW or GW.
UNIT_SCALES = (1.0, 1e-3, 1e-6, 1e3, 1e6)
PERCENT_SCALES = (1.0, 100.0, 0.01)


@dataclass
class Tile:
    area: str | None
    visible: bool
    width: float | None = None
    height: float | None = None


@dataclass
class TurnState:
    """
    What a check sees of one turn: the dashboard before and after it, the
    nodes it added and changed, its tool calls and model usage, and lookups
    of the live dashboard for node outputs, tiles and reference queries.
    """

    prompt: str
    graph: dict[str, Any]
    before: dict[str, Any]
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    errors: int = 0
    """Tool calls whose result was an error or a refusal."""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    output: str | None = None
    error: str | None = None
    model: str | None = None
    tiles: dict[str, Tile] = field(default_factory=dict)
    grid: list[str] = field(default_factory=list)
    """The visible grid tiles in their order on the dashboard."""
    port: Callable[[str], Any] = lambda port: None
    """The current value of an output port, ``node.port``, or None when it has none."""
    reference: Callable[[str], pd.DataFrame] | None = None
    """Runs SQL on the fixture's own tables, for the numbers a check expects."""

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        return self.graph.get("nodes") or {}

    @property
    def edges(self) -> list[dict[str, str]]:
        return self.graph.get("edges") or []

    @property
    def removed(self) -> list[str]:
        return [name for name in (self.before.get("nodes") or {}) if name not in self.nodes]

    @property
    def refusals(self) -> int:
        return self.errors

    def component(self, name: str) -> str | None:
        return (self.nodes.get(name) or {}).get("component")

    def config(self, name: str) -> dict[str, Any]:
        return (self.nodes.get(name) or {}).get("config") or {}

    def of_kind(self, kind: str, names: Iterable[str] | None = None) -> list[str]:
        components = KINDS.get(kind, (kind,))
        return [n for n in (self.nodes if names is None else names) if self.component(n) in components]

    def visible(self, name: str) -> bool:
        tile = self.tiles.get(name)
        return tile is not None and tile.visible

    def upstream(self, name: str) -> list[str]:
        """The output ports, ``node.port``, feeding any input of ``name``."""
        return [edge["from"] for edge in self.edges if edge["to"].split(".")[0] == name]

    def feeds(self, name: str) -> list[str]:
        return list(dict.fromkeys(edge["to"].split(".")[0] for edge in self.edges if edge["from"].split(".")[0] == name))

    def ancestors(self, name: str) -> set[str]:
        found, queue = set(), [name]
        while queue:
            for port in self.upstream(queue.pop()):
                node = port.split(".")[0]
                if node not in found:
                    found.add(node)
                    queue.append(node)
        return found

    def shown(self, name: str) -> bool:
        """Whether ``name`` is visible or feeds, directly or not, a visible node."""
        queue, seen = [name], set()
        while queue:
            node = queue.pop()
            if node in seen:
                continue
            seen.add(node)
            if self.visible(node):
                return True
            queue.extend(self.feeds(node))
        return False

    def data(self, name: str) -> pd.DataFrame | None:
        """The rows a node outputs, or the rows its pipeline or data input reads for a view."""
        component = self.component(name)
        if component in DATA_COMPONENTS:
            frame = self.port(f"{name}.data")
            return frame if isinstance(frame, pd.DataFrame) else None
        for port in self.upstream(name):
            frame = self.data(port.split(".")[0])
            if frame is not None:
                return frame
        return None

    def kpi_value(self, name: str) -> Any:
        """The number a KPI shows: its wired value, or its literal one."""
        for edge in self.edges:
            if edge["to"] == f"{name}.value":
                return self.port(edge["from"])
        return (self.nodes.get(name) or {}).get("inputs", {}).get("value")


@dataclass(frozen=True)
class Check:
    id: str
    description: str
    test: Callable[[TurnState], str | None]
    workstream: str
    """The plan's workstream that addresses this kind of failure, e.g. ``"W1"``."""
    quality: bool = False
    """A design or cost check (titles, bar counts, request budget), scored apart from what was asked."""
    guard: bool = False
    """A check that the turn did no harm, which an agent that does nothing passes too."""

    def run(self, state: TurnState) -> tuple[bool, str]:
        try:
            reason = self.test(state)
        except Exception as exc:
            reason = f"check raised {type(exc).__name__}: {exc}"
        return reason is None, reason or ""


def parse_graph(text: str) -> dict[str, Any]:
    return yaml.safe_load(text) or {}


# -- numbers ---------------------------------------------------------------------------------------


def _numbers(frame: pd.DataFrame) -> list[pd.Series]:
    return [pd.to_numeric(frame[c], errors="coerce") for c in frame.columns
            if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]


def _close(a: float, b: float, rtol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-9)


# The most rows a result may have for one of its cells to count as a scalar answer: an aggregate,
# not a raw table where some cell matches by chance.
MAX_ANSWER_ROWS = 10


def holds_value(frame: pd.DataFrame | None, value: float, scales: Sequence[float] = UNIT_SCALES,
                rtol: float = 0.005, max_rows: int = MAX_ANSWER_ROWS) -> bool:
    """Whether any numeric cell of ``frame``, a result of at most ``max_rows`` rows, is ``value`` at one of ``scales``."""
    if frame is None or frame.empty or len(frame) > max_rows:
        return False
    targets = [value * scale for scale in scales]
    for column in _numbers(frame):
        for cell in column.dropna().tolist():
            if any(_close(float(cell), target, rtol) for target in targets):
                return True
    return False


def matches_series(frame: pd.DataFrame | None, expected: pd.DataFrame, scales: Sequence[float] = UNIT_SCALES,
                   rtol: float = 0.01) -> bool:
    """
    Whether ``frame`` holds ``expected``'s numbers by its keys: ``expected``
    has key columns first and the value last; some column of ``frame`` holds
    every key's value (as text, so a year matches a year label) and some
    numeric column every value at one scale. Rows ``frame`` leaves out, e.g.
    a top N, are not counted against it; it must hold at least one.
    """
    if frame is None or frame.empty or expected.empty:
        return False
    keys = list(expected.columns[:-1])
    expected_map = {tuple(_norm(v) for v in row[:-1]): float(row[-1]) for row in expected.itertuples(index=False)}
    key_columns = []
    for key in keys:
        wanted = {k[keys.index(key)] for k in expected_map}
        found = next((c for c in frame.columns if frame[c].map(_norm).isin(wanted).mean() > 0.5), None)
        if found is None:
            return False
        key_columns.append(found)
    for series in _numbers(frame):
        if series.name in key_columns:
            continue
        for scale in scales:
            hits = misses = 0
            for row_keys, cell in zip(frame[key_columns].itertuples(index=False), series, strict=False):
                ref = expected_map.get(tuple(_norm(v) for v in row_keys))
                if ref is None or pd.isna(cell):
                    continue
                if _close(float(cell), ref * scale, rtol):
                    hits += 1
                else:
                    misses += 1
            if hits and not misses:
                return True
    return False


def holds_row(frame: pd.DataFrame | None, expected: pd.Series, rtol: float = 1e-6,
              max_rows: int = MAX_ANSWER_ROWS) -> bool:
    """
    Whether ``frame``, of at most ``max_rows`` rows, has a row holding each
    of ``expected``'s values in some column: text by its text, numbers
    within ``rtol``.
    """
    if frame is None or frame.empty or len(frame) > max_rows:
        return False
    numeric = {c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])}
    def found(row, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return any(not pd.isna(row[c]) and _close(float(row[c]), float(value), rtol) for c in numeric)
        return any(_norm(row[c]) == _norm(value) for c in frame.columns)

    return any(all(found(row, value) for value in expected.tolist()) for _, row in frame.iterrows())


def _norm(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value).strip().lower()


def _candidates(state: TurnState, where: str) -> list[str]:
    """Data nodes whose numbers count: those the turn added or changed, or those the dashboard shows."""
    names = [n for n in state.nodes if state.component(n) in DATA_COMPONENTS]
    if where == "turn":
        touched = set(state.added) | set(state.changed)
        return [n for n in names if n in touched or any(v in touched for v in state.feeds(n))]
    if where == "shown":
        return [n for n in names if state.shown(n)]
    return names


# -- check constructors ----------------------------------------------------------------------------


def has_view(kind: str, count: int = 1, *, workstream: str = "W1", id: str | None = None) -> Check:
    """At least ``count`` visible nodes of ``kind``."""
    def test(state):
        shown = [n for n in state.of_kind(kind) if state.visible(n)]
        if len(shown) < count:
            hidden = [n for n in state.of_kind(kind) if not state.visible(n)]
            return f"{len(shown)} visible {kind} node(s), wanted {count}" + (f"; hidden: {hidden}" if hidden else "")
        return None
    return Check(id or f"{kind}_visible", f"{count}+ visible {kind}", test, workstream)


def kpis(count: int, *, top: bool = True, workstream: str = "W1") -> Check:
    """``count`` visible, labelled KPI nodes with a number, in the top bar or the first grid row when ``top``."""
    def test(state):
        found = [n for n in state.of_kind("kpi") if state.visible(n)]
        if len(found) < count:
            others = [n for n in state.added if state.component(n) == VIEW_ID and "text" in str(state.config(n).get("spec", {}).get("mark"))]
            return f"{len(found)} visible KPI node(s), wanted {count}" + (f"; text-mark charts instead: {others}" if others else "")
        unlabelled = [n for n in found if state.config(n).get("label", "KPI") in ("", "KPI")]
        if unlabelled:
            return f"KPIs without a label: {unlabelled}"
        empty = [n for n in found if not isinstance(state.kpi_value(n), (int, float))]
        if empty:
            return f"KPIs without a number: {empty}"
        if top:
            first_row = _rows(state)[0] if state.grid else []
            off = [n for n in found if state.tiles[n].area != "top" and n not in first_row]
            if off:
                return f"KPIs neither in the top bar nor in the first row: {off}"
        return None
    return Check("kpis", f"{count} labelled KPIs{' at the top' if top else ''}", test, workstream)


def kpi_values(*sqls: str | tuple[str, ...], workstream: str = "W4", scales: Sequence[float] = UNIT_SCALES) -> Check:
    """
    Each reference scalar is the number of some KPI (or, without KPIs, of
    some shown one-row result). A tuple of queries gives the readings of an
    ambiguous request, any of which counts, e.g. a share of all rows or of
    the rows with a value.
    """
    def test(state):
        shown = [state.kpi_value(n) for n in state.of_kind("kpi") if state.visible(n)]
        frames = [state.data(n) for n in _candidates(state, "shown")]
        missing = []
        for readings in sqls:
            readings = (readings,) if isinstance(readings, str) else readings
            values = [_scalar(state, sql) for sql in readings]
            if any(isinstance(v, (int, float)) and any(_close(float(v), value * s, 0.005) for s in scales)
                   for v in shown for value in values):
                continue
            if any(f is not None and len(f) == 1 and holds_value(f, value, scales) for f in frames for value in values):
                continue
            missing.append(" or ".join(f"{value:,.6g} ({sql})" for value, sql in zip(values, readings, strict=True)))
        return f"no KPI shows {', '.join(missing)}" if missing else None
    return Check("kpi_values", "KPI numbers match the reference", test, workstream)


def _scalar(state: TurnState, sql: str) -> float:
    frame = state.reference(sql)
    return float(frame.iloc[0, 0])


def shows_value(sql: str, *, where: str = "turn", scales: Sequence[float] = UNIT_SCALES,
                workstream: str = "W4", id: str = "value") -> Check:
    """
    The reference scalar of ``sql`` is in a small result (at most
    ``MAX_ANSWER_ROWS`` rows) of a data node the turn added or changed
    (``where="turn"``), shows (``"shown"``) or anywhere (``"any"``).
    """
    def test(state):
        value = _scalar(state, sql)
        names = _candidates(state, where)
        if any(holds_value(state.data(n), value, scales) for n in names):
            return None
        seen = {n: _preview(state.data(n)) for n in names[:4]}
        return f"{value:,.6g} not in {where} nodes {seen}"
    return Check(id, f"shows {sql}", test, workstream)


def shows_row(sql: str, *, where: str = "shown", workstream: str = "W4", id: str = "row") -> Check:
    """
    The first row of ``sql``, e.g. the record a top N must include, is a
    row of a small result of some data node: each of its values in some
    column, so a result that only holds its number by chance does not count.
    """
    def test(state):
        expected = state.reference(sql).iloc[0]
        names = _candidates(state, where)
        if any(holds_row(state.data(n), expected) for n in names):
            return None
        seen = {n: _preview(state.data(n)) for n in names[:4]}
        return f"no {where} node has the row {expected.tolist()}; saw {seen}"
    return Check(id, f"shows the row of {sql}", test, workstream)


def shows_series(sql: str, *, where: str = "shown", scales: Sequence[float] = UNIT_SCALES,
                 workstream: str = "W4", id: str = "numbers") -> Check:
    """Some data node holds the keyed numbers of ``sql`` (keys first, value last), at one unit scale."""
    def test(state):
        expected = state.reference(sql)
        names = _candidates(state, where)
        if any(matches_series(state.data(n), expected, scales) for n in names):
            return None
        return f"no {where} node matches {_preview(expected)}; saw {{{', '.join(f'{n}: {_preview(state.data(n))}' for n in names[:4])}}}"
    return Check(id, f"numbers match {sql}", test, workstream)


def _preview(frame: pd.DataFrame | None) -> str:
    if frame is None:
        return "None"
    return frame.head(3).to_csv(index=False).strip().replace("\n", " | ")[:160]


def dashboard_title(text: str | None = None, *, workstream: str = "W1") -> Check:
    def test(state):
        title = str(state.graph.get("title") or "")
        if not title or title == "Untitled":
            return f"dashboard title is {title or 'empty'!r}"
        if text is not None and title.strip().lower() != text.lower():
            return f"dashboard title is {title!r}, wanted {text!r}"
        return None
    return Check("title", f"dashboard titled {text or ''}".strip(), test, workstream)


def heading_on_top(*, workstream: str = "W3") -> Check:
    """The dashboard is titled, or a heading is in the top bar or the first grid tile."""
    def test(state):
        if str(state.graph.get("title") or "Untitled") != "Untitled":
            return None
        headings = [n for n in state.of_kind("heading") if state.visible(n)]
        if not headings:
            return "no title and no visible heading"
        if any(state.tiles[n].area == "top" or (state.grid and state.grid[0] == n) for n in headings):
            return None
        return f"heading {headings} is not at the top (grid order {state.grid[:4]}...)"
    return Check("heading_top", "a title at the top", test, workstream)


def charts_titled(*, workstream: str = "W5") -> Check:
    def test(state):
        charts = [n for n in state.of_kind("chart") if state.visible(n)]
        missing = [n for n in charts if not _spec_title(state.config(n).get("spec"))]
        if not charts:
            return "no visible chart"
        return f"charts without a title: {missing}" if missing else None
    return Check("chart_titles", "every chart has a title", test, workstream, quality=True)


def _spec_title(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    title = spec.get("title")
    if isinstance(title, dict):
        title = title.get("text")
    return str(title) if title else None


def _layers(spec: Any) -> list[dict]:
    if not isinstance(spec, dict):
        return []
    layers = [spec]
    for key in ("layer", "concat", "hconcat", "vconcat"):
        for sub in spec.get(key) or []:
            layers.extend(_layers(sub))
    if isinstance(spec.get("spec"), dict):
        layers.extend(_layers(spec["spec"]))
    return layers


def _encodings(spec: Any) -> list[dict]:
    return [layer.get("encoding") or {} for layer in _layers(spec)]


def _mark(layer: dict) -> str | None:
    mark = layer.get("mark")
    return mark.get("type") if isinstance(mark, dict) else mark


def chart_title(name: str | None, text: str, *, workstream: str = "W2") -> Check:
    """The chart ``name`` (or any visible chart) is titled ``text``."""
    def test(state):
        names = [name] if name else [n for n in state.of_kind("chart") if state.visible(n)]
        titles = {n: _spec_title(state.config(n).get("spec")) for n in names if n in state.nodes}
        if any((t or "").strip().lower() == text.lower() for t in titles.values()):
            return None
        return f"titles are {titles}, wanted {text!r}"
    return Check("chart_title", f"chart titled {text!r}", test, workstream)


def legend(*, workstream: str = "W5") -> Check:
    """Every visible chart with more than one series (a colour, or several layers) shows a legend."""
    def test(state):
        bad = []
        charts = [n for n in state.of_kind("chart") if state.visible(n)]
        if not charts:
            return "no visible chart"
        for n in charts:
            spec = state.config(n).get("spec")
            encodings = _encodings(spec)
            color = [e.get("color") for e in encodings if isinstance(e.get("color"), dict)]
            layered = len([layer for layer in _layers(spec) if _mark(layer)]) > 1
            if color and all(c.get("legend", True) is None for c in color):
                bad.append(f"{n}: legend turned off")
            elif layered and not color:
                bad.append(f"{n}: layers without a colour legend")
        return "; ".join(bad) or None
    return Check("legend", "series are named in a legend", test, workstream, quality=True)


def chart_mark(mark: str, name: str | None = None, *, workstream: str = "W2") -> Check:
    def test(state):
        names = [name] if name else [n for n in state.of_kind("chart") if state.visible(n)]
        marks = {n: [_mark(layer) for layer in _layers(state.config(n).get("spec")) if _mark(layer)] for n in names if n in state.nodes}
        if any(mark in m for m in marks.values()):
            return None
        return f"marks are {marks}, wanted {mark}"
    return Check(f"mark_{mark}", f"a {mark} chart", test, workstream)


def axes(name: str, *, x: str, y: str, workstream: str = "W2") -> Check:
    """The chart ``name`` puts field ``x`` on its x axis and ``y`` on its y axis."""
    def test(state):
        for encoding in _encodings(state.config(name).get("spec")):
            fx, fy = (encoding.get("x") or {}).get("field"), (encoding.get("y") or {}).get("field")
            if fx and fy:
                return None if (fx, fy) == (x, y) else f"x={fx}, y={fy}, wanted x={x}, y={y}"
        return f"{name} has no x/y encoding"
    return Check("axes", f"{name} plots {y} against {x}", test, workstream)


# Hue ranges, in degrees, of the colour names a check may ask for.
HUES = {"red": (-20, 15), "orange": (15, 45), "yellow": (45, 70), "green": (70, 170), "blue": (190, 250),
        "purple": (250, 300)}


def _colors(spec: Any) -> list[str]:
    """The colours a Vega-Lite spec draws its marks in: mark properties, value encodings, scale ranges and the mark config."""
    found = []
    for layer in _layers(spec):
        marks = [layer.get("mark")] + [(layer.get("config") or {}).get(key) for key in ("mark", _mark(layer) or "")]
        for mark in marks:
            if isinstance(mark, dict):
                found += [mark[key] for key in ("color", "fill", "stroke") if isinstance(mark.get(key), str)]
        for channel in ("color", "fill", "stroke"):
            enc = (layer.get("encoding") or {}).get(channel)
            if not isinstance(enc, dict):
                continue
            if isinstance(enc.get("value"), str):
                found.append(enc["value"])
            scale_range = (enc.get("scale") or {}).get("range")
            if isinstance(scale_range, list):
                found += [c for c in scale_range if isinstance(c, str)]
    return found


def _is_color(text: str, name: str) -> bool:
    """Whether the CSS colour ``text`` is ``name``, by name (``seagreen`` is green) or by the hue of a hex colour."""
    text = text.strip().lower()
    if name in text:
        return True
    match = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", text)
    if not match or name not in HUES:
        return False
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(d * 2 for d in digits)
    r, g, b = (int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))
    high, low = max(r, g, b), min(r, g, b)
    if high - low < 0.15:
        return False
    if high == r:
        hue = 60 * ((g - b) / (high - low))
    elif high == g:
        hue = 60 * ((b - r) / (high - low) + 2)
    else:
        hue = 60 * ((r - g) / (high - low) + 4)
    hue = hue - 360 if hue > 340 else hue
    start, end = HUES[name]
    return start <= hue < end


def color(name: str, value: str, *, workstream: str = "W2") -> Check:
    """Chart ``name`` draws its marks in ``value``, e.g. ``green``, by a mark colour, value encoding or colour scale."""
    def test(state):
        colors = _colors(state.config(name).get("spec"))
        if colors and all(_is_color(c, value.lower()) for c in colors):
            return None
        return f"{name} draws in {colors or 'the default colour'}, wanted {value}"
    return Check("color", f"{name} is {value}", test, workstream)


def years_unformatted(*, workstream: str = "W5") -> Check:
    """A year field on an axis is not drawn as a quantity with thousands separators (``1,992``)."""
    def test(state):
        bad = []
        charts = [n for n in state.of_kind("chart") if state.visible(n)]
        if not charts:
            return "no visible chart"
        for n in charts:
            for encoding in _encodings(state.config(n).get("spec")):
                for channel in ("x", "y"):
                    enc = encoding.get(channel)
                    if not isinstance(enc, dict) or "year" not in str(enc.get("field", "")).lower():
                        continue
                    axis = enc.get("axis") or {}
                    if enc.get("type") == "quantitative" and not (isinstance(axis, dict) and axis.get("format") in ("d", "%Y", "")) and not enc.get("timeUnit"):
                        bad.append(f"{n}.{channel}")
        return f"years drawn as quantities: {bad}" if bad else None
    return Check("years", "years are not drawn as 1,992", test, workstream, quality=True)


def max_bars(limit: int = 20, *, workstream: str = "W5") -> Check:
    """No visible bar chart draws more than ``limit`` categories."""
    def test(state):
        bad = {}
        bars = [n for n in state.of_kind("chart") if state.visible(n)
                and "bar" in [_mark(layer) for layer in _layers(state.config(n).get("spec"))]]
        if not bars:
            return "no visible bar chart"
        for n in bars:
            frame = state.data(n)
            if frame is not None and len(frame) > limit:
                bad[n] = len(frame)
        return f"bar charts with too many bars: {bad}" if bad else None
    return Check("max_bars", f"at most {limit} bars", test, workstream, quality=True)


def tables_rounded(decimals: int = 2, *, workstream: str = "W5") -> Check:
    """Floats in visible tables are rounded, in the SQL or by a formatter, to at most ``decimals`` places."""
    def test(state):
        bad = []
        tables = [n for n in state.of_kind("table") if state.visible(n)]
        if not tables:
            return "no visible table"
        for n in tables:
            frame = state.data(n)
            formatters = (state.config(n).get("spec") or {}).get("formatters") or {}
            for column in [] if frame is None else frame.columns:
                series = frame[column]
                if column in formatters or not pd.api.types.is_float_dtype(series):
                    continue
                if ((series.dropna() * 10 ** decimals).round(6) % 1 != 0).any():
                    bad.append(f"{n}.{column}")
        return f"unrounded float columns: {bad}" if bad else None
    return Check("rounded", f"table floats rounded to {decimals} places", test, workstream, quality=True)


def filter_on(column: str, *, workstream: str = "W1") -> Check:
    """
    A filter on ``column`` reaches the dashboard's views: a global filter
    in the filter bar, or a Filter node on ``column`` with a widget wired to
    its value, upstream of a visible view.
    """
    def test(state):
        for entry in state.graph.get("filters") or []:
            if entry.get("column") == column:
                return None
        problems = []
        for n in state.of_kind("filter"):
            if state.config(n).get("field") != column:
                continue
            wired = any(e["to"] in (f"{n}.value", f"{n}.selection") for e in state.edges)
            if wired and state.shown(n):
                return None
            problems.append(f"filter node {n} on {column} is {'shown' if state.shown(n) else 'not shown'} "
                            f"and {'wired' if wired else 'has no widget'}")
        for n in state.of_kind("chart"):
            if state.visible(n) and _bound_filter(state.config(n).get("spec"), column):
                return None
        if problems:
            return "; ".join(problems)
        text = [n for n in state.added if state.component(n) in (MARKDOWN_ID, HEADING_ID)]
        return f"no filter on {column}" + (f"; added text instead: {text}" if text else "")
    return Check("filter", f"a filter on {column}", test, workstream)


def _bound_filter(spec: Any, column: str) -> bool:
    """Whether a Vega-Lite spec filters by a selection on ``column`` bound to an input, e.g. a dropdown."""
    for layer in _layers(spec):
        bound = set()
        for param in layer.get("params") or []:
            select = param.get("select") if isinstance(param, dict) else None
            if not isinstance(select, dict) or column not in str(select.get("fields", select.get("encodings", ""))):
                continue
            if param.get("bind") is not None or select.get("bind") is not None:
                bound.add(param.get("name"))
        filters = [t.get("filter") for t in layer.get("transform") or [] if isinstance(t, dict)]
        if bound and any(isinstance(f, dict) and f.get("param") in bound for f in filters):
            return True
    return False


def no_hidden_leftovers(*, workstream: str = "W6") -> Check:
    """Every node the turn added is visible or feeds a visible node."""
    def test(state):
        left = [n for n in state.added if not state.shown(n)]
        return f"hidden nodes left behind: {left}" if left else None
    return Check("no_leftovers", "no hidden nodes left behind", test, workstream, guard=True)


def no_unused(*, workstream: str = "W6") -> Check:
    """No node on the dashboard, besides sources, is hidden and feeds nothing visible."""
    def test(state):
        left = [n for n in state.nodes if state.component(n) != "Lumen/source" and not state.shown(n)]
        return f"unused nodes: {left}" if left else None
    return Check("no_unused", "nothing unused on the dashboard", test, workstream)


def no_duplicates(*, workstream: str = "W2") -> Check:
    """No two visible views show the same thing: equal config over equal data, or KPIs with one label."""
    def test(state):
        seen: dict[str, str] = {}
        dupes = []
        for n in state.nodes:
            if not state.visible(n) or state.component(n) not in sum((KINDS[k] for k in VIEW_KINDS), ()):
                continue
            config = state.config(n)
            if state.component(n) == KPI_ID:
                key = f"kpi:{config.get('label', '').lower()}"
            else:
                sources = sorted(state.config(p.split(".")[0]).get("sql", p) for p in state.upstream(n))
                key = f"{state.component(n)}:{config}:{sources}"
            if key in seen:
                dupes.append((seen[key], n))
            seen[key] = n
        return f"duplicate views: {dupes}" if dupes else None
    return Check("no_duplicates", "no duplicate views", test, workstream, guard=True)


def no_new_nodes(*, kinds: Sequence[str] | None = None, workstream: str = "W6", id: str = "no_new_nodes") -> Check:
    """The turn added no nodes (of ``kinds``)."""
    def test(state):
        added = state.added if kinds is None else [n for k in kinds for n in state.of_kind(k, state.added)]
        return f"added {added}" if added else None
    what = "/".join(kinds) if kinds else "nodes"
    return Check(id, f"no new {what}", test, workstream, guard=True)


def edited_in_place(name: str | None = None, *, kind: str = "chart", workstream: str = "W2") -> Check:
    """
    The turn changed the existing node ``name`` (or an existing ``kind``
    node) and placed no new node of its kind, rather than removing and
    re-creating it.
    """
    def test(state):
        new = state.of_kind(kind, state.added)
        if name is not None:
            if name not in state.nodes:
                return f"{name} was removed" + (f"; placed {new} instead" if new else "")
            if name not in state.changed:
                return f"{name} unchanged" + (f"; placed {new} instead" if new else "")
        elif not state.of_kind(kind, state.changed):
            return f"no existing {kind} changed" + (f"; placed {new} instead" if new else "")
        if new:
            return f"also placed new {kind} {new}"
        return None
    return Check("in_place", f"edits {name or 'the ' + kind} in place", test, workstream)


def removed(name: str, *, workstream: str = "W2") -> Check:
    def test(state):
        return f"{name} is still on the dashboard" if name in state.nodes else None
    return Check(f"removed_{name}", f"{name} removed", test, workstream)


def _rows(state: TurnState) -> list[list[str]]:
    rows, row, used = [], [], 0.0
    for name in state.grid:
        width = state.tiles[name].width or 100.0
        if row and used + width > 100.5:
            rows.append(row)
            row, used = [], 0.0
        row.append(name)
        used += width
    if row:
        rows.append(row)
    return rows


def same_row(*names: str, ordered: bool = True, guard: bool = False, workstream: str = "W3") -> Check:
    """
    The tiles ``names`` are side by side in one grid row, in this order when
    ``ordered``; a ``guard`` when the turn is asked to keep them so.
    """
    def test(state):
        rows = _rows(state)
        for row in rows:
            if all(n in row for n in names):
                order = [row.index(n) for n in names]
                if ordered and order != sorted(order):
                    return f"{names} in one row but in the order {row}"
                return None
        return f"not in one row: rows are {rows}"
    return Check(f"row_{names[0]}", f"{' | '.join(names)} in one row", test, workstream, guard=guard)


def width(name: str, percent: float, *, tolerance: float = 4.0, workstream: str = "W3") -> Check:
    def test(state):
        tile = state.tiles.get(name)
        if tile is None or not tile.visible:
            return f"{name} has no visible tile"
        actual = tile.width or 100.0
        return None if abs(actual - percent) <= tolerance else f"{name} is {actual:g}% wide, wanted {percent:g}%"
    return Check(f"width_{name}", f"{name} {percent:g}% wide", test, workstream)


def same_height(*names: str, workstream: str = "W3") -> Check:
    """The tiles ``names`` are visible and set to one height; tiles at their default height do not count as matched."""
    def test(state):
        hidden = [n for n in names if not state.visible(n)]
        if hidden:
            return f"no visible tile for {hidden}"
        heights = {n: state.tiles[n].height for n in names}
        if None in heights.values():
            return f"heights not set: {heights}"
        return None if len(set(heights.values())) == 1 else f"heights differ: {heights}"
    return Check("same_height", f"{' and '.join(names)} equally tall", test, workstream, guard=True)


def before_views(kind: str = "kpi", *, workstream: str = "W3") -> Check:
    """Every visible ``kind`` tile is in the top bar or before every other visible grid tile."""
    def test(state):
        names = [n for n in state.of_kind(kind) if state.visible(n)]
        if not names:
            return f"no visible {kind}"
        grid = [n for n in state.grid]
        others = [i for i, n in enumerate(grid) if n not in names and state.component(n) not in KINDS["heading"]]
        late = [n for n in names if state.tiles[n].area != "top" and others and grid.index(n) > min(others)]
        return f"{kind} tiles after other tiles: {late}" if late else None
    return Check(f"{kind}_first", f"{kind} tiles first", test, workstream)


def selection_link(chart: str, view: str, *, workstream: str = "W1") -> Check:
    """A click on ``chart`` filters ``view``: the chart's selection drives a filter node upstream of ``view``."""
    def test(state):
        filters = [e["to"].split(".")[0] for e in state.edges if e["from"] == f"{chart}.selection"]
        for name in filters:
            if name in state.ancestors(view):
                return None
        return f"{chart}.selection drives {filters or 'nothing'}, which does not feed {view}"
    return Check("selection", f"{chart} selection filters {view}", test, workstream)


def kpi_units(*units: str, workstream: str = "W1") -> Check:
    """Visible KPIs name each of ``units`` in a label, prefix or suffix."""
    def test(state):
        text = " ".join(
            " ".join(str(state.config(n).get(key, "")) for key in ("label", "prefix", "suffix", "caption"))
            for n in state.of_kind("kpi") if state.visible(n)
        ).lower()
        missing = [u for u in units if not re.search(rf"(^|[\s(]){re.escape(u.lower())}($|[\s)])", text)]
        return f"no KPI shows the unit(s) {missing}: {text!r}" if missing else None
    return Check("units", f"KPIs show {', '.join(units)}", test, workstream)


def titled_node(name: str, text: str, *, workstream: str = "W1") -> Check:
    """Node ``name`` carries ``text`` as a title in its config."""
    def test(state):
        config = state.config(name)
        found = []

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("title", "label", "text") and isinstance(item, str):
                        found.append(item)
                    walk(item)

        walk(config)
        return None if any(t.strip().lower() == text.lower() for t in found) else f"{name} titles: {found}"
    return Check("node_title", f"{name} titled {text!r}", test, workstream)


def multi_series(name: str, *, workstream: str = "W2") -> Check:
    """Chart ``name`` draws more than one series: several marked layers, or a colour field."""
    def test(state):
        layers = [layer for layer in _layers(state.config(name).get("spec")) if _mark(layer)]
        colored = any(isinstance((layer.get("encoding") or {}).get("color"), dict) for layer in layers)
        return None if len(layers) > 1 or colored else f"{name} draws one series"
    return Check("series", f"{name} draws several series", test, workstream)


def unchanged(*, workstream: str = "W6") -> Check:
    def test(state):
        touched = state.added + state.changed + state.removed
        return f"changed the dashboard: {touched}" if touched else None
    return Check("unchanged", "dashboard left as it was", test, workstream, guard=True)


def kept(name: str, *, workstream: str = "W6") -> Check:
    def test(state):
        return None if name in state.nodes and state.visible(name) else f"{name} is gone or hidden"
    return Check(f"kept_{name}", f"{name} still shown", test, workstream, guard=True)


def max_requests(limit: int = 12, *, workstream: str = "W6") -> Check:
    def test(state):
        return None if state.requests <= limit else f"{state.requests} model requests, limit {limit}"
    return Check("requests", f"at most {limit} requests", test, workstream, quality=True)


def no_error(*, workstream: str = "W6") -> Check:
    def test(state):
        return state.error
    return Check("no_error", "the turn finished", test, workstream, guard=True)


# Phrases of a reply that says the assistant cannot do what was asked.
APOSTROPHE = "['\u2019]"
DECLINES = re.compile(
    rf"\b(can{APOSTROPHE}?t|cannot|can not|unable to|not able to|isn{APOSTROPHE}?t (possible|supported|available)|"
    rf"not (possible|supported|available)|no way to|don{APOSTROPHE}?t (have|support)|"
    rf"doesn{APOSTROPHE}?t (have|support))\b",
    re.IGNORECASE,
)


def declines(*, workstream: str = "W6") -> Check:
    """The reply says the assistant cannot do what was asked, for a request no tool can carry out."""
    def test(state):
        text = state.output or ""
        return None if DECLINES.search(text) else f"the reply does not decline: {text[:160]!r}"
    return Check("declines", "says it cannot do this", test, workstream)


def answer_names(sql: str, aliases: dict[str, str] | None = None, *, workstream: str = "W6",
                 id: str = "answer") -> Check:
    """
    The reply names the reference answer, the first cell of ``sql``, or its
    alias, e.g. a state's name for its code, as a whole word.
    """
    def test(state):
        value = str(state.reference(sql).iloc[0, 0])
        words = [value, *([aliases[value]] if aliases and value in aliases else [])]
        text = state.output or ""
        if any(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) for word in words):
            return None
        return f"the reply does not name {' or '.join(words)}: {text[:160]!r}"
    return Check(id, f"the reply names the answer to {sql}", test, workstream)
