"""
Lumen ``Analysis`` classes (``lumen/ai/analysis.py``) as a FlowDash node.

``AnalysisNode`` wraps host-registered ``Analysis`` subclasses or instances by
composition without modifying ``lumen/ai/analysis.py``. The host hands the
editor the analyses a dashboard may run under ``ANALYSES_KEY``
(``"lumen.analyses"``) in the editor's context (or via ``bind_analysis``), and a
node only names one of them by ``analysis``; it never imports classes from
dashboard YAML.

Depending on what ``Analysis.__call__`` returns, the node populates either its
``result`` (pipeline) and ``data`` outputs (for a ``Pipeline`` or ``DataFrame``)
or its ``view`` output and tile body (for a Lumen ``View``, Panel ``Viewable``,
or renderable object).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import weakref

from collections.abc import Mapping, Sequence
from concurrent.futures import (
    ThreadPoolExecutor, TimeoutError as FutureTimeout,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import panel as pn
import param

from panel_flowdash import dashboard_context, register
from panel_flowdash.dataflow_engine import failure_text
from panel_flowdash.util import is_failure

from ..pipeline import Pipeline
from ..sources.base import InMemorySource
from ..util import as_pandas, get_dataframe_schema, is_narwhals
from .checks import CheckDiagnostic, pipeline_columns
from .choices import palette_presets
from .nodes import (
    NODE_TIMEOUT, PIPELINE_TYPE, PREVIEW_ROWS, RESULT_LIMIT,
    GlobalFilterTarget, LumenNode, _clone, _pipeline_preview, _preview_frame,
    fetch, row_ceiling,
)

if TYPE_CHECKING:
    from ..ai.analysis import Analysis

ANALYSIS_ID = "Lumen/analysis"
ANALYSES_KEY = "lumen.analyses"

logger = logging.getLogger(__name__)

# Live AnalysisNodes per editor, refreshed when ``bind_analysis`` binds an analysis by name.
_ANALYSIS_NODES: weakref.WeakKeyDictionary[Any, weakref.WeakSet] = weakref.WeakKeyDictionary()

# Default timeout (seconds) for running an async Analysis hook synchronously from ``_resolve``.
ANALYSIS_TIMEOUT = 30


def _analysis_base() -> type[Analysis]:
    """Import ``Analysis`` lazily so importing ``lumen.flowdash`` needs no AI extras."""
    from ..ai.analysis import Analysis

    return Analysis


def _is_analysis(value: Any) -> bool:
    base = _analysis_base()
    return isinstance(value, base) or (isinstance(value, type) and issubclass(value, base))


def _entry_name(entry: Any) -> str:
    name = getattr(entry, "name", "")
    if isinstance(name, str) and name:
        return name
    return entry.__name__ if isinstance(entry, type) else type(entry).__name__


def _normalize_analyses(raw: Any) -> dict[str, Any]:
    """Normalize a mapping or sequence of ``Analysis`` classes/instances into ``{name: entry}``."""
    if not raw:
        return {}
    if isinstance(raw, Mapping):
        entries = [(str(name), entry) for name, entry in raw.items() if entry is not None]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = [(_entry_name(entry), entry) for entry in raw if entry is not None]
    else:
        return {}
    valid: dict[str, Any] = {}
    for name, entry in entries:
        if not name or not _is_analysis(entry):
            logger.warning("Ignoring non-Analysis entry %r in %s.", name, ANALYSES_KEY)
            continue
        valid[name] = entry
    return valid


def host_analyses(context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    The ``Analysis`` classes or instances the host lets a dashboard run, by
    name, from ``context`` or else ``dashboard_context()``.
    """
    context = dashboard_context() if context is None else context
    raw = context.get(ANALYSES_KEY)
    if not raw:
        return {}
    return _normalize_analyses(raw)


def bound_analyses(editor: Any) -> dict[str, Any]:
    """The analyses bound in ``editor``'s context, by name."""
    return host_analyses(editor.context)


def bind_analysis(editor: Any, analysis: Any, name: str | None = None) -> str:
    """
    Let ``Lumen/analysis`` nodes of ``editor`` run ``analysis`` under ``name``
    (defaulting to ``analysis.name``), refreshing any existing nodes naming it.
    """
    if not _is_analysis(analysis):
        raise TypeError(f"Expected a Lumen Analysis subclass or instance, got {type(analysis).__name__}.")
    key = name or _entry_name(analysis)
    existing = host_analyses(editor.context)
    existing[key] = analysis
    editor.context[ANALYSES_KEY] = existing
    for node in list(_ANALYSIS_NODES.get(editor, ())):
        if node.analysis == key:
            node._refresh()
    return key


def unbind_analysis(editor: Any, name: str) -> None:
    """Forget an analysis bound with ``bind_analysis``."""
    existing = host_analyses(editor.context)
    if name in existing:
        existing.pop(name, None)
        editor.context[ANALYSES_KEY] = existing


def missing_analysis(name: str, analyses: Mapping[str, Any]) -> str:
    """What an analysis node reports when the host provides only ``analyses``."""
    if analyses:
        return f"The app defines no analysis `{name}`; its analyses are: {', '.join(f'`{a}`' for a in sorted(analyses))}."
    return 'The app provides no Lumen analyses. Pass `analyses=[...]` to `build_routes`, or set `editor_options = {"context": {"lumen.analyses": {...}}}`.'


def analysis_param_names(entry: Any) -> list[str]:
    """
    The user-configurable parameter names of ``entry``, matching
    ``Analysis.controls()``: parameters defined on the subclass with
    non-negative precedence, excluding base ``Analysis`` internals.
    """
    base = _analysis_base()
    return [p for p in entry.param if p not in base.param and (entry.param[p].precedence is None or entry.param[p].precedence >= 0)]


def analysis_doc(entry: Any) -> str:
    """The cleaned docstring of an ``Analysis`` class or instance, or empty string."""
    cls = entry if isinstance(entry, type) else type(entry)
    doc = inspect.getdoc(cls) or ""
    if doc == inspect.getdoc(_analysis_base()):
        return ""
    return doc


def analysis_param_summary(entry: Any) -> list[str]:
    """Human-readable descriptions of each configurable parameter on ``entry``."""
    names = analysis_param_names(entry)
    cls = entry if isinstance(entry, type) else type(entry)
    is_instance = not isinstance(entry, type)
    field_params = set(getattr(cls, "_field_params", ()))
    summaries: list[str] = []
    for pname in names:
        pobj = entry.param[pname]
        parts: list[str] = []
        if pobj.doc:
            parts.append(" ".join(pobj.doc.split()))
        kind = type(pobj).__name__.lower()
        if pname in field_params:
            parts.append("column name")
        else:
            parts.append(kind)
        default = getattr(entry, pname) if is_instance else pobj.default
        if default is not None:
            parts.append(f"default {default!r}")
        if isinstance(pobj, param.Selector) and pobj.objects and pname not in field_params:
            opts = list(pobj.objects)[:10]
            parts.append(f"choices {opts}")
        if isinstance(pobj, (param.Number, param.Integer)) and pobj.bounds:
            lo, hi = pobj.bounds
            if lo is not None or hi is not None:
                parts.append(f"bounds [{lo}, {hi}]")
        summaries.append(f"{pname} ({'; '.join(parts)})")
    return summaries


def missing_columns(entry: Any, columns: Sequence[str]) -> list[str]:
    """Required column specs of ``entry`` that are not present in ``columns``."""
    col_set = set(columns)
    missing: list[str] = []
    for spec in getattr(entry, "columns", ()) or ():
        if isinstance(spec, tuple):
            if not any(c in col_set for c in spec):
                missing.append("one of (" + ", ".join(repr(c) for c in spec) + ")")
        elif spec not in col_set:
            missing.append(repr(spec))
    return missing


def build_analysis(
    entry: Any,
    params: Mapping[str, Any] | None = None,
    *,
    name: str | None = None,
    columns: Sequence[str] | None = None,
) -> Analysis:
    """
    Build a fresh ``Analysis`` instance from ``entry`` (class or configured
    instance), populating ``_field_params`` with ``columns`` and validating
    ``params`` against the subclass's declared public parameters.
    """
    label = name or _entry_name(entry)
    if params is not None and not isinstance(params, Mapping):
        raise ValueError(f"Parameters for analysis {label!r} must be a mapping, got {type(params).__name__}.")
    allowed = analysis_param_names(entry)
    given = dict(params or {})
    unknown = [k for k in given if k not in allowed]
    if unknown:
        hint = f"Its parameters are: {', '.join(allowed)}." if allowed else f"Analysis {label!r} takes no parameters."
        raise ValueError(f"Analysis {label!r} has no parameter {', '.join(map(repr, unknown))}. {hint}")

    cls = entry if isinstance(entry, type) else type(entry)
    instance = cls.instance()

    # Populate column-selector choices before applying params so Selector validation passes.
    if columns is not None:
        col_list = list(columns)
        for field in getattr(cls, "_field_params", ()) or ():
            if field in instance.param:
                instance.param[field].objects = col_list

    # Preserve non-default values configured on a host-registered Analysis instance.
    if not isinstance(entry, type):
        for pname in allowed:
            if pname in given:
                continue
            val = getattr(entry, pname)
            if val != cls.param[pname].default:
                try:
                    setattr(instance, pname, val)
                except Exception:
                    pass

    for key, value in given.items():
        try:
            setattr(instance, key, value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid value for parameter {key!r} of analysis {label!r}: {exc}") from exc
    return instance


def _run_coroutine_sync(coro: Any, timeout: float = ANALYSIS_TIMEOUT) -> Any:
    """Run ``coro`` to completion from synchronous code without blocking an active loop."""
    pool = ThreadPoolExecutor(1)
    future = pool.submit(asyncio.run, coro)
    try:
        return future.result(timeout)
    except FutureTimeout:
        raise TimeoutError(f"Analysis did not finish within {timeout} seconds.") from None
    finally:
        pool.shutdown(wait=False)


async def check_analysis_applies(entry: Any, pipeline: Pipeline) -> bool:
    """Await ``entry.applies(pipeline)``, supporting both async and sync implementations."""
    result = entry.applies(pipeline)
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


def _check_applies_sync(entry: Any, pipeline: Pipeline, timeout: float = ANALYSIS_TIMEOUT) -> bool:
    result = entry.applies(pipeline)
    if inspect.isawaitable(result):
        return bool(_run_coroutine_sync(result, timeout=timeout))
    return bool(result)


def _slug_table(name: str) -> str:
    slug = re.sub(r"\W+", "_", name).strip("_").lower()
    return slug or "analysis_result"


def _frame_pipeline(frame: pd.DataFrame, name: str) -> Pipeline:
    """Wrap a DataFrame returned by an ``Analysis`` in an in-memory ``Pipeline``."""
    table = _slug_table(name)
    source = InMemorySource(tables={table: frame})
    schema = get_dataframe_schema(frame)["items"]["properties"] if len(frame.columns) else {}
    return Pipeline(source=source, table=table, schema=schema, data=frame, name=table)


@dataclass(frozen=True)
class AnalysisResult:
    """Classified output of one ``Analysis.__call__`` execution."""

    kind: Literal["pipeline", "data", "view", "empty"]
    pipeline: Pipeline | None = None
    data: pd.DataFrame | None = None
    view: Any = None
    analysis: Any = None


def classify_result(raw: Any, analysis_name: str, instance: Any) -> AnalysisResult:
    """Classify the return value of ``Analysis.__call__`` into pipeline/data or view outputs."""
    from ..views.base import View as LumenView

    if raw is None:
        return AnalysisResult("empty", analysis=instance)
    if isinstance(raw, Pipeline):
        return AnalysisResult("pipeline", pipeline=raw, analysis=instance)
    if isinstance(raw, pd.DataFrame) or is_narwhals(raw):
        df = as_pandas(raw)
        return AnalysisResult("data", pipeline=_frame_pipeline(df, analysis_name), data=df, analysis=instance)
    if isinstance(raw, LumenView):
        if getattr(raw, "_panel", None) is None:
            raw.update()
        return AnalysisResult("view", view=raw.panel, analysis=instance)
    if isinstance(raw, pn.viewable.Viewable):
        return AnalysisResult("view", view=raw, analysis=instance)
    return AnalysisResult("view", view=pn.panel(raw), analysis=instance)


def _bounded_pipeline(pipeline: Pipeline, limit: int, timeout: float) -> Pipeline:
    """
    A copy of ``pipeline`` with its rows fetched under ``limit``, so the
    analysis shares ``fetch``'s cache and respects the host's row ceiling.
    """
    rows = fetch(pipeline, limit, timeout)
    return _clone(pipeline, data=rows)


def run_analysis(
    entry: Any,
    pipeline: Pipeline,
    params: Mapping[str, Any] | None = None,
    *,
    name: str | None = None,
    limit: int = RESULT_LIMIT,
    timeout: float = NODE_TIMEOUT,
) -> AnalysisResult:
    """
    Validate and execute ``entry`` on ``pipeline`` with ``params``, returning a
    classified :class:`AnalysisResult`.
    """
    if is_failure(pipeline):
        raise ValueError(f"The input failed: {failure_text(pipeline)}")
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected a Lumen pipeline, got {type(pipeline).__name__}.")
    label = name or _entry_name(entry)
    columns = pipeline_columns(pipeline)
    if missing := missing_columns(entry, columns):
        raise ValueError(f"Analysis {label!r} requires column(s) {', '.join(missing)}, not in pipeline columns ({', '.join(columns) or 'none'}).")
    instance = build_analysis(entry, params, name=label, columns=columns)
    bounded = _bounded_pipeline(pipeline, limit, timeout)
    if not _check_applies_sync(entry, bounded, timeout=max(timeout, ANALYSIS_TIMEOUT)):
        raise ValueError(f"Analysis {label!r} does not apply to this pipeline.")
    # Fresh context dict so an analysis mutating context never alters editor state.
    context: dict[str, Any] = {
        "pipeline": bounded,
        "source": bounded.source,
        "table": bounded.table,
        "data": bounded.data,
    }
    raw = instance(bounded, context)
    if inspect.isawaitable(raw):
        raw = _run_coroutine_sync(raw, timeout=max(timeout, ANALYSIS_TIMEOUT))
    return classify_result(raw, label, instance)


def check_analysis_config(config: dict[str, Any], inputs: dict[str, Any]) -> Any:
    """
    Check a ``Lumen/analysis`` config without running the analysis: the named
    analysis must be registered by the host, its parameters must be valid, and
    any required ``columns`` must exist on the connected pipeline.
    """
    name = config.get("analysis")
    if not name:
        return None
    analyses = host_analyses()
    entry = analyses.get(name)
    if entry is None:
        return CheckDiagnostic(missing_analysis(name, analyses), field="analysis")
    pipeline = inputs.get("pipeline")
    columns: list[str] | None = None
    if isinstance(pipeline, Pipeline) and not is_failure(pipeline):
        columns = pipeline_columns(pipeline)
        if missing := missing_columns(entry, columns):
            return CheckDiagnostic(
                f"Analysis {name!r} requires column(s) {', '.join(missing)}, not in pipeline columns ({', '.join(columns) or 'none'}).",
                field="analysis",
            )
    params = config.get("params")
    if params is not None and not isinstance(params, Mapping):
        return CheckDiagnostic("params must be a mapping of parameter names to values.", field="params")
    try:
        build_analysis(entry, params, name=name, columns=columns)
    except ValueError as exc:
        return CheckDiagnostic(str(exc), field="params")
    return None


def analysis_choices(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Offer the names of the analyses the host registered for a ``Lumen/analysis`` node."""
    analyses = host_analyses()
    return {"analysis": sorted(analyses)} if analyses else {}


def analysis_presets(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A ``Lumen/analysis`` palette entry per analysis the host provides."""
    presets = []
    for name, entry in host_analyses(context).items():
        doc = analysis_doc(entry)
        first_line = doc.splitlines()[0].strip() if doc else f"Run the {name} analysis on a pipeline."
        cols = [str(c) for c in (getattr(entry, "columns", ()) or ())]
        tooltip = f"Analysis {name}" + (f" (requires: {', '.join(cols)})" if cols else "")
        presets.append(
            {
                "title": f"{name} (analysis)",
                "description": first_line,
                "tooltip": tooltip,
                "tags": ["analysis", name],
                "config": {"analysis": name},
            }
        )
    return presets


def _track_analysis(node: LumenNode) -> None:
    if node.editor is not None:
        _ANALYSIS_NODES.setdefault(node.editor, weakref.WeakSet()).add(node)


@register(
    component=True,
    page=False,
    title="Lumen Analysis",
    icon="insights",
    description="Runs a host Analysis on a pipeline.",
    config=["analysis", "params"],
    check_config=check_analysis_config,
    config_choices=analysis_choices,
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
    provides=[
        {"key": "result", "type": PIPELINE_TYPE},
        {"key": "data", "type": "DataFrame"},
        {"key": "view"},
    ],
    **palette_presets(analysis_presets),
)
class AnalysisNode(GlobalFilterTarget, LumenNode):
    pipeline = param.Parameter(doc="The upstream pipeline.")

    analysis = param.String(default="", doc="Name of an Analysis registered by the host.")

    params = param.Dict(default={}, doc="Parameter values passed to the selected Analysis.")

    _filter_inputs = ("pipeline",)

    def __init__(self, **params):
        super().__init__(**params)
        _track_analysis(self)

    def _filter_pipeline(self) -> Any:
        return self.pipeline

    @param.depends("pipeline", "analysis", "params", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self) -> AnalysisResult | None:
        if self.pipeline is None or not self.analysis:
            return None
        analyses = host_analyses(self._context)
        entry = analyses.get(self.analysis)
        if entry is None:
            raise ValueError(missing_analysis(self.analysis, analyses))
        return run_analysis(
            entry,
            self.pipeline,
            self.params,
            name=self.analysis,
            limit=row_ceiling(self.editor),
        )

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def result(self):
        result = self._value()
        if result is None or is_failure(result):
            return result
        return result.pipeline

    @param.output(param.DataFrame)
    @param.depends("_object", "_error")
    def data(self):
        result = self._value()
        if result is None or is_failure(result):
            return result
        if result.data is not None:
            return result.data
        if result.pipeline is not None:
            return fetch(result.pipeline, row_ceiling(self.editor))
        return None

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def view(self):
        result = self._value()
        if result is None or is_failure(result):
            return result
        return result.view

    def _body(self):
        result = self._object
        if result is None:
            if self.pipeline is None:
                return pn.pane.Markdown("Waiting for a pipeline")
            return pn.pane.Markdown("Pick an analysis")
        if self._global_filters and isinstance(self.pipeline, Pipeline):
            analyses = host_analyses(self._context)
            entry = analyses.get(self.analysis)
            if entry is not None:
                try:
                    result = run_analysis(
                        entry,
                        self._shown(self.pipeline),
                        self.params,
                        name=self.analysis,
                        limit=row_ceiling(self.editor),
                    )
                except Exception as exc:
                    return pn.pane.Alert(str(exc), alert_type="danger")
        if result.view is not None:
            return result.view
        if result.data is not None:
            return pn.widgets.Tabulator(result.data.head(PREVIEW_ROWS), disabled=True, sizing_mode="stretch_both")
        if result.pipeline is not None:
            preview = fetch(result.pipeline, PREVIEW_ROWS)
            return pn.widgets.Tabulator(preview, disabled=True, sizing_mode="stretch_both")
        return pn.pane.Markdown(f"{self.analysis}: no output")

    def __flowdash_preview__(self, output: str | None = None) -> Any:
        if self._error is not None or self._object is None:
            return None
        result: AnalysisResult = self._object
        if output == "view" or (output is None and result.view is not None):
            if result.view is None:
                return None
            return f"{self.analysis} view ({type(result.view).__name__})"
        if output == "data" and result.data is not None:
            return _preview_frame(result.data.head(PREVIEW_ROWS), len(result.data))
        if result.pipeline is not None:
            return _pipeline_preview(self, result.pipeline)
        if result.data is not None:
            return _preview_frame(result.data.head(PREVIEW_ROWS), len(result.data))
        return f"{self.analysis}: no output"
