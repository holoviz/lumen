"""Explore locally saved Lumen evaluation runs.

Serve with: pixi run -e evals panel serve tests/evals/dashboard.py --show
"""

import html
import json

from pathlib import Path

import pandas as pd
import panel as pn

from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.palettes import Category10
from bokeh.plotting import figure
from panel_material_ui import (
    MenuList, Page, Select, Typography,
)

RESULTS = Path(__file__).parent / "results"


def model_colors(data):
    palette = Category10[10]
    return {model: palette[index % len(palette)] for index, model in enumerate(sorted(data["Model"].unique()))}


def load_runs(paths):
    runs = {}
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if "run" in report and "cases" in report:
            if path.stem in runs:
                raise ValueError(f"Duplicate result filename: {path.stem}")
            runs[path.stem] = report
    if not runs:
        raise ValueError("No evaluation results found.")
    return runs


def summarize(run):
    cases = run["cases"]
    failures = {failure["name"] for failure in run.get("failures", [])}
    usage = [case["usage"] for case in cases if case.get("usage")]
    input_tokens = sum(item["input_tokens"] for item in usage)
    costs = [item.get("cost_usd") for item in usage]
    return {
        "passed": sum(case["name"] not in failures and bool(case["assertions"]) and all(case["assertions"].values()) for case in cases),
        "cases": len(cases),
        "seconds": sum(case.get("duration") or 0 for case in cases),
        "input": input_tokens if usage else None,
        "output": sum(item["output_tokens"] for item in usage) if usage else None,
        "cached_percent": 100 * sum(item["cached_tokens"] for item in usage) / input_tokens if input_tokens else None,
        "cost": sum(costs) if usage and all(cost is not None for cost in costs) else None,
    }


def group_runs(runs, suite):
    grouped = {}
    for run_id, report in runs.items():
        meta = report["run"]
        if meta.get("dataset") != suite:
            continue
        signature = meta.get("case_fingerprint") or tuple(sorted(case["name"] for case in report["cases"]))
        key = (meta.get("provider"), meta.get("model"), meta.get("commit"), meta.get("api"), signature,
               run_id if meta.get("dirty") else None)
        grouped.setdefault(key, []).append((run_id, report))
    result = {}
    for attempts in grouped.values():
        latest = max(attempts, key=lambda entry: entry[1]["run"].get("timestamp", ""))
        label = latest[0] if len(attempts) == 1 else f"{latest[0]} (+{len(attempts) - 1})"
        result[label] = attempts
    return result


def run_table(groups):
    rows = []
    for run_id, attempts in groups.items():
        meta = max(attempts, key=lambda entry: entry[1]["run"].get("timestamp", ""))[1]["run"]
        stats = [summarize(report) for _, report in attempts]
        costs = [stat["cost"] for stat in stats]
        total_input = sum(stat["input"] or 0 for stat in stats)
        cached = sum((stat["cached_percent"] or 0) * (stat["input"] or 0) / 100 for stat in stats)
        count = len(attempts)
        rows.append({
            "Run": run_id, "Model": meta.get("model", "Unknown"), "Commit": (meta.get("commit") or "")[:8],
            "Timestamp": max(report["run"].get("timestamp", "") for _, report in attempts)[:19].replace("T", " "),
            "Suite": meta.get("dataset", "Unknown"), "Attempts": count,
            "Passed": sum(stat["passed"] for stat in stats) / count, "Cases": stats[0]["cases"],
            "Seconds": round(sum(stat["seconds"] for stat in stats) / count, 1),
            "Cost ($)": sum(costs) / count if all(cost is not None for cost in costs) else None,
            "Input": round(total_input / count) if all(stat["input"] is not None for stat in stats) else None,
            "Output": round(sum(stat["output"] for stat in stats) / count) if all(stat["output"] is not None for stat in stats) else None,
            "Cached (%)": 100 * cached / total_input if total_input else None,
        })
    return pd.DataFrame(rows).sort_values("Timestamp", ascending=False, kind="stable").reset_index(drop=True)


def case_table(groups):
    rows = []
    for run_id, attempts in groups.items():
        meta = attempts[-1][1]["run"]
        for name in sorted(case["name"] for case in attempts[0][1]["cases"]):
            cases = [next(case for case in report["cases"] if case["name"] == name) for _, report in attempts]
            usage = [case.get("usage") or {} for case in cases]
            costs = [item.get("cost_usd") for item in usage]
            inputs = [item.get("input_tokens") for item in usage]
            outputs = [item.get("output_tokens") for item in usage]
            passing = sum(case["name"] not in {failure["name"] for failure in report.get("failures", [])} and bool(case.get("assertions")) and all(case["assertions"].values())
                          for case, (_, report) in zip(cases, attempts, strict=True))
            rows.append({
                "Run": run_id, "Model": meta.get("model", "Unknown"), "Commit": (meta.get("commit") or "")[:8],
                "Case": name, "Checks": f"{passing}/{len(cases)}", "Attempts": len(cases),
                "Seconds": round(sum(case.get("duration") or 0 for case in cases) / len(cases), 1),
                "Cost ($)": sum(costs) / len(cases) if all(cost is not None for cost in costs) else None,
                "Input": round(sum(inputs) / len(cases)) if all(value is not None for value in inputs) else None,
                "Output": round(sum(outputs) / len(cases)) if all(value is not None for value in outputs) else None,
                "Cached (%)": 100 * sum(item.get("cached_tokens", 0) for item in usage) / sum(inputs) if all(value is not None for value in inputs) and sum(inputs) else None,
            })
    return pd.DataFrame(rows).sort_values(["Case", "Run"], kind="stable").reset_index(drop=True)


def comparison_chart(data, colors):
    plot = figure(height=310, sizing_mode="stretch_width", toolbar_location=None,
                  title="Cost vs runtime by run", x_axis_label="Total runtime (seconds)", y_axis_label="Estimated cost (USD)")
    for model, group in data.dropna(subset=["Cost ($)"]).groupby("Model", sort=False):
        source = ColumnDataSource({"run": group["Run"].tolist(), "model": group["Model"].tolist(),
                                   "commit": group["Commit"].tolist(),
                                   "seconds": group["Seconds"].tolist(), "cost": group["Cost ($)"].tolist(),
                                   "passed": group["Passed"].tolist(), "cases": group["Cases"].tolist()})
        glyph = plot.scatter("seconds", "cost", source=source, size=13, color=colors[model], legend_label=model)
        plot.add_tools(HoverTool(renderers=[glyph], tooltips=[("Model", "@model"), ("Run", "@run"), ("Commit", "@commit"),
                                                           ("Checks", "@passed / @cases"), ("Runtime", "@seconds{0.0} s"),
                                                           ("Cost", "@cost{$0.0000}")]))
    if plot.legend:
        plot.legend.visible = False
    plot.yaxis.formatter = NumeralTickFormatter(format="$0.000")
    plot.outline_line_color = None
    return plot


def usage_chart(data, colors):
    plot = figure(height=290, sizing_mode="stretch_width", toolbar_location=None,
                  title="Input vs output tokens by run", x_axis_label="Input tokens", y_axis_label="Output tokens")
    for model, group in data.dropna(subset=["Input", "Output"]).groupby("Model", sort=False):
        source = ColumnDataSource({"run": group["Run"].tolist(), "model": group["Model"].tolist(),
                                   "commit": group["Commit"].tolist(),
                                   "input": group["Input"].tolist(), "output": group["Output"].tolist(),
                                   "cached": group["Cached (%)"].tolist()})
        glyph = plot.scatter("input", "output", source=source, size=12, color=colors[model], legend_label=model)
        plot.add_tools(HoverTool(renderers=[glyph], tooltips=[("Model", "@model"), ("Run", "@run"), ("Commit", "@commit"),
                                                           ("Input", "@input{0,0}"), ("Output", "@output{0,0}"),
                                                           ("Cached", "@cached{0.0}%")]))
    if plot.legend:
        plot.legend.visible = False
    plot.outline_line_color = None
    return plot


def performance_chart(data, colors, metric="Seconds"):
    cost = metric == "Cost ($)"
    plot = figure(height=310, sizing_mode="stretch_width", toolbar_location=None,
                  title="Pass rate vs estimated cost" if cost else "Pass rate vs runtime",
                  x_axis_label="Estimated cost (USD)" if cost else "Total runtime (seconds)",
                  y_axis_label="Cases passing (%)", y_range=(0, 105))
    for model, group in data.dropna(subset=[metric]).groupby("Model", sort=False):
        source = ColumnDataSource({"run": group["Run"].tolist(), "model": group["Model"].tolist(),
                                   "commit": group["Commit"].tolist(), "x": group[metric].tolist(),
                                   "rate": (100 * group["Passed"] / group["Cases"]).tolist(),
                                   "passed": group["Passed"].tolist(), "cases": group["Cases"].tolist()})
        glyph = plot.scatter("x", "rate", source=source, size=13, color=colors[model], legend_label=model)
        plot.add_tools(HoverTool(renderers=[glyph], tooltips=[("Model", "@model"), ("Run", "@run"),
                                                           ("Commit", "@commit"), ("Passing", "@passed / @cases"),
                                                           ("Cost", "@x{$0.0000}") if cost else ("Runtime", "@x{0.0} s")]))
    if plot.legend:
        plot.legend.visible = False
    if cost:
        plot.xaxis.formatter = NumeralTickFormatter(format="$0.000")
    plot.outline_line_color = None
    return plot


def case_chart(data, colors):
    plot = figure(height=290, sizing_mode="stretch_width", toolbar_location=None,
                  title="Case cost vs runtime", x_axis_label="Runtime (seconds)", y_axis_label="Estimated cost (USD)")
    for model, group in data.dropna(subset=["Cost ($)"]).groupby("Model", sort=False):
        source = ColumnDataSource({"run": group["Run"].tolist(), "model": group["Model"].tolist(),
                                   "case": group["Case"].tolist(),
                                   "seconds": group["Seconds"].tolist(), "cost": group["Cost ($)"].tolist(),
                                   "checks": group["Checks"].tolist()})
        glyph = plot.scatter("seconds", "cost", source=source, size=9, color=colors[model], alpha=0.75, legend_label=model)
        plot.add_tools(HoverTool(renderers=[glyph], tooltips=[("Model", "@model"), ("Case", "@case"), ("Run", "@run"),
                                                           ("Checks", "@checks"), ("Runtime", "@seconds{0.0} s"),
                                                           ("Cost", "@cost{$0.00000}")]))
    if plot.legend:
        plot.legend.visible = False
    plot.yaxis.formatter = NumeralTickFormatter(format="$0.000")
    plot.outline_line_color = None
    return plot


def case_usage_chart(data, colors):
    plot = figure(height=290, sizing_mode="stretch_width", toolbar_location=None,
                  title="Input vs output tokens by case", x_axis_label="Input tokens", y_axis_label="Output tokens")
    for model, group in data.dropna(subset=["Input", "Output"]).groupby("Model", sort=False):
        source = ColumnDataSource({"run": group["Run"].tolist(), "model": group["Model"].tolist(),
                                   "case": group["Case"].tolist(),
                                   "input": group["Input"].tolist(), "output": group["Output"].tolist(),
                                   "cached": group["Cached (%)"].tolist()})
        glyph = plot.scatter("input", "output", source=source, size=9, color=colors[model], alpha=0.75, legend_label=model)
        plot.add_tools(HoverTool(renderers=[glyph], tooltips=[("Model", "@model"), ("Case", "@case"), ("Run", "@run"),
                                                           ("Input", "@input{0,0}"), ("Output", "@output{0,0}"),
                                                           ("Cached", "@cached{0.0}%")]))
    if plot.legend:
        plot.legend.visible = False
    plot.outline_line_color = None
    return plot


def _details(case):
    if case is None:
        return "Select a case to inspect its outputs."
    checks = case.get("assertions", {})
    failed = ", ".join(name for name, passed in checks.items() if not passed)
    parts = [f"**Failed checks:** {failed}" if failed else "**All checks passed**" if checks else "**Evaluation failed**"]
    if case.get("error"):
        parts.append(f"**Error:** {html.escape(case['error'])}")
    for index, turn in enumerate(case.get("turns", []), 1):
        parts.append(f"### Turn {index}: {html.escape(turn['prompt'])}")
        parts.append(f"Status: {turn['status']} · Agents: {', '.join(turn['actors']) or 'None'} · Views: {', '.join(turn['view_types']) or 'None'}")
        for label, value in (("Answer", turn.get("answer")), ("SQL", turn.get("sql")), ("Rows", turn.get("rows"))):
            if value is not None:
                escaped = html.escape(value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, indent=2))
                parts.append(f"**{label}**\n\n<pre style='white-space:pre-wrap;overflow-wrap:anywhere;max-height:220px;overflow:auto'>{escaped}</pre>")
    return "\n\n".join(parts)


class Comparison(pn.viewable.Viewer):
    def __init__(self, paths=None):
        super().__init__()
        pn.extension("tabulator", sizing_mode="stretch_width")
        self.runs = load_runs(paths if paths is not None else sorted(
            path for path in RESULTS.glob("*.json") if "smoke" not in path.stem and path.stem != "latest"
        ))
        suites = sorted({run["run"].get("dataset", "Unknown") for run in self.runs.values()})
        newest = max(self.runs.values(), key=lambda run: run["run"].get("timestamp", ""))
        self.suite = Select(label="Suite", options=suites, value=newest["run"].get("dataset", "Unknown"))
        self.groups = group_runs(self.runs, self.suite.value)
        self.summary = run_table(self.groups)
        self.data = case_table(self.groups)
        colors = model_colors(self.summary)
        options = dict(show_index=False, theme="materialize", pagination="local", page_size=10,
                       selectable=1, disabled=True, sizing_mode="stretch_width")
        self.run_table = pn.widgets.Tabulator(self.summary, height=min(345, 100 + 36 * len(self.summary)), min_width=1100, **options)
        self.case_table = pn.widgets.Tabulator(self.data, height=345, min_width=1000, **options)
        self.search = pn.widgets.TextInput(name="Find case", placeholder="Filter by case name")
        self.search.param.watch(self._filter_cases, "value")
        self.detail_title = pn.pane.Markdown("Select a case result", sizing_mode="stretch_width")
        self.details = pn.pane.Markdown(_details(None), sizing_mode="stretch_width")
        self.case_table.param.watch(self._update, "selection")
        self.menu = MenuList(items=[{"label": "Runs", "icon": "analytics"}, {"label": "Cases", "icon": "view_list"}], active=0)
        self.menu.param.watch(self._navigate, "active")
        self.suite.param.watch(self._select_suite, "value")
        self.cost_plot = pn.pane.Bokeh(comparison_chart(self.summary, colors), sizing_mode="stretch_width")
        self.cost_performance_plot = pn.pane.Bokeh(performance_chart(self.summary, colors, "Cost ($)"), sizing_mode="stretch_width")
        self.time_performance_plot = pn.pane.Bokeh(performance_chart(self.summary, colors), sizing_mode="stretch_width")
        self.usage_plot = pn.pane.Bokeh(usage_chart(self.summary, colors), sizing_mode="stretch_width")
        self.case_cost_plot = pn.pane.Bokeh(case_chart(self.data, colors), sizing_mode="stretch_width")
        self.case_usage_plot = pn.pane.Bokeh(case_usage_chart(self.data, colors), sizing_mode="stretch_width")
        self.runs_view = pn.Column(
            pn.pane.Markdown("Runs with the same model, commit and case set are averaged. Costs use estimated standard text-token rates; cache share is cached input / input tokens."),
            Typography("Runs", variant="h5"),
            pn.Column(self.run_table, styles={"overflowX": "auto"}),
            self.cost_plot,
            self.cost_performance_plot,
            self.time_performance_plot,
            self.usage_plot,
            sizing_mode="stretch_width", margin=16,
        )
        self.cases_view = pn.Column(
            pn.pane.Markdown("Search cases, select a result, and inspect the captured answer, SQL and rows."),
            Typography("Cases", variant="h5"),
            self.search,
            pn.Column(self.case_table, styles={"overflowX": "auto"}),
            self.case_cost_plot,
            self.case_usage_plot,
            self.detail_title,
            self.details,
            sizing_mode="stretch_width", margin=16,
        )
        self.content = pn.Column(self.runs_view, sizing_mode="stretch_width")
        self._page = Page(title="Lumen eval runs", sidebar=[self.suite, self.menu], main=[self.content])

    def _select_suite(self, event):
        self.groups = group_runs(self.runs, event.new)
        self.summary = run_table(self.groups)
        self.data = case_table(self.groups)
        colors = model_colors(self.summary)
        with pn.io.hold():
            self.run_table.value = self.summary
            self.run_table.height = min(345, 100 + 36 * len(self.summary))
            self.case_table.selection = []
            self._filter_cases(None)
            self.detail_title.object = "Select a case result"
            self.details.object = _details(None)
            self.cost_plot.object = comparison_chart(self.summary, colors)
            self.cost_performance_plot.object = performance_chart(self.summary, colors, "Cost ($)")
            self.time_performance_plot.object = performance_chart(self.summary, colors)
            self.usage_plot.object = usage_chart(self.summary, colors)
            self.case_cost_plot.object = case_chart(self.data, colors)
            self.case_usage_plot.object = case_usage_chart(self.data, colors)

    def _navigate(self, event):
        self.content.objects = [self.runs_view if event.new in (0, (0,)) else self.cases_view]

    def _filter_cases(self, event):
        query = self.search.value.strip()
        self.case_table.selection = []
        self.case_table.value = self.data[self.data["Case"].str.contains(query, case=False, regex=False)] if query else self.data

    def _update(self, _event=None):
        selected = self.case_table.selected_dataframe
        if selected.empty:
            return
        row = selected.iloc[0]
        attempts = self.groups[row["Run"]]
        self.detail_title.object = f"### {row['Case']} · {row['Run']}"
        self.details.object = "\n\n".join(
            f"#### {run_id}\n\n" + _details(next(case for case in report["cases"] if case["name"] == row["Case"]))
            for run_id, report in attempts
        )

    def __panel__(self):
        return self._page


if pn.state.served:
    Comparison().servable()
