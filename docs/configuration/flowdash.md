# :material-graph: FlowDash app (experimental)

`lumen serve --flowdash` serves your data in a [FlowDash](https://github.com/panel-extensions/panel-flowdash) app instead of the Lumen AI explorer. Dashboards in it are graphs of Lumen nodes: a source, tables, filters, SQL queries, data tables and Vega-Lite charts, wired to each other and to FlowDash's widgets. This mode is experimental, and node names and configs may still change.

## Launch

``` bash
lumen serve penguins.csv --flowdash
```

The data arguments are read as for Lumen AI (files, URLs, `duckdb://` and SQLAlchemy connection strings, see [CLI](cli.md)). The usual Panel serve options such as `--port` and `--show` apply.

| Option | Meaning |
| --- | --- |
| `--flowdash` | Serve the FlowDash app. Without it `lumen serve` is unchanged. |
| `--db-path PATH` | The SQLite database dashboards are saved in; `dashboards.db` in the working directory by default. |
| `--agent-model MODEL` | A pydantic-ai model, e.g. `openai:gpt-6-luna`, for Lumen's FlowDash agent. Without it the app has no agent. |

Lumen YAML dashboards are served without `--flowdash`, as before. The Lumen AI options (`--provider`, `--api-key`, `--model`, `--temperature` and the like) configure the explorer and are refused with `--flowdash`.

## Sources

A dashboard cannot open connections of its own. The app binds the data you served, and a **Lumen Source** node names it by `ref`: the first source is `data`, further ones `data_2`, `data_3` and so on. All CSV, Parquet and JSON files go into one DuckDB source, so `lumen serve a.csv b.parquet --flowdash` gives one source `data` with two tables.

## Nodes

| Node | Inputs | Config | Outputs |
| --- | --- | --- | --- |
| Lumen Source (`Lumen/source`) | | `ref` | `source` |
| Lumen Table (`Lumen/table`) | `source` | `table`, `limit` | `pipeline` |
| Lumen Filter (`Lumen/filter`) | `pipeline`, `value` | `field` | `filtered` |
| Lumen SQL (`Lumen/sql`) | `source` | `sql`, `table`, `limit` | `pipeline`, `data` |
| Lumen Data (`Lumen/data`) | `pipeline` | | `data` |
| Lumen Vega-Lite view (`Lumen/view`) | `pipeline` | `spec` | |

Edges between Lumen nodes carry a Lumen pipeline rather than a table of rows, so the work happens in the source's engine. A table node fetches nothing by itself. A filter node fed by a widget, say a Select on `species`, adds `WHERE "species" = 'Adelie'` and the row limit to the query the source runs, rather than Lumen filtering all rows after fetching them. A filter's value can be a single value, a list (from a MultiChoice) or a `(start, end)` range; an empty value keeps every row.

Rows are fetched where they are needed: by a **Lumen Data** node, which outputs a DataFrame for any FlowDash component that takes one, by a chart, or by a node's own preview when it is shown. Every query is capped at 100,000 rows; a table or SQL node's `limit` can lower that.

SQL nodes accept one read-only `SELECT` or `WITH` query over the source's own tables, and chart specs may not load data or URLs. These checks apply however the config arrives: typed in the editor, set by the agent or imported from a file.

## In your own app

The nodes are a FlowDash component library, so any FlowDash app can offer them:

``` python
from lumen.flowdash.app import build_routes
import panel as pn

pn.serve(build_routes(["penguins.csv"], db_path="dashboards.db"))
```

In a `flowdash serve` project, set `component_libraries = ["lumen.flowdash"]` in its `__init__.py` (or pass `--component-library lumen.flowdash`) and bind the sources with the editor class from `lumen.flowdash.app`:

``` python
from lumen.flowdash.app import LumenFlowDash
from lumen.sources.duckdb import DuckDBSource

component_libraries = ["lumen.flowdash"]
agent_sensitive_components = ["Lumen/source"]
editor_class = LumenFlowDash
editor_options = {"lumen_sources": {"data": DuckDBSource(uri="warehouse.duckdb")}}
```
