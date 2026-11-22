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
| `--project FILE` | The project file whose `metrics:` section defines the app's [metrics](#metrics). Without it, `flowdash.yaml` next to the data files is read, if there is one. |
| `--agent-model MODEL[=LABEL]` | A pydantic-ai model, e.g. `openai:gpt-6-luna`, for Lumen's FlowDash agent. Repeat it to let people pick the model below the chat's input; the first is the default and `=LABEL` names it in the picker. Without it the app has no agent. |

With two models the assistant's chat shows a model picker, and the browser remembers the pick:

``` bash
lumen serve penguins.csv --flowdash \
  --agent-model openrouter:openai/gpt-6-luna=Luna \
  --agent-model anthropic:claude-haiku-4-5=Haiku
```

A model whose provider package or API key is missing is listed disabled. With one model there is no picker.

Lumen YAML dashboards are served without `--flowdash`, as before. The Lumen AI options (`--provider`, `--api-key`, `--model`, `--temperature` and the like) configure the explorer and are refused with `--flowdash`.

## Sources

A dashboard cannot open connections of its own. The app hands the editor the data you served, and a node names a source by `ref`, the source's name. A **Lumen Table** node picks a source by `ref` and one of its tables, so reading a table takes one node; the component list offers each table of each source as an entry such as "turbines (local)" with both filled in. A **Lumen Source** node is for SQL nodes, which query a source, and for several table nodes that share one source node through its output; a table node with a source connected reads that source instead of its `ref`. All CSV, Parquet and JSON files, and those in a folder you serve, go into one DuckDB source named after the data: the file's stem for one file (`lumen serve turbines.csv --flowdash` gives `turbines`), the folder's name for a folder (`lumen serve data/ --flowdash` gives `data`), and `local` for several files (`data` when some are URLs). Each file's table is its stem without the extension: `turbines.csv` gives `turbines`, in any directory. A database file or connection is named by its file or database name (`shop.db` gives `shop`).

Names that clash get `_2`, `_3` and so on in the order you list the data, for tables and sources alike: `lumen serve 2025/sales.csv 2026/sales.csv --flowdash` gives the tables `sales` and `sales_2`, and `shop.db shop.csv` the sources `shop` and `shop_2`. The first one listed keeps the name, so add new data at the end: a file added later never takes over a name a saved dashboard reads. Listing a file of the same name before one a dashboard already reads does renumber them, and that dashboard then reads the new file; `lumen serve` logs which file each numbered table reads (`sales: y/sales.csv, sales_2: x/sales.csv`) whenever file names clash. Each source `lumen serve` reads is named by its ref, so a source node shows the name a dashboard uses; a source passed to `build_routes` from Python keeps its own name, and its node shows the ref.

Dashboards saved with earlier names keep working where the name is unambiguous. Refs: `data` names the first source and `data_2`, `data_3` and so on the next ones, in the order earlier releases read them (database files and connections first, then the data files), and `files`, `local`, the name of a served folder, each file's stem and the tagged names `shop_db` and `turbines_files` also resolve, so a dashboard saved on a folder keeps its source when files are served beside it. Tables: the earlier `turbines_csv` and the path as given (`x_sales_csv` for `x/sales.csv`) still work in queries, though only the new names are listed. A name two files or two sources had resolves to neither, so the query or source node reports it rather than reading the wrong data, and a source's own ref always wins over an earlier name.

## Metrics

A metric is a named, described query over one of the app's sources, defined once in the `metrics:` section of the project file, `flowdash.yaml`, and shared by every dashboard and the agent. The file is panel-flowdash's project file: Lumen registers its `metrics:` section, and sections nothing registered are ignored with a warning, so the app's other settings live in the same file.

``` yaml
metrics:
  capacity:
    description: Installed wind capacity.
    source: turbines                 # the source's ref
    table: turbines
    measures:
      capacity_mw: {expr: "SUM(t_cap) / 1000.0", format: ",.1f", description: Rated capacity in MW}
      turbines: "COUNT(*)"
    dimensions:
      state: {column: t_state, description: Two-letter state code}
      online: {column: online, type: time, grains: [year, quarter, month]}
    parameters:
      since: {type: integer, default: 2005, column: p_year, op: ">="}

  top_projects:
    description: Projects ranked by installed capacity.
    source: turbines
    sql: SELECT p_name, SUM(t_cap) AS kw FROM turbines GROUP BY p_name ORDER BY kw DESC LIMIT :n
    parameters:
      n: {type: integer, default: 10, bounds: [1, 50]}
```

A metric with a `table` lets a **Lumen Metric** node pick its measures and the dimensions to group by, and truncate a time dimension to a `grain`. Its `limit` and `order_by` (columns, `-` first for descending) are parameters too, so a widget can set a top N; declare `limit: {type: integer, default: 10, bounds: [1, 50]}` to give them a default and bounds, and the node's config is the default otherwise. A metric with `sql` is one fixed read-only query, for rankings, windows and joins, whose `:name` placeholders are its parameters; each needs a `default` or `required: true`. Parameter types are `string`, `number`, `integer`, `boolean`, `date`, `datetime`, `list[<type>]` and `range[<type>]`, with an optional `default`, `required`, `bounds` and `options`. `options` and `bounds` check what a dashboard asks for; they are not access control, since anyone who can edit a dashboard can also write SQL on the same source. `format` is a [d3-format](https://d3js.org/d3-format) string, carried in the pipeline's schema with each description.

The node's `parameters` input takes a mapping of parameter values set in the inspector, e.g. `{"since": 2010}`, or the outputs of **Lumen Parameter** nodes wired to it, any number of them. A Lumen Parameter node names a value, such as a slider's, as one parameter: wire the slider to its `value` and its `parameters` output to the metric. The mapping also filters a column dimension by its name, e.g. `{"state": ["TX", "CA"]}`, with the value converted to the column's type. Values are converted to the declared type and refused outside the bounds or options, and enter the query as SQL literals, never as text. A value of None, such as an empty widget, takes the parameter's default, and an empty list filters nothing. In a SQL metric that means its `IN` or `NOT IN` condition is dropped when it is ANDed into the `WHERE` or `HAVING` of the query's outer `SELECT` (or of a `UNION` branch), and other items of the same `IN` are kept; anywhere else, such as under `OR`, in a subquery, a CTE, `NOT EXISTS`, a join or an aggregate's `FILTER`, an empty list is refused, since dropping the condition there could filter everything.

A filter node on a metric filters the rows before they are aggregated, so a filter on `t_state` feeds a chart of capacity by year; a filter on a measure filters the result. On a SQL metric, a filter on a column of a table the query's outer `FROM` or `JOIN` reads filters that table inside the query; tables read by a subquery or a CTE are left alone, so a share keeps its total.

The file is checked when the app starts: each metric's columns, its query with the defaults and every grain must work on its source, and a problem is reported with its line and column, e.g. `flowdash.yaml:10:7: Unknown column 't_sate' in dimension 'state'. Did you mean 't_state'?`. A metric over a source the app does not serve is left out with a warning.

A source node's preview lists its tables with their row and column counts and column names. Reading a table's schema counts its rows, so in the app a background worker reads the schemas the source has not stored yet, at most 20 tables, and the card fills in once they are there. Selecting a source node shows its spec as YAML, read only, in the code panel under the canvas (with a panel-flowdash that supports it). Credentials are never shown: passwords in connection URIs, settings named like a password, secret, token, key or credentials, strings passed to the driver such as `connect_args`, secrets in SQL run on connect such as DuckDB's `CREATE SECRET`, and values read from environment variables all read `***`.

## Nodes

| Node | Inputs | Config | Outputs |
| --- | --- | --- | --- |
| Lumen Source (`Lumen/source`) | | `ref` | `source` |
| Lumen Table (`Lumen/table`) | `source` (optional) | `ref`, `table`, `limit` | `pipeline` |
| Lumen Filter (`Lumen/filter`) | `pipeline`, `value`, `selection` | `field` | `filtered` |
| Lumen SQL (`Lumen/sql`) | `source` | `sql`, `table`, `limit` | `pipeline`, `data` |
| Lumen Data (`Lumen/data`) | `pipeline` | | `data` |
| Lumen Vega-Lite view (`Lumen/view`) | `pipeline` | `spec`, `debounce` | `selection` |
| Lumen Metric (`Lumen/metric`) | `parameters` | `metric`, `measures`, `dimensions`, `grain`, `order_by`, `limit` | `pipeline`, `data` |
| Lumen Parameter (`Lumen/parameter`) | `value` | `parameter` | `parameters` |

Edges between Lumen nodes carry a Lumen pipeline rather than a table of rows, so the work happens in the source's engine. A table node fetches nothing by itself. A filter node fed by a widget, say a Select on `species`, adds `WHERE "species" = 'Adelie'` and the row limit to the query the source runs, rather than Lumen filtering all rows after fetching them. A filter's value can be a single value, a list (from a MultiChoice) or a `(start, end)` range; an empty value keeps every row.

Rows are fetched where they are needed: by a **Lumen Data** node, which outputs a DataFrame for any FlowDash component that takes one, by a chart, or by a node's own preview when it is shown. Every query is capped at 100,000 rows; a table or SQL node's `limit` can lower that.

### Brushing a chart

A chart can filter the rest of the dashboard. Give its spec a selection param, such as an interval brush, `"params": [{"name": "brush", "select": {"type": "interval", "encodings": ["x"]}}]`, and wire its `selection` output to a filter node's `selection` input. Feed the chart and the filter from the same table or SQL node, and the KPIs, tables and other charts from the filter:

```
table --> chart (selection) --> filter <-- table
                                   |
                                   +--> data, KPIs, other charts
```

The chart keeps showing all its rows with the brush highlighted, and the filter adds the brushed predicates to the source's query: `WHERE "category" IN ('A', 'B')` for a brush across bars, `"price" BETWEEN ...` for a range, and `"date" >= ... AND "date" < ...` for the whole months a `yearmonth` brush covers. Clearing the brush restores every row. The chart reports a drag once it pauses for `debounce` milliseconds, 150 by default, so a drag runs one query rather than one per mouse move. Feeding the chart from the filter it drives would be a cycle, which the editor refuses.

The chart's check warns about selections a filter cannot apply exactly: a brush on bins without a fixed `step`, a brush on a repeating time unit such as `month`, and a point selection without `fields`.

### Global filters

With a panel-flowdash that has the dashboard filter bar, its filters reach the Lumen charts, tables and SQL results a dashboard shows, matched by column name, with no wiring. Each node adds the filters to the query of what it shows, as a filter node would: `WHERE "state" IN ('TX', 'OK')` or `"amount" BETWEEN ...`, with values written as quoted literals of the source's dialect. Only what the node shows is filtered; its outputs keep every row, so nodes downstream are not filtered twice. A select filter offers the column's distinct values, at most 1000, which the source computes with `SELECT DISTINCT` rather than by fetching rows, and a range filter its minimum and maximum. A chart fed by an aggregate without the column, say totals by category under a filter on state, is left unfiltered and says *Not filtered by State*, and a view can opt out of a filter in its **Tile** settings.

### Checks

Each node's config is checked before it is applied, whether it is typed in the editor's inspector or code panel, or set by an agent, and again when a dashboard is validated. A change that fails is refused and the node keeps its last config. Source and table nodes may only name a source the app provides, table nodes a table of their source, filter nodes a column of their input, and charts only columns of their input, with no data or URLs of their own.

SQL nodes accept one read-only `SELECT` or `WITH` query, checked in steps:

- Before a source is connected, the query must parse in the dialect of one of the app's sources and be read-only.
- Once a source is connected, it may only read the source's own tables.
- On DuckDB and Snowflake, which can stop a running statement, the engine then binds the query with `LIMIT 0`, so a misspelled column is refused without running the query. If the engine does not answer within 3 seconds the change is applied with the warning "Could not verify the query" (panel-flowdash releases without check warnings apply it silently), and the node reports any error when it runs the query.
- Other sources are never asked, since a slow check could not be stopped. There the query's columns are compared with the table schemas Lumen already read, when it has them, ignoring case, so a column is only refused when no column of that name exists.

A query placed before its source is connected is checked against the source when the node runs; the node reports what the engine refused, and validating the dashboard lists it.

## In your own app

The nodes are a FlowDash component library, so any FlowDash app can offer them:

``` python
from lumen.flowdash.app import build_routes
import panel as pn

pn.serve(build_routes(["penguins.csv"], db_path="dashboards.db"))
```

In a `flowdash serve` project, set `component_libraries = ["lumen.flowdash"]` in its `__init__.py` (or pass `--component-library lumen.flowdash`) and give the editor its sources under the `lumen.sources` key of its context:

``` python
from lumen.sources.duckdb import DuckDBSource

component_libraries = ["lumen.flowdash"]
editor_options = {"context": {"lumen.sources": {"data": DuckDBSource(uri="warehouse.duckdb")}}}
```

Each session's editor copies the `lumen.sources` dict, so adding or removing a source in one session does not reach another; the source objects themselves are shared. Without sources, a source node says how to provide them.

Lumen's agent reads every source the app provides by its ref, also on a dashboard with no source node yet, and its `add_sql` places a source node for a source when no node shows it. Placing or re-pointing a source or table node with the dashboard tools (`add_node`, `set_config`) needs no approval, since a `ref` can only name a source the app already lets the agent read. A project that wants approval for them can still list them in `agent_sensitive_components`.

The agent's `list_tables` lists the app's metrics with their descriptions, formats and parameters, and `add_metric` places a metric node, which needs no approval since it only names a metric the app defines. The agent prefers a metric that answers the question to writing SQL. A project sets the metrics under the `lumen.metrics` key of the context, e.g. `read_metrics(load_project("flowdash.yaml", context={"lumen.sources": sources}))` from `lumen.flowdash.metrics` and `lumen.flowdash.project`, which checks them against `sources`; `build_routes(project=...)` does that for you.
