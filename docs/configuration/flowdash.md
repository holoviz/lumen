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

A dashboard cannot open connections of its own. The app hands the editor the data you served, and a **Lumen Source** node names it by `ref`. A ref comes from the data's name, so it stays the same when you add, remove or reorder what you serve, or run `lumen serve` from another directory: a database file or connection is named by its file or database name (`shop.db` gives `shop`), and all CSV, Parquet and JSON files go into one DuckDB source named by the file's stem (`lumen serve turbines.csv --flowdash` gives `turbines`) or, with several files, `files` (`lumen serve a.csv b.parquet --flowdash` gives one source `files` with two tables). Each file's table is named by its file name, `turbines_csv` for `turbines.csv` in any directory. Files of the same name from different directories are each prefixed with as many parent directories as tell them apart (`2025_sales_csv` and `2026_sales_csv` for `2025/sales.csv` and `2026/sales.csv`), so a query saved against `sales_csv` reports the missing table rather than reading one of them.

When two sources claim the same name, counting each data file's stem, neither gets it: each is named with a tag of its kind instead, so `lumen serve shop.db shop.csv --flowdash` gives `shop_db` and `shop_files`, and a dashboard saved with `ref: shop` reports the missing source rather than reading the other one. `data`, `data_2` and so on are never given as refs either (`data.csv` gives `data_files`).

Dashboards saved before refs were named this way keep working: `data` names the first source and `data_2`, `data_3` and so on the next ones, in the order `lumen serve` reads them (database files and connections first, then the CSV, Parquet and JSON files). `files`, each file's stem and the tagged names also resolve, so a dashboard saved with one file keeps its source when you add a second.

## Nodes

| Node | Inputs | Config | Outputs |
| --- | --- | --- | --- |
| Lumen Source (`Lumen/source`) | | `ref` | `source` |
| Lumen Table (`Lumen/table`) | `source` | `table`, `limit` | `pipeline` |
| Lumen Filter (`Lumen/filter`) | `pipeline`, `value`, `selection` | `field` | `filtered` |
| Lumen SQL (`Lumen/sql`) | `source` | `sql`, `table`, `limit` | `pipeline`, `data` |
| Lumen Data (`Lumen/data`) | `pipeline` | | `data` |
| Lumen Vega-Lite view (`Lumen/view`) | `pipeline` | `spec`, `debounce` | `selection` |

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

### Checks

Each node's config is checked before it is applied, whether it is typed in the editor's inspector or set by an agent, and again when a dashboard is validated. A change that fails is refused and the node keeps its last config. Source nodes may only name a source the app provides, table nodes a table of their source, filter nodes a column of their input, and charts only columns of their input, with no data or URLs of their own.

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
agent_sensitive_components = ["Lumen/source"]
editor_options = {"context": {"lumen.sources": {"data": DuckDBSource(uri="warehouse.duckdb")}}}
```

Each session's editor copies the `lumen.sources` dict, so adding or removing a source in one session does not reach another; the source objects themselves are shared. Without sources, a source node says how to provide them.
