import io

import pandas as pd
import panel as pn

from lumen.ai.memory import memory
from lumen.sources.duckdb import DuckDBSource


def add_source_controls(replace_controls: bool = True):
    def add_table(event):
        with menu.param.update(loading=True):
            if input_tabs.active == 0 and upload.value:
                if upload.filename.endswith("csv"):
                    df = pd.read_csv(io.BytesIO(upload.value), parse_dates=True)
                elif upload.filename.endswith((".parq", ".parquet")):
                    df = pd.read_parquet(io.BytesIO(upload.value))
                elif upload.filename.endswith(".json"):
                    df = pd.read_json(io.BytesIO(upload.value))
                elif upload.filename.endswith(".xlsx"):
                    df = pd.read_excel(io.BytesIO(upload.value))
                # TODO: add url support
                duckdb_source._connection.from_df(df).to_table(name.value)
                if duckdb_source.tables is None:
                    duckdb_source.tables = {}
                duckdb_source.tables[name.value] = f"SELECT * FROM {name.value}"
                memory["current_source"] = duckdb_source
                memory["current_table"] = name.value
                memory["available_sources"].add(duckdb_source)
            if replace_controls:
                name.value = ""
                upload.value = None
                upload.filename = ""
                src = memory["current_source"]
                tables[:] = [
                    (t, pn.widgets.Tabulator(src.get(t), sizing_mode="stretch_width"))
                    for t in src.get_tables()
                ]
                input_column.visible = False

    def add_name(event):
        if event.new is None:
            return
        if "/" in event.new:
            name.value = event.new.split("/")[-1].split(".")[0].replace("-", "_")
        elif not name.value and event.new:
            name.value = event.new.split(".")[0].replace("-", "_")
            name.visible = True

    def enable_add(event):
        add.visible = not bool(name.value)

    name = pn.widgets.TextInput(name="Name your table", visible=False)
    upload = pn.widgets.FileInput(align="end", accept=".csv,.parquet,.parq,.json,.xlsx")
    add = pn.widgets.Button(
        name="Add table",
        icon="table-plus",
        visible=False,
        button_type="success",
    )
    input_tabs = pn.Tabs(("Upload", upload), sizing_mode="stretch_width")
    tables = pn.Tabs(sizing_mode="stretch_width")
    duckdb_source = DuckDBSource(uri=":memory:")
    input_column = pn.Column(
        input_tabs,
        name,
        add,
        sizing_mode="stretch_width",
    )
    menu = pn.Column(
        input_column,
        tables,
        sizing_mode="stretch_width",
    )
    upload.param.watch(add_name, "filename")
    upload.param.watch(enable_add, "value")
    add.on_click(add_table)
    return menu
