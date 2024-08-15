import io

import pandas as pd
import panel as pn

from lumen.ai.memory import memory
from lumen.sources.duckdb import DuckDBSource


def add_source_controls(replace_controls: bool = True):
    def wrap_upload():
        return io.BytesIO(upload.value)

    def add_table(event):
        with menu.param.update(loading=True):
            if input_tabs.active == 0 and upload.value:
                uploaded_bytes = wrap_upload()
                if upload.filename.endswith("csv"):
                    df = pd.read_csv(uploaded_bytes, parse_dates=True)
                elif upload.filename.endswith((".parq", ".parquet")):
                    df = pd.read_parquet(uploaded_bytes)
                elif upload.filename.endswith(".json"):
                    df = pd.read_json(uploaded_bytes)
                elif upload.filename.endswith(".xlsx"):
                    sheet_name = input_sheet.value or input_sheet.options[0]
                    df = pd.read_excel(uploaded_bytes, sheet_name=sheet_name)
                # TODO: add url support
                duckdb_source = DuckDBSource(uri=":memory:", ephemeral=True)
                duckdb_source._connection.from_df(df).to_view(name.value)
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
        add.visible = True

    def select_sheet(event):
        if not upload.filename.endswith(".xlsx"):
            return
        sheets = pd.read_excel(wrap_upload(), sheet_name=None)
        sheet_names = list(sheets.keys())
        input_sheet.options = sheet_names
        input_sheet.visible = True

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
    input_sheet = pn.widgets.Select(name="Sheet", visible=False)
    input_column = pn.Column(
        input_tabs,
        input_sheet,
        name,
        add,
        sizing_mode="stretch_width",
    )
    menu = pn.Column(
        input_column,
        tables,
        sizing_mode="stretch_width",
    )
    upload.param.watch(select_sheet, "filename")
    upload.param.watch(add_name, "filename")
    add.on_click(add_table)
    return menu
