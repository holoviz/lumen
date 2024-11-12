import io
import zipfile

import pandas as pd
import param

from panel.layout import Column, FlexBox, Tabs
from panel.viewable import Viewer
from panel.widgets import (
    Button, FileDropper, NestedSelect, Select, Tabulator, TextInput,
)

from ..sources.duckdb import DuckDBSource
from .memory import _Memory, memory


class TableControls(Viewer):

    filename = param.String(default="", doc="Filename")
    table = param.String(default="", doc="Table name")
    extension = param.String(default="", doc="File extension")
    sheet = param.ObjectSelector(default=None, objects=[], doc="Sheet")

    _load = param.Event(doc="Load table")

    def __init__(self, file: io.BytesIO, **params):
        filename = params["filename"]
        if "/" in filename:
            filename, extension = filename.rsplit("/")[-1].split(".", maxsplit=1)
        elif not self.filename and filename:
            filename, extension = filename.rsplit(".", maxsplit=1)
        params["filename"] = filename
        params["table"] = filename.replace("-", "_")
        params["extension"] = extension
        super().__init__(**params)
        self.file = file
        self._name_input = TextInput.from_param(
            self.param.table, name="Table name"
        )
        self._sheet_select = Select.from_param(
            self.param.sheet, name="Sheet", visible=False
        )
        self.box = FlexBox(
            self._name_input,
            self._sheet_select,
        )
        self.param.trigger("table")  # fix name
        self.param.trigger("_load")  # to prevent blocking

    @param.depends("_load", watch=True)
    async def _select_sheet(self):
        if not self.extension.endswith("xlsx"):
            return
        import openpyxl

        wb = openpyxl.load_workbook(self.file, read_only=True)
        with param.parameterized.batch_call_watchers(self):
            self.param.sheet.objects = wb.sheetnames
            self.sheet = wb.sheetnames[0]
            self._sheet_select.visible = True

    @param.depends("table", watch=True)
    async def _replace_with_underscore(self):
        self.table = "".join(
            c if c.isalnum() or c == "." else "_" for c in self.table
        ).strip("_").lower()

    def __panel__(self):
        return self.box


class SourceControls(Viewer):

    add = param.Event(doc="Add tables")

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    multiple = param.Boolean(default=False, doc="Allow multiple files")

    replace_controls = param.Boolean(default=False, doc="Replace controls")

    select_existing = param.Boolean(default=True, doc="Select existing table")

    _last_table = param.String(default="", doc="Last table added")

    def __init__(self, **params):
        super().__init__(**params)

        self.tables_tabs = Tabs(sizing_mode="stretch_width")
        self._file_input = FileDropper(
            height=100,
            multiple=self.param.multiple,
            margin=(0, 10, 0, 0),
            sizing_mode="stretch_width",
            # accepted_filetypes=[".csv", ".parquet", ".parq", ".json", ".xlsx"],
        )
        self._file_input.param.watch(self._generate_table_controls, "value")
        self._upload_tabs = Tabs(sizing_mode="stretch_width")

        self._input_tabs = Tabs(
            ("Upload", Column(self._file_input, self._upload_tabs)),
            sizing_mode="stretch_both",
        )

        if self.select_existing:
            nested_sources_tables = {
                source.name: source.get_tables() for source in self._memory["available_sources"]
            }
            first_table = {k: nested_sources_tables[k][0] for k in list(nested_sources_tables)[:1]}
            self._select_table = NestedSelect(
                name="Table",
                value=first_table,
                options=nested_sources_tables,
                levels=["source", "table"],
                sizing_mode="stretch_width",
            )
            self._input_tabs.append(("Select", self._select_table))
            self._select_table.param.watch(self._generate_table_controls, "value")

        self._add_button = Button.from_param(
            self.param.add,
            name="Use tables",
            icon="table-plus",
            visible=False,
            button_type="success",
        )
        self.menu = Column(
            self._input_tabs if self.select_existing else self._input_tabs[0],
            self._add_button,
            self.tables_tabs,
            sizing_mode="stretch_width",
        )

        self._table_controls = []

    @property
    def _memory(self):
        return memory if self.memory is None else self.memory

    def _generate_table_controls(self, event):
        if self._input_tabs.active == 0:
            self._upload_tabs.clear()
            self._table_controls.clear()
            for filename, file in self._file_input.value.items():
                table_controls = TableControls(
                    io.BytesIO(file) if isinstance(file, bytes) else io.StringIO(file),
                    filename=filename,
                )
                self._upload_tabs.append((filename, table_controls))
                self._table_controls.append(table_controls)

            if len(self._upload_tabs) > 0:
                self._add_button.visible = True
            else:
                self._add_button.visible = False
        elif self._input_tabs.active == 1:
            self._add_button.visible = True

    def _add_table(
        self,
        duckdb_source: DuckDBSource,
        file: io.BytesIO | io.StringIO,
        table_controls: TableControls,
    ):
        conn = duckdb_source._connection
        extension = table_controls.extension
        table = table_controls.table
        sql_expr = f"SELECT * FROM {table}"
        params = {}
        conversion = None
        if extension.endswith("csv"):
            df = pd.read_csv(file, parse_dates=True)
        elif extension.endswith(("parq", "parquet")):
            df = pd.read_parquet(file)
        elif extension.endswith("json"):
            df = pd.read_json(file)
        elif extension.endswith("xlsx"):
            sheet = table_controls.sheet
            df = pd.read_excel(file, sheet_name=sheet)
        elif extension.endswith(('geojson', 'wkt', 'zip')):
            if extension.endswith('zip'):
                zf = zipfile.ZipFile(file)
                if not any(f.filename.endswith('shp') for f in zf.filelist):
                    raise ValueError("Could not interpret zip file contents")
                file.seek(0)
            import geopandas as gpd
            geo_df = gpd.read_file(file)
            df = pd.DataFrame(geo_df)
            df['geometry'] = geo_df['geometry'].to_wkb()
            params['initializers'] = init = ["""
            INSTALL spatial;
            LOAD spatial;
            """]
            conn.execute(init[0])
            cols = ', '.join(f'"{c}"' for c in df.columns if c != 'geometry')
            conversion = f'CREATE TEMP TABLE {table} AS SELECT {cols}, ST_GeomFromWKB(geometry) as geometry FROM {table}_temp'
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        duckdb_source.param.update(params)
        df_rel = conn.from_df(df)
        if conversion:
            conn.register(f'{table}_temp', df_rel)
            conn.execute(conversion)
            conn.unregister(f'{table}_temp')
        else:
            df_rel.to_view(table)
        duckdb_source.tables[table] = sql_expr
        self._memory["current_source"] = duckdb_source
        self._memory["current_table"] = table
        if "available_sources" in self._memory:
            self._memory["available_sources"].append(duckdb_source)
            self._memory.trigger("available_sources")
        else:
            self._memory["available_sources"] = [duckdb_source]
        self._last_table = table

    @param.depends("add", watch=True)
    def add_tables(self):
        with self.menu.param.update(loading=True):
            duckdb_source = DuckDBSource(uri=":memory:", ephemeral=True, name='Uploaded')
            if duckdb_source.tables is None:
                duckdb_source.tables = {}
            if self._input_tabs.active == 0:
                for i in range(len(self._upload_tabs)):
                    table_controls = self._table_controls[i]
                    self._add_table(duckdb_source, table_controls.file, table_controls)

                if self.replace_controls:
                    src = self._memory["current_source"]
                    self.tables_tabs[:] = [
                        (t, Tabulator(src.get(t), sizing_mode="stretch_both"))
                        for t in src.get_tables()
                    ]
                    self.menu[0].visible = False
                    self._add_button.visible = False
            elif self.select_existing and self._input_tabs.active == 1:
                table = self._select_table.value["table"]
                duckdb_source.tables[table] = f"SELECT * FROM {table}"
                self._memory["current_source"] = duckdb_source
                self._memory["current_table"] = table
                self._memory["available_sources"].append(duckdb_source)
                self._last_table = table

    def __panel__(self):
        return self.menu
