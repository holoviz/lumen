import io
import zipfile

import pandas as pd
import param

from panel.layout import (
    Column, FlexBox, Row, Tabs,
)
from panel.pane.markup import Markdown
from panel.viewable import Viewer
from panel.widgets import (
    Button, FileDropper, NestedSelect, Select, Tabulator, TextInput,
    ToggleIcon,
)

from ..sources.duckdb import DuckDBSource
from .memory import _Memory, memory

TABLE_EXTENSIONS = ("csv", "parquet", "parq", "json", "xlsx", "geojson", "wkt", "zip")
DOCUMENT_EXTENSIONS = ("doc", "docx", "pdf", "txt", "md", "rst")


class MediaControls(Viewer):

    alias = param.String(default="", doc="What to name the uploaded file when querying it")

    filename = param.String(default="", doc="Filename")

    extension = param.String(default="", doc="File extension")

    _load = param.Event(doc="Load table")

    def __init__(self, file_obj: io.BytesIO, **params):
        filename = params["filename"]
        if "/" in filename:
            filename, extension = filename.rsplit("/")[-1].split(".", maxsplit=1)
        elif not self.filename and filename:
            filename, extension = filename.rsplit(".", maxsplit=1)
        params["filename"] = filename
        params["alias"] = filename.replace("-", "_")
        params["extension"] = extension
        super().__init__(**params)
        self.file_obj = file_obj

    def __panel__(self):
        return self.box


class DocumentControls(MediaControls):

    def __init__(self, file_obj: io.BytesIO, **params):
        super().__init__(file_obj, **params)
        self._name_input = TextInput.from_param(
            self.param.alias, name="Document alias",
        )
        self._metadata_input = TextInput(name="Metadata", description="Comments or notes about this document")
        self.box = FlexBox(
            self._name_input,
            self._metadata_input,
        )


class TableControls(MediaControls):

    sheet = param.Selector(default=None, objects=[], doc="Sheet")

    def __init__(self, file_obj: io.BytesIO, **params):
        super().__init__(file_obj, **params)
        self._name_input = TextInput.from_param(
            self.param.alias, name="Table alias",
        )
        self._sheet_select = Select.from_param(
            self.param.sheet, name="Sheet", visible=False
        )
        self.box = FlexBox(
            self._name_input,
            self._sheet_select,
        )
        self.param.trigger("filename")  # fix name
        self.param.trigger("_load")  # to prevent blocking

    @param.depends("_load", watch=True)
    async def _select_sheet(self):
        if not self.extension.endswith("xlsx"):
            return
        import openpyxl

        wb = openpyxl.load_workbook(self.file_obj, read_only=True)
        with param.parameterized.batch_call_watchers(self):
            self.param.sheet.objects = wb.sheetnames
            self.sheet = wb.sheetnames[0]
            self._sheet_select.visible = True

    @param.depends("alias", watch=True)
    async def _replace_with_underscore(self):
        self.alias = "".join(
            c if c.isalnum() or c == "." else "_" for c in self.alias
        ).strip("_").lower()


class SourceControls(Viewer):

    add = param.Event(doc="Use uploaded file(s)")

    cancel = param.Event(doc="Cancel")

    cancellable = param.Boolean(default=True, doc="Show cancel button")

    clear_uploads = param.Boolean(default=False, doc="Clear uploaded file tabs")

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
        self._file_input.param.watch(self._generate_media_controls, "value")
        self._upload_tabs = Tabs(sizing_mode="stretch_width")

        self._input_tabs = Tabs(
            ("Upload", Column(self._file_input, self._upload_tabs)),
            sizing_mode="stretch_both",
        )

        if self.select_existing:
            nested_sources_tables = {
                source.name: source.get_tables() for source in self._memory["sources"]
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
            self._select_table.param.watch(self._generate_media_controls, "value")

        self._add_button = Button.from_param(
            self.param.add,
            name="Use file(s)",
            icon="table-plus",
            visible=False,
            button_type="success",
        )

        self._cancel_button = Button.from_param(
            self.param.cancel,
            name="Cancel",
            icon="circle-x",
            visible=self.param.cancellable,
        )

        self._message_placeholder = Markdown(css_classes=["message"] if self.replace_controls else [], visible=False)

        self.menu = Column(
            self._input_tabs if self.select_existing else self._input_tabs[0],
            self._add_button,
            self._cancel_button,
            self.tables_tabs,
            self._message_placeholder,
            sizing_mode="stretch_width",
        )

        self._media_controls = []

    @property
    def _memory(self):
        return memory if self.memory is None else self.memory

    def _generate_media_controls(self, event):
        if self._input_tabs.active == 0:
            self._upload_tabs.clear()
            self._media_controls.clear()
            for filename, file in self._file_input.value.items():
                file_obj = io.BytesIO(file) if isinstance(file, bytes) else io.StringIO(file)
                if filename.lower().endswith(TABLE_EXTENSIONS):
                    table_controls = TableControls(
                        file_obj,
                        filename=filename,
                    )
                elif filename.lower().endswith(DOCUMENT_EXTENSIONS):
                    table_controls = DocumentControls(
                        file_obj,
                        filename=filename,
                    )
                self._upload_tabs.append((filename, table_controls))
                self._media_controls.append(table_controls)

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
        table = table_controls.alias
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
        self._memory["source"] = duckdb_source
        self._memory["table"] = table
        if "sources" in self._memory:
            self._memory["sources"].append(duckdb_source)
            self._memory.trigger("sources")
        else:
            self._memory["sources"] = [duckdb_source]
        self._last_table = table

    def _add_document(
        self,
        file: io.BytesIO,
        document_controls: DocumentControls
    ):
        extension = document_controls.extension.lower()
        if extension in ("txt", "md", "rst"):
            file.seek(0)
            text = file.read().decode("utf-8", errors="replace")
        elif extension in ("doc", "docx"):
            # Ensure python-docx is installed: pip install python-docx
            from docx import Document
            file.seek(0)
            doc = Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif extension == "pdf":
            # Ensure pdfminer.six is installed: pip install pdfminer.six
            from pdfminer.high_level import extract_text
            file.seek(0)
            text = extract_text(file)
        else:
            raise ValueError(f"Unsupported document extension: {extension}")

        metadata = f"Filename: {document_controls.filename}\n\n" + document_controls._metadata_input.value
        document = {"text": text, "metadata": metadata}
        if "document_sources" in self._memory:
            self._memory["document_sources"].append(document)
            self._memory.trigger("document_sources")
        else:
            self._memory["document_sources"] = [document]

    @param.depends("add", watch=True)
    def add_medias(self):
        if self._input_tabs.active == 1 and not self.select_existing:
            return
        with self.menu.param.update(loading=True):
            if self._input_tabs.active == 1:
                duckdb_source = DuckDBSource(uri=":memory:", ephemeral=True, name='Uploaded', tables={})
                table = self._select_table.value["table"]
                duckdb_source.tables[table] = f"SELECT * FROM {table}"
                self._memory["source"] = duckdb_source
                self._memory["table"] = table
                self._memory["sources"].append(duckdb_source)
                self._last_table = table
                return

            source = None
            n_tables = 0
            n_docs = 0
            for i in range(len(self._upload_tabs)):
                media_controls = self._media_controls[i]
                if media_controls.extension.endswith(TABLE_EXTENSIONS):
                    if source is None:
                        source = DuckDBSource(uri=":memory:", ephemeral=True, name='Uploaded', tables={})
                    self._add_table(source, media_controls.file_obj, media_controls)
                    n_tables += 1
                elif media_controls.extension.endswith(DOCUMENT_EXTENSIONS):
                    self._add_document(media_controls.file_obj, media_controls)
                    n_docs += 1

            if self.replace_controls:
                src = self._memory.get("source")
                if src:
                    self.tables_tabs[:] = [
                        (t, Tabulator(src.get(t), sizing_mode="stretch_both"))
                        for t in src.get_tables()
                    ]
                self.menu[0].visible = False
                self._add_button.visible = False
                self._cancel_button.visible = False
                if n_tables == 0:
                    self.tables_tabs.visible = False
                    self.menu.height = 70

            if self.clear_uploads:
                self._upload_tabs.clear()
                self._media_controls.clear()
                self._add_button.visible = False

            self._message_placeholder.param.update(
                object=f"Successfully uploaded {len(self._upload_tabs)} files ({n_tables} table(s), {n_docs} document(s)).",
                visible=True,
            )

    def __panel__(self):
        return self.menu


class RetryControls(Viewer):

    active = param.Boolean(False, doc="Click to retry")

    reason = param.String(doc="Reason for retry")

    def __init__(self, **params):
        super().__init__(**params)
        icon = ToggleIcon.from_param(
            self.param.active,
            name=" ",
            description="Prompt LLM to retry",
            icon="repeat-once",
            active_icon="x",
            margin=5,
        )
        self._text_input = TextInput(
            placeholder="Enter feedback and press the <Enter> to retry.",
            visible=icon.param.value,
            max_length=200,
            margin=(5, 0),
        )
        row = Row(icon, self._text_input)
        self._row = row

        self._text_input.param.watch(self._enter_reason, "enter_pressed")

    def _enter_reason(self, _):
        self.param.update(
            reason=self._text_input.value,
            active=False,
        )

    def __panel__(self):
        return self._row
