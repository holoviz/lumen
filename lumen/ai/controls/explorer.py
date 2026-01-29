from __future__ import annotations

import param

from panel.io import state
from panel.layout import Row
from panel.viewable import Viewer
from panel_material_ui import AutocompleteInput

from ...pipeline import Pipeline
from ..config import SOURCE_TABLE_SEPARATOR
from .editors import SQLEditor


class TableExplorer(Viewer):
    """
    TableExplorer provides a high-level entrypoint to explore tables in a split UI.
    It allows users to load tables, explore them using Graphic Walker, and then
    interrogate the data via a chat interface.
    """

    add_exploration = param.Event(label="Explore table")

    table_slug = param.Selector(label="Select table(s) to preview.")

    context = param.Dict(default={})

    def __init__(self, **params):
        self._initialized = False
        super().__init__(**params)
        self._table_autocomplete = AutocompleteInput(
            value="",
            options=[],
            restrict=True,
            case_sensitive=False,
            search_strategy="includes",
            min_characters=0,
            margin=(10, 0),
            placeholder="Select or search for a table...",
            sizing_mode='stretch_width',
        )
        self._table_autocomplete.param.watch(self._on_table_selected, 'value')
        self._input_row = Row(
            self._table_autocomplete
        )
        self.source_map = {}
        self._layout = self._input_row

    def _on_table_selected(self, event):
        """Handle table selection from autocomplete."""
        if event.new:
            self.table_slug = event.new
            self.param.trigger('add_exploration')
            # Clear the autocomplete after selection
            self._table_autocomplete.value = ""

    @param.depends("context", watch=True)
    async def sync(self, context=None):
        init = not self._initialized
        self._initialized = True
        context = context or self.context
        if "sources" in context:
            sources = context["sources"]
        elif "source" in context:
            sources = [context["source"]]
        else:
            return
        selected = [self.table_slug] if self.table_slug else []
        deduplicate = len(sources) > 1
        new = {}

        # Build the source map for UI display
        for source in sources:
            tables = source.get_tables()
            for table in tables:
                if deduplicate:
                    table = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table}'

                if (table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)[-1] not in self.source_map and
                    not init and state.loaded):
                    selected.append(table)
                new[table] = source

        self.source_map.clear()
        self.source_map.update(new)
        selected = selected[-1] if len(selected) == 1 else None

        self._table_autocomplete.options = list(self.source_map)
        if selected:
            self.table_slug = selected
        self._input_row.visible = bool(self.source_map)
        self._initialized = True

    def create_sql_output(self) -> SQLEditor | None:
        if not self.table_slug:
            return

        source = self.source_map[self.table_slug]
        if SOURCE_TABLE_SEPARATOR in self.table_slug:
            _, table = self.table_slug.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
        else:
            table = self.table_slug
        new_table = f"select_{table}"
        sql_expr = f"SELECT * FROM \"{table}\""
        new_source = source.create_sql_expr_source({new_table: sql_expr})
        pipeline = Pipeline(source=new_source, table=new_table)
        return SQLEditor(spec=sql_expr, component=pipeline)

    def __panel__(self):
        return self._layout
