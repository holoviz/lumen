import param

from panel.layout import Column
from panel.pane import PDF
from panel_material_ui import Markdown

from ..context import ContextModel, TContext
from ..schemas import Metaset
from .base_list import BaseListAgent


class DocumentListInputs(ContextModel):

    metaset: Metaset


class DocumentListAgent(BaseListAgent):
    """
    The DocumentListAgent lists all available documents provided by the user.
    """

    conditions = param.List(
        default=[
            "Use when user asks to list or see all available documents",
            "NOT when user asks about specific document content",
        ]
    )

    purpose = param.String(default="""
        Displays a list of all available documents.""")

    input_schema = DocumentListInputs

    _column_name = "Documents"

    _message_format = "Tell me about: {item}"

    def _create_row_content(self, context: TContext, source_name: str):
        """Create row content function that displays document preview."""
        metaset = context.get('metaset')
        source_catalog = context.get('source_catalog')

        def row_content(row):
            filename = row[self._column_name]

            # Check if we have raw PDF bytes in the source catalog
            if source_catalog and hasattr(source_catalog, '_available_metadata'):
                for meta in source_catalog._available_metadata:
                    if meta["filename"] == filename and "raw_bytes" in meta:
                        return PDF(meta["raw_bytes"], height=500, max_width=1200, sizing_mode="stretch_width")

            # Fall back to markdown preview from extracted content
            # First try full content from source catalog
            full_content = None
            if source_catalog and hasattr(source_catalog, '_available_metadata'):
                for meta in source_catalog._available_metadata:
                    if meta["filename"] == filename:
                        full_content = meta.get("content")
                        break

            if full_content:
                return Column(
                    Markdown(full_content, sizing_mode='stretch_width'),
                    max_height=500,
                    scroll='y-auto',
                    sizing_mode='stretch_width',
                )

            # Fall back to metaset chunks
            if not metaset or not metaset.has_docs:
                return Markdown("*No preview available*")

            doc_chunks = [doc for doc in metaset.docs if doc.filename == filename]
            if not doc_chunks:
                return Markdown("*No preview available*")

            full_text = "\n\n".join(chunk.text for chunk in doc_chunks)
            return Column(
                Markdown(full_text, sizing_mode='stretch_width'),
                max_height=500,
                scroll='y-auto',
                sizing_mode='stretch_width',
            )

        return row_content

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        metaset = context.get("metaset")
        if not metaset:
            return False
        return metaset.has_docs

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        metaset = context.get("metaset")
        if not metaset or not metaset.has_docs:
            return {}

        # Extract unique filenames from document chunks
        filenames = sorted(set(doc.filename for doc in metaset.docs))
        return {"Documents": filenames} if filenames else {}
