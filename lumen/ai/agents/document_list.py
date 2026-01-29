import param

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

        def row_content(row):
            filename = row[self._column_name]

            if not metaset or not metaset.has_docs:
                return Markdown("*No preview available*")

            # Find all chunks for this document
            doc_chunks = [doc for doc in metaset.docs if doc.filename == filename]

            if not doc_chunks:
                return Markdown("*No preview available*")

            # Show preview from first chunk
            first_chunk = doc_chunks[0]
            preview_text = first_chunk.text[:500]  # First 500 chars
            if len(first_chunk.text) > 500:
                preview_text += "..."

            content = f"**Preview:**\n\n{preview_text}"

            # Show chunk count if multiple
            if len(doc_chunks) > 1:
                content += f"\n\n*({len(doc_chunks)} chunks available)*"

            return Markdown(content, sizing_mode='stretch_width')

        return row_content

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        metaset = context.get("metaset")
        if not metaset:
            return False
        return metaset.has_docs and len(metaset.docs) > 1

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        metaset = context.get("metaset")
        if not metaset or not metaset.has_docs:
            return {}

        # Extract unique filenames from document chunks
        filenames = sorted(set(doc.filename for doc in metaset.docs))
        return {"Documents": filenames} if filenames else {}
