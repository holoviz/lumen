import param

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
