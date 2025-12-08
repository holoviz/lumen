
from typing import Any

import param

from ..context import ContextModel, TContext
from .base_list import BaseListAgent


class DocumentListInputs(ContextModel):

    document_sources: dict[str, Any]


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

    _column_name = "Documents"

    _message_format = "Tell me about: {item}"

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        sources = context.get("document_sources")
        if not sources:
            return False
        return len(sources) > 1

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        # extract the filename, following this pattern `Filename: 'filename'``
        documents = [doc["metadata"].get("filename", "untitled") for doc in context.get("document_sources", [])]
        return {"Documents": documents} if documents else {}
