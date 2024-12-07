from __future__ import annotations

from typing import Any

import param

from .actor import Actor
from .embeddings import NumpyEmbeddings
from .llm import Message
from .vector_store import NumpyVectorStore, VectorStore


class Tool(Actor):
    """
    A Tool can be invoked by another Actor to provide additional
    context or respond to a question.
    """

    provides = param.List(default=[], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    requires = param.List(default=[], readonly=True, doc="""
        List of context that this Tool requires to be run.""")


class TableLookup(Tool):
    """
    The TableLookup tool creates a vector store of all available tables
    and responds with a list of the most relevant tables given the user
    query.
    """

    min_similarity = param.Number(default=0.3, doc="""
        The minimum similarity to include a table.""")

    n = param.Integer(default=3, bounds=(0, None), doc="""
        The number of table results to return.""")

    requires = param.List(default=["sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    provides = param.List(default=["closest_tables"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    vector_store = param.ClassSelector(class_=VectorStore, constant=True, doc="""
        Vector store object which is queried to provide additional context
        before responding.""")

    def __init__(self, **params):
        if 'vector_store' not in params:
            params['vector_store'] = NumpyVectorStore(embeddings=NumpyEmbeddings())
        super().__init__(**params)
        self._memory.on_change('sources', self._update_vector_store_table_list)

    def _update_vector_store_table_list(self, _, __, sources):
        for source in sources:
            for table in source.get_tables():
                source_table = f'{source.name}//{table}'
                if not self.vector_store.query(source_table, threshold=1):
                    self.vector_store.add([{"text": source_table}])

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        self._memory["closest_tables"] = closest_tables = [
            result["text"].split("//")[1] for result in
            self.vector_store.query(messages[-1]["content"], top_k=self.n, threshold=self.min_similarity)
        ]

        message = "The most relevant tables are:\n"
        if not closest_tables:
            self._memory["closest_tables"] = closest_tables = [
                result["text"].split("//")[1] for result in
                self.vector_store.query(messages[-1]["content"], top_k=self.n, threshold=0)
            ]
            message = "No relevant tables found, but here are some other tables:\n"
        return message + "\n".join(f"- `{table}`" for table in closest_tables)
