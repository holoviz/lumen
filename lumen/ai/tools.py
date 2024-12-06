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

    vector_store = param.ClassSelector(class_=VectorStore, constant=True, doc="""
        Vector store object which is queried to provide additional context
        before responding.""")

    def __init__(self, **params):
        if 'vector_store' not in params:
            params['vector_store'] = NumpyVectorStore(embeddings=NumpyEmbeddings())
        super().__init__(**params)
        for src in self._memory['sources']:
            for table in src.get_tables():
                self.vector_store.add([{"text": table}])

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        results = self.vector_store.query(messages[-1]["content"], top_k=self.n)
        self._memory['closest_tables'] = tables = [
            result['text'] for result in results if result["similarity"] > self.min_similarity
        ]
        return "The most relevant tables give the user query are: {}" + "\n".join(
            [f"- `{table}`" for table in tables]
        )
