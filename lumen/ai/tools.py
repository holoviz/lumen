from __future__ import annotations

from typing import Any

import param

from .actor import Actor, ContextProvider
from .config import PROMPTS_DIR
from .embeddings import NumpyEmbeddings
from .llm import Message
from .translate import function_to_model
from .vector_store import NumpyVectorStore, VectorStore


class Tool(Actor, ContextProvider):
    """
    A Tool can be invoked by another Actor to provide additional
    context or respond to a question. Unlike an Agent they never
    interact with or on behalf of a user directly.
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


class FunctionTool(Tool):

    formatter = param.Parameter(default="{function}({args}) result: {output}")

    function = param.Callable(default=None, doc="""
        The function to call.""")

    provides = param.List(default=[])

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "FunctionTool" / "main.jinja2"
            },
        }
    )

    def __init__(self, function, **params):
        super().__init__(
            function=function,
            name=function.__name__,
            purpose=f"Invokes the function with the following docstring: {function.__doc__}." if function.__doc__ else "",
            **params
        )

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        model = function_to_model(self.function, skipped=list(self._memory))
        prompt = await self._render_prompt("main", messages)
        kwargs = await self.llm.invoke(
            messages,
            system=prompt,
            response_model=model,
            allow_partial=False,
            max_retries=3,
        )
        result = self.function(**dict(kwargs))
        if self.provides:
            self._memory.update({result[key] for key in self.provides})
        return f"{self.function.__name__} result: **{result}**"
