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
    """
    FunctionTool wraps arbitrary functions and makes them available as a tool
    for an LLM to call. It inspects the arguments of the function and generates
    a pydantic Model that the LLM will populate.

    The function may also consume information in memory, e.g. the current
    table or pipeline by declaring the requires parameter.

    The function may also return context to add to the current working memory
    by returning a dictionary and declaring the `provides` parameter. Any
    keys listed in the provides parameter will be copied into working memory.
    """

    formatter = param.Parameter(default="{function}({arguments}) returned: {output}", doc="""
        Formats the return value for inclusion in the global context.
        Accepts the 'function', 'arguments' and 'output' as formatting variables.""")

    function = param.Callable(default=None, doc="""
        The function to call.""")

    provides = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it provides to current working memory.""")

    requires = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it requires to be in memory.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "FunctionTool" / "main.jinja2"
            },
        }
    )

    def __init__(self, function, **params):
        model = function_to_model(function, skipped=self.requires)
        super().__init__(
            function=function,
            name=function.__name__,
            purpose=model.__doc__ or "",
            **params
        )
        self._model = model

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        prompt = await self._render_prompt("main", messages)
        kwargs = {}
        if any(field not in self.requires for field in self._model.model_fields):
            kwargs = await self.llm.invoke(
                messages,
                system=prompt,
                response_model=self._model,
                allow_partial=False,
                max_retries=3,
            )
        arguments = dict(kwargs, **{k: self._memory[k] for k in self.requires})
        if param.parameterized.iscoroutinefunction(self.function):
            result = await self.function(**kwargs)
        else:
            result = self.function(**kwargs)
        if self.provides:
            if len(self.provides) == 1 and not isinstance(result, dict):
                self._memory[self.provides[0]] = result
            else:
                self._memory.update({result[key] for key in self.provides})
        return self.formatter.format(
            function=self.function.__name__,
            arguments=', '.join(f'{k}={v!r}' for k, v in arguments.items()),
            output=result
        )
