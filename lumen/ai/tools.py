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


class VectorLookupTool(Tool):
    """
    Baseclass for tools that search a vector database for relevant
    chunks.
    """

    min_similarity = param.Number(default=0.1, doc="""
        The minimum similarity to include a document.""")

    n = param.Integer(default=3, bounds=(0, None), doc="""
        The number of document results to return.""")

    vector_store = param.ClassSelector(class_=VectorStore, constant=True, doc="""
        Vector store object which is queried to provide additional context
        before responding.""")

    __abstract = True

    def __init__(self, **params):
        if 'vector_store' not in params:
            params['vector_store'] = NumpyVectorStore(embeddings=NumpyEmbeddings())
        super().__init__(**params)


class DocumentLookup(VectorLookupTool):
    """
    The DocumentLookup tool creates a vector store of all available documents
    and responds with a list of the most relevant documents given the user query.
    Always use this for more context.
    """

    purpose = param.String(doc="""
        Looks up relevant""")

    requires = param.List(default=["document_sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._memory.on_change("document_sources", self._update_vector_store)
        if "document_sources" in self._memory:
            self._update_vector_store(None, None, self._memory["document_sources"])

    def _update_vector_store(self, _, __, sources):
        for source in sources:
            if not self.vector_store.query(source["text"], threshold=1):
                self.vector_store.add([{"text": source["text"], "metadata": source.get("metadata", "")}])

    async def respond(self, messages: list[Message], **kwargs: Any) -> str:
        query = messages[-1]["content"]
        results = self.vector_store.query(query, top_k=self.n, threshold=self.min_similarity)
        closest_doc_chunks = [
            f"{result['text']} (Relevance: {result['similarity']:.1f} - Metadata: {result['metadata']}"
            for result in results
        ]
        if not closest_doc_chunks:
            return ""
        message = "Please augment your response with the following context:\n"
        message += "\n".join(f"- {doc}" for doc in closest_doc_chunks)
        return message


class TableLookup(VectorLookupTool):
    """
    The TableLookup tool creates a vector store of all available tables
    and responds with a list of the most relevant tables given the user
    query.
    """

    requires = param.List(default=["sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    provides = param.List(default=["closest_tables"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._memory.on_change('sources', self._update_vector_store)
        if "sources" in self._memory:
            self._update_vector_store(None, None, self._memory["sources"])

    def _update_vector_store(self, _, __, sources):
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
            model_spec = self.prompts["main"].get("llm_spec", "default")
            kwargs = await self.llm.invoke(
                messages,
                system=prompt,
                model_spec=model_spec,
                response_model=self._model,
                allow_partial=False,
                max_retries=3,
            )
        arguments = dict(kwargs, **{k: self._memory[k] for k in self.requires})
        if param.parameterized.iscoroutinefunction(self.function):
            result = await self.function(**arguments)
        else:
            result = self.function(**arguments)
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