from __future__ import annotations

import asyncio
import re

from typing import Any

import param

from panel.viewable import Viewable

from ..views.base import View
from .actor import Actor, ContextProvider
from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
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
        self._memory.on_change('document_sources', self._update_vector_store)

    def _update_vector_store(self, _, __, sources):
        for source in sources:
            metadata = source.get("metadata", {})
            filename = metadata.get("filename")
            if filename:
                # overwrite existing items with the same filename
                existing_items = self.vector_store.filter_by({'filename': filename})
                if existing_ids := [item['id'] for item in existing_items]:
                    self.vector_store.delete(existing_ids)
            self.vector_store.add([{"text": source["text"], "metadata": source.get("metadata", {})}])

    async def respond(self, messages: list[Message], **kwargs: Any) -> str:
        query = re.findall(r"'(.*?)'", messages[-1]["content"])[0]
        results = self.vector_store.query(query, top_k=self.n, threshold=self.min_similarity)
        closest_doc_chunks = [
            f"{result['text']} (Relevance: {result['similarity']:.1f} - Metadata: {result['metadata']}"
            for result in results
        ]
        if not closest_doc_chunks:
            return ""
        message = "Please augment your response with the following context if relevant:\n"
        message += "\n".join(f"- {doc}" for doc in closest_doc_chunks)
        return message


class TableLookup(VectorLookupTool):
    """
    TableLookup tool that creates a vector store of all available tables
    and responds with relevant tables for user queries.
    """

    requires = param.List(default=["sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    provides = param.List(default=["closest_tables"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    include_metadata = param.Boolean(default=True, doc="""
        Whether to include table descriptions in the embeddings and responses.""")

    include_columns = param.Boolean(default=False, doc="""
        Whether to include column names and descriptions in the embeddings.""")

    max_concurrent = param.Integer(default=2, doc="""
        Maximum number of concurrent metadata fetch operations.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._table_metadata = {}
        self._metadata_tasks = set()  # Track ongoing metadata fetch tasks
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._memory.on_change('sources', self._update_vector_store)

        if sources := self._memory.get("sources"):
            self._update_vector_store(None, None, sources)
        elif source := self._memory.get("source"):
            self._update_vector_store(None, None, [source])

    async def _fetch_and_store_metadata(self, source, table_name):
        """Fetch metadata for a table and store it in the vector store."""
        async with self._semaphore:
            metadata = await asyncio.to_thread(source.get_metadata, table_name)
        multi_source = len(self._memory.get("sources", [])) > 1
        if multi_source:
            enriched_text = table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table_name}"
        else:
            enriched_text = table_slug = table_name
        self._table_metadata[table_slug] = metadata

        vector_metadata = {"source": source.name, "table": table_name}
        if self.vector_store.filter_by(vector_metadata):
            return

        if description := metadata.get('description', ''):
            enriched_text += f"\nDescription: {description}"
        if self.include_columns and (columns := metadata.get('columns', {})):
            enriched_text += "\nColumns:"
            for col_name, col_info in columns.items():
                col_text = f"\n- {col_name}"
                if 'description' in col_info:
                    col_text += f": {col_info['description']}"
                enriched_text += col_text
        self.vector_store.add([{"text": enriched_text, "metadata": vector_metadata}])

    def _update_vector_store(self, _, __, sources):
        for source in sources:
            for table in source.get_tables():
                metadata = {"table": table, "source": source.name}
                if not self.include_metadata:
                    if self.vector_store.filter_by(metadata):
                        # TODO: Add ability to update existing source table?
                        continue
                    table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"
                    self.vector_store.add([{"text": table_slug, "metadata": metadata}])
                else:
                    task = asyncio.create_task(
                        self._fetch_and_store_metadata(source, table)
                    )
                    self._metadata_tasks.add(task)
                    task.add_done_callback(lambda t: self._metadata_tasks.discard(t))

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        """
        Fetches the closest tables based on the user query, depending on whether
        any tables reach the min_similarity threshold we adjust the messsage.
        """
        query = messages[-1]["content"]
        results = self.vector_store.query(query, top_k=self.n)
        multi_source = len(self._memory.get("sources", []))
        any_matches = any(result['similarity'] >= self.min_similarity for result in results)

        closest_tables, descriptions = [], []
        for result in results:
            if any_matches and result['similarity'] < self.min_similarity:
                continue
            table_name = result['metadata']['table']
            if multi_source:
                source_name = result['metadata']['source']
                table_name = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"
            description = f"- `{table_name}`: {result['similarity']:.3f} similarity"
            if table_metadata := self._table_metadata.get(table_name):
                description += f"\n    {table_metadata['description']}"
            closest_tables.append(table_name)
            descriptions.append(description)

        if any_matches:
            message = "The most relevant tables are:\n"
        else:
            message = "No relevant tables found, but here are some other tables:\n"
        self._memory["closest_tables"] = closest_tables
        return message + "\n".join(descriptions)


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

    function = param.Callable(default=None, allow_refs=False, doc="""
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
            purpose=params.pop("purpose", model.__doc__) or "",
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
        if isinstance(result, (View, Viewable)):
            return result
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
