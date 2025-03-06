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

    include_descriptions = param.Boolean(default=True, doc="""
        Whether to include table and column descriptions in responses.""")

    max_concurrent = param.Integer(default=5, doc="""
        Maximum number of concurrent metadata fetch operations.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._table_metadata = {}  # Store metadata for all tables
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
            try:
                metadata = await asyncio.to_thread(source.get_metadata, table_name)
                source_table = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table_name}'
                self._table_metadata[source_table] = metadata

                enriched_text = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table_name}"
                if description := metadata.get('description', ''):
                    enriched_text += f"\nDescription: {description}"

                if columns := metadata.get('columns', {}):
                    enriched_text += "\nColumns:"
                    for col_name, col_info in columns.items():
                        col_text = f"\n- {col_name}"
                        if 'description' in col_info:
                            col_text += f": {col_info['description']}"
                        enriched_text += col_text

                if not self.vector_store.query(source_table, threshold=1):
                    self.vector_store.add([{"text": enriched_text}])
            except Exception:
                pass

    def _update_vector_store(self, _, __, sources):
        for source in sources:
            for table in source.get_tables():
                table_name = table.split('.')[-1] if '.' in table else table
                source_table = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table}'

                if not self.vector_store.query(source_table, threshold=1):
                    self.vector_store.add([{"text": source_table}])

                task = asyncio.create_task(
                    self._fetch_and_store_metadata(source, table_name)
                )
                self._metadata_tasks.add(task)
                task.add_done_callback(lambda t: self._metadata_tasks.discard(t))
                # TODO: Add ability to update existing source table?

    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> str:
        closest_tables = []
        message = "The most relevant tables are:\n"

        results = self.vector_store.query(
            messages[-1]["content"],
            top_k=self.n,
            threshold=self.min_similarity,
        )

        for result in results:
            if SOURCE_TABLE_SEPARATOR not in result["text"]:
                closest_tables.append(result["text"])
            else:
                first_line = result["text"].split("\n")[0] if "\n" in result["text"] else result["text"]
                table_name = first_line.split(SOURCE_TABLE_SEPARATOR, 1)[1]
                closest_tables.append(table_name)

        if not closest_tables:
            message = "No relevant tables found, but here are some other tables:\n"
            results = self.vector_store.query(messages[-1]["content"], top_k=self.n, threshold=0)
            for result in results:
                first_line = result["text"].split("\n")[0] if "\n" in result["text"] else result["text"]
                table_name = first_line.split(SOURCE_TABLE_SEPARATOR, 1)[1]
                closest_tables.append(table_name)

        self._memory["closest_tables"] = closest_tables

        table_descriptions = []
        for table in closest_tables:
            table_desc = f"- `{table}`"
            if self.include_descriptions:
                for source_table, metadata in self._table_metadata.items():
                    if table in source_table and metadata.get('description'):
                        table_desc += f": {metadata['description']}"
                        break
            table_descriptions.append(table_desc)

        return message + "\n".join(table_descriptions)


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
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
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
