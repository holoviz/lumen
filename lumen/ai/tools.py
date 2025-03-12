from __future__ import annotations

import asyncio
import re

from textwrap import indent
from typing import Any

import param

from panel.viewable import Viewable

from lumen.sources.base import Source

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

    always_use = param.Boolean(default=False, doc="""
        Whether to always use this tool, even if it is not explicitly
        required by the current context.""")


class VectorLookupTool(Tool):
    """
    Baseclass for tools that search a vector database for relevant
    chunks.
    """

    min_similarity = param.Number(default=0.05, doc="""
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

    purpose = param.String(default="""
        Looks up relevant documents based on the user query.""")

    requires = param.List(default=["document_sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._memory.on_change('document_sources', self._update_vector_store)

    async def _update_vector_store(self, _, __, sources):
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

    purpose = param.String(default="""
        Looks up relevant tables based on the user query.""")

    requires = param.List(default=["sources"], readonly=True, doc="""
        List of context that this Tool requires to be run.""")

    provides = param.List(default=["closest_tables", "sources_raw_metadata"], readonly=True, doc="""
        List of context values this Tool provides to current working memory.""")

    batched = param.Boolean(default=True, doc="""
        Whether to batch the table metadata fetches.""")

    include_metadata = param.Boolean(default=True, doc="""
        Whether to include table descriptions in the embeddings and responses.""")

    include_columns = param.Boolean(default=True, doc="""
        Whether to include column names and descriptions in the embeddings.""")

    include_misc = param.Boolean(default=False, doc="""
        Whether to include miscellaneous metadata in the embeddings,
        besides table and column descriptions.""")

    max_concurrent = param.Integer(default=1, doc="""
        Maximum number of concurrent metadata fetch operations.""")

    def __init__(self, **params):
        super().__init__(**params)
        if "sources_raw_metadata" not in self._memory:
            self._memory["sources_raw_metadata"] = {}

        self._table_metadata = {}
        self._metadata_tasks = set()  # Track ongoing metadata fetch tasks
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._memory.on_change('sources', self._update_vector_store)
        self._memory.trigger('sources')

    async def _fetch_and_store_metadata(self, source: Source, table_name: str):
        """Fetch metadata for a table and store it in the vector store."""
        async with self._semaphore:
            if self.batched:
                source_metadata = self._memory["sources_raw_metadata"][source.name]
                if isinstance(source_metadata, asyncio.Task):
                    source_metadata = await source_metadata
                    self._memory["sources_raw_metadata"][source.name] = source_metadata
            else:
                source_metadata = await asyncio.to_thread(source.get_metadata, table_name, batched=False)

        table_name_key = next((
            key for key in (table_name.upper(), table_name.lower(), table_name)
            if key in source_metadata), None
        )
        if not table_name_key:
            return
        table_name = table_name_key
        table_metadata = source_metadata[table_name]

        # IMPORTANT: re-insert using table slug
        # we need to store a copy of table_metadata so it can be used to inject context into the LLM
        table_slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table_name}"
        self._table_metadata[table_slug] = table_metadata.copy()
        self._table_metadata[table_slug]["source_name"] = source.name  # need this to rebuild the slug

        vector_metadata = {"source": source.name, "table_name": table_name}
        if existing_items := self.vector_store.filter_by(vector_metadata):
            if existing_items and existing_items[0].get("metadata", {}).get("enriched"):
                return
            self.vector_store.delete([item['id'] for item in existing_items])

        # ---
        # TODO: maybe migrate to jinja2 template?
        # the following is for the embeddings/vector store use only (i.e. filter from 1000s of tables)
        vector_metadata["enriched"] = True
        enriched_text = f"Table: {table_name}"
        if description := table_metadata.pop('description', ''):
            enriched_text += f"\nDescription: {description}"
        if self.include_columns and (columns := table_metadata.pop('columns', {})):
            enriched_text += "\nColumns:"
            for col_name, col_info in columns.items():
                col_text = f"\n- {col_name}"
                if col_info.get("description"):
                    col_text += f": {col_info['description']}"
                enriched_text += col_text
            enriched_text += "\n"
        if self.include_misc:
            enriched_text += "\nMiscellaneous:"
            for key, value in table_metadata.items():
                if key == "enriched":
                    continue
                enriched_text += f"\n- {key}: {value}"
        self.vector_store.add([{"text": enriched_text, "metadata": vector_metadata}])

    async def _update_vector_store(self, _, __, sources):
        for source in sources:
            if self.include_metadata and self._memory["sources_raw_metadata"].get(source.name) is None:
                self._memory["sources_raw_metadata"][source.name] = asyncio.create_task(
                    asyncio.to_thread(source.get_metadata)
                )

            tables = source.get_tables()
            # since enriching the metadata might take time, first add basic metadata (table name)
            for table_name in tables:
                metadata = {"source": source.name, "table_name": table_name}
                if self.vector_store.filter_by(metadata):
                    # TODO: Add ability to update existing source table?
                    continue
                self.vector_store.add([{"text": table_name, "metadata": metadata}])

            # if metadata is included, upsert the existing
            if self.include_metadata:
                for table in tables:
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
        any_matches = any(result['similarity'] >= self.min_similarity for result in results)

        # this is where we inject context into the LLM!
        closest_tables, descriptions = [], []
        for result in results:
            if any_matches and result['similarity'] < self.min_similarity:
                continue
            source_name = result['metadata']["source"]
            table_name = result['metadata']["table_name"]
            table_slug = f"{source_name}{SOURCE_TABLE_SEPARATOR}{table_name}"
            description = f"- `{table_slug}` ({result['similarity']:.3f} similarity):\n"
            if table_metadata := self._table_metadata.get(table_slug):
                if table_description := table_metadata.get("description"):
                    description += f"  Description: {table_description}"
                columns_description = "Columns:"
                for col_name, col_info in table_metadata.get("columns", {}).items():
                    col_dtype = f" ({col_info.get('data_type', '')})" if col_info.get("data_type") else ""
                    col_desc = f": {col_info['description']}" if col_info.get("description") else ""
                    columns_description += f"\n- {col_name}{col_dtype}{col_desc}"
                description += f"\n{indent(columns_description, ' ' * 2)}"
            closest_tables.append(table_slug)
            descriptions.append(description)

        if any_matches:
            message = "Based on the vector search, the most relevant tables are:\n"
        else:
            message = "Based on the vector search, was unable to find tables relevant to the query.\n"
            if closest_tables:
                message += "Here are some other tables:\n"
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
