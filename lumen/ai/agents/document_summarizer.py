from __future__ import annotations

from typing import Any

import param

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..editors import DocumentEditor
from ..llm import Message
from ..schemas import Metaset
from .base_lumen import BaseLumenAgent


class DocumentSummarizerInputs(ContextModel):
    metaset: Metaset


class DocumentSummarizerOutputs(ContextModel):
    document_summary: str
    view: dict[str, Any]


class DocumentSummarizerAgent(BaseLumenAgent):
    """
    Synthesizes relevant document context into a user-focused summary.
    """

    conditions = param.List(
        default=[
            "Use when user asks questions answerable from documentation or uploaded metadata files",
            "Use for summarization of documentation",
            "NOT when user only asks to list available documents",
        ]
    )

    not_with = param.List(default=["DbtslAgent", "SQLAgent"])

    purpose = param.String(default="""
        Synthesizes relevant documents into an answer focused on the user query.
        Also renders a document editor with tabbed previews of matched files.""")

    user = param.String(default="Summarizer")

    input_schema = DocumentSummarizerInputs
    output_schema = DocumentSummarizerOutputs

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "DocumentSummarizerAgent" / "main.jinja2",
            },
        }
    )

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        metaset = context.get("metaset")
        return bool(metaset and metaset.has_docs)

    def _collect_documents(self, context: TContext, metaset: Metaset) -> list[dict[str, Any]]:
        source_catalog = context.get("source_catalog")
        available_metadata = getattr(source_catalog, "_available_metadata", []) if source_catalog else []
        by_filename = {meta.get("filename"): meta for meta in available_metadata}

        doc_chunks: dict[str, list[Any]] = {}
        for chunk in metaset.get_docs():
            doc_chunks.setdefault(chunk.filename, []).append(chunk)

        # Keep ordering by best similarity score across chunks.
        ranked_filenames = sorted(
            doc_chunks,
            key=lambda fname: max((c.similarity for c in doc_chunks[fname]), default=0),
            reverse=True,
        )

        documents = []
        for filename in ranked_filenames:
            source_meta = by_filename.get(filename, {})
            chunks = doc_chunks[filename]
            fallback_content = "\n\n".join(
                f"- {chunk.text.strip()}" for chunk in chunks if chunk.text and chunk.text.strip()
            )
            similarity = max((c.similarity for c in chunks), default=0)
            documents.append(
                {
                    "filename": filename,
                    "similarity": similarity,
                    "content": source_meta.get("content") or fallback_content or "*No textual content available.*",
                    "raw_bytes": source_meta.get("raw_bytes"),
                }
            )
        return documents

    def _build_document_context(self, documents: list[dict[str, Any]], max_chars_per_doc: int = 4000) -> str:
        sections = []
        for doc in documents:
            filename = doc.get("filename", "unknown")
            similarity = doc.get("similarity", 0.0)
            content = (doc.get("content") or "").strip()
            if len(content) > max_chars_per_doc:
                content = f"{content[:max_chars_per_doc]}\n...[truncated]..."
            sections.append(
                f'<document filename="{filename}" relevance="{similarity:.3f}">\n{content}\n</document>'
            )
        return "\n\n".join(sections)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        metaset = context.get("metaset")
        if not metaset or not metaset.has_docs:
            fallback = "I could not find relevant documents to summarize."
            return [fallback], {"document_summary": fallback}

        documents = self._collect_documents(context, metaset)
        document_editor = DocumentEditor(
            documents=documents,
            query=messages[-1]["content"] if messages else None,
            title="Relevant document tabs",
        )
        system_prompt = await self._render_prompt(
            "main",
            messages,
            context,
            document_context=self._build_document_context(documents),
        )
        summary_message = await self._stream(messages, system_prompt)
        return [summary_message, document_editor], {
            "document_summary": summary_message.object, **(await document_editor.render_context())
        }
