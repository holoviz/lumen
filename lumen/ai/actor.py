from abc import abstractmethod
from typing import Any

import param

from .llm import Message
from .memory import memory
from .utils import log_debug, render_template
from .vector_store import VectorStore


class Actor(param.Parameterized):

    vector_store = param.ClassSelector(class_=VectorStore, doc="""
        Vector store object which is queried to provide additional context
        before responding.""")

    prompt_overrides = param.Dict(
        default={},
        doc="""
        Overrides the prompt's 'instructions' or 'context' jinja2 blocks.
        Is a nested dictionary with the prompt name (e.g. main) as the key
        and the block names as the inner keys with the new content as the
        values.""",
    )

    prompt_templates = param.Dict(
        default={},
        doc="""
        The paths to the prompt's jinja2 templates.""",
    )

    def __init__(self, **params):
        super().__init__(**params)
        def update_vector_store_table_list(_, __, sources, init=False):
            for src in sources:
                for table in src.get_tables():
                    metadata = {"category": "table_list", "table": table}
                    if not init and self.vector_store.filter_by(filters=metadata):
                            continue
                    self.vector_store.add([{"text": table, "metadata": metadata}])
        memory.on_change('available_sources', update_vector_store_table_list)

    def _add_embeddings(self, messages: list[Message], context: dict[str, Any]) -> Any:
        for message in messages:
            if message["role"] == "user":
                break
        content = message["content"]

        if "current_table" in memory and self.vector_store.filter_by(filters={"table": memory["current_table"]}):
            # specific table embeddings, with lower threshold
            embeddings = self.vector_store.query(
                content, top_k=3, filters={"table": memory["current_table"]}, threshold=0.05
            )
        else:
            embeddings = self.vector_store.query(
                content, top_k=3, threshold=0.1, filters={"category": "table"})
        context["embeddings"] = embeddings
        return context

    def _render_prompt(self, prompt_name: str, **context) -> str:
        context["memory"] = self._memory
        prompt = render_template(
            self.prompt_templates[prompt_name],
            prompt_overrides=self.prompt_overrides.get(prompt_name, {}),
            **context,
        )
        log_debug(f"\033[92mRendered prompt\033[0m '{prompt_name}':\n{prompt}")
        return prompt

    async def _render_main_prompt(self, messages: list[Message], **context) -> str:
        """
        Renders the main prompt using the provided prompts.
        """
        context = self._add_embeddings(messages, context)
        main_prompt = self._render_prompt("main", **context)
        return main_prompt

    @abstractmethod
    async def respond(self, messages: list[Message], **kwargs: dict[str, Any]) -> Any:
        """
        Responds to the provided messages.
        """
