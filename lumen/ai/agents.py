import asyncio
import json
import traceback

from collections.abc import Callable
from functools import partial
from typing import (
    Annotated, Any, ClassVar, Literal, NotRequired,
)

import pandas as pd
import panel as pn
import param
import requests

from panel.chat import ChatInterface
from panel.viewable import Viewable, Viewer
from panel_material_ui import Button, Tabs
from panel_material_ui.chat import ChatMessage, ChatStep
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ..config import dump_yaml, load_yaml
from ..dashboard import Config
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..sources.duckdb import DuckDBSource
from ..state import state
from ..transforms.sql import SQLLimit
from ..views import (
    Panel, VegaLiteView, View, hvPlotUIView,
)
from .actor import ContextProvider
from .config import (
    LUMEN_CACHE_DIR, PROMPTS_DIR, SOURCE_TABLE_SEPARATOR,
    VECTOR_STORE_ASSETS_URL, VEGA_LITE_EXAMPLES_NUMPY_DB_FILE,
    VEGA_LITE_EXAMPLES_OPENAI_DB_FILE, VEGA_MAP_LAYER, VEGA_ZOOMABLE_MAP_ITEMS,
    MissingContextError, RetriesExceededError,
)
from .context import ContextModel, TContext
from .controls import SourceControls
from .llm import Llm, Message, OpenAI
from .models import (
    DbtslQueryParams, DistinctQuery, PartialBaseModel,
    QueryCompletionValidation, RetrySpec, SampleQuery, VegaLiteSpec,
    VegaLiteSpecUpdate, make_discovery_model, make_sql_model,
)
from .schemas import DbtslMetaset, Metaset, get_metaset
from .services import DbtslMixin
from .tools import ToolUser
from .translate import param_to_pydantic
from .utils import (
    apply_changes, clean_sql, describe_data, get_data, get_pipeline,
    get_root_exception, get_schema, load_json, log_debug, parse_table_slug,
    report_error, retry_llm_output, stream_details,
)
from .vector_store import DuckDBVectorStore
from .views import (
    AnalysisOutput, LumenOutput, SQLOutput, VegaLiteOutput,
)


class Agent(Viewer, ToolUser, ContextProvider):
    """
    Agents are actors responsible for taking a user query and
    performing a particular task, either by adding context or
    generating outputs.

    Agents have access to an LLM and are given context and can
    solve tasks by executing a series of prompts or by rendering
    contents such as forms or widgets to gather user input.
    """

    agents = param.List(doc="""
        List of agents this agent can invoke.""")

    debug = param.Boolean(default=False, doc="""
        Whether to enable verbose error reporting.""")

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM implementation to query.""")

    user = param.String(default="Agent", doc="""
        The name of the user that will be respond to the user query.""")

    # Panel extensions this agent requires to be loaded
    _extensions = ()

    # Maximum width of the output
    _max_width = 1200

    __abstract = True

    def __init__(self, **params):
        def _exception_handler(exception):
            traceback.print_exception(exception)
            if self.interface is None:
                return
            messages = self.interface.serialize()
            if messages and str(exception) in messages[-1]["content"]:
                return

            self.interface.send(
                f"Error cannot be resolved:\n\n{exception}", user="System", respond=False
            )

        super().__init__(**params)
        if not self.debug:
            pn.config.exception_handler = _exception_handler

        if state.config is None:
            state.config = Config(raise_with_notifications=True)
        else:
            state.config.raise_with_notifications = True

    async def _interface_callback(self, contents: list | str, user: str, instance: ChatInterface):
        await self.respond(contents)
        self._retries_left = 1

    @property
    def _steps_layout(self):
        """
        Stream to a dummy column to be able to suppress the steps output.
        """
        return pn.Column() if not self.steps_layout else self.steps_layout

    def __panel__(self):
        return self.interface

    async def _stream(self, messages: list[Message], system_prompt: str) -> Any:
        message = None
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        output = self.llm.stream(messages, system=system_prompt, model_spec=model_spec, field="output")
        async for output_chunk in output:
            if self.interface is None:
                if message is None:
                    message = ChatMessage(output_chunk, user=self.user)
                else:
                    message.object = output_chunk
            else:
                message = self.interface.stream(output_chunk, replace=True, message=message, user=self.user, max_width=self._max_width)
        return message

    async def _gather_prompt_context(self, prompt_name: str, messages: list, context: TContext, **kwargs):
        context = await super()._gather_prompt_context(prompt_name, messages, context, **kwargs)
        if "tool_context" not in context:
            context["tool_context"] = await self._use_tools(prompt_name, messages, context)
        return context

    # Public API

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        """
        Additional checks to determine if the agent should be used.
        """
        return True

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """
        Provides a response to the user query.

        The type of the response may be a simple string or an object.

        Arguments
        ---------
        messages: list[Message]
            The list of messages corresponding to the user query and any other
            system messages to be included.
        context: TContext
            A mapping containing context for the agent to perform its task.
        step_title: str | None
            If the Agent response is part of a longer query this describes
            the step currently being processed.
        """
        system_prompt = await self._render_prompt("main", messages, context)
        return [await self._stream(messages, system_prompt)], context


class SourceOutputs(ContextModel):

    document_sources: list[Any]

    source: Source

    sources: list[Source]


class SourceAgent(Agent):
    """
    SourceAgent renders a form that allows a user to upload or
    provide a URI to one or more datasets.
    """

    conditions = param.List(
        default=[
            "Use ONLY when user explicitly asks to upload or connect to NEW data sources (NOT if data sources already exist)",
        ])

    purpose = param.String(default="Allows a user to upload new datasets, data, or documents.")

    source_controls: ClassVar[SourceControls] = SourceControls

    _extensions = ("filedropper",)

    output_schema = SourceOutputs

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], SourceOutputs]:
        source_controls = self.source_controls(context=context, cancellable=True, replace_controls=True)
        output = pn.Column(source_controls)
        if "source" not in context:
            help_message = "No datasets or documents were found, **please upload at least one to continue**..."
        else:
            help_message = "**Please upload new dataset(s)/document(s) to continue**, or click cancel if this was unintended..."
        output.insert(0, help_message)
        self.interface.send(output, respond=False, user=self.__class__.__name__)
        while not source_controls._add_button.clicks > 0:
            await asyncio.sleep(0.05)
            if source_controls._cancel_button.clicks > 0:
                self.interface.undo()
                self.interface.disabled = False
                return [], {}
        return [], source_controls.outputs


class ChatAgent(Agent):
    """
    ChatAgent provides general information about available data
    and other topics  to the user.
    """

    conditions = param.List(
        default=[
            "Use for high-level data information or general conversation",
            "Use for technical questions about programming, functions, methods, libraries, APIs, software tools, or 'how to' code usage",
            "NOT for data-specific questions that require querying data",
        ]
    )

    purpose = param.String(
        default="""
        Engages in conversations about high-level data topics, programming questions,
        technical documentation, and general conversation. Handles questions about
        specific functions, methods, libraries, and provides coding guidance and
        technical explanations.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "ChatAgent" / "main.jinja2",
            },
        }
    )

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        prompt_context = {"tool_context": await self._use_tools("main", messages, context)}
        if "metaset" not in context and "source" in context and "table" in context:
            context["metaset"] = await get_metaset(
                [context["source"]], [context["table"]]
            )
        system_prompt = await self._render_prompt("main", messages, context, **prompt_context)
        return [await self._stream(messages, system_prompt)], {}


class AnalystInputs(ContextModel):

    data: NotRequired[Any]

    source: Source

    pipeline: Pipeline

    sql: NotRequired[str]


class AnalystAgent(ChatAgent):
    conditions = param.List(
        default=[
            "Use for interpreting and analyzing results from executed queries",
            "NOT for initial data queries, data exploration, or technical programming questions",
        ]
    )

    purpose = param.String(
        default="""
        Responsible for analyzing results and providing clear, concise, and actionable insights.
        Focuses on breaking down complex data findings into understandable points for
        high-level decision-making. Emphasizes detailed interpretation, trends, and
        relationships within the data, while avoiding general overviews or
        superficial descriptions.""")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "AnalystAgent" / "main.jinja2", "tools": []},
        }
    )

    input_schema = AnalystInputs

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        messages, out_context = await super().respond(messages, context, step_title=step_title)
        if len(context.get("data", [])) == 0 and context.get("sql"):
            context["sql"] = f"{context['sql']}\n-- No data was returned from the query."
        return messages, out_context


class ListAgent(Agent):
    """
    Abstract base class for agents that display a list of items to the user.
    """

    purpose = param.String(default="""
        Renders a list of items to the user and lets the user pick one.""")

    _extensions = ("tabulator",)

    _column_name = None

    _message_format = None

    __abstract = True

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        """Return dict of items grouped by source/category"""

    def _use_item(self, event, interface=None):
        """Handle when a user clicks on an item in the list"""
        if event.column != "show":
            return

        tabulator = self._tabs[self._tabs.active]
        item = tabulator.value.iloc[event.row, 0]

        if self._message_format is None:
            raise ValueError("Subclass must define _message_format")

        message = self._message_format.format(item=repr(item))
        interface.send(message)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        items = self._get_items(context)

        # Create tabs with one tabulator per source
        tabs = []
        for source_name, source_items in items.items():
            if not source_items:
                continue

            header_filters = False
            if len(source_items) > 10:
                column_filter = {"type": "input", "placeholder": f"Filter by {self._column_name.lower()}..."}
                header_filters = {self._column_name: column_filter}

            df = pd.DataFrame({self._column_name: source_items})
            item_list = pn.widgets.Tabulator(
                df,
                buttons={"show": '<i class="fa fa-eye"></i>'},
                show_index=False,
                min_height=150,
                min_width=350,
                widths={self._column_name: "90%"},
                disabled=True,
                page_size=10,
                pagination="remote",
                header_filters=header_filters,
                sizing_mode="stretch_width",
                name=source_name
            )
            item_list.on_click(partial(self._use_item, interface=self.interface))
            tabs.append(item_list)

        self._tabs = Tabs(*tabs, sizing_mode="stretch_width")

        self.interface.stream(
            pn.Column(
                f"The available {self._column_name.lower()}s are listed below. Click on the eye icon to show the {self._column_name.lower()} contents.",
                self._tabs
            ), user=self.__class__.__name__
        )
        return [self._tabs], {}


class TableListInputs(ContextModel):

    source: Source

    visible_slugs: NotRequired[set[str]]


class TableListAgent(ListAgent):
    """
    The TableListAgent lists all available data and lets the user pick one.
    """

    conditions = param.List(default=[
        "Use when user explicitly asks to 'list data', 'show available data', or 'what data do you have'",
        "NOT for showing actual data contents, querying, or analyzing data",
    ])

    not_with = param.List(default=["DbtslAgent", "SQLAgent"])

    purpose = param.String(default="""
        Displays a list of all available data & datasets. Not useful for identifying which dataset to use for analysis.""")

    _column_name = "Data"

    _message_format = "Show the data: {item}"

    input_schema = TableListInputs

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        return len(context.get('visible_slugs', set())) > 1

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        if "closest_tables" in context:
            # If we have closest_tables from search, return as a single group
            return {"Search Results": context["closest_tables"]}

        # Group tables by source
        visible_slugs = context.get('visible_slugs', set())
        if not visible_slugs:
            return {}

        tables_by_source = {}
        for slug in visible_slugs:
            if SOURCE_TABLE_SEPARATOR in slug:
                source_name, table_name = slug.split(SOURCE_TABLE_SEPARATOR, 1)
            else:
                # Fallback if separator not found
                source_name = "Unknown"
                table_name = slug

            if source_name not in tables_by_source:
                tables_by_source[source_name] = []
            tables_by_source[source_name].append(table_name)

        # Sort tables within each source
        for source in tables_by_source:
            tables_by_source[source].sort()

        return tables_by_source


class DocumentListInputs(ContextModel):

    document_sources: dict[str, Any]


class DocumentListAgent(ListAgent):
    """
    The DocumentListAgent lists all available documents provided by the user.
    """

    conditions = param.List(
        default=[
            "Use when user asks to list or see all available documents",
            "NOT when user asks about specific document content",
        ]
    )

    purpose = param.String(default="""
        Displays a list of all available documents.""")

    _column_name = "Documents"

    _message_format = "Tell me about: {item}"

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        sources = context.get("document_sources")
        if not sources:
            return False
        return len(sources) > 1

    def _get_items(self, context: TContext) -> dict[str, list[str]]:
        # extract the filename, following this pattern `Filename: 'filename'``
        documents = [doc["metadata"].get("filename", "untitled") for doc in context.get("document_sources", [])]
        return {"Documents": documents} if documents else {}


class LumenBaseAgent(Agent):

    prompts = param.Dict(
        default={
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "LumenBaseAgent" / "retry_output.jinja2"},
        }
    )

    user = param.String(default="Lumen")

    _max_width = None
    _output_type = LumenOutput

    @retry_llm_output()
    async def revise(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        view: LumenOutput | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        **kwargs
    ) -> str:
        """
        Retry the output by line, allowing the user to provide instruction on why the output was not satisfactory, or an error.
        """
        if view is not None:
            spec = view.spec
            language = view.language
        if spec is None:
            raise ValueError("Must provide previous spec to revise.")
        lines = spec.splitlines()
        numbered_text = "\n".join(f"{i:2d}: {line}" for i, line in enumerate(lines, 1))
        system = await self._render_prompt(
            "retry_output",
            messages,
            context,
            numbered_text=numbered_text,
            language=language,
            feedback=instruction,
            errors=errors,
            **kwargs
        )
        retry_model = self._lookup_prompt_key("retry_output", "response_model")
        result = await self.llm.invoke(
            messages,
            system=system,
            response_model=retry_model,
            model_spec="edit"
        )
        spec = load_yaml(apply_changes(lines, result.edits))
        if view is not None:
            view.validate_spec(spec)
        if isinstance(spec, str):
            return spec
        yaml_spec = dump_yaml(spec)
        return yaml_spec


class SQLInputs(ContextModel):

    data: NotRequired[Any]

    source: Source

    sources: Annotated[list[Source], ("accumulate", "source")]

    sql: NotRequired[str]

    metaset: Metaset

    visible_slugs: NotRequired[set[str]]


class SQLOutputs(ContextModel):

    data: Any

    table: str

    sql: str

    pipeline: Pipeline


class SQLAgent(LumenBaseAgent):

    conditions = param.List(
        default=[
            "Use for displaying, examining, or querying data resulting in a data pipeline",
            "Use for calculations that require data (e.g., 'calculate average', 'sum by category')",
            "Commonly used with AnalystAgent to analyze query results",
            "NOT for non-data questions or technical programming help",
            "NOT useful if the user is using the same data for plotting",
        ]
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    not_with = param.List(default=["DbtslAgent", "TableLookup", "TableListAgent"])

    exploration_enabled = param.Boolean(default=True, doc="""
        Whether to enable SQL exploration mode. When False, only attempts oneshot SQL generation.""")

    purpose = param.String(
        default="""
        Handles the display of data and the creation, modification, and execution
        of SQL queries to address user queries about the data. Executes queries in
        a single step, encompassing tasks such as table joins, filtering, aggregations,
        and calculations. If additional columns are required, SQLAgent can join the
        current table with other tables to fulfill the query requirements."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": make_sql_model,
                "template": PROMPTS_DIR / "SQLAgent" / "main.jinja2",
            },
            "select_discoveries": {
                "response_model": make_discovery_model,
                "template": PROMPTS_DIR / "SQLAgent" / "select_discoveries.jinja2",
            },
            "check_sufficiency": {
                "response_model": make_discovery_model,
                "template": PROMPTS_DIR / "SQLAgent" / "check_sufficiency.jinja2",
            },
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "SQLAgent" / "retry_output.jinja2"},
        }
    )

    user = param.String(default="SQL")

    _extensions = ("codeeditor", "tabulator")

    _output_type = SQLOutput

    input_schema = SQLInputs
    output_schema = SQLOutputs

    async def _validate_sql(
        self,
        context: TContext,
        sql_query: str,
        expr_slug: str,
        source: BaseSQLSource,
        messages: list[Message],
        step: ChatStep,
        max_retries: int = 2,
        discovery_context: str | None = None,
    ) -> str:
        """Validate and potentially fix SQL query."""
        # Clean SQL
        try:
            sql_query = clean_sql(sql_query, source.dialect)
        except Exception as e:
            step.stream(f"\n\n❌ SQL cleaning failed: {e}")

        # Validate with retries
        for i in range(max_retries):
            try:
                step.stream(f"\n\n`{expr_slug}`\n```sql\n{sql_query}\n```")
                source.execute(sql_query)
                step.stream("\n\n✅ SQL validation successful")
                return sql_query
            except Exception as e:
                if i == max_retries - 1:
                    step.stream(f"\n\n❌ SQL validation failed after {max_retries} attempts: {e}")
                    raise e

                # Retry with LLM fix
                step.stream(f"\n\n⚠️ SQL validation failed (attempt {i+1}/{max_retries}): {e}")
                feedback = f"{type(e).__name__}: {e!s}"
                if "KeyError" in feedback:
                    feedback += " The data does not exist; select from available data sources."

                retry_result = await self.revise(
                    feedback, messages, context, spec=sql_query, language=f"sql.{source.dialect}",
                    discovery_context=discovery_context
                )
                sql_query = clean_sql(retry_result, source.dialect)
        return sql_query

    async def _execute_query(
        self, source: BaseSQLSource, context: TContext, expr_slug: str, sql_query: str, tables: list[str],
        is_final: bool, should_materialize: bool, step: ChatStep
    ) -> tuple[Pipeline, Source, str]:
        """Execute SQL query and return pipeline and summary."""
        # Create SQL source
        table_defs = {table: source.tables[table] for table in tables if table in source.tables}
        table_defs[expr_slug] = sql_query

        sql_expr_source = source.create_sql_expr_source(
            table_defs, materialize=should_materialize
        )

        if should_materialize:
            context["source"] = sql_expr_source

        # Create pipeline
        if is_final:
            sql_transforms = [SQLLimit(limit=1_000_000, write=source.dialect, pretty=True, identify=False)]
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)
        else:
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug)

        # Get data summary
        df = await get_data(pipeline)
        summary = await describe_data(df, reduce_enums=False)
        if len(summary) >= 1000:
            summary = summary[:1000-3] + "..."
        summary_formatted = f"\n```\n{summary}\n```"
        if should_materialize:
            summary_formatted += f"\n\nMaterialized data: `{sql_expr_source.name}{SOURCE_TABLE_SEPARATOR}{expr_slug}`"
        stream_details(f"{summary_formatted}", step, title=expr_slug)

        return pipeline, sql_expr_source, summary

    async def _finalize_execution(
        self,
        pipeline: Pipeline,
        sql: str,
        step_title: str | None,
        raise_if_empty: bool = False
    ) -> tuple[LumenOutput, SQLOutputs]:
        """Finalize execution for final step."""

        df = await get_data(pipeline)
        if df.empty and raise_if_empty:
            raise ValueError(f"\nQuery `{sql}` returned empty results; ensure all the WHERE filter values exist in the dataset.")

        view = self._output_type(
            component=pipeline, title=step_title, spec=sql
        )
        return view, {
            "data": await describe_data(df),
            "sql": sql,
            "pipeline": pipeline,
            "table": pipeline.table,
            "source": pipeline.source
        }

    def _merge_sources(
        self, sources: dict[tuple[str, str], BaseSQLSource], tables: list[tuple[str, str]]
    ) -> tuple[BaseSQLSource, list[str]]:
        if len(set(m.source for m in tables)) == 1:
            m = tables[0]
            return sources[(m.source, m.table)], [m.table]
        mirrors = {}
        for m in tables:
            if not any(m.table.rstrip(")").rstrip("'").rstrip('"').endswith(ext)
                       for ext in [".csv", ".parquet", ".parq", ".json", ".xlsx"]):
                renamed_table = m.table.replace(".", "_")
            else:
                renamed_table = m.table
            mirrors[renamed_table] = (sources[(m.source, m.table)], m.table)
        return DuckDBSource(uri=":memory:", mirrors=mirrors), list(mirrors)

    async def _render_execute_query(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], BaseSQLSource],
        step_title: str,
        success_message: str,
        discovery_context: str | None = None,
        raise_if_empty: bool = False,
        output_title: str | None = None
    ) -> tuple[LumenOutput, SQLOutputs]:
        """
        Helper method that generates, validates, and executes final SQL queries.

        Parameters
        ----------
        messages : list[Message]
            List of user messages
        sources : dict[tuple[str, str], BaseSQLSource]
            Data sources to execute against
        step_title : str
            Title for the execution step
        success_message : str
            Message to show on successful completion
        discovery_context : str, optional
            Optional discovery context to include in prompt
        raise_if_empty : bool, optional
            Whether to raise error if query returns empty results
        output_title : str, optional
            Title to use for the output

        Returns
        -------
        Pipeline
            Pipeline object from successful execution
        """
        with self._add_step(title=step_title, steps_layout=self._steps_layout) as step:
            # Generate SQL using common prompt pattern
            dialects = set(src.dialect for src in sources.values())
            dialect = "duckdb" if len(dialects) > 1 else next(iter(dialects))
            system_prompt = await self._render_prompt(
                "main",
                messages,
                context,
                dialect=dialect,
                step_number=1,
                is_final_step=True,
                current_step="",
                sql_query_history={},
                current_iteration=1,
                sql_plan_context=None,
                errors=None,
                discovery_context=discovery_context,
            )

            # Generate SQL using existing model
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            sql_response_model = self._get_model("main", sources=list(sources))

            output = await self.llm.invoke(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=sql_response_model,
            )
            if not output:
                raise ValueError("No output was generated.")

            if len(sources) == 1:
                source = next(iter(sources.values()))
                tables = [next(table for _, table in sources)]
            else:
                source, tables = self._merge_sources(sources, output.tables)
            sql_query = output.query.strip()
            expr_slug = output.table_slug.strip()

            validated_sql = await self._validate_sql(
                context, sql_query, expr_slug, source, messages,
                step, discovery_context=discovery_context
            )

            pipeline, sql_expr_source, summary = await self._execute_query(
                source, context, expr_slug, validated_sql, tables=tables,
                is_final=True, should_materialize=True, step=step
            )

            view, out_context = await self._finalize_execution(
                pipeline,
                validated_sql,
                output_title,
                raise_if_empty=raise_if_empty
            )

            if isinstance(step, ChatStep):
                step.param.update(
                    status="success",
                    success_title=success_message
                )

        return view, out_context

    async def _select_discoveries(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], BaseSQLSource],
        error_context: str,
    ) -> BaseModel:
        """Let LLM choose which discoveries to run based on error and available tables."""
        system_prompt = await self._render_prompt(
            "select_discoveries",
            messages,
            context,
            error_context=error_context,
        )

        discovery_models = self._get_model("select_discoveries", sources=list(sources))
        discovery_model = discovery_models[0]
        model_spec = self.prompts["select_discoveries"].get("llm_spec", self.llm_spec_key)
        selection = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=discovery_model
        )
        return selection

    async def _check_discovery_sufficiency(
        self,
        messages: list[Message],
        context: TContext,
        error_context: str,
        discovery_results: list[tuple[str, str]],
    ) -> BaseModel:
        """Check if discovery results are sufficient to fix the error."""
        system_prompt = await self._render_prompt(
            "check_sufficiency",
            messages,
            context,
            error_context=error_context,
            discovery_results=discovery_results,
        )

        # Get sources from context for the discovery models
        metaset = context["metaset"]
        selected_slugs = list(metaset.catalog)
        visible_slugs = context.get('visible_slugs', [])
        if visible_slugs:
            selected_slugs = [slug for slug in selected_slugs if slug in visible_slugs]
        sources_list = [
            tuple(table_slug.split(SOURCE_TABLE_SEPARATOR))
            for table_slug in selected_slugs
        ]

        discovery_models = self._get_model("check_sufficiency", sources=sources_list)
        sufficiency_model = discovery_models[1]

        model_spec = self.prompts["check_sufficiency"].get("llm_spec", self.llm_spec_key)
        sufficiency = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=sufficiency_model,
        )
        return sufficiency

    async def _execute_discovery_query(
        self,
        query_model: SampleQuery | DistinctQuery,
        sources: dict[tuple[str, str], Source],
    ) -> tuple[str, str]:
        """Execute a single discovery query and return formatted result."""
        # Generate the actual SQL
        format_kwargs = query_model.dict(exclude={'query'})
        sql = query_model.query.format(**format_kwargs)
        table = query_model.slug.table
        source = sources[(query_model.slug.source, table)]
        if isinstance(query_model, SampleQuery):
            query_type = f"Sample from {table}"
        elif isinstance(query_model, DistinctQuery):
            query_type = f"Distinct values from {table!r} table column {query_model.column!r}"
        try:
            df = source.execute(sql)
        except Exception as e:
            return f"Discovery query {query_type[0].lower()}{query_type[1:]} failed", str(e)
        if isinstance(query_model, SampleQuery):
            result = f"\n```\n{df.head().to_string(max_cols=10)}\n```"
        else:
            result = f" `{df.iloc[:10, 0].tolist()}`"
        return query_type, result

    async def _run_discoveries_parallel(
        self,
        discovery_queries: list[SampleQuery | DistinctQuery],
        sources: dict[tuple[str, str], Source],
    ) -> list[str]:
        """Execute all discoveries concurrently and stream results as they complete."""
        # Create all tasks
        tasks = [
            asyncio.create_task(self._execute_discovery_query(query, sources))
            for query in discovery_queries
        ]

        results = []
        for completed_task in asyncio.as_completed(tasks):
            query_type, result = await completed_task
            with self._add_step(title=query_type, steps_layout=self._steps_layout) as step:
                step.stream(result)
            results.append((query_type, result))
        return results

    async def _explore_tables(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], Source],
        error_context: str,
        step_title: str | None = None,
    ) -> tuple[LumenOutput, SQLOutputs]:
        """Run adaptive exploration: initial discoveries → sufficiency check → optional follow-ups → final answer."""
        # Step 1: LLM selects initial discoveries
        with self._add_step(title="Selecting initial discoveries", steps_layout=self._steps_layout) as step:
            selection = await self._select_discoveries(messages, context, sources, error_context)
            step.stream(f"Strategy: {selection.reasoning}\n\nSelected {len(selection.queries)} initial discoveries")

        # Step 2: Run initial discoveries in parallel
        initial_results = await self._run_discoveries_parallel(selection.queries, sources)

        # Step 3: Check if discoveries are sufficient
        with self._add_step(title="Evaluating discovery sufficiency", steps_layout=self._steps_layout) as step:
            sufficiency = await self._check_discovery_sufficiency(messages, context, error_context, initial_results)
            step.stream(f"Assessment: {sufficiency.reasoning}")

            if sufficiency.sufficient:
                step.stream("\n✅ Sufficient context - proceeding to final answer")
                final_results = initial_results
            else:
                step.stream(f"\n⚠️ Insufficient - running {len(sufficiency.follow_up_needed)} follow-ups")

        # Step 4: Run follow-up discoveries if needed
        if not sufficiency.sufficient and sufficiency.follow_up_needed:
            follow_up_results = await self._run_discoveries_parallel(sufficiency.follow_up_needed, sources)
            final_results = initial_results + follow_up_results
        else:
            final_results = initial_results

        # Step 5: Generate final answer with all discovery context
        discovery_context = "\n".join([
            f"- **{query_type}:** {result}" for query_type, result in final_results
        ])

        return await self._render_execute_query(
            messages,
            context,
            sources=sources,
            step_title="Generating final answer with discoveries",
            success_message="Final answer generated with discovery context",
            discovery_context=discovery_context,
            raise_if_empty=False,
            output_title=step_title
        )


    async def revise(
        self,
        feedback: str,
        messages: list[Message],
        context: TContext,
        view: LumenOutput | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        **kwargs
    ) -> str:
        result = await super().revise(
            feedback, messages, context, view=view, spec=spec, language=language, **kwargs
        )
        if view is not None:
            result = clean_sql(result, view.component.source.dialect)
        return result

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], SQLOutputs]:
        """Execute SQL generation with one-shot attempt first, then exploration if needed."""
        sources = context["sources"]
        metaset = context["metaset"]
        selected_slugs = list(metaset.catalog)
        visible_slugs = context.get('visible_slugs', [])
        if visible_slugs:
            selected_slugs = [slug for slug in selected_slugs if slug in visible_slugs]
        sources = {
            tuple(table_slug.split(SOURCE_TABLE_SEPARATOR)): parse_table_slug(table_slug, sources)[0]
            for table_slug in selected_slugs
        }
        if not sources:
            return [], {}
        try:
            # Try one-shot approach first
            out, out_context = await self._render_execute_query(
                messages,
                context,
                sources=sources,
                step_title="Attempting one-shot SQL generation...",
                success_message="One-shot SQL generation successful",
                discovery_context=None,
                raise_if_empty=True,
                output_title=step_title
            )
        except Exception as e:
            if not self.exploration_enabled:
                # If exploration is disabled, re-raise the error instead of falling back
                raise e
            # Fall back to exploration mode if enabled
            out, out_context = await self._explore_tables(
                messages, context, sources, str(e), step_title
            )
        return [out], out_context



class DbtslOutputs(SQLOutputs):

    dbtsl_metaset: DbtslMetaset


class DbtslAgent(LumenBaseAgent, DbtslMixin):
    """
    Responsible for creating and executing queries against a dbt Semantic Layer
    to answer user questions about business metrics.
    """

    conditions = param.List(
        default=[
            "Always use this when dbtsl_metaset is available",
        ]
    )

    purpose = param.String(
        default="""
        Responsible for displaying data to answer user queries about
        business metrics using dbt Semantic Layers. This agent can compile
        and execute metric queries against a dbt Semantic Layer."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": DbtslQueryParams,
                "template": PROMPTS_DIR / "DbtslAgent" / "main.jinja2",
            },
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "LumenBaseAgent" / "retry_output.jinja2"},
        }
    )

    requires = param.List(default=["source", "dbtsl_metaset"], readonly=True)

    source = param.ClassSelector(
        class_=BaseSQLSource,
        doc="""
        The source associated with the dbt Semantic Layer.""",
    )

    user = param.String(default="DBT")

    _extensions = ("codeeditor", "tabulator")

    _output_type = LumenOutput

    output_schema = DbtslOutputs

    def __init__(self, source: Source, **params):
        super().__init__(source=source, **params)

    @retry_llm_output()
    async def _create_valid_query(
        self, messages: list[Message], title: str | None = None, errors: list | None = None
    ) -> DbtslOutputs:
        """
        Create a valid dbt Semantic Layer query based on user messages.
        """
        system_prompt = await self._render_prompt(
            "main",
            messages,
            errors=errors,
        )

        out_context = {}
        with self._add_step(title=title or "dbt Semantic Layer query", steps_layout=self._steps_layout) as step:
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=DbtslQueryParams,
            )

            query_params = None
            try:
                async for output in response:
                    step_message = output.chain_of_thought or ""
                    step.stream(step_message, replace=True)

                query_params = {
                    "metrics": output.metrics,
                    "group_by": output.group_by,
                    "limit": output.limit,
                    "order_by": output.order_by,
                    "where": output.where,
                }
                expr_slug = output.expr_slug

                # Ensure required fields are present
                if "metrics" not in query_params or not query_params["metrics"]:
                    raise ValueError("Query must include at least one metric")

                # Ensure limit exists with a default value if not provided
                if "limit" not in query_params or not query_params["limit"]:
                    query_params["limit"] = 10000

                formatted_params = json.dumps(query_params, indent=2)
                step.stream(f"\n\n`{expr_slug}`\n```json\n{formatted_params}\n```")

                out_context["dbtsl_query_params"] = query_params
            except asyncio.CancelledError as e:
                step.failed_title = "Cancelled dbt Semantic Layer query generation"
                raise e

        if step.failed_title and step.failed_title.startswith("Cancelled"):
            raise asyncio.CancelledError()
        elif not query_params:
            raise ValueError("No dbt Semantic Layer query was generated.")

        try:
            # Execute the query against the dbt Semantic Layer
            client = self._instantiate_client()
            async with client.session():
                sql_query = await client.compile_sql(
                    metrics=query_params.get("metrics", []),
                    group_by=query_params.get("group_by"),
                    limit=query_params.get("limit"),
                    order_by=query_params.get("order_by"),
                    where=query_params.get("where"),
                )

                # Log the compiled SQL for debugging
                step.stream(f"\nCompiled SQL:\n```sql\n{sql_query}\n```", replace=False)

            sql_expr_source = self.source.create_sql_expr_source({expr_slug: sql_query})
            out_context["sql"] = sql_query

            # Apply transforms
            sql_transforms = [SQLLimit(limit=1_000_000, write=self.source.dialect, pretty=True, identify=False)]
            transformed_sql_query = sql_query
            for sql_transform in sql_transforms:
                transformed_sql_query = sql_transform.apply(transformed_sql_query)
                if transformed_sql_query != sql_query:
                    stream_details(f"```sql\n{transformed_sql_query}\n```", step, title=f"{sql_transform.__class__.__name__} Applied")

            # Create pipeline and get data
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)

            df = await get_data(pipeline)
            metaset = await get_metaset([sql_expr_source], [expr_slug])

            # Update context
            out_context["data"] = await describe_data(df)
            out_context["source"] = sql_expr_source
            out_context["pipeline"] = pipeline
            out_context["table"] = pipeline.table
            out_context["dbtsl_metaset"] = metaset
        except Exception as e:
            report_error(e, step)
            raise e
        return out_context

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], DbtslOutputs]:
        """
        Responds to user messages by generating and executing a dbt Semantic Layer query.
        """
        try:
            out_context = await self._create_valid_query(messages, step_title)
        except RetriesExceededError as e:
            traceback.print_exception(e)
            context["__error__"] = str(e)
            return None

        pipeline = out_context["pipeline"]
        view = self._output_type(
            component=pipeline, title=step_title, spec=out_context["sql"]
        )
        return [view], out_context



class ViewInputs(ContextModel):

    data: Any

    pipeline: Pipeline

    metaset: NotRequired[Metaset]

    table: str


class ViewOutputs(ContextModel):

    view = Any


class BaseViewAgent(LumenBaseAgent):

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "BaseViewAgent" / "main.jinja2"},
        }
    )

    input_schema = ViewInputs
    output_schema = ViewOutputs

    def __init__(self, **params):
        self._last_output = None
        super().__init__(**params)

    def _build_errors_context(self, pipeline: Pipeline, context: TContext, errors: list[str] | None) -> dict:
        errors_context = {}
        if errors:
            errors = ("\n".join(f"{i + 1}. {error}" for i, error in enumerate(errors))).strip()
            try:
                last_output = load_yaml(self._last_output["yaml_spec"])
            except Exception:
                last_output = ""

            catalog = None
            if "metaset" in context:
                catalog = context["metaset"].catalog
            elif "dbtsl_metaset" in context:
                catalog = context["dbtsl_metaset"].catalog

            columns_context = ""
            if catalog is not None:
                for table_slug, catalog_entry in catalog.items():
                    table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[-1]
                    if table_name in pipeline.table:
                        columns = [col.name for col in catalog_entry.columns]
                        columns_context += f"\nSQL: {catalog_entry.sql_expr}\nColumns: {', '.join(columns)}\n\n"
            errors_context = {
                "errors": errors,
                "last_output": last_output,
                "columns_context": columns_context,
            }
        return errors_context

    @retry_llm_output()
    async def _create_valid_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        schema: dict[str, Any],
        step_title: str | None = None,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        errors_context = self._build_errors_context(pipeline, context, errors)
        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        system = await self._render_prompt(
            "main",
            messages,
            context,
            table=pipeline.table,
            doc=doc,
            **errors_context,
        )

        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        response = self.llm.stream(
            messages,
            system=system,
            model_spec=model_spec,
            response_model=self._get_model("main", schema=schema),
        )

        e = None
        error = ""
        spec = ""
        with self._add_step(
            title=step_title or "Generating view...", steps_layout=self._steps_layout
        ) as step:
            async for output in response:
                chain_of_thought = output.chain_of_thought or ""
                step.stream(chain_of_thought, replace=True)

            self._last_output = spec = dict(output)

            for i in range(3):
                try:
                    spec = await self._extract_spec(context, spec)
                    break
                except Exception as e:
                    e = get_root_exception(e, exceptions=(MissingContextError,))
                    if isinstance(e, MissingContextError):
                        raise e

                    error = str(e)
                    traceback.print_exception(e)
                    context = f"```\n{dump_yaml(load_yaml(self._last_output['yaml_spec']))}\n```"
                    report_error(e, step, language="json", context=context, status="failed")
                    with self._add_step(
                        title="Re-attempted view generation",
                        steps_layout=self._steps_layout,
                    ) as retry_step:
                        view = await self.revise(e, messages, context, dump_yaml(spec), language="yaml")
                        if "yaml_spec: " in view:
                            view = view.split("yaml_spec: ")[-1].rstrip('"').rstrip("'")
                        retry_step.stream(f"\n\n```yaml\n{view}\n```")
                    if i == 2:
                        raise

        self._last_output = spec

        if error:
            raise ValueError(error)

        log_debug(f"{self.name} settled on spec: {spec!r}.")
        return spec

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        return dict(spec)

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], ViewOutputs]:
        """
        Generates a visualization based on user messages and the current data pipeline.
        """
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        spec = await self._create_valid_spec(messages, context, pipeline, schema, step_title)
        view = self.view_type(pipeline=pipeline, **spec)
        out = self._output_type(component=view, title=step_title)
        return [out], {"view": dict(spec, type=self.view_type.view_type)}


class hvPlotAgent(BaseViewAgent):

    conditions = param.List(
        default=[
            "Use for exploratory data analysis, interactive plots, and dynamic filtering",
            "Use for quick, iterative data visualization during analysis",
        ]
    )

    purpose = param.String(default="Generates a plot of the data given a user prompt.")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "hvPlotAgent" / "main.jinja2"},
        }
    )

    view_type = hvPlotUIView

    def _get_model(self, prompt_name: str, schema: dict[str, Any]) -> type[BaseModel]:
        # Find parameters
        excluded = self.view_type._internal_params + [
            "controls",
            "type",
            "source",
            "pipeline",
            "transforms",
            "download",
            "field",
            "selection_group",
        ]
        model = param_to_pydantic(
            self.view_type,
            base_model=PartialBaseModel,
            excluded=excluded,
            schema=schema,
            extra_fields={
                "chain_of_thought": (str, FieldInfo(description="Your thought process behind the plot.")),
            },
        )
        return model[self.view_type.__name__]

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        pipeline = context["pipeline"]
        spec = {key: val for key, val in spec.items() if val is not None}
        spec["type"] = "hvplot_ui"
        self.view_type.validate(spec)
        spec.pop("type", None)

        # Add defaults
        spec["responsive"] = True
        data = await get_data(pipeline)
        if len(data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            spec["rasterize"] = True
            spec["cnorm"] = "log"
        return spec


class VegaLiteAgent(BaseViewAgent):

    conditions = param.List(
        default=[
            "Use for publication-ready visualizations or when user specifically requests Vega-Lite charts",
            "Use for polished charts intended for presentation or sharing",
        ]
    )

    purpose = param.String(default="Generates a vega-lite plot specification from the input data pipeline.")

    prompts = param.Dict(
        default={
            "main": {"response_model": VegaLiteSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2"},
            "interaction_polish": {"response_model": VegaLiteSpecUpdate, "template": PROMPTS_DIR / "VegaLiteAgent" / "interaction_polish.jinja2"},
            "annotate_plot": {"response_model": VegaLiteSpecUpdate, "template": PROMPTS_DIR / "VegaLiteAgent" / "annotate_plot.jinja2"},
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "retry_output.jinja2"},
        }
    )

    user = param.String(default="Vega")

    vector_store_path = param.Path(default=None, check_exists=False, doc="""
        Path to a custom vector store for storing and retrieving Vega-Lite examples;
        if not provided a default store will be used depending on the LLM--
        OpenAIEmbeddings for OpenAI LLM or NumpyEmbeddings for all others.""")

    view_type = VegaLiteView

    _extensions = ("vega",)

    _output_type = VegaLiteOutput

    def __init__(self, **params):
        self._vector_store: DuckDBVectorStore | None = None
        super().__init__(**params)

    def _get_vector_store(self):
        """Get or initialize the vector store (lazy initialization)."""
        if self._vector_store is not None:
            return self._vector_store

        if self.vector_store_path:
            uri = self.vector_store_path
        else:
            db_file = VEGA_LITE_EXAMPLES_OPENAI_DB_FILE if isinstance(self.llm, OpenAI) else VEGA_LITE_EXAMPLES_NUMPY_DB_FILE
            uri = LUMEN_CACHE_DIR / db_file
            if not uri.exists():
                response = requests.get(f"{VECTOR_STORE_ASSETS_URL}{db_file}", timeout=5)
                response.raise_for_status()
                uri.write_bytes(response.content)
        # Use a read-only connection to avoid lock conflicts
        self._vector_store = DuckDBVectorStore(uri=str(uri), read_only=True)
        return self._vector_store

    def _deep_merge_dicts(self, base_dict: dict[str, Any], update_dict: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries, with update_dict taking precedence.

        Special handling:
        - If update_dict contains 'layer', append new annotation layers rather than merging
        - When merging layers, ensure each layer has both 'mark' and 'encoding'
        """
        if not update_dict:
            return base_dict

        result = base_dict.copy()

        # Special handling for layer arrays
        if "layer" in update_dict and "layer" in result:
            base_layers = result["layer"]
            update_layers = update_dict["layer"]

            # Check if update layers are new annotations (have different mark types from base)
            # If first update layer has a different mark type, treat as append operation
            is_append_operation = False
            if update_layers and base_layers:
                first_update_mark = self._get_layer_mark_type(update_layers[0])
                first_base_mark = self._get_layer_mark_type(base_layers[0])

                # If marks are different, or if update has marks like 'rule', 'rect', 'text'
                # which are typically annotations, treat as append
                annotation_marks = {'rule', 'rect', 'text'}
                if first_update_mark != first_base_mark or first_update_mark in annotation_marks:
                    is_append_operation = True

            if is_append_operation:
                # Append all update layers as new annotation layers
                # Normalize each layer to ensure proper mark structure
                normalized_updates = [self._normalize_layer_mark(layer.copy()) for layer in update_layers]
                result["layer"] = base_layers + normalized_updates
            else:
                # Original merge behavior for updating existing layers
                merged_layers = []
                for i, update_layer in enumerate(update_layers):
                    if i < len(base_layers):
                        # Merge with corresponding base layer
                        base_layer = base_layers[i]
                        merged_layer = self._deep_merge_dicts(base_layer, update_layer)

                        # Ensure layer has mark (carry over from base if not in update)
                        if "mark" not in merged_layer and "mark" in base_layer:
                            merged_layer["mark"] = base_layer["mark"]

                        merged_layers.append(merged_layer)
                    else:
                        # New layer added by update
                        merged_layers.append(self._normalize_layer_mark(update_layer.copy()))

                # Keep any remaining base layers not updated
                merged_layers.extend(base_layers[len(update_layers):])
                result["layer"] = merged_layers
        else:
            # Standard recursive merge for non-layer properties
            for key, value in update_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge_dicts(result[key], value)
                else:
                    result[key] = value

        # If we're merging in a 'layer', remove conflicting top-level properties
        if "layer" in update_dict:
            result.pop("mark", None)
            result.pop("encoding", None)

        return result

    def _get_layer_mark_type(self, layer: dict) -> str | None:
        """Extract mark type from a layer, handling both dict and string formats."""
        if "mark" in layer:
            mark = layer["mark"]
            if isinstance(mark, dict):
                return mark.get("type")
            return mark
        return None

    def _normalize_layer_mark(self, layer: dict) -> dict:
        """Ensure layer has a properly formatted single mark property.

        Handles cases where:
        - Mark appears multiple times (duplicate keys in parsed YAML/dict)
        - Mark is a string but needs to be an object with 'type'
        - Layer might have conflicting mark definitions
        """
        if "mark" not in layer:
            return layer

        mark = layer["mark"]

        # If mark is a string, convert to object with type
        if isinstance(mark, str):
            layer["mark"] = {"type": mark}
        # If mark is a dict, ensure it has 'type'
        elif isinstance(mark, dict) and "type" not in mark:
            # Malformed mark without type - try to infer or leave as-is
            # This shouldn't happen in valid Vega-Lite specs
            pass

        return layer

    async def _update_spec_step(
        self,
        step_name: str,
        step_desc: str,
        vega_spec: dict[str, Any],
        prompt_name: str,
        messages: list[Message],
        context: TContext,
        doc: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Update a Vega-Lite spec with incremental changes for a specific step."""
        with self._add_step(title=step_desc, steps_layout=self._steps_layout) as step:
            vega_spec = dump_yaml(vega_spec, default_flow_style=False)
            system_prompt = await self._render_prompt(
                prompt_name,
                messages,
                context,
                vega_spec=vega_spec,
                doc=doc,
                table=context["pipeline"].table,
            )

            model_spec = self.prompts.get(prompt_name, {}).get("llm_spec", self.llm_spec_key)
            result = await self.llm.invoke(
                messages=messages,
                system=system_prompt,
                response_model=VegaLiteSpecUpdate,
                model_spec=model_spec,
            )

            step.stream(f"Reasoning: {result.chain_of_thought}")
            step.stream(f"Update:\n```yaml\n{result.yaml_update}\n```", replace=False)
            update_dict = load_yaml(result.yaml_update)
        return step_name, update_dict

    def _add_zoom_params(self, vega_spec: dict) -> None:
        """Add zoom parameters to vega spec."""
        if "params" not in vega_spec:
            vega_spec["params"] = []

        existing_param_names = {
            param.get("name") for param in vega_spec["params"]
            if isinstance(param, dict) and "name" in param
        }

        for p in VEGA_ZOOMABLE_MAP_ITEMS["params"]:
            if p.get("name") not in existing_param_names:
                vega_spec["params"].append(p)

    def _setup_projection(self, vega_spec: dict) -> None:
        """Setup map projection settings."""
        if "projection" not in vega_spec:
            vega_spec["projection"] = {"type": "mercator"}
        vega_spec["projection"].update(VEGA_ZOOMABLE_MAP_ITEMS["projection"])

    def _handle_map_compatibility(self, vega_spec: dict, vega_spec_str: str) -> None:
        """Handle map projection compatibility and add geographic outlines."""
        has_world_map = "world-110m.json" in vega_spec_str
        uses_albers_usa = vega_spec["projection"]["type"] == "albersUsa"

        if has_world_map and uses_albers_usa:
            # albersUsa incompatible with world map
            vega_spec["projection"] = "mercator"
        elif not has_world_map and not uses_albers_usa:
            # Add world map outlines if needed
            vega_spec["layer"].append(VEGA_MAP_LAYER["world"])

    def _add_geographic_items(self, vega_spec: dict, vega_spec_str: str) -> dict:
        """Add geographic visualization items to vega spec."""
        self._add_zoom_params(vega_spec)
        self._setup_projection(vega_spec)
        self._handle_map_compatibility(vega_spec, vega_spec_str)
        return vega_spec

    @classmethod
    def _extract_as_keys(cls, transforms: list[dict]) -> list[str]:
        """
        Extracts all 'as' field names from a list of Vega-Lite transform definitions.

        Parameters
        ----------
        transforms : list[dict]
            A list of Vega-Lite transform objects.

        Returns
        -------
        list[str]
            A list of field names from 'as' keys (flattened, deduplicated).
        """
        as_fields = []
        for t in transforms:
            # Top-level 'as'
            if "as" in t:
                if isinstance(t["as"], list):
                    as_fields.extend(t["as"])
                elif isinstance(t["as"], str):
                    as_fields.append(t["as"])
            for key in ("aggregate", "joinaggregate", "window"):
                if key in t and isinstance(t[key], list):
                    for entry in t[key]:
                        if "as" in entry:
                            as_fields.append(entry["as"])

        return list(dict.fromkeys(as_fields))

    @retry_llm_output()
    async def _generate_basic_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc_examples: list,
        doc: str,
        errors: list | None = None
    ) -> dict[str, Any]:
        """Generate the basic VegaLite spec structure."""
        errors_context = self._build_errors_context(pipeline, context, errors)
        with self._add_step(title="Creating basic plot structure", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "main",
                messages,
                context,
                table=pipeline.table,
                doc=doc,
                doc_examples=doc_examples,
                **errors_context,
            )
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=VegaLiteSpec,
            )

            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            current_spec = await self._extract_spec(context, {"yaml_spec": output.yaml_spec})
            step.success_title = "Complete visualization with titles and colors created"
        return current_spec

    async def _extract_spec(self, context: TContext, spec: dict[str, Any]):
        # .encode().decode('unicode_escape') fixes a JSONDecodeError in Python
        # where it's expecting property names enclosed in double quotes
        # by properly handling the escaped characters in your JSON string
        if yaml_spec := spec.get("yaml_spec"):
            vega_spec = load_yaml(yaml_spec)
        elif json_spec := spec.get("json_spec"):
            vega_spec = load_json(json_spec)
        if "$schema" not in vega_spec:
            vega_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
        if "width" not in vega_spec:
            vega_spec["width"] = "container"
        if "height" not in vega_spec:
            vega_spec["height"] = "container"
        self._output_type.validate_spec(vega_spec)

        # using string comparison because these keys could be in different nested levels
        vega_spec_str = dump_yaml(vega_spec)
        # Handle different types of interactive controls based on chart type
        if "latitude:" in vega_spec_str or "longitude:" in vega_spec_str:
            vega_spec = self._add_geographic_items(vega_spec, vega_spec_str)
        # elif ("point: true" not in vega_spec_str or "params" not in vega_spec) and vega_spec_str.count("encoding:") == 1:
        #     # add pan/zoom controls to all plots except geographic ones and points overlaid on line plots
        #     # because those result in an blank plot without error
        #     vega_spec["params"] = [{"bind": "scales", "name": "grid", "select": "interval"}]
        return {"spec": vega_spec, "sizing_mode": "stretch_both", "min_height": 300}

    async def _get_doc_examples(self, user_query: str) -> list[str]:
        # Query vector store for relevant examples
        doc_examples = []
        vector_store = self._get_vector_store()
        if vector_store:
            if user_query:
                doc_results = await vector_store.query(user_query, top_k=5)
                # Extract text and specs from results
                k = 0
                for result in doc_results:
                    if "metadata" in result and "spec" in result["metadata"]:
                        # Include the text description/title before the spec
                        text_description = result.get("text", "")
                        spec = result["metadata"]["spec"]
                        if "hconcat" in spec or "vconcat" in spec or "repeat" in spec or "params" in spec:
                            # Skip complex multi-view specs for simplicity
                            continue
                        doc_examples.append(f"{text_description}\n```yaml\n{spec}\n```")
                        k += 1
                    if k >= 3:  # Limit to top 3 examples
                        break
        return doc_examples

    async def revise(
        self,
        feedback: str,
        messages: list[Message],
        context: TContext,
        view: LumenOutput | None = None,
        spec: str | None = None,
        language: str | None = None,
        errors: list[str] | None = None,
        **kwargs
    ) -> str:
        if errors is not None:
            kwargs["errors"] = errors
        doc_examples = await self._get_doc_examples(feedback)
        context["doc_examples"] = doc_examples
        return await super().revise(
            feedback, messages, context, view=view, spec=spec, language=language, **kwargs
        )

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """
        Generates a VegaLite visualization using progressive building approach with real-time updates.
        """
        pipeline = context.get("pipeline")
        if not pipeline:
            raise ValueError("Context did not contain a pipeline.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        user_query = messages[-1].get("content", "") if messages[-1].get("role") == "user" else ""
        try:
            doc_examples = await self._get_doc_examples(user_query)
        except Exception:
            doc_examples = []

        # Step 1: Generate basic spec
        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        # Produces {"spec": {$schema: ..., ...}, "sizing_mode": ..., ...}
        full_dict = await self._generate_basic_spec(messages, context, pipeline, doc_examples, doc)

        # Step 2: Show complete plot immediately
        view = self.view_type(pipeline=pipeline, **full_dict)
        out = self._output_type(component=view, title=step_title)

        # Step 3: enhancements (LLM-driven creative decisions)
        steps = {
            "interaction_polish": "Add helpful tooltips and ensure responsive, accessible user experience",
        }

        # Execute steps sequentially
        for step_name, step_desc in steps.items():
            # Only pass the vega lite 'spec' portion to prevent ballooning context
            step_name, update_dict = await self._update_spec_step(
                step_name, step_desc, out.spec, step_name, messages, context, doc=doc
            )
            try:
                # Validate merged spec
                test_spec = self._deep_merge_dicts(full_dict["spec"], update_dict)
                await self._extract_spec(context, {"yaml_spec": dump_yaml(test_spec)})
            except Exception as e:
                log_debug(f"Skipping invalid {step_name} update due to error: {e}")
                continue
            spec = self._deep_merge_dicts(full_dict["spec"], update_dict)
            out.spec = dump_yaml(spec)
            log_debug(f"📊 Applied {step_name} updates and refreshed visualization")

        return [out], {"view": dict(full_dict, type=view.view_type)}

    async def annotate(
        self,
        instruction: str,
        messages: list[Message],
        context: TContext,
        spec: dict
    ) -> dict:
        """
        Apply annotations based on user request.

        Parameters
        ----------
        instruction: str
            User's description of what to annotate
        messages: list[Message]
            Chat history for context
        context: TContext
            Session context
        spec : dict
            The current VegaLite specification (full dict with 'spec' key)

        Returns
        -------
        dict
            Updated specification with annotations
        """
        # Add user's annotation request to messages context
        annotation_messages = messages + [{
            "role": "user",
            "content": f"Add annotations: {instruction}"
        }]

        vega_spec = dump_yaml(spec["spec"], default_flow_style=False)
        system_prompt = await self._render_prompt(
            "annotate_plot",
            annotation_messages,
            context,
            vega_spec=vega_spec,
        )

        model_spec = self.prompts.get("annotate_plot", {}).get("llm_spec", self.llm_spec_key)
        result = await self.llm.invoke(
            messages=annotation_messages,
            system=system_prompt,
            response_model=VegaLiteSpecUpdate,
            model_spec=model_spec,
        )
        update_dict = load_yaml(result.yaml_update)

        # Merge and validate
        final_dict = spec.copy()
        try:
            final_dict["spec"] = self._deep_merge_dicts(final_dict["spec"], update_dict)
            await self._extract_spec(context, {"yaml_spec": dump_yaml(final_dict["spec"])})
        except Exception as e:
            log_debug(f"Skipping invalid annotation update due to error: {e}")
        return final_dict


class AnalysisInputs(ContextModel):

    data: NotRequired[Any]

    pipeline: Pipeline


class AnalysisOutputs(ContextModel):

    analysis: Callable

    view: Any


class AnalysisAgent(LumenBaseAgent):

    analyses = param.List([])

    conditions = param.List(
        default=[
            "Use for custom analysis, advanced analytics, or domain-specific analysis methods",
            "Use when built-in SQL/visualization agents are insufficient",
            "NOT for simple queries or basic visualizations",
        ]
    )

    purpose = param.String(default="Perform custom analyses on the data.")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "AnalysisAgent" / "main.jinja2"},
        }
    )

    _output_type = AnalysisOutput

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None
    ) -> tuple[list[Any], TContext]:
        pipeline = context.get("pipeline")
        analyses = {a.name: a for a in self.analyses if await a.applies(pipeline)}
        if not analyses:
            log_debug("No analyses apply to the current data.")
            return None

        # Short cut analysis selection if there's an exact match
        if len(messages):
            analysis = messages[0].get("content").replace("Apply ", "")
            if analysis in analyses:
                analyses = {analysis: analyses[analysis]}

        if len(analyses) > 1:
            with self._add_step(title="Choosing the most relevant analysis...", steps_layout=self._steps_layout) as step:
                type_ = Literal[tuple(analyses)]
                analysis_model = create_model(
                    "Analysis", correct_name=(type_, FieldInfo(description="The name of the analysis that is most appropriate given the user query."))
                )
                system_prompt = await self._render_prompt(
                    "main",
                    messages,
                    analyses=analyses,
                    data=context.get("data"),
                )
                model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
                analysis_name = (
                    await self.llm.invoke(
                        messages,
                        system=system_prompt,
                        model_spec=model_spec,
                        response_model=analysis_model,
                        allow_partial=False,
                    )
                ).correct_name
                step.stream(f"Selected {analysis_name}")
                step.success_title = f"Selected {analysis_name}"
        else:
            analysis_name = next(iter(analyses))

        view = None
        out_context = {}
        with self._add_step(title=step_title or "Creating view...", steps_layout=self._steps_layout) as step:
            await asyncio.sleep(0.1)  # necessary to give it time to render before calling sync function...
            analysis_callable = analyses[analysis_name].instance(agents=self.agents, interface=self.interface)

            out_context["analysis"] = analysis_callable
            data = await get_data(pipeline)
            for field in analysis_callable._field_params:
                analysis_callable.param[field].objects = list(data.columns)

            if analysis_callable.autorun:
                if asyncio.iscoroutinefunction(analysis_callable.__call__):
                    view = await analysis_callable(pipeline, context)
                else:
                    view = await asyncio.to_thread(analysis_callable, pipeline, context)
                if isinstance(view, Viewable):
                    view = Panel(object=view, pipeline=context.get("pipeline"))
                spec = view.to_spec()
                if isinstance(view, View):
                    view_type = view.view_type
                    out_context["view"] = dict(spec, type=view_type)
                elif isinstance(view, Pipeline):
                    out_context["pipeline"] = view
                # Ensure data reflects processed pipeline
                if "pipeline" in out_context:
                    pipeline = out_context["pipeline"]
                    data = await get_data(pipeline)
                    if len(data) > 0:
                        out_context["data"] = await describe_data(data)
                yaml_spec = dump_yaml(spec)
                step.stream(f"Generated view\n```yaml\n{yaml_spec}\n```")
                step.success_title = "Generated view"
            else:
                step.success_title = "Configure the analysis"

        analysis = out_context["analysis"]
        if view is None and analysis.autorun:
            self.interface.stream("Failed to find an analysis that applies to this data.")
        else:
            out = self._output_type(
                component=view, title=step_title, analysis=analysis, pipeline=out_context.get("pipeline"), context=context
            )
        return [] if view is None else [out], out_context


class ValidationInputs(ContextModel):

    data: NotRequired[Any]

    sql: NotRequired[str]

    view: NotRequired[Any]


class ValidationOutputs(ContextModel):

    validation_result: str


class ValidationAgent(Agent):
    """
    ValidationAgent focuses solely on validating whether the executed plan
    fully answered the user's original query. It identifies missing elements
    and suggests next steps when validation fails.
    """

    conditions = param.List(
        default=[
            "Use to validate whether executed plans fully answered user queries",
            "Use to identify missing elements from the original user request",
            "NOT for data analysis, pattern identification, or technical programming questions",
        ]
    )

    purpose = param.String(
        default="""
        Validates whether executed plans fully answered the user's original query.
        Identifies missing elements, assesses completeness, and suggests next steps
        when validation fails. Acts as a quality gate for plan execution."""
    )

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "ValidationAgent" / "main.jinja2", "response_model": QueryCompletionValidation, "tools": []},
        }
    )

    input_schema = ValidationInputs

    output_schema = ValidationOutputs

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], ValidationOutputs]:
        interface = self.interface
        def on_click(event):
            if messages:
                user_messages = [msg for msg in reversed(messages) if msg.get("role") == "user"]
                original_query = user_messages[0].get("content", "").split("-- For context...")[0]
            suggestions_list = '\n- '.join(result.suggestions)
            interface.send(f"Follow these suggestions to fulfill the original intent {original_query}\n\n{suggestions_list}")

        executed_steps = None
        if "plan" in context:
            executed_steps = [
                f"{step[0].__class__.__name__}: {step.instruction}" for step in context["plan"]
            ]

        system_prompt = await self._render_prompt("main", messages, context, executed_steps=executed_steps)
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

        result = await self.llm.invoke(
            messages=messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=QueryCompletionValidation,
        )
        response_parts = []
        if result.correct:
            return [result], {"validation_result": result}

        response_parts.append(f"**Query Validation: ✗ Incomplete** - {result.chain_of_thought}")
        if result.missing_elements:
            response_parts.append(f"**Missing Elements:** {', '.join(result.missing_elements)}")
        if result.suggestions:
            response_parts.append("**Suggested Next Steps:**")
            for i, suggestion in enumerate(result.suggestions, 1):
                response_parts.append(f"{i}. {suggestion}")

        button = Button(name="Rerun", on_click=on_click)
        footer_objects = [button]
        formatted_response = "\n\n".join(response_parts)
        interface.stream(formatted_response, user=self.user, max_width=self._max_width, footer_objects=footer_objects)
        return [result], {"validation_result": result}
