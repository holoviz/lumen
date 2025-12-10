"""
RESTSQLAgent - SQL agent for REST API data sources with dynamic URL parameter generation.

Extends SQLAgent to handle RESTDuckDBSource by allowing LLMs to generate both SQL queries
and URL parameters dynamically based on user questions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import param

from pydantic import BaseModel, Field, create_model

from ...sources.rest_duckdb import RESTDuckDBSource
from ..config import PROMPTS_DIR
from ..context import TContext
from ..llm import Message
from .sql import SQLAgent, SQLQuery


def make_rest_params_model(table_name: str, table_config: dict) -> type[BaseModel]:
    """
    Create a Pydantic model for a single REST table's URL parameters.

    Args:
        table_name: Name of the table (e.g., 'daily', 'raob')
        table_config: Dict with keys:
            - url_params: dict of param_name -> default_value
            - required_params: list of required param names (optional)
            - constant_params: list of params LLM shouldn't change, default ['format']
            - param_descriptions: dict of param_name -> description string (optional)

    Returns:
        Pydantic model class like DailyParams, RaobParams, etc.
    """
    url_params = table_config.get('url_params', {})
    required_params = table_config.get('required_params', [])
    constant_params = table_config.get('constant_params', ['format'])
    param_descriptions = table_config.get('param_descriptions', {})

    # Build field definitions for create_model
    fields = {}
    for param_name, default_value in url_params.items():
        # Skip constant params - they'll be merged back at execution
        if param_name in constant_params:
            continue

        # Determine if required
        is_required = param_name in required_params

        # Build description
        description = param_descriptions.get(param_name, f"Parameter: {param_name}")

        # Create field with examples to guide LLM
        if is_required:
            field_info = Field(..., description=description, examples=[str(default_value)])
        else:
            field_info = Field(default=str(default_value), description=description, examples=[str(default_value)])

        fields[param_name] = (str, field_info)

    # Create model with capitalized name
    model_name = f"{table_name.capitalize()}Params"
    return create_model(model_name, **fields)


def make_rest_sql_model(rest_tables: dict[str, dict]) -> type[BaseModel]:
    """
    Create RESTSQLQuery model combining SQLQuery base with typed url_params.

    Args:
        rest_tables: Dict mapping table_name -> table_config

    Returns:
        Pydantic model class RESTSQLQuery with:
            - query: str (inherited from SQLQuery)
            - table_slug: str (inherited from SQLQuery)
            - url_params: RESTParams | None (new field)
    """
    if not rest_tables:
        # No REST tables, return basic SQLQuery
        return SQLQuery

    # Create params model for each REST table
    table_param_models = {}
    for table_name, table_config in rest_tables.items():
        param_model = make_rest_params_model(table_name, table_config)
        table_param_models[table_name] = (param_model | None, Field(default=None))

    # Create RESTParams with optional fields for each table
    RESTParams = create_model('RESTParams', **table_param_models)

    # Create RESTSQLQuery extending SQLQuery
    RESTSQLQuery = create_model(
        'RESTSQLQuery',
        url_params=(
            RESTParams | None,
            Field(
                default=None,
                description="URL parameters for REST tables. Only include params for tables you're actually querying."
            )
        ),
        __base__=SQLQuery
    )

    return RESTSQLQuery


class RESTSQLAgent(SQLAgent):
    """SQL agent for RESTDuckDBSource with dynamic URL parameter generation."""

    conditions = param.List(default=[
        "Use when querying REST API-backed data sources",
        "Use when user asks about different locations, time periods, or other parameterized queries",
    ])

    purpose = param.String(default="""
        Handles queries against REST API data sources. Dynamically generates
        URL parameters (station codes, date ranges, etc.) based on user questions,
        fetches the appropriate data, then executes SQL against it.
    """)

    # Override prompts to use REST-specific template and response model
    prompts = param.Dict(default={
        "main": {
            "response_model": make_rest_sql_model,
            "template": PROMPTS_DIR / "RESTSQLAgent" / "main.jinja2",
        },
        # Reuse SQLAgent's other prompts
        "select_discoveries": SQLAgent.param.prompts.default["select_discoveries"],
        "check_sufficiency": SQLAgent.param.prompts.default["check_sufficiency"],
        "revise_output": SQLAgent.param.prompts.default["revise_output"],
    })

    def _get_rest_tables(self, source: RESTDuckDBSource) -> dict[str, dict]:
        """Extract REST table configs from source."""
        rest_tables = {}
        for table_name, table_config in source.tables.items():
            if isinstance(table_config, dict) and 'url' in table_config:
                rest_tables[table_name] = table_config
        return rest_tables

    def _get_model(self, prompt_name: str, **kwargs):
        """Override to build response model with REST table configs."""
        if prompt_name != "main":
            return super()._get_model(prompt_name, **kwargs)

        # The model will be built dynamically in _render_execute_query
        # where we have access to the actual source objects
        model_fn = self.prompts["main"]["response_model"]
        return model_fn

    def _extract_tables_from_sql(self, sql_query: str, rest_table_names: list[str]) -> list[str]:
        """Parse SQL to find which REST tables are referenced."""
        # Simple string matching approach
        tables_used = []
        sql_upper = sql_query.upper()
        for table_name in rest_table_names:
            if table_name.upper() in sql_upper:
                tables_used.append(table_name)
        return tables_used

    def _apply_url_params(
        self,
        source: RESTDuckDBSource,
        url_params: BaseModel | None,
        tables_used: list[str]
    ) -> None:
        """
        Materialize REST tables with LLM-generated params before SQL execution.

        Args:
            source: The RESTDuckDBSource
            url_params: The url_params from LLM response (RESTParams model)
            tables_used: List of table names referenced in SQL
        """
        if not url_params:
            return

        rest_tables = self._get_rest_tables(source)

        for table_name in tables_used:
            if table_name not in rest_tables:
                continue

            # Get LLM-generated params for this table
            table_params = getattr(url_params, table_name, None)
            if not table_params:
                continue

            # Convert Pydantic model to dict (try both v1 and v2 methods)
            if hasattr(table_params, 'model_dump'):
                llm_params = table_params.model_dump()
            elif hasattr(table_params, 'dict'):
                llm_params = table_params.dict()
            else:
                llm_params = dict(table_params)

            # Merge with constant params from source config
            table_config = rest_tables[table_name]
            constant_params = table_config.get('constant_params', ['format'])
            full_params = {}

            # Add constant params
            for param_name in constant_params:
                if param_name in table_config.get('url_params', {}):
                    full_params[param_name] = table_config['url_params'][param_name]

            # Add LLM-generated params
            full_params.update(llm_params)

            # Materialize the table with full params
            source.get(table_name, url_params=full_params)

    async def _gather_prompt_context(self, prompt_name: str, messages: list, context: TContext, **kwargs):
        """Override to add REST-specific context."""
        prompt_context = await super()._gather_prompt_context(prompt_name, messages, context, **kwargs)

        if prompt_name == "main":
            # Add REST tables config for template
            sources = context.get("sources", [])
            rest_tables = {}

            # Extract REST tables from all sources
            for source in sources:
                if isinstance(source, RESTDuckDBSource):
                    rest_tables.update(self._get_rest_tables(source))

            if rest_tables:
                prompt_context["rest_tables"] = rest_tables

            # Add current date for temporal queries
            prompt_context["current_date"] = datetime.now().strftime("%Y-%m-%d")

        return prompt_context

    async def _render_execute_query(
        self,
        messages: list[Message],
        context: TContext,
        sources: dict[tuple[str, str], Any],
        step_title: str,
        success_message: str,
        discovery_context: str | None = None,
        raise_if_empty: bool = False,
        output_title: str | None = None
    ):
        """
        Override to apply URL params before validation.

        This is the key difference from SQLAgent - we need to materialize
        REST tables with the correct URL parameters before validating SQL.
        """
        # First, check if we have any REST sources
        has_rest_source = any(
            isinstance(source, RESTDuckDBSource)
            for source in sources.values()
        )

        if not has_rest_source:
            # No REST sources, use parent implementation
            return await super()._render_execute_query(
                messages, context, sources, step_title, success_message,
                discovery_context, raise_if_empty, output_title
            )

        # We have REST sources - need custom implementation
        with self._add_step(title=step_title, steps_layout=self._steps_layout) as step:
            # Get REST table configs
            rest_tables = {}
            for source in sources.values():
                if isinstance(source, RESTDuckDBSource):
                    rest_tables.update(self._get_rest_tables(source))

            # Generate SQL using REST-specific prompt
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
                rest_tables=rest_tables,
                current_date=datetime.now().strftime("%Y-%m-%d"),
            )

            # Generate SQL using REST model
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

            # Build response model with REST tables
            sql_response_model = make_rest_sql_model(rest_tables)

            output = await self.llm.invoke(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=sql_response_model,
            )

            if not output:
                raise ValueError("No output was generated.")

            # Get source and tables
            if len(sources) == 1:
                source = next(iter(sources.values()))
                tables = [next(table for _, table in sources)]
            else:
                source, tables = self._merge_sources(sources, output.tables if hasattr(output, 'tables') else [])

            sql_query = output.query.strip()
            expr_slug = output.table_slug.strip()

            # Apply URL params BEFORE validation
            if hasattr(output, 'url_params') and output.url_params:
                tables_used = self._extract_tables_from_sql(sql_query, list(rest_tables.keys()))
                self._apply_url_params(source, output.url_params, tables_used)
                step.stream(f"\n\n🔧 Applied URL parameters for tables: {', '.join(tables_used)}")

            # Now validate and execute
            validated_sql = await self._validate_sql(
                context, sql_query, expr_slug, source, messages,
                step, discovery_context=discovery_context
            )

            pipeline, sql_expr_source, summary = await self._execute_query(
                source, context, expr_slug, validated_sql, tables=tables,
                is_final=True, should_materialize=True, step=step
            )

            view = await self._finalize_execution(
                pipeline,
                validated_sql,
                output_title,
                raise_if_empty=raise_if_empty
            )

            if hasattr(step, 'param'):
                step.param.update(
                    status="success",
                    success_title=success_message
                )

        return view
