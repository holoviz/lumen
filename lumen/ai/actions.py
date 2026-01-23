from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import param

from panel_material_ui import TextAreaInput, TextInput, Typography

from lumen.ai.agents.chat import ChatAgent

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource
from .report import Action
from .schemas import Metaset, get_metaset
from .utils import describe_data
from .views import SQLOutput

if TYPE_CHECKING:
    from .context import TContext


class SQLQueryInputs(TypedDict):

    source: BaseSQLSource


class SQLQueryOutputs(TypedDict):
    source: BaseSQLSource
    pipeline: Pipeline
    data: dict
    metaset: Metaset
    table: str


class SQLQuery(Action):
    """
    An `SQLQuery` is an `Action` that executes a SQL expression on a Source
    and generates an LumenOutput to be rendered.
    """

    generate_caption = param.Boolean(default=True, doc="""
        Whether to generate a caption for the data.""")

    source = param.ClassSelector(class_=BaseSQLSource, doc="""
        The Source to execute the SQL expression on.""")

    sql_expr = param.String(default="", doc="""
        The SQL expression to use for the action.""")

    table_params = param.List(default=[], doc="""
        List of parameters to pass to the SQL expression.
        Parameters are used with placeholders (?) in the SQL expression.""")

    table = param.String(doc="""
        The name of the table generated from the SQL expression.""")

    user_content = param.String(default="Generate a short caption for the data", doc="""
        Additional instructions to provide to the analyst agent, i.e. what to focus on.""")

    inputs = SQLQueryInputs
    outputs = SQLQueryOutputs

    def _render_controls(self):
        return [
            TextInput.from_param(
                self.param.table, sizing_mode="stretch_width", margin=(10, 0)
            ),
            TextAreaInput.from_param(
                self.param.sql_expr, sizing_mode="stretch_width", margin=(10, 0)
            ),
        ]

    def __repr__(self):
        params = []
        if self.sql_expr:
            params.append(f"sql_expr='{self.sql_expr}'")
        if self.table:
            params.append(f"table='{self.table}'")
        if self.title:
            params.append(f"title='{self.title}'")
        return f"{self.__class__.__name__}({', '.join(params)})"

    async def _execute(self, context: TContext, **kwargs) -> tuple[list[Any], SQLQueryOutputs]:
        """
        Executes the action.

        Arguments
        ----------
        **kwargs: dict
            Additional keyword arguments to pass to the action.

        Returns
        -------
        The outputs of the action.
        """
        source = self.source
        if source is None:
            if context is None or 'source' not in context:
                raise ValueError(
                    "SQLQuery could not resolve a source. Either provide "
                    "an explicit source or ensure another action or actor "
                    "provides a source."
                )
            source = context['source']
        if not self.table:
            raise ValueError("SQLQuery must declare a table name.")

        # Pass table_params if provided
        params = {self.table: self.table_params} if self.table_params else None
        source = source.create_sql_expr_source({self.table: self.sql_expr}, params=params)
        pipeline = Pipeline(source=source, table=self.table)
        out_context = {
            "source": source,
            "pipeline": pipeline,
            "data": await describe_data(pipeline.data),
            "metaset": await get_metaset([source], [self.table]),
            "table": self.table,
        }
        outputs = [SQLOutput(component=pipeline, spec=self.sql_expr)]
        if self.generate_caption:
            caption_out, _ = await ChatAgent(llm=self.llm).respond(
                [{"role": "user", "content": self.user_content}], out_context
            )
            caption = caption_out[0]
            outputs.append(Typography(caption.object))
        return outputs, out_context
