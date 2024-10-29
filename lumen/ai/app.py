from __future__ import annotations

import param

from panel.config import config, panel_extension
from panel.io.state import state
from panel.layout import Row
from panel.viewable import Viewer

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from .agents import (
    AnalysisAgent, ChatAgent, SourceAgent, SQLAgent,
)
from .assistant import Assistant, PlanningAssistant
from .llm import Llm, OpenAI
from .memory import memory

DataT = str | Source | Pipeline


class LumenAI(Viewer):

    analyses = param.List(default=[])

    assistant = param.ClassSelector(class_=Assistant, default=PlanningAssistant, is_instance=False)

    agents = param.List(default=[])

    default_agents = param.List(default=[
        ChatAgent, SourceAgent, SQLAgent
    ])

    llm = param.ClassSelector(class_=Llm, default=OpenAI())

    template = param.Selector(
        default=config.param.template.names['fast'],
        objects=config.param.template.names
    )

    title = param.String(default='Lumen.ai')

    def __init__(
        self,
        data: DataT | list[DataT] | None = None,
        **params
    ):
        super().__init__(**params)
        agents = self.default_agents + self.agents
        if self.analyses:
            agents.append(AnalysisAgent(analyses=self.analyses))
        self._assistant = self.assistant(
            agents=agents,
            llm=self.llm
        )
        self._resolve_data(data)

    def _resolve_data(self, data: DataT | list[DataT] | None):
        if data is None:
            return
        elif not isinstance(data, list):
            data = [data]
        sources = []
        mirrors, tables = {}, {}
        remote = False
        for src in data:
            if isinstance(src, Source):
                sources.append(src)
            elif isinstance(src, Pipeline):
                mirrors[src.name] = src
            elif isinstance(src, str):
                if src.startswith('http'):
                    remote = True
                if src.endswith(('.parq', '.parquet')):
                    table = f"read_parquet('{src}')"
                elif src.endswith(".csv"):
                    table = f"read_csv('{src}')"
                elif src.endswith(".json"):
                    table = f"read_json_auto('{src}')"
                else:
                    raise ValueError(
                        "Could not determine how to load {} file."
                    )
                tables[src] = table
        if tables or mirrors:
            initializers = ["INSTALL httpfs;", "LOAD httpfs;"] if remote else []
            source = DuckDBSource(tables=tables, mirrors=mirrors, uri=':memory:', initializers=initializers)
            sources.append(source)
        if not sources:
            raise ValueError(
                'Must provide at least one data source.'
            )
        memory['available_sources'] = sources
        if sources:
            memory['current_source'] = sources[0]

    def show(self, **kwargs):
        return self._create_view(server=True).show(**kwargs)

    def _create_view(self, server: bool | None = None):
        if (state.curdoc and state.curdoc.session_context) or server is True:
            panel_extension(
                *{ext for agent in self._assistant.agents for ext in agent._extensions}, template=self.template
            )
            config.template = self.template
            template = state.template
            template.title = self.title
            template.main.append(self._assistant)
            template.sidebar.append(self._assistant.controls())
            return template
        return super()._create_view()

    def servable(self, title: str | None = None, **kwargs):
        if (state.curdoc and state.curdoc.session_context):
            self._create_view()
            return self
        return self._create_view().servable(title, **kwargs)

    def __panel__(self):
        return Row(
            Row(self._assistant.controls(), max_width=300),
            self._assistant
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        panel_extension(
            *{ext for exts in self._assistant.agents for ext in exts}, design='material', notifications=True
        )
        return self._create_view()._repr_mimebundle_(include, exclude)
