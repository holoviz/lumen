import asyncio

from types import FunctionType
from typing import Self

import panel as pn
import param

from panel.layout.base import ListLike, NamedListLike
from panel.viewable import Viewable, Viewer
from panel_material_ui import (
    Accordion, ChatFeed, ChatMessage, Container, Paper, Progress, Tabs,
    Typography,
)

from ..views.base import Panel, View
from .actor import Actor
from .llm import Llm
from .memory import _Memory
from .tools import FunctionTool, Tool
from .views import LumenOutput


class Task(Viewer):
    """
    A Task is a single unit of work that can be executed and rendered.
    """

    history = param.List()

    interface = param.ClassSelector(class_=ChatFeed)

    instruction = param.String(default="")

    llm = param.ClassSelector(class_=Llm)

    memory = param.ClassSelector(class_=_Memory)

    title = param.String()

    running = param.Boolean()

    status = param.String(default="idle")

    steps_layout = param.ClassSelector(default=None, class_=(ListLike, NamedListLike), allow_None=True, doc="""
        The layout progress updates will be streamed to.""")

    subtasks = param.List()

    level = 3

    def __init__(self, *subtasks, **params):
        if not subtasks:
            subtasks = params.pop('subtasks', [])
        else:
            subtasks = list(subtasks)
        super().__init__(subtasks=[FunctionTool(task) if isinstance(task, FunctionType) else task for task in subtasks], **params)
        self._init_view()
        self._populate_view()

    def _init_view(self):
        self._view = self._output = pn.Column(sizing_mode='stretch_width')

    def _populate_view(self):
        self._view[:] = []

    def clear(self):
        for subtask in self.subtasks:
            if isinstance(subtask, Task):
                subtask.clear()
        self._populate_view()

    def _render_output(self, out):
        if isinstance(out, ChatMessage):
            return out.object
        elif isinstance(out, (Viewable, View)):
            return out
        elif isinstance(out, LumenOutput):
            return Tabs(
                ('Specification', out),
                ('Output', pn.param.ParamMethod(out.render, inplace=True, sizing_mode='stretch_both')),
                active=1, sizing_mode='stretch_width'
            )

    async def _run_task(self, i: int, task: Self | Actor, **kwargs):
        pre = len(self.memory['outputs'])
        outputs = []
        memory = task.memory or self.memory
        messages = self.history + [{'content': self.instruction, 'role': 'user'}]
        with task.param.update(
            interface=self.interface, llm=task.llm or self.llm, memory=memory,
            steps_layout=self.steps_layout
        ):
            if isinstance(task, Actor):
                out = await task.respond(messages, **kwargs)
                # Handle Tool specific behaviors
                if isinstance(task, Tool):
                    # Handle View/Viewable results regardless of agent type
                    if isinstance(out, (View, Viewable)):
                        if isinstance(out, Viewable):
                            out = Panel(object=out, pipeline=self.memory.get('pipeline'))
                        out = LumenOutput(
                            component=out, title=self.title
                        )
                        message_kwargs = dict(value=out, user=task.name)
                        if self.interface:
                            self.interface.stream(**message_kwargs)
                            self.memory['outputs'] = self.memory['outputs'] + [out]
                if isinstance(out, (Viewable, View, LumenOutput)):
                    new = [out]
                else:
                    new = self.memory['outputs'][pre-1:]
                outputs += new
                self._view.extend([self._render_output(out) for out in new])
            else:
                with task.param.update(running=True, history=messages):
                    outputs += await task.execute(**kwargs)
        return outputs

    async def execute(self, **kwargs):
        if 'outputs' not in self.memory:
            self.memory['outputs'] = []
        outputs = [f"{'#'*self.level} {self.title}"] if self.title else []
        for i, task in enumerate(self.subtasks):
            try:
                outputs += await self._run_task(i, task, **kwargs)
            except Exception:
                self.status = "error"
            else:
                self.status = "success"
        return outputs

    def __panel__(self):
        return self._output


class Section(Task):
    """
    A Section consists of multiple Tasks which are executed and rendered in sequence.
    """

    subtasks = param.List(item_type=Task)

    level = 2

    def _init_view(self):
        self._view = self._output = pn.Column()

    def _populate_view(self):
        self._view[:] = [subtask for subtask in self.subtasks if isinstance(subtask, Task)]

    async def _run_task(self, i: int, task: Task | Actor):
        self.memory['outputs'] = []
        instructions = "\n".join(f"{i+1}. task.instruction" for i, task in enumerate(self.subtasks))
        self.memory['reasoning'] = f"{self.title}\n\n{instructions}"
        return await super()._run_task(i, task)

    @param.depends('running', watch=True)
    async def _running(self):
        await asyncio.sleep(0.05)
        if not self.running:
            return
        loader = Progress(variant='indeterminate', sizing_mode='stretch_width')
        self._view.insert(0, loader)
        while self.running:
            await asyncio.sleep(0.05)
        self._view.remove(loader)


class Report(Task):
    """
    A Report consists of multiple Sections which are executed and rendered in sequence.
    """

    subtasks = param.List(item_type=Section)

    level = 1

    def _init_view(self):
        self._view = Accordion(sizing_mode='stretch_width')
        self._output = Paper(
            Typography(self.title, variant='h3'),
            self._view
        )

    def _populate_view(self):
        self._view[:] = objects = [(subtask.title, subtask) for subtask in self.subtasks if isinstance(subtask, Task)]
        self._view.active = list(range(len(objects)))

    async def _run_task(self, i: int, task: Task | Actor):
        self._view.active = self._view.active + [i]
        return await super()._run_task(i, task)

    def __panel__(self):
        return Container(self._output)
