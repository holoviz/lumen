import asyncio
import io
import traceback as tb

from functools import partial
from types import FunctionType

import panel as pn
import param

from panel.layout.base import (
    Column, ListLike, NamedListLike, Row,
)
from panel.pane import Markdown
from panel.viewable import Viewable, Viewer
from panel_material_ui import (
    Accordion, Alert, ChatFeed, ChatMessage, Container, Dialog, Divider,
    FileDownload, IconButton, Progress, Tabs, TextAreaInput, TextInput,
    Typography,
)
from typing_extensions import Self

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource
from ..views.base import Panel, View
from .actor import Actor
from .agents import AnalystAgent
from .config import MissingContextError
from .export import (
    format_output, make_md_cell, make_preamble, write_notebook,
)
from .llm import Llm
from .memory import _Memory
from .tools import FunctionTool, Tool
from .utils import describe_data, wrap_logfire_on_method
from .views import LumenOutput, SQLOutput


class Task(Viewer):
    """
    A Task is a single unit of work that can be executed and rendered.
    """

    abort_on_error = param.Boolean(default=False, doc="""
        If True, the report will abort if an error occurs.""")

    history = param.List(doc="""
        Conversation history to include as context for the task.""")

    interface = param.ClassSelector(class_=ChatFeed)

    instruction = param.String(default="", doc="""
        The instruction to give to the task.""")

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM to use for the task.""")

    memory = param.ClassSelector(class_=_Memory)

    outputs = param.List()

    title = param.String(doc="""
        The title of the task.""")

    running = param.Boolean(doc="""
        Whether the task is currently running.""")

    status = param.Selector(objects=["idle", "success", "error"], default="idle", doc="""
        The current status of the task.""")

    steps_layout = param.ClassSelector(default=None, class_=(ListLike, NamedListLike), allow_None=True, doc="""
        The layout progress updates will be streamed to.""")

    subtasks = param.List(doc="""
        The subtasks of the task.""")

    level = 3

    def __init__(self, *subtasks, **params):
        if not subtasks:
            subtasks = params.pop('subtasks', [])
        else:
            subtasks = list(subtasks)
        super().__init__(subtasks=[FunctionTool(task) if isinstance(task, FunctionType) else task for task in subtasks], **params)
        self._init_view()
        self._populate_view()

    def __init_subclass__(cls, **kwargs):
        """
        Apply wrap_logfire to all the subclasses' execute automatically
        """
        super().__init_subclass__(**kwargs)
        wrap_logfire_on_method(cls, "execute")

    def __repr__(self):
        params = []
        if self.instruction:
            params.append(f"instruction='{self.instruction}'")
        if self.title:
            params.append(f"title='{self.title}'")
        subtasks = [f"\n    {task!r}" for task in self.subtasks]
        return f"{self.__class__.__name__}({', '.join(params)}{''.join(subtasks)})"

    def _init_view(self):
        self._view = self._output = Column(sizing_mode='stretch_width', styles={'min-height': 'unset'}, height_policy='fit')

    def _populate_view(self):
        self._view[:] = []

    def clear(self):
        for subtask in self.subtasks:
            if isinstance(subtask, Task):
                subtask.clear()
        self._populate_view()
        self.outputs.clear()

    def _render_output(self, out):
        if isinstance(out, ChatMessage):
            return Typography(out.object, margin=(20, 10))
        elif isinstance(out, (Viewable, View)):
            return out
        elif isinstance(out, LumenOutput):
            return Tabs(
                ('Specification', out),
                ('Output', pn.param.ParamMethod(out.render, inplace=True, sizing_mode='stretch_width')),
                active=1, sizing_mode='stretch_width', min_height=0, height_policy='fit'
            )

    def _add_outputs(self, i: int, outputs: list, **kwargs):
        views = []
        for out in outputs:
            view = self._render_output(out)
            if view is not None:
                views.append(view)
        self._view.extend(views)
        self.outputs += outputs

    def _add_child_outputs(self, previous, event):
        self.outputs = previous + event.new

    async def _run_task(self, i: int, task: Self | Actor, **kwargs):
        pre = len(self.memory['outputs'])
        outputs = []
        memory = task.memory or self.memory
        messages = self.history

        with task.param.update(
            interface=self.interface, llm=task.llm or self.llm, memory=memory,
            steps_layout=self.steps_layout
        ):
            if isinstance(task, Actor):
                try:
                    out = await task.respond(messages, **kwargs)
                except MissingContextError:
                    # Re-raise MissingContextError to allow retry logic at Plan level
                    raise
                except Exception as e:
                    tb.print_exception(e)
                    alert = Alert(
                        f'Executing task {type(task).__name__} failed.', alert_type='error', sizing_mode="stretch_width"
                    )
                    self._add_outputs(i, [alert])
                    return outputs
                # Handle Tool specific behaviors
                if isinstance(task, Tool) and isinstance(out, (View, Viewable)):
                    # Handle View/Viewable results regardless of agent type
                    if isinstance(out, Viewable):
                        out = Panel(object=out, pipeline=self.memory.get('pipeline'))
                    out = LumenOutput(component=out, title=self.title)
                    message_kwargs = dict(value=out, user=task.name)
                    if self.interface:
                        self.interface.stream(**message_kwargs)
                        self.memory['outputs'] = self.memory['outputs'] + [out]
                new = self.memory['outputs'][pre:]
                if not new and isinstance(out, (Viewable, View, LumenOutput)):
                    new = [out]
                outputs += new
                self._add_outputs(i, new, **kwargs)
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
            except MissingContextError:
                # Re-raise MissingContextError to allow retry logic at Plan level
                raise
            except Exception as e:
                tb.print_exception(e)
                self.status = "error"
                if self.abort_on_error:
                    break
            else:
                self.status = "success"
        return outputs

    def editor(self, level=0, show_title=True):
        return Column(
            *self._render_controls(),
            *((
                Divider(sizing_mode="stretch_width", margin=10),
                Accordion(
                    *self._render_subtasks(level=level), margin=(10, 10, 5, 10), sizing_mode="stretch_width"
                ),
              ) if self.subtasks else ()
            )
        )

    def _render_controls(self):
        return [
            TextInput.from_param(
                self.param.title, sizing_mode="stretch_width"
            ),
            TextAreaInput.from_param(
                self.param.instruction, sizing_mode="stretch_width"
            ),
        ]

    def _render_subtasks(self, level: int = 0) -> Viewable:
        subtasks = []
        for task in self.subtasks:
            task_type = type(task).__name__
            if isinstance(task, Task):
                subtasks.append((
                    f'{task.title} {task_type}',
                    task.editor(level=level+1, show_title=False)
                ))
            elif isinstance(task, FunctionTool):
                subtasks.append((f"{task_type}: {task.function.__name__}", ''))
            elif isinstance(task, Actor):
                subtasks.append((task_type, ''))
        return subtasks

    def to_notebook(self):
        cells = make_preamble("", extensions=['tabulator'], title=self.title)
        for out in self.outputs:
            if isinstance(out, Typography):
                level = int(out.variant[1:]) if out.variant and out.variant.startswith('h') else 0
                prefix = f"{'#'*level} " if level else ''
                cells.append(make_md_cell(f"{prefix}{out.object}"))
            elif isinstance(out, Markdown):
                cells.append(make_md_cell(out))
            elif isinstance(out, LumenOutput):
                cells += format_output(out)
            elif isinstance(out, Viewable):
                cells += format_output(Panel(out))
        return write_notebook(cells)

    def __panel__(self):
        return self._output


class Action(Task):
    """
    An Action is a Task that does not involve the usage of an LLM to
    perform the task, e.g. executing a specified SQL query instead of
    asking an LLM to generate the query.
    """


class SQLQuery(Action):

    add_output = param.Boolean(default=True)

    generate_caption = param.Boolean(default=True)

    source = param.ClassSelector(class_=BaseSQLSource)

    sql_expr = param.String(default="")

    table = param.String()

    async def _rerun(self, i: int, old: list, event: param.Event, **kwargs):
        pass

    def _render_controls(self):
        return [
            TextInput.from_param(
                self.param.table, sizing_mode="stretch_width"
            ),
            TextAreaInput.from_param(
                self.param.sql_expr, sizing_mode="stretch_width"
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

    async def execute(self, **kwargs):
        source = self.source
        if source is None:
            if 'source' not in self.memory:
                raise ValueError(
                    "SQLAction could not resolve a source. Either provide "
                    "an explicit source or ensure another action or actor "
                    "provides a source."
                )
            source = self.memory['source']
        source = source.create_sql_expr_source({self.table: self.sql_expr})
        self.memory["source"] = source
        self.memory["sources"].append(source)
        self.memory["pipeline"] = pipeline = Pipeline(source=source, table=self.table)
        self.memory["data"] = await describe_data(pipeline.data)
        self.memory["table"] = self.table
        out = SQLOutput(component=pipeline, spec=self.sql_expr)
        outputs = [Typography(f"### {self.title}", variant='h4'), out] if self.title else [out]
        if self.generate_caption:
            caption = await AnalystAgent(llm=self.llm).respond([{"role": "user", "content": "Generate a short caption for the data"}])
            outputs.append(Typography(caption.object, margin=(20, 10)))
        if self.add_output:
            self._add_outputs(0, outputs, **kwargs)
        return [outputs]


class Section(Task):
    """
    A Section consists of multiple Tasks which are executed and rendered in sequence.
    """

    subtasks = param.List(item_type=Task)

    level = 2

    def __init__(self, *subtasks, **params):
        self._watchers = {}
        super().__init__(*subtasks, **params)

    def __repr__(self):
        params = []
        if self.title:
            params.append(f"title='{self.title}'")
        subtasks = [f"\n    {task!r}" for task in self.subtasks]
        return f"{self.__class__.__name__}({', '.join(params)}{''.join(subtasks)})"

    def _render_controls(self):
        return [
            TextInput.from_param(
                self.param.title, sizing_mode="stretch_width"
            )
        ]

    def _init_view(self):
        self._dialog = Dialog(
            self.editor(),
            show_close_button=True,
            close_on_click=True,
            title=f'{self.title} Section Configuration',
        )
        self._settings = IconButton(icon='settings', on_click=self._open_settings, size="small", color="default")
        self._view = Column(sizing_mode='stretch_width', styles={'min-height': 'unset'}, height_policy='fit')
        self._view[:] = ["```python\n{repr(self)}\n```"]
        self._output = Column(
            Row(self._settings, self._dialog, styles={'position': 'absolute', 'top': '-70px', 'right': '20px'}),
            self._view,
            sizing_mode='stretch_width',
            styles={'min-height': 'unset'},
            height_policy='fit'
        )

    def _populate_view(self):
        self._view[:] = [subtask for subtask in self.subtasks if isinstance(subtask, Task)]

    def _open_settings(self, event):
        self._dialog.open = True

    def _add_child_outputs(self, i: int, previous: list, event: param.Event, **kwargs):
        for out in (event.old or []):
            if out not in event.new:
                out.param.unwatch(self._watchers[out])
        for out in event.new:
            if out not in self._watchers and isinstance(out, LumenOutput):
                context = dict(
                    interface=self.interface, llm=self.llm, memory=self.memory.clone(),
                    steps_layout=self.steps_layout
                )
                self._watchers[out] = out.param.watch(partial(self._rerun, i+1, context), 'spec')
        self.outputs = previous + event.new

    async def _rerun(self, i: int, context: dict, _: param.Event, **kwargs):
        for task in self.subtasks[i:]:
            task.clear()
        with self.param.update(running=True):
            for j, task in enumerate(self.subtasks[i:]):
                try:
                    with self.param.update(context):
                        await self._run_task(i+j, task, **kwargs)
                except Exception as e:
                    tb.print_exception(e)
                    self.status = "error"
                    if self.abort_on_error:
                        break
                else:
                    self.status = "success"

    async def _run_task(self, i: int, task: Task | Actor, **kwargs):
        self.memory['outputs'] = []
        instructions = "\n".join(f"{i+1}. task.instruction" for i, task in enumerate(self.subtasks))
        self.memory['reasoning'] = f"{self.title}\n\n{instructions}"
        if isinstance(task, Task):
            watcher = task.param.watch(partial(self._add_child_outputs, i, list(self.outputs), **kwargs), 'outputs')
        try:
            outputs = await super()._run_task(i, task)
        finally:
            if isinstance(task, Task):
                task.param.unwatch(watcher)
        return outputs

    @param.depends('running', watch=True)
    async def _running(self):
        await asyncio.sleep(0.05)
        if not self.running:
            return
        loader = Progress(variant='indeterminate', sizing_mode='stretch_width', margin=0)
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
        self._title = Typography(self.param.title, variant="h1", margin=(0, 0, 0, 10))
        self._view = Accordion(sizing_mode="stretch_width", min_height=0, margin=(0, 5, 5, 5))
        self._run = IconButton(icon="play_arrow", on_click=self._execute, margin=0, size="large")
        self._clear = IconButton(icon="clear", on_click=lambda _: self.clear(), margin=0, size="large")
        self._collapse = IconButton(
            styles={"margin-left": "auto"}, on_click=self._expand_all, icon="unfold_less", size="large", color="default", margin=(0, 0, 10, 0)
        )
        self._settings = IconButton(
            icon="settings", on_click=self._open_settings, size="large", color="default", margin=0
        )
        self._export = FileDownload(
            callback=self._notebook_export, label="\u200b", variant='text', icon='get_app', icon_size="2.4em",
            color="default", sx={".MuiButton-startIcon": {"mr": 0}}, margin=(8, 0, 10, 0)
        )
        self._dialog = Dialog(
            TextInput.from_param(self.param.title, margin=0),
            show_close_button=True, close_on_click=True,
            title=f"{self.title} Report Settings",
        )
        self._menu = Row(
            self._title,
            self._run,
            self._clear,
            self._collapse,
            self._export,
            self._settings,
            self._dialog,
            sizing_mode="stretch_width"
        )
        self._output = Column(
            self._view,
            margin=(0, 0, 0, 5),
            sizing_mode="stretch_both"
        )

    def _notebook_export(self):
        return io.StringIO(self.to_notebook())

    async def _execute(self, *args):
        with self._run.param.update(loading=True):
            await self.execute()

    def _expand_all(self, event):
        if self._collapse.icon == "unfold_less":
            self._view.active = []
            self._collapse.icon = "expand"
        else:
            self._view.active = list(range(len(self._view)))
            self._collapse.icon = "unfold_less"

    def _open_settings(self, event):
        self._dialog.open = True

    def _populate_view(self):
        self._view[:] = objects = [(subtask.title, subtask) for subtask in self.subtasks if isinstance(subtask, Task)]
        self._view.active = list(range(len(objects)))

    async def _run_task(self, i: int, task: Section):
        self._view.active = self._view.active + [i]
        watcher = task.param.watch(partial(self._add_child_outputs, self.outputs), "outputs")
        try:
            outputs = await super()._run_task(i, task)
        finally:
            task.param.unwatch(watcher)
        return outputs

    def __panel__(self):
        return Column(
            self._menu,
            Container(
                self._output, sizing_mode="stretch_both", height_policy="max",
                stylesheets=[":host > div { overflow-y: auto; }"]
            )
        )
