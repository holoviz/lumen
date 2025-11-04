from __future__ import annotations

import asyncio
import io
import traceback as tb

from abc import abstractmethod
from collections.abc import Iterator
from functools import partial
from types import FunctionType
from typing import Any, final

import panel as pn
import param

from panel.layout.base import (
    Column, ListLike, NamedListLike, Row,
)
from panel.pane import Markdown
from panel.viewable import Viewable, Viewer
from panel_material_ui import (
    Accordion, Alert, Button, ChatFeed, ChatMessage, Container, Dialog,
    Divider, FileDownload, IconButton, Progress, Select, Tabs, TextAreaInput,
    TextInput, Typography,
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
from .schemas import get_metaset
from .tools import FunctionTool, Tool
from .utils import (
    describe_data, extract_block_source, get_block_names, get_root_exception,
    wrap_logfire_on_method,
)
from .views import LumenOutput


class Task(Viewer):
    """
    A `Task` defines a single unit of work that can be executed and rendered.
    """

    abort_on_error = param.Boolean(default=False, doc="""
        If True, the report will abort if an error occurs.""")

    history = param.List(doc="""
        Conversation history to include as context for the task.""")

    instruction = param.String(default="", doc="""
        The instruction to give to the task.""")

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The chat interface to use for the task.""")

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM to use for the task.""")

    memory = param.ClassSelector(class_=_Memory, doc="""
        The memory to use for the task.""")

    outputs = param.List(doc="""
        The outputs of the task.""")

    title = param.String(doc="""
        The title of the task.""")

    running = param.Boolean(doc="""
        Whether the task is currently running.""")

    status = param.Selector(objects=["idle", "success", "error"], default="idle", doc="""
        The current status of the task.""")

    steps_layout = param.ClassSelector(default=None, class_=(ListLike, NamedListLike), allow_None=True, doc="""
        The layout progress updates will be streamed to.""")

    def __init_subclass__(cls, **kwargs):
        """
        Apply wrap_logfire to all the subclasses' execute automatically
        """
        super().__init_subclass__(**kwargs)
        wrap_logfire_on_method(cls, "execute")

    def __init__(self, **params):
        super().__init__(**params)
        self._init_view()

    def __repr__(self):
        params = []
        if self.instruction:
            params.append(f"instruction='{self.instruction}'")
        if self.title:
            params.append(f"title='{self.title}'")
        return f"{self.__class__.__name__}({', '.join(params)})"

    def _init_view(self):
        self._view = self._container = Column(sizing_mode='stretch_width', styles={'min-height': 'unset'}, height_policy='fit')

    def _populate_view(self):
        self._view[:] = []

    def reset(self):
        """Resets the view, removing generated outputs."""
        self._view[:] = []
        self.outputs.clear()

    def _render_controls(self):
        return [
            TextInput.from_param(
                self.param.title, sizing_mode="stretch_width", margin=(10, 0)
            ),
            TextAreaInput.from_param(
                self.param.instruction, sizing_mode="stretch_width", margin=(10, 0)
            ),
        ]

    def _render_output(self, out):
        if isinstance(out, str):
            return Typography(out, margin=(20, 10))
        elif isinstance(out, ChatMessage):
            return Typography(out.object, margin=(20, 10))
        elif isinstance(out, (Viewable, View)):
            return out
        elif isinstance(out, LumenOutput):
            return Tabs(
                ('Specification', out),
                ('Output', pn.param.ParamMethod(out.render, inplace=True, sizing_mode='stretch_width')),
                active=1, sizing_mode='stretch_width', min_height=0, height_policy='fit'
            )

    def editor(self, show_title: bool = True) -> Viewable:
        """
        Returns the editor for the task.

        Arguments
        ----------
        show_title: bool
            Whether to show the title of the task.

        Returns
        -------
        The editor for the task.
        """
        return Column(*self._render_controls())

    def __panel__(self):
        return self._container

    @abstractmethod
    async def _execute(self, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement the execute.")

    async def execute(self, **kwargs) -> list[Any]:
        """
        Executes the task.

        Arguments
        ----------
        **kwargs: dict
            Additional keyword arguments to pass to the task.

        Returns
        -------
        The outputs of the task.
        """
        return await self._execute(**kwargs)


class TaskGroup(Task):
    """
    A `TaskGroup` defines a collection of `Task` objects which are executed and rendered in sequence.
    """

    _tasks = param.List(item_type=(Task, Actor))

    level = 3

    __abstract = True

    def __init__(self, *tasks, **params):
        if not tasks:
            tasks = params.pop('tasks', [])
        else:
            tasks = list(tasks)
        outputs, _tasks = [], []
        for task in tasks:
            if isinstance(task, FunctionType):
                task = FunctionTool(task)
            elif isinstance(task, Task):
                outputs += task.outputs
            _tasks.append(task)
        super().__init__(_tasks=_tasks, outputs=outputs, **params)
        self._current = 0
        self._init_view()
        self._populate_view()

    def __repr__(self):
        params = []
        if self.instruction:
            params.append(f"instruction='{self.instruction}'")
        if self.title:
            params.append(f"title='{self.title}'")
        tasks = [f"\n    {task!r}" for task in self._tasks]
        return f"{self.__class__.__name__}({', '.join(params)}{''.join(tasks)})"

    def __iter__(self) -> Iterator[Task | Actor]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index) -> Task | Actor:
        return self._tasks[index]

    def _populate_view(self):
        """Populates the view on initialization or reset.

        Should either pre-populate the view with all tasks
        or act as a no-op since outputs are added progressively
        using _add_outputs.
        """

    def _add_outputs(self, i: int, task: Task | Actor, outputs: list, **kwargs):
        if isinstance(task, Task):
            self._view.append(task)
            self.outputs += task.outputs
        else:
            views = []
            for out in outputs:
                view = self._render_output(out)
                if view is not None:
                    views.append(view)
            self._view.extend(views)
            self.outputs += outputs

    def _watch_child_outputs(self, previous, event):
        pass

    def append(self, task: Task | Actor):
        """
        Appends a task to the collection.

        Arguments
        ----------
        task: Task | Actor
            The task to append.
        """
        self._tasks.append(task)
        self._populate_view()

    def insert(self, index, task: Task | Actor):
        """
        Inserts a task at the given index.

        Arguments
        ----------
        index: int
            The index to insert the task at.
        task: Task | Actor
            The task to insert.
        """
        self._tasks.insert(index, task)
        self._populate_view()

    def reset(self):
        """
        Resets the view, removing generated outputs.
        """
        self._current = 0
        self.outputs.clear()
        self._populate_view()
        for task in self._tasks:
            if isinstance(task, Task):
                task.reset()

    async def _run_task(self, i: int, task: Self | Actor, **kwargs) -> list[Any]:
        pre = 0 if self.memory is None else len(self.memory['outputs'])
        outputs = []
        memory = task.memory or self.memory
        messages = list(self.history)
        if self.instruction:
            user_msg = None
            for msg in messages[::-1]:
                if msg.get("role") == "user":
                    user_msg = msg
                    break
            if not user_msg:
                messages.append({"role": "user", "content": self.instruction})
            elif self.instruction not in user_msg.get("content"):
                user_msg["content"] = f'{user_msg["content"]}\n\nInstruction: {self.instruction}'

        with task.param.update(
            interface=self.interface,
            llm=task.llm or self.llm,
            memory=memory,
            steps_layout=self.steps_layout
        ):
            if isinstance(task, Actor):
                try:
                    out = await task.respond(messages, **kwargs)
                except Exception as e:
                    e = get_root_exception(e, exceptions=(MissingContextError,))
                    if isinstance(e, MissingContextError):
                        raise e
                    tb.print_exception(e)
                    alert = Alert(
                        f'Executing task {type(task).__name__} failed.', alert_type='error',
                        sizing_mode="stretch_width"
                    )
                    return [alert]
                # Handle Tool specific behaviors
                if isinstance(task, Tool):
                    # Handle View/Viewable results regardless of agent type
                    if isinstance(out, (View, Viewable)):
                        if isinstance(out, Viewable):
                            pipeline = None if self.memory is None else self.memory.get('pipeline')
                            out = Panel(object=out, pipeline=pipeline)
                        out = LumenOutput(
                            component=out, title=self.title
                        )
                        message_kwargs = dict(value=out, user=task.name)
                        if self.interface:
                            self.interface.stream(**message_kwargs)
                            self.memory['outputs'] = self.memory['outputs'] + [out]
                new = self.memory['outputs'][pre:]
                if not new and isinstance(out, (Viewable, View, LumenOutput)):
                    new = [out]
                outputs += new
            else:
                with task.param.update(running=True, history=messages):
                    outputs += await task.execute(**kwargs)
        return outputs

    def _render_tasks(self) -> Viewable:
        tasks = []
        for task in self._tasks:
            task_type = type(task).__name__
            if isinstance(task, Task):
                tasks.append((
                    f'{task_type}: {task.title}',
                    task.editor(show_title=False)
                ))
            elif isinstance(task, FunctionTool):
                tasks.append((f"{task_type}: {task.function.__name__}", ''))
            elif isinstance(task, Actor):
                tasks.append((task_type, self._actor_prompt(task)))
        return tasks

    def _actor_prompt(self, actor: Actor):
        prompt = Select(
            options=list(actor.prompts), label="Select prompt to modify",
            margin=(10, 0), sizing_mode="stretch_width"
        )
        template = pn.rx(actor.prompts)[prompt]['template']
        block = Select(
            options=template.rx.pipe(get_block_names), label='Select block to modify',
            margin=(10, 0, 10, 10), sizing_mode="stretch_width"
        )

        def update_prompt(prompt, block, event):
            if prompt not in actor.template_overrides:
                actor.template_overrides[prompt] = {}
            actor.template_overrides[prompt][block] = event.new

        def edit_prompt(event):
            source = extract_block_source(template.rx.value, block.value)
            editor = TextAreaInput(
                label=f"Edit {prompt.value} prompt's {block.value} block",
                value=source, sizing_mode='stretch_width', margin=(10, 0), height=300
            )
            cancel = IconButton(
                icon="cancel",
                on_click=lambda _: (layout.remove(edit_layout), actor.template_overrides.pop(prompt.value, {}).pop(block.value, None)),
                styles={'position': 'absolute', 'right': '0px', 'zIndex': "9999"}
            )
            edit_layout = Column(cancel, editor)
            editor.param.watch(partial(update_prompt, prompt.value, block.value), 'value')
            layout.append(edit_layout)

        edit = Button(
            icon="edit_note", icon_size='2em', on_click=edit_prompt, height=54,
            margin=(10, 0, 10, 10)
        )
        layout = Column(Row(prompt, block, edit))
        return layout

    def editor(self, show_title=True):
        """
        Returns the editor for the tasks.

        Arguments
        ----------
        show_title: bool
            Whether to show the title of the tasks.

        Returns
        -------
        The editor for the tasks.
        """
        return Column(
            *self._render_controls(),
            *((
                Divider(sizing_mode="stretch_width", margin=(10, 0)),
                Accordion(
                    *self._render_tasks(), margin=(10, 0, 10, 0), title_variant="h4",
                    sizing_mode="stretch_width"
                ),
              ) if self._tasks else ()
            )
        )

    async def _execute(self, **kwargs):
        """
        Executes the tasks.

        Arguments
        ---------
        **kwargs: dict
            Additional keyword arguments to pass to the tasks.
        """
        if self.memory is not None and 'outputs' not in self.memory:
            self.memory['outputs'] = []
        if self._current != 0:
            outputs = list(self.outputs)
        else:
            outputs = [Typography(f"{'#'*self.level} {self.title}", margin=(10, 10, 0, 10))] if self.title else []
            if outputs:
                self._add_outputs(-1, None, outputs, **kwargs)
        for i, task in enumerate(self._tasks):
            if i < self._current:
                continue
            new = []
            try:
                new = await self._run_task(i, task, **kwargs)
            except Exception as e:
                e = get_root_exception(e, exceptions=(MissingContextError,))
                if isinstance(e, MissingContextError):
                    # Re-raise MissingContextError to allow retry logic at Plan level
                    raise e
                tb.print_exception(e)
                self.status = "error"
                if self.abort_on_error:
                    break
            else:
                self.status = "success"
                outputs += new
            self._add_outputs(i, task, new, **kwargs)
            self._current = i + 1
        return outputs

    def to_notebook(self):
        """
        Returns the notebook representation of the tasks.
        """
        if len(self) and not len(self.outputs):
            raise RuntimeError(
                "Report has not been executed, run report before exporting to_notebook."
            )
        cells, extensions = [], ['tabulator']
        for out in self.outputs:
            ext = None
            if isinstance(out, Typography):
                level = int(out.variant[1:]) if out.variant and out.variant.startswith('h') else 0
                prefix = f"{'#'*level} " if level else ''
                cell = make_md_cell(f"{prefix}{out.object}")
            elif isinstance(out, Markdown):
                cell = make_md_cell(out.object)
            elif isinstance(out, LumenOutput):
                cell, ext = format_output(out)
            elif isinstance(out, Viewable):
                cell, ext = format_output(Panel(out))
            cells.append(cell)
            if ext and ext not in extensions:
                extensions.append(ext)
        cells = make_preamble("", extensions=extensions) + cells
        return write_notebook(cells)


class Section(TaskGroup):
    """
    A `Section` is a `TaskGroup` representing a sequence of related tasks.
    """

    level = 2

    def __init__(self, *tasks, **params):
        self._watchers = {}
        super().__init__(*tasks, **params)

    def __repr__(self) -> str:
        params = []
        if self.title:
            params.append(f"title='{self.title}'")
        tasks = [f"\n    {task!r}" for task in self._tasks]
        return f"{self.__class__.__name__}({', '.join(params)}{''.join(tasks)})"

    def _add_outputs(self, i: int, task: Task | Actor, outputs: list, **kwargs):
        self.outputs += outputs

    def _render_controls(self):
        return [
            TextInput.from_param(
                self.param.title, sizing_mode="stretch_width", margin=(10, 0)
            )
        ]

    def _init_view(self):
        self._dialog = Dialog(
            self.editor(),
            show_close_button=True,
            close_on_click=True,
            title=f'Section Configuration: {self.title}',
        )
        self._settings = IconButton(
            icon='settings', on_click=self._open_settings, size="small", color="default"
        )
        self._view = Column(
            sizing_mode='stretch_width', styles={'min-height': 'unset'}, height_policy='fit'
        )
        self._container = Column(
            Row(
                self._settings,
                self._dialog,
                styles={'position': 'absolute', 'top': '-57.5px', 'right': '20px'}
            ),
            self._view,
            sizing_mode='stretch_width',
            styles={'min-height': 'unset'},
            height_policy='fit'
        )

    def _populate_view(self):
        self._view[:] = self._tasks

    def _open_settings(self, event):
        self._dialog.open = True

    def _watch_child_outputs(self, i: int, previous: list, event: param.Event, **kwargs):
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

    async def _rerun(self, i: int, context: dict, _: param.Event, **kwargs):
        for task in self._tasks[i:]:
            task.reset()
        with self.param.update(running=True):
            for j, task in enumerate(self._tasks[i:]):
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

    async def _run_task(self, i: int, task: Task | Actor, **kwargs) -> list[Any]:
        if self.memory:
            self.memory['outputs'] = []
            instructions = "\n".join(f"{i+1}. {task.instruction}" if hasattr(task, 'instruction') else f"{i+1}. <no instruction>" for i, task in enumerate(self._tasks))
            self.memory['reasoning'] = f"{self.title}\n\n{instructions}"
        if isinstance(task, Task):
            watcher = task.param.watch(partial(self._watch_child_outputs, i, list(self.outputs), **kwargs), 'outputs')
        try:
            outputs = await super()._run_task(i, task, **kwargs)
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


class Report(TaskGroup):
    """
    A `Report` is a `TaskGroup` consisting of a sequence of `Section` objects.

    The `Report` UI renders buttons to execute, clear, export and configure the
    Report.
    """

    _tasks = param.List(item_type=Section)

    level = 1

    def _init_view(self):
        self._title = Typography(
            self.param.title, variant="h1", margin=(0, 10, 0, 10)
        )
        self._view = Accordion(
            sizing_mode="stretch_width", min_height=0, margin=(0, 5, 5, 5),
            sx={"& .MuiAccordionDetails-root": {"p": "0 calc(2 * var(--mui-spacing)) 1em !important"}}
        )
        self._run = IconButton(
            icon="play_arrow", on_click=self._execute, margin=0, size="large",
            description="Execute Report"
        )
        self._clear = IconButton(
            icon="clear", on_click=lambda _: self.reset(), margin=0, size="large",
            description="Clear outputs"
        )
        self._collapse = IconButton(
            styles={"margin-left": "auto"}, on_click=self._expand_all, icon="unfold_less",
            size="large", color="default", margin=(0, 0, 10, 0), description="Collapse/Expand Sections"
        )
        self._settings = IconButton(
            icon="settings", on_click=self._open_settings, size="large", color="default",
            margin=0, description="Configure Report"
        )
        self._export = FileDownload(
            callback=self._notebook_export, label="\u200b", variant='text', icon='get_app',
            icon_size="2.4em", color="default", margin=(8, 0, 10, 0),
            sx={".MuiButton-startIcon": {"mr": 0, "color": "var(--mui-palette-default-dark)"}},
            description="Export Report to .ipynb", filename=f"{self.title or 'Report'}.ipynb"
        )
        self._dialog = Dialog(
            TextInput.from_param(self.param.title, margin=(10, 0, 0, 0), sizing_mode="stretch_width"),
            show_close_button=True,
            close_on_click=True,
            title=f"Report Settings: {self.title}",
        )
        self._menu = Row(
            self._title,
            self._run,
            self._clear,
            self._collapse,
            self._export,
            self._settings,
            sizing_mode="stretch_width"
        )
        self._container = Column(
            self._view,
            self._dialog,
            margin=(0, 0, 0, 5),
            sizing_mode="stretch_both"
        )

    @param.depends('title', watch=True)
    def _update_filename(self):
        self._export.filename = f"{self.title or 'Report'}.ipynb"

    def _add_outputs(self, i: int, task: Task | Actor, outputs: list, **kwargs):
        self.outputs += outputs

    def _notebook_export(self):
        return io.StringIO(self.to_notebook())

    async def _execute(self, *args):
        with self._run.param.update(loading=True):
            return await super()._execute()

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
        self._view[:] = objects = [(task.title, task) for task in self._tasks]
        self._view.active = list(range(len(objects)))

    async def _run_task(self, i: int, task: Section, **kwargs):
        self._view.active = self._view.active + [i]
        watcher = task.param.watch(partial(self._watch_child_outputs, self.outputs), "outputs")
        try:
            outputs = await super()._run_task(i, task, **kwargs)
        finally:
            task.param.unwatch(watcher)
        return outputs

    def __panel__(self):
        return Column(
            self._menu,
            Container(
                self._container, sizing_mode="stretch_both", height_policy="max",
                stylesheets=[":host > div { overflow-y: auto; }"], min_height=600
            )
        )


class Action(Task):
    """
    An `Action` implements an execute method that performs some unit of work
    and optionally generates outputs to be rendered.
    """

    render_outputs = param.Boolean(default=True, doc="""
         Whether the outputs should be rendered.""")

    @final
    async def execute(self, **kwargs):
        outputs = await super().execute(**kwargs)
        if self.render_outputs:
            self._view[:] = [self._render_output(out) for out in outputs]
        self.outputs += outputs
        return outputs


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

    async def _execute(self, **kwargs):
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
            if self.memory is None or 'source' not in self.memory:
                raise ValueError(
                    "SQLQuery could not resolve a source. Either provide "
                    "an explicit source or ensure another action or actor "
                    "provides a source."
                )
            source = self.memory['source']
        if not self.table:
            raise ValueError("SQLQuery must declare a table name.")

        # Pass table_params if provided
        params = {self.table: self.table_params} if self.table_params else None
        source = source.create_sql_expr_source({self.table: self.sql_expr}, params=params)
        pipeline = Pipeline(source=source, table=self.table)
        if self.memory is not None:
            self.memory["source"] = source
            if "sources" not in self.memory:
                self.memory["sources"] = []
            self.memory["sources"].append(source)
            self.memory["pipeline"] = pipeline
            self.memory["data"] = await describe_data(pipeline.data)
            self.memory["sql_metaset"] = await get_metaset([source], [self.table])
            self.memory["table"] = self.table
        out = LumenOutput(component=pipeline)
        outputs = [Typography(f"### {self.title}", variant='h4', margin=(10, 10, 0, 10)), out] if self.title else [out]
        if self.generate_caption:
            caption = await AnalystAgent(llm=self.llm).respond(
                [{"role": "user", "content": self.user_content}]
            )
            outputs.append(Typography(caption.object))
        return outputs
