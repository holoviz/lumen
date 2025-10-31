from __future__ import annotations

import asyncio
import io
import traceback as tb

from abc import abstractmethod
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType
from typing import Any, TypedDict, final

import panel as pn
import param

from panel.io import hold
from panel.layout.base import (
    Column, ListLike, NamedListLike, Row,
)
from panel.pane import Markdown
from panel.viewable import Viewable, Viewer
from panel_material_ui import (
    Accordion, Button, ChatFeed, ChatMessage, Container, Dialog, Divider,
    FileDownload, IconButton, Progress, Select, TextAreaInput, TextInput,
    Typography,
)
from typing_extensions import Self

from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource
from ..views.base import Panel, View
from .actor import Actor, ContextProvider, TContext
from .agents import AnalystAgent, LumenBaseAgent, VegaLiteAgent
from .config import MissingContextError
from .context import (
    LWW, ContextError, ValidationIssue, collect_task_outputs,
    input_dependency_keys, merge_contexts, validate_task_inputs,
    validate_taskgroup_exclusions,
)
from .controls import AnnotationControls, RetryControls
from .export import (
    format_output, make_md_cell, make_preamble, write_notebook,
)
from .llm import Llm
from .schemas import SQLMetaset, get_metaset
from .tools import FunctionTool, Tool
from .utils import (
    describe_data, extract_block_source, get_block_names,
    wrap_logfire_on_method,
)
from .views import LumenOutput, SQLOutput, VegaLiteOutput


class Task(Viewer):
    """
    A `Task` defines a single unit of work that can be executed and rendered.
    """

    abort_on_error = param.Boolean(default=True, doc="""
        If True, the report will abort if an error occurs.""")

    context = param.Dict()

    history = param.List(doc="""
        Conversation history to include as context for the task.""")

    instruction = param.String(default="", doc="""
        The instruction to give to the task.""")

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The chat interface to use for the task.""")

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM to use for the task.""")

    parent = param.Parameter()

    running = param.Boolean(doc="""
        Whether the task is currently running.""")

    status = param.Selector(objects=["idle", "success", "error"], default="idle", doc="""
        The current status of the task.""")

    steps_layout = param.ClassSelector(default=None, class_=(ListLike, NamedListLike), allow_None=True, doc="""
        The layout progress updates will be streamed to.""")

    title = param.String(doc="""
        The title of the task.""")

    views = param.List(doc="""
        The generated viewable outputs of the task.""")

    def __init_subclass__(cls, **kwargs):
        """
        Apply wrap_logfire to all the subclasses' execute automatically
        """
        super().__init_subclass__(**kwargs)
        wrap_logfire_on_method(cls, "execute")

    def __init__(self, **params):
        super().__init__(**params)
        self._prepared = False
        self._init_view()

    def __repr__(self):
        params = []
        if self.instruction:
            params.append(f"instruction='{self.instruction}'")
        if self.title:
            params.append(f"title='{self.title}'")
        return f"{self.__class__.__name__}({', '.join(params)})"

    def _init_view(self):
        self._view = self._container = Column(
            sizing_mode='stretch_width', styles={'min-height': 'unset'}, height_policy='fit'
        )

    def _populate_view(self):
        self._view[:] = []

    def reset(self):
        """Resets the view, removing generated outputs."""
        self._view[:] = []
        self.views.clear()

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
        elif isinstance(out, (Viewable, View, LumenOutput)):
            return out

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
    async def _execute(self, context: TContext, **kwargs) -> tuple[list[Any], TContext]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement the execute.")

    async def prepare(self, context: TContext | None = None):
        self._prepared = True

    async def execute(self, context: TContext | None = None, **kwargs) -> tuple[list[Any], TContext]:
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
        context = dict(self.context or {}, **(context or {}))
        if not self._prepared:
            await self.prepare(context)
        return await self._execute(context, **kwargs)


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
        views, _tasks, _contexts = [], [], []
        for task in tasks:
            if isinstance(task, FunctionType):
                task = FunctionTool(task)
            elif isinstance(task, Task):
                views += task.views
            task.parent = self
            _tasks.append(task)
            _contexts.append({})
        super().__init__(_tasks=_tasks, views=views, **params)
        self._current = 0
        self._watchers = {}
        self._task_outputs = {}
        self._task_contexts = {}
        self._task_rendered = {}
        self._init_view()
        self._populate_view()

    def validate(
        self, context: TContext | None = None,
        available_types: dict[str, Any] | None = None,
        path: str | None = None,
        raise_on_error: bool = True
    ):
        """
        Validate the task group and its subtasks.

        Parameters
        ----------
        context : TContext | None, optional
            The context to validate against, by default None
        available_types : dict[str, Any] | None, optional
            Dictionary of available types for validation, by default None
        path : str | None, optional
            Path to the current task group for error reporting, by default None
        raise_on_error : bool, optional
            Whether to raise an error if validation issues are found, by default True

        Returns
        -------
        tuple[list[ValidationIssue], dict[str, Any]]
            A tuple containing:
            - List of validation issues found
            - Dictionary of output types from all tasks
        """
        cur_path = path or self.name
        issues: list[ValidationIssue] = validate_taskgroup_exclusions(self, path=cur_path)
        value_ctx = dict((self.context or {}), **(context or {}))
        types_out: dict[str, Any] = dict(available_types or {})
        for idx, t in enumerate(self):
            subpath = f"{cur_path}[{idx}] -> {t.name}"
            if isinstance(t, TaskGroup):
                sub_issues, sub_types = t.validate(
                    value_ctx,
                    available_types=types_out,
                    path=subpath,
                )
            else:
                sub_issues = validate_task_inputs(
                    t, value_ctx, types_out, subpath
                )
                sub_types = collect_task_outputs(t)
            issues.extend(sub_issues)
            types_out.update(sub_types)
        if path is None and issues and raise_on_error:
            raise ContextError(issues)
        return issues, types_out

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

    async def _retry_invoke(
        self, i: int, task: Task | Actor, context: TContext, view: LumenOutput,
        config: dict[str, Any], event: param.parameterized.Event
    ):
        invalidation_keys = set(task.output_schema.__annotations__)
        self.invalidate(invalidation_keys, start=i+1)
        if isinstance(task, LumenBaseAgent):
            with view.editor.param.update(loading=True):
                messages = list(self.history)
                task_context = self._get_context(i, context, task)
                view.spec = await task.revise(
                    event.new, messages, task_context, view
                )
        root = self
        while root.parent is not None:
            root = root.parent
        with root.param.update(config):
            await root.execute()

    async def _annotate_invoke(
        self, i: int, task: Task | Actor, context: TContext, view: VegaLiteOutput,
        config: dict[str, Any], event: param.parameterized.Event
    ):
        if isinstance(task, VegaLiteAgent):
            with view.editor.param.update(loading=True):
                messages = list(self.history)
                task_context = self._get_context(i, context, task)
                # Call the annotate method with the current spec dict
                view.spec = await task.annotate(
                    event.new, messages, task_context, view
                )
        root = self
        while root.parent is not None:
            root = root.parent
        with root.param.update(config):
            await root.execute()

    def _add_outputs(
        self, i: int, task: Task | Actor, views: list, context: TContext, out_context: TContext | None, **kwargs
    ):
        # Attach retry and annotation controls
        for view in views:
            if not isinstance(view, LumenOutput):
                continue
            retry_controls = RetryControls()
            view.footer = [retry_controls]
            self._watchers[i] = retry_controls.param.watch(
                partial(self._retry_invoke, i, task, context, view, {'interface': self.interface}),
                "instruction"
            )

            # Add annotation controls for VegaLite views
            if isinstance(view, VegaLiteOutput):
                annotation_controls = AnnotationControls()
                view.footer = [*view.footer, annotation_controls]
                self._watchers[f"{i}_annotate"] = annotation_controls.param.watch(
                    partial(self._annotate_invoke, i, task, context, view, {'interface': self.interface}),
                    "instruction"
                )

        # Track context and outputs
        if i >= 0:
            self._task_contexts[task] = out_context
        self._task_outputs[task] = list(views)

        # Find view and output to insert the new outputs after
        if i >= 0:
            prev_task = None
            for j in reversed(range(i)):
                prev_task = self._tasks[j]
                if prev_task in self._task_rendered:
                    prev_out = self._task_rendered.get(prev_task)
                    break
            else:
                prev_out = self._task_rendered.get(None)
            for j in reversed(range(i)):
                prev_task = self._tasks[j]
                if isinstance(prev_task, Task) and prev_task.views:
                    prev_view = prev_task.views[-1]
                    break
                elif self._task_outputs.get(prev_task):
                    prev_view = self._task_outputs[prev_task][-1]
                    break
            else:
                task_out = self._task_outputs.get(None)
                prev_view = task_out[-1] if task_out else None
        else:
            prev_view = prev_out = None

        idx = 0 if prev_out is None else (self._view.index(prev_out) + 1)
        if isinstance(task, Task):
            self._view.insert(idx, task)
            self._task_rendered[task] = self._view[idx]
            views = task.views
        else:
            rendered = []
            for out in views:
                view = self._render_output(out)
                if view is not None:
                    rendered.append(view)
            if rendered:
                if len(rendered) > 1:
                    rendered_out = Column(*rendered)
                else:
                    rendered_out = rendered[0]
                self._task_rendered[task] = rendered_out
                self._view.insert(idx, rendered_out)
        new_views = list(self.views)
        view_idx = 0 if prev_view is None else new_views.index(prev_view)+1
        for vi, view in enumerate(views):
            new_views.insert(view_idx+vi, view)
        self.views = new_views

    def _watch_child_outputs(
        self, i: int, previous: list, context: TContext, event: param.Event, **kwargs
    ):
        pass

    def append(self, task: Task | Actor):
        """
        Appends a task to the collection.

        Arguments
        ----------
        task: Task | Actor
            The task to append.
        """
        if isinstance(task, Task):
            task.parent = self
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
        if isinstance(task, Task):
            task.parent = self
        self._tasks.insert(index, task)
        self._populate_view()

    def merge(self, other: TaskGroup):
        """
        Merges another task group into the current task group.

        Parameters
        ---------
        other: TaskGroup
            The other task group to merge with.

        Returns
        -------
        self: TaskGroup
            The current task group.
        """
        for task in other:
            other.parent = self
            self._tasks.append(task)
        self._task_contexts.update(other._task_contexts)
        self._task_rendered.update(other._task_rendered)
        self._task_outputs.update(other._task_outputs)
        self._view[:] = list(self._view) + list(other._view)
        self.views = self.views + other.views
        return self

    async def prepare(self, context: TContext | None = None):
        context = context or (self.context or {})
        for task in self._tasks:
            await task.prepare(context)
        self._prepared = True

    def reset(self):
        """
        Resets the view, removing generated outputs.
        """
        self._current = 0
        with hold():
            self._task_rendered.clear()
            self._task_contexts.clear()
            self._task_outputs.clear()
            self.views.clear()
            self._view.clear()
            for task in self._tasks:
                if isinstance(task, Task):
                    task.reset()
            self._populate_view()

    async def _run_task(
        self, i: int, task: Self | Actor, context: TContext, **kwargs
    ) -> tuple[list[Any], TContext]:
        outputs = []
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

        subcontext = self._get_context(i, context, task)
        with task.param.update(
            interface=self.interface,
            llm=task.llm or self.llm,
            steps_layout=self.steps_layout
        ):
            if isinstance(task, Actor):
                try:
                    out, out_context = await task.respond(messages, subcontext, **kwargs)
                except MissingContextError:
                    # Re-raise MissingContextError to allow retry logic at Plan level
                    raise
                except Exception as e:
                    raise e
                # Handle Tool specific behaviors
                if isinstance(task, Tool):
                    # Handle View/Viewable results regardless of agent type
                    rendered = []
                    for o in out:
                        if not isinstance(o, (View, Viewable)):
                            continue
                        if isinstance(o, Viewable):
                            pipeline = None if context is None else context.get('pipeline')
                            o = Panel(object=o, pipeline=pipeline)
                        o = LumenOutput(
                            component=o, title=self.title
                        )
                        message_kwargs = dict(value=o, user=task.name)
                        if self.interface:
                            self.interface.stream(**message_kwargs)
                        rendered.append(o)
                    out = rendered
            else:
                with task.param.update(running=True, history=messages):
                    out, out_context = await task.execute(subcontext, **kwargs)
            self._task_outputs[task] = out
            outputs += out
        if isinstance(task, (Actor, Action)):
            unprovided = [
                p for p in task.output_schema.__required_keys__
                if p not in out_context
            ]
            if unprovided:
                raise RuntimeError(f"{task.__class.__name__} failed to provide declared context.")
        return outputs, out_context

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

    def _get_context(self, i: int, context: TContext | None, task: Task | Actor) -> TContext | None:
        contexts = ([self.context] if self.context else []) + ([context] if context else [])
        contexts += [self._task_contexts[task] for task in self._tasks[:i]]
        if isinstance(task, Actor):
            return merge_contexts(task.input_schema, contexts)
        elif isinstance(task, TaskGroup):
            subcontexts = []
            for subtask in task:
                subtask_context = self._get_context(i, context, subtask)
                if subtask_context is not None:
                    subcontexts.append(subtask_context)
            return merge_contexts(LWW, subcontexts)
        elif isinstance(task, Action):
            return merge_contexts(task.input_schema, contexts)
        else:
            raise TypeError("Abstract Task does not implement _get_context.")

    async def _execute(self, context: TContext, **kwargs):
        """
        Executes the tasks.

        Arguments
        ---------
        context: TContext
            The context given to the task
        **kwargs: dict
            Additional keyword arguments to pass to the tasks.
        """
        if None in self._task_outputs:
            views = self._task_outputs[None]
        else:
            title = Typography(f"{'#'*self.level} {self.title}", margin=(10, 10, 0, 10))
            views = [title] if self.title else []
            if views:
                self._add_outputs(-1, None, views, context, None, **kwargs)
        for i, task in enumerate(self._tasks):
            self._current = i
            if task in self._task_outputs:
                views += self._task_outputs[task]
                continue
            new = []
            try:
                new, new_context = await self._run_task(i, task, context, **kwargs)
            except MissingContextError:
                # Re-raise MissingContextError to allow retry logic at Plan level
                raise
            except Exception as e:
                tb.print_exception(e)
                self.status = "error"
                new_context = {"__error__": str(e)}
                self._task_contexts[task] = new_context
                if self.abort_on_error:
                    if self.parent is not None:
                        raise e
                    break
                else:
                    continue
            else:
                status = task.status if isinstance(task, Task) and task.status != "success" else "success"
                self.status = status
                views += new
            self._add_outputs(
                i, task, new, context, new_context, **kwargs
            )
        if self.status != "error":
            self._current += 1
        contexts = ([self.context] if self.context else [])
        contexts += [self._task_contexts[task] for task in self if task in self._task_contexts]
        return views, merge_contexts(LWW, contexts)

    def invalidate(
        self, keys: Iterable[str], start: int = 0, propagate: bool = True
    ) -> tuple[bool, set[str]]:
        """
        Invalidates tasks and propagates context dependencies within a TaskGroup.

        Parameters
        ----------
        keys : Iterable[str]
            A set of context keys that have been modified or invalidated by the user.
            These represent outputs whose dependent tasks should be re-evaluated.
        start : int
            Index of the task to start invalidating.
        propagate: bool
            Whether to propagate the invalidation to the parent task group.

        Returns
        -------
        tuple[bool, set[str]]
            A tuple ``(invalidated, keys)`` where:
            - ``invalidated`` is ``True`` if any tasks in this group (or nested groups)
                were marked invalid due to overlapping input dependencies.
            - ``keys`` is the full, cumulative set of invalidated context keys after
                propagating dependencies through all affected tasks.

        Notes
        -----
        - If a task's input keys intersect with the provided invalidation keys,
        that task is considered stale and will be rerun.
        - When a task is invalidated, its recorded outputs are removed from
        ``self._task_outputs`` and its output keys are added to the invalidation set,
        ensuring that downstream tasks depending on those outputs are also invalidated.
        - Nested TaskGroups are traversed recursively, and their invalidations
        propagate upward to the parent group.
        """
        if propagate and self.parent is not None and self in self.parent:
            parent_idx = self.parent._tasks.index(self)
            if start >= len(self):
                # If the last task is being invalidated,
                # only subsequent tasks on the parent
                # have to be invalidated
                parent_idx += 1
            self.parent.invalidate(keys, start=parent_idx)
            return
        keys = set(keys)
        invalidated = False
        views = list(self.views)
        rendered_views = list(self._view)
        for i, task in enumerate(self):
            if i < start:
                continue
            if isinstance(task, ContextProvider):
                deps = input_dependency_keys(task.input_schema)
                if not (deps & keys):
                    continue
                invalidated = True
                self._task_contexts.pop(task, None)
                outputs = self._task_outputs.pop(task, [])
                rendered = self._task_rendered.pop(task, None)
                keys |= set(task.output_schema.__annotations__)
                if rendered is not None:
                    rendered_views.remove(rendered)
                views = [view for view in views if view not in outputs]
                if isinstance(task, Task):
                    task.reset()
            else:
                old = list(task.views)
                subtask_invalidated, subtask_keys = task.invalidate(keys, propagate=not propagate)
                new = list(task.views)
                invalidated = invalidated or subtask_invalidated
                keys |= subtask_keys
                views = [view for view in views if not (view in old and view not in new)]
        self.views = views
        self._view[:] = rendered_views
        return invalidated, keys

    def to_notebook(self):
        """
        Returns the notebook representation of the tasks.
        """
        if len(self) and not len(self._task_rendered):
            raise RuntimeError(
                "Report has not been executed, run report before exporting to_notebook."
            )
        cells, extensions = [], ['tabulator']
        for out in self.views:
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

    def _add_outputs(self, i: int, task: Task | Actor, views: list, context: TContext, out_context: TContext | None, **kwargs):
        if out_context is not None:
            self._task_contexts[task] = out_context
            self._task_outputs[task] = views
            self._task_rendered[task] = task
        self.views = self.views + views

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

    async def _run_task(self, i: int, task: Task | Actor, context: TContext | None, **kwargs) -> list[Any]:
        if context is not None:
            instructions = "\n".join(
                f"{i+1}. {task.instruction}" if hasattr(task, 'instruction') else f"{i+1}. <no instruction>"
                for i, task in enumerate(self._tasks)
            )
            context['reasoning'] = f"{self.title}\n\n{instructions}"
        return await super()._run_task(i, task, context, **kwargs)

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

    def __init__(self, *tasks, **params):
        if not tasks:
            tasks = params.pop('tasks', [])
        else:
            tasks = list(tasks)
        super().__init__(*tasks, **params)
        pn.state.execute(self.prepare)

    def _init_view(self):
        self._title = Typography(
            self.param.title, variant="h1", margin=(0, 10, 0, 10)
        )
        self._view = Accordion(
            sizing_mode="stretch_width", min_height=0, margin=(0, 5, 5, 5),
            sx={"& .MuiAccordionDetails-root": {"p": "0 calc(2 * var(--mui-spacing)) 1em !important"}}
        )
        self._run = IconButton(
            icon="play_arrow", on_click=self._execute_event, margin=0, size="large",
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

    async def _execute_event(self, event):
        await self.execute()

    @param.depends('title', watch=True)
    def _update_filename(self):
        self._export.filename = f"{self.title or 'Report'}.ipynb"

    def _add_outputs(self, i: int, task: Task | Actor, views: list, context: TContext, out_context: dict | None, **kwargs):
        if out_context is not None:
            self._task_contexts[task] = out_context
            self._task_outputs[task] = views
            self._task_rendered[task] = task
        self.views = self.views + views

    def _notebook_export(self):
        return io.StringIO(self.to_notebook())

    async def _execute(self, context: TContext, *args):
        with self._run.param.update(loading=True):
            return await super()._execute(context)

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

    async def _run_task(self, i: int, task: Section, context: TContext, **kwargs):
        self._view.active = self._view.active + [i]
        watcher = task.param.watch(partial(self._watch_child_outputs, i, self.views, dict(context)), "views")
        try:
            outputs = await super()._run_task(i, task, context, **kwargs)
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


class Action(Task, ContextProvider):
    """
    An `Action` implements an execute method that performs some unit of work
    and optionally generates outputs to be rendered.
    """

    render_outputs = param.Boolean(default=True, doc="""
         Whether the outputs should be rendered.""")

    @final
    async def execute(self, context: TContext | None = None, **kwargs):
        views, out_context = await super().execute(context, **kwargs)
        if self.render_outputs:
            self._view[:] = [self._render_output(out) for out in views]
        self.views = self.views + views
        return views, out_context


class SQLQueryInputs(TypedDict):

    source: BaseSQLSource


class SQLQueryOutputs(TypedDict):
    source: BaseSQLSource
    pipeline: Pipeline
    data: dict
    sql_metaset: SQLMetaset
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
        source = source.create_sql_expr_source({self.table: self.sql_expr})
        pipeline = Pipeline(source=source, table=self.table)
        out_context = {
            "source": source,
            "pipeline": pipeline,
            "data": await describe_data(pipeline.data),
            "sql_metaset": await get_metaset([source], [self.table]),
            "table": self.table,
        }
        out = SQLOutput(component=pipeline, spec=self.sql_expr)
        title = Typography(f"### {self.title}", variant='h4', margin=(10, 10, 0, 10))
        outputs = [title, out] if self.title else [out]
        if self.generate_caption:
            caption_out, _ = await AnalystAgent(llm=self.llm).respond(
                [{"role": "user", "content": self.user_content}], context
            )
            caption = caption_out[0]
            outputs.append(Typography(caption.object))
        return outputs, out_context
