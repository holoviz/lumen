"""What FlowDash does around each of its own tool calls, done around Lumen's tools too."""
from __future__ import annotations

import dataclasses
import logging

from collections.abc import Collection
from typing import Any

from panel_agents import current_run_id
from panel_flowdash import agent_edits
from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn

logger = logging.getLogger(__name__)


class FlowDashToolCalls(AbstractCapability[FlowDashDeps]):
    """
    Runs Lumen's tools as panel-flowdash runs its dashboard tools, through
    the hooks ``FlowDashDeps`` and the operations offer:

    - a request whose user moved to another dashboard is told to stop
      (``check_pin``), and the tools in ``editing`` wait for another
      request's edits of the dashboard to end (``wait_to_edit``);
    - the edits count as the agent's, in the turn of its run
      (``agent_edits``), and the edit history is flushed after each call;
    - operations that change more than asked, e.g. ``ExplorationOps``
      copying an earlier answer's node to change it, take that back when the
      call fails (``abort_call``) and report it when it succeeds
      (``take_notes``);
    - the edit listeners hear about each call of an editing tool;
    - a refusal of the operations, a ``PermissionError``, comes back for the
      model to work around rather than ending the run.

    Parameters
    ----------
    tools
        The names of the tools to run this way; FlowDash's own tools already are.
    editing
        Those of them that change the dashboard.
    """

    def __init__(self, tools: Collection[str], editing: Collection[str]):
        self.tools = frozenset(tools)
        self.editing = frozenset(editing)

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        if tool_def.name not in self.tools:
            return await handler(args)
        deps = ctx.deps
        if (refusal := deps.check_pin(ctx.run_id)) is not None:
            return refusal
        run = current_run_id() or ctx.run_id
        editing = tool_def.name in self.editing
        if editing:
            refusal = await deps.wait_to_edit(run)
            if refusal is None:
                refusal = deps.check_pin(ctx.run_id)
            if refusal is not None:
                return refusal
        ops = deps.ops
        begin, abort = getattr(ops, "begin_call", None), getattr(ops, "abort_call", None)
        take_notes = getattr(ops, "take_notes", None)
        if begin is not None:
            begin()
        listen = editing and bool(deps.edit_listeners)
        before = _snapshot(ops) if listen else None
        try:
            with agent_edits(run, deps.question):
                try:
                    result = await handler(args)
                except BaseException:
                    if abort is not None:
                        abort()
                    raise
        except PermissionError as exc:
            raise ModelRetry(f"Refused: {exc}") from exc
        except OperationError as exc:
            raise ModelRetry(str(exc)) from exc
        finally:
            ops.editor.edit_history.flush()
        notes = take_notes() if take_notes is not None else None
        if notes:
            result = _noted(result, notes)
        if listen:
            after = _snapshot(ops)
            for listener in list(deps.edit_listeners):
                try:
                    listener(before, after)
                except Exception:
                    logger.exception("An edit listener failed")
        return result


def _noted(result: Any, notes: str) -> Any:
    if isinstance(result, str):
        return f"{result} {notes}"
    if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
        return dataclasses.replace(result, return_value=f"{result.return_value} {notes}")
    return result


def _snapshot(ops: Any) -> str | None:
    try:
        return ops.get_dashboard()
    except (OperationError, PermissionError):
        return None
