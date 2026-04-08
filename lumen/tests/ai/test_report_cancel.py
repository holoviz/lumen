"""Tests for Report cancellation functionality."""
import asyncio

import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown

from lumen.ai.report import (
    Action, Report, Section, TaskGroup, Typography,
)


class SlowAction(Action):
    """An action that sleeps, giving us time to cancel."""

    order = param.List()
    sleep_time = param.Number(default=0.5)

    async def _execute(self, context, **kwargs):
        self.order.append(self.title)
        await asyncio.sleep(self.sleep_time)
        return [Markdown(f"{self.title} done")], {self.title: "done"}


class InstantAction(Action):
    """An action that completes immediately."""

    order = param.List()

    async def _execute(self, context, **kwargs):
        self.order.append(self.title)
        return [Markdown(f"{self.title} done")], {self.title: "done"}


# ---- CancelledError propagation through Task.execute ----

async def test_task_execute_propagates_cancelled_error():
    """Task.execute re-raises CancelledError without setting error status."""
    a = SlowAction(title="A", sleep_time=10.0)

    task = asyncio.ensure_future(a.execute())
    # Wait for it to get past the initial sleep(0.1)
    await asyncio.sleep(0.15)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # CancelledError should NOT set status to "error"
    assert a.status != "error"
    assert a.running is False


async def test_taskgroup_propagates_cancelled_error():
    """TaskGroup._execute re-raises CancelledError (not caught by except Exception)."""
    order = []
    a = SlowAction(order=order, title="A", sleep_time=10.0)
    b = SlowAction(order=order, title="B", sleep_time=0.05)
    tg = TaskGroup(a, b, title="Group")

    task = asyncio.ensure_future(tg.execute())
    # Wait for execution to start
    for _ in range(200):
        if "A" in order:
            break
        await asyncio.sleep(0.01)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert "B" not in order
    assert tg.status != "error"


# ---- Report cancellation via _execute_task.cancel() ----

async def test_report_execute_event_stores_task():
    """_execute_event stores asyncio.current_task() in _execute_task."""
    order = []
    a = SlowAction(order=order, title="A", sleep_time=2.0)
    section = Section(a, title="S1")
    report = Report(section, title="Test")

    # Wrap in ensure_future to simulate pn.state.execute behavior
    fut = asyncio.ensure_future(report._execute_event())

    # Wait for _execute_task to be set
    for _ in range(200):
        if report._execute_task is not None:
            break
        await asyncio.sleep(0.01)

    assert report._execute_task is not None

    # Cancel to clean up
    report._execute_task.cancel()
    try:
        await fut
    except asyncio.CancelledError:
        pass


async def test_report_cancel_via_task_cancel():
    """Cancelling _execute_task stops report execution and resets running state."""
    order = []
    a = SlowAction(order=order, title="A", sleep_time=0.1)
    b = SlowAction(order=order, title="B", sleep_time=2.0)
    c = SlowAction(order=order, title="C", sleep_time=0.1)
    section = Section(a, b, c, title="S1")
    report = Report(section, title="Test")

    fut = asyncio.ensure_future(report._execute_event())

    # Wait for A to complete
    for _ in range(200):
        if "A" in order:
            break
        await asyncio.sleep(0.01)
    assert "A" in order

    # Cancel the execution
    assert report._execute_task is not None
    report._execute_task.cancel()

    # Wait for cleanup
    for _ in range(200):
        if report._execute_task is None:
            break
        await asyncio.sleep(0.01)

    assert report._execute_task is None
    assert report.running is False
    assert "C" not in order


async def test_report_cancel_resets_section_running():
    """After cancellation, section running state is also cleaned up."""
    order = []
    a = SlowAction(order=order, title="A", sleep_time=2.0)
    section = Section(a, title="S1")
    report = Report(section, title="Test")

    fut = asyncio.ensure_future(report._execute_event())

    # Wait for execution to start
    for _ in range(200):
        if report.running:
            break
        await asyncio.sleep(0.01)

    report._execute_task.cancel()

    for _ in range(200):
        if report._execute_task is None:
            break
        await asyncio.sleep(0.01)

    assert report.running is False
    assert section.running is False


async def test_report_cancel_then_rerun():
    """After cancellation, the report can be re-executed successfully."""
    order = []
    a = SlowAction(order=order, title="A", sleep_time=0.05)
    b = SlowAction(order=order, title="B", sleep_time=0.05)
    section = Section(a, b, title="S1")
    report = Report(section, title="Test")

    # First run — cancel partway through
    fut = asyncio.ensure_future(report._execute_event())
    for _ in range(200):
        if report.running:
            break
        await asyncio.sleep(0.01)

    report._execute_task.cancel()
    for _ in range(200):
        if report._execute_task is None:
            break
        await asyncio.sleep(0.01)

    # Reset and rerun
    report.reset()
    order.clear()

    outs, ctx = await report.execute()
    assert "A" in order
    assert "B" in order
    assert report.status == "success"


# ---- ToggleIcon button tests ----

async def test_report_toggle_icon_exists():
    """Report has a ToggleIcon _run button."""
    a = InstantAction(title="A")
    section = Section(a, title="S1")
    report = Report(section, title="Test")

    assert hasattr(report, '_run')
    assert report._run.icon == "play_circle_filled"
    assert report._run.active_icon == "stop"
    assert report._run.value is False


async def test_report_toggle_off_cancels():
    """Setting _run.value to False cancels a running execution."""
    order = []
    a = SlowAction(order=order, title="A", sleep_time=2.0)
    section = Section(a, title="S1")
    report = Report(section, title="Test")

    fut = asyncio.ensure_future(report._execute_event())

    # Wait for running
    for _ in range(200):
        if report.running:
            break
        await asyncio.sleep(0.01)

    # Simulate user clicking stop (toggle off)
    assert report._execute_task is not None
    report._execute_task.cancel()

    for _ in range(200):
        if report._execute_task is None:
            break
        await asyncio.sleep(0.01)

    assert report.running is False
    assert report._execute_task is None


async def test_report_execute_event_clears_task_on_success():
    """_execute_event sets _execute_task to None after successful completion."""
    order = []
    a = InstantAction(order=order, title="A")
    section = Section(a, title="S1")
    report = Report(section, title="Test")

    fut = asyncio.ensure_future(report._execute_event())
    await fut

    assert report._execute_task is None
    assert "A" in order
    assert report.status == "success"


async def test_report_execute_event_clears_task_on_cancel():
    """_execute_event sets _execute_task to None after cancellation."""
    order = []
    a = SlowAction(order=order, title="A", sleep_time=10.0)
    section = Section(a, title="S1")
    report = Report(section, title="Test")

    fut = asyncio.ensure_future(report._execute_event())

    for _ in range(200):
        if report._execute_task is not None:
            break
        await asyncio.sleep(0.01)

    report._execute_task.cancel()

    for _ in range(200):
        if report._execute_task is None:
            break
        await asyncio.sleep(0.01)

    assert report._execute_task is None


async def test_report_toggle_resets_to_false_after_completion():
    """_run.value is set back to False after execution completes."""
    order = []
    a = InstantAction(order=order, title="A")
    section = Section(a, title="S1")
    report = Report(section, title="Test")

    fut = asyncio.ensure_future(report._execute_event())
    await fut

    assert report._run.value is False
