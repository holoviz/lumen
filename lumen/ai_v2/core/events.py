"""
Event system for Lumen AI 2.0 agents
"""

from collections.abc import Callable

from .models import ExecutionEvent


def print_execution_event(event: ExecutionEvent, time_format: str = "time") -> None:
    """Handler for new ExecutionEvent objects with milestone progress tracking

    Args:
        event: The execution event to print
        time_format: Format for timestamps:
            - "time": HH:MM:SS (default)
            - "datetime": MM/DD HH:MM:SS
            - "relative": relative to start (e.g., "+2.5s")
            - "none": no timestamp
    """
    # Convert timestamp to readable format
    from datetime import datetime

    if time_format == "time":
        time_str = f"[{datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')}]"
    elif time_format == "datetime":
        time_str = f"[{datetime.fromtimestamp(event.timestamp).strftime('%m/%d %H:%M:%S')}]"
    elif time_format == "relative":
        # This would need a start time reference - for now, use elapsed
        time_str = f"[+{event.timestamp:.1f}s]"
    else:  # "none"
        time_str = ""

    if event.error:
        print(f"❌ {time_str} {event.milestone} (iter {event.iteration}): {event.action}")  # noqa: T201
        print(f"   💥 Error: {event.error}")  # noqa: T201
    elif event.completed:
        print(f"✅ {time_str} {event.milestone}: {event.action} COMPLETED")  # noqa: T201
        if event.duration_ms > 0:
            print(f"   ⏱️ Duration: {event.duration_ms:.1f}ms")  # noqa: T201
    else:
        print(f"🔄 {time_str} {event.milestone} (iter {event.iteration}): {event.action}")  # noqa: T201
        if event.result_type:
            print(f"   📝 Result: {event.result_type}")  # noqa: T201
        if event.duration_ms > 0:
            print(f"   ⏱️ Duration: {event.duration_ms:.1f}ms")  # noqa: T201


DEFAULT_EXECUTION_HANDLER = print_execution_event

# Type aliases for handlers
ExecutionHandler = Callable[[ExecutionEvent], None]
