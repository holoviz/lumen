from __future__ import annotations

import param

from panel.viewable import Viewer
from panel_material_ui import Column, LinearProgress


class Progress(Viewer):
    """
    Progress reporter with simple API and customizable rendering.

    This class provides a simple interface for reporting progress in async
    operations. It can be used standalone or as part of a control.

    The rendering is handled by `_render_bar()` and `_render_description()`
    methods which can be overridden in subclasses for custom UI.

    Examples
    --------
    Indeterminate progress (spinner):

    >>> progress("Loading metadata...")

    Determinate with percentage:

    >>> progress("Downloading...", value=50)

    Determinate with current/total (auto-calculates %):

    >>> progress("Downloading...", current=500, total=1000)

    Increment pattern for loops:

    >>> progress("Processing files...", total=len(files))
    >>> for f in files:
    ...     process(f)
    ...     progress.increment()

    Clear progress:

    >>> progress.clear()
    """

    current = param.Number(default=0, doc="Current progress value.")

    total = param.Number(default=None, allow_None=True, doc="Total value for percentage calculation.")

    message = param.String(default="", doc="Progress description message.")

    value = param.Number(default=None, allow_None=True, bounds=(0, 100), doc="Direct percentage value (0-100).")

    visible = param.Boolean(default=False, doc="Whether progress indicators are visible.")

    def __init__(self, **params):
        super().__init__(**params)
        self._bar = self._render_bar()
        self._desc = self._render_description()

    def _render_bar(self):
        """
        Render the progress bar component.

        Override this method to customize the progress bar appearance.

        Returns
        -------
        LinearProgress or similar widget
        """
        return LinearProgress(
            visible=False,
            margin=(0, 10, 5, 10),
            sizing_mode="stretch_width"
        )

    def _render_description(self):
        """
        Render the progress description component.

        Override this method to customize the description appearance.

        Returns
        -------
        Typography or similar widget
        """
        from panel_material_ui import Typography
        return Typography(
            styles={"margin-left": "auto", "margin-right": "auto"},
            visible=False
        )

    def __call__(
        self,
        message: str | None = None,
        *,
        value: float | None = None,
        current: float | None = None,
        total: float | None = None,
    ):
        """
        Update progress state.

        Parameters
        ----------
        message : str, optional
            Description text (e.g., "Downloading...")
        value : float, optional
            Direct percentage (0-100)
        current : float, optional
            Current progress value (used with total)
        total : float, optional
            Total value for percentage calculation
        """
        # Update message
        if message is not None:
            self.message = message
            self._desc.object = message
            self._desc.visible = bool(message)

        # Calculate percentage
        pct = None
        if value is not None:
            self.value = value
            pct = value
        elif current is not None and total is not None:
            self.current = current
            self.total = total
            pct = (current / total * 100) if total > 0 else 0
        elif total is not None:
            # Setting total resets current, enables increment pattern
            self.total = total
            self.current = 0
            pct = 0

        # Update bar
        if pct is not None:
            self._bar.variant = "determinate"
            self._bar.value = min(pct, 100)
        else:
            self._bar.variant = "indeterminate"

        self._bar.visible = True
        self.visible = True

    def increment(self, amount: float = 1):
        """
        Increment progress by amount.

        Must call `progress(total=N)` first to set the total.

        Parameters
        ----------
        amount : float
            Amount to increment by (default 1).

        Raises
        ------
        ValueError
            If total has not been set.
        """
        if self.total is None:
            raise ValueError("Call progress(total=N) before using increment()")
        self.current += amount
        pct = (self.current / self.total * 100) if self.total > 0 else 0
        self._bar.value = min(pct, 100)

    def clear(self):
        """Hide progress indicators and reset state."""
        self._bar.visible = False
        self._desc.visible = False
        self._desc.object = ""
        self.param.update(
            total=None,
            current=0,
            value=None,
            message="",
            visible=False
        )

    def reset(self):
        """Alias for clear()."""
        self.clear()

    @property
    def bar(self):
        """The progress bar widget."""
        return self._bar

    @property
    def description(self):
        """The description widget."""
        return self._desc

    def __panel__(self):
        return Column(self._bar, self._desc, sizing_mode="stretch_width")
