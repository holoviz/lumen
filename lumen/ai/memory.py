from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakKeyDictionary

import panel as pn

from panel import state
from panel.viewable import Viewer

from ..base import Component

if TYPE_CHECKING:
    from bokeh.document import Document


class _Memory(Viewer):

    _session_contexts: ClassVar[WeakKeyDictionary[Document, Any]] = WeakKeyDictionary()

    _views: ClassVar[WeakKeyDictionary[Document, Any]] = WeakKeyDictionary()

    _global_context = {}

    @property
    def _curcontext(self):
        if state.curdoc:
            if state.curdoc in self._session_contexts:
                context = self._session_contexts[state.curdoc]
            else:
                self._session_contexts[state.curdoc] = context = {}
            return context
        else:
            return self._global_context

    def __contains__(self, key):
        return key in self._curcontext

    def __getitem__(self, key):
        return self._curcontext[key]

    def __setitem__(self, key, value):
        self._curcontext[key] = value
        if state.curdoc in self._views:
            self._views[state.curdoc][:] = [self._create_view()]

    def get(self, key, default=None):
        return self._curcontext.get(key, default)

    def _render_item(self, item):
        if isinstance(item, Component):
            item = item.to_spec()
        if isinstance(item, str):
            item = f'```yaml\n{item}\n```'
        return pn.panel(item, sizing_mode='stretch_width')

    def _create_view(self):
        return pn.Accordion(*(
            (name, self._render_item(item))
            for name, item in self._curcontext.items()
        ), sizing_mode='stretch_width', active=list(range(len(self._curcontext))))

    def __panel__(self):
        self._views[state.curdoc] = view = pn.Column(self._create_view())
        return view


memory = _Memory()
