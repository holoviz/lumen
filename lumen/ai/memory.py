from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakKeyDictionary

import panel as pn

from panel import state
from panel.viewable import Viewer

from ..base import Component
from ..config import SessionCache

if TYPE_CHECKING:
    from bokeh.document import Document


class _Memory(SessionCache, Viewer):

    _views: ClassVar[WeakKeyDictionary[Document, Any]] = WeakKeyDictionary()

    _global_view = None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if state.curdoc in self._views or (state.curdoc is None and self._global_view is not None):
            self._update_view(key, value)

    def _render_item(self, key, item):
        if isinstance(item, Component):
            item = item.to_spec()
            if 'password' in item:
                item['password'] = '•'*len(item['password'])
        if isinstance(item, str):
            item = f'```yaml\n{item}\n```'
        return pn.panel(item, name=key, sizing_mode='stretch_width', styles={'overflow': 'scroll'})

    def _render_memories(self):
        return pn.Accordion(*(
            self._render_item(name, item)
            for name, item in self._curcontext.items()
        ), sizing_mode='stretch_width', active=list(range(len(self._curcontext))))

    def _update_view(self, key, value):
        view = self._views[state.curdoc] if state.curdoc else self._global_view
        accordion = view[0]
        new_item = self._render_item(key, value)
        i = 0
        for i, item in enumerate(accordion):
            if item.name == key:
                accordion[i] = new_item
                break
        else:
            accordion.append(new_item)
            accordion.active = accordion.active + [i+1]

    def __panel__(self):
        view = pn.Column(self._render_memories())
        if state.curdoc is None:
            self._global_view = view
        else:
            self._views[state.curdoc] = view
        return view


memory = _Memory()
