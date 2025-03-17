from __future__ import annotations

import weakref

from collections import defaultdict
from functools import partial

import param

from panel.io.state import state

from ..config import SessionCache


class _Memory(SessionCache):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_callbacks = weakref.WeakKeyDictionary()
        self._global_callbacks = defaultdict(list)
        self._session_rx = weakref.WeakKeyDictionary()
        self._global_rx = {}

    @property
    def _callbacks(self):
        if state.curdoc:
            if state.curdoc not in self._session_callbacks:
                self._session_callbacks[state.curdoc] = defaultdict(list)
            return self._session_callbacks[state.curdoc]
        return self._global_callbacks

    @property
    def _rx(self):
        if state.curdoc:
            if state.curdoc not in self._session_rx:
                self._session_rx[state.curdoc] = {}
            return self._session_rx[state.curdoc]
        return self._global_rx

    def __setitem__(self, key, new):
        if key in self:
            old = self[key]
        else:
            old = None
        super().__setitem__(key, new)
        self._trigger_update(key, old, new)

    def cleanup(self):
        if state.curdoc:
            self._session_callbacks[state.curdoc].clear()
            self._session_rx[state.curdoc].clear()
        else:
            self._global_callbacks.clear()
            self._global_rx.clear()

    def on_change(self, key, callback):
        self._callbacks[key].append(callback)

    def remove_on_change(self, key, callback):
        self._callbacks[key].remove(callback)

    def rx(self, key):
        if key in self._rx:
            return self._rx[key]
        self._rx[key] = rxp = param.rx(self[key])
        return rxp

    def trigger(self, key):
        self._trigger_update(key, self[key], self[key])

    def _trigger_update(self, key, old, new):
        for cb in self._callbacks[key]:
            if param.parameterized.iscoroutinefunction(cb):
                param.parameterized.async_executor(partial(cb, key, old, new))
            else:
                cb(key, old, new)
        if key in self._rx:
            self._rx[key].rx.value = new


memory = _Memory()
