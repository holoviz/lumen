from __future__ import annotations

from collections import defaultdict

import param

from ..config import SessionCache


class _Memory(SessionCache):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks = defaultdict(list)
        self._rx = {}

    def __setitem__(self, key, new):
        if key in self:
            old = self[key]
        else:
            old = None
        super().__setitem__(key, new)
        self._trigger_update(key, old, new)

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
            cb(key, old, new)
        if key in self._rx:
            self._rx[key].rx.value = new


memory = _Memory()
