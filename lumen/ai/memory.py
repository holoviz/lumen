from __future__ import annotations

from collections import defaultdict

import param

from ..config import SessionCache


class _Memory(SessionCache):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks = defaultdict(list)
        self._rx = {}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._trigger_update(key, value)

    def on_change(self, key, callback):
        self._callbacks[key].append(callback)

    def rx(self, key):
        if key in self._rx:
            return self._rx[key]
        self._rx[key] = rxp = param.rx(self[key])
        return rxp

    def trigger(self, key):
        self._trigger_update(key, self[key])

    def _trigger_update(self, key, value):
        for cb in self._callbacks[key]:
            cb(value)
        if key in self._rx:
            self._rx[key].rx.value = value


memory = _Memory()
