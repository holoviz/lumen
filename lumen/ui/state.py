from __future__ import annotations

from typing import (
    TYPE_CHECKING, Any, ClassVar, Dict,
)
from weakref import WeakKeyDictionary

import panel as pn
import param  # type: ignore

if TYPE_CHECKING:
    from bokeh.document import Document  # type: ignore
    from panel.template import FastListTemplate

    from lumen.sources.base import Source

    from .views import ViewGallery


class session_state(param.Parameterized):

    components = param.String(default='./components')

    _modals: ClassVar[WeakKeyDictionary[Document, pn.Column]] = WeakKeyDictionary()

    _sources: ClassVar[WeakKeyDictionary[Document, Dict[str, Source]]] = WeakKeyDictionary()

    _specs: ClassVar[WeakKeyDictionary[Document, Dict[str, Any]]] = WeakKeyDictionary()

    _templates: ClassVar[WeakKeyDictionary[Document, FastListTemplate]] = WeakKeyDictionary()

    _views: ClassVar[WeakKeyDictionary[Document, ViewGallery]] = WeakKeyDictionary()

    @property
    def spec(self):
        return self._specs.get(pn.state.curdoc)

    @spec.setter
    def spec(self, spec):
        self._specs[pn.state.curdoc] = spec

    @property
    def sources(self):
        return self._sources.get(pn.state.curdoc)

    @sources.setter
    def sources(self, sources):
        self._sources[pn.state.curdoc] = sources

    @property
    def modal(self):
        return self._modals.get(pn.state.curdoc)

    @modal.setter
    def modal(self, modal: pn.Column):
        self._modals[pn.state.curdoc] = modal

    @property
    def template(self):
        return self._templates.get(pn.state.curdoc)

    @template.setter
    def template(self, template: FastListTemplate):
        self._templates[pn.state.curdoc] = template

    @property
    def views(self):
        return self._views.get(pn.state.curdoc)

    @views.setter
    def views(self, views: ViewGallery):
        self._views[pn.state.curdoc] = views


state = pn.state.as_cached('session_state', session_state)
