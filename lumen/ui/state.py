from weakref import WeakKeyDictionary

import panel as pn
import param


class session_state(param.Parameterized):

    components = param.String(default='./components')

    _modals = WeakKeyDictionary()

    _sources = WeakKeyDictionary()

    _specs = WeakKeyDictionary()

    _templates = WeakKeyDictionary()

    _views = WeakKeyDictionary()

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
    def modal(self, modal):
        self._modals[pn.state.curdoc] = modal

    @property
    def template(self):
        return self._templates.get(pn.state.curdoc)

    @template.setter
    def template(self, template):
        self._templates[pn.state.curdoc] = template

    @property
    def views(self):
        return self._views.get(pn.state.curdoc)

    @views.setter
    def views(self, views):
        self._views[pn.state.curdoc] = views


state = pn.state.as_cached('session_state', session_state)
