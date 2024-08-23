from __future__ import annotations

import importlib
import os
import sys
import weakref

from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakKeyDictionary

import panel as pn
import param  # type: ignore

from panel import state
from panel.template import DarkTheme, DefaultTheme
from panel.template.base import BasicTemplate
from panel.widgets.indicators import Indicator

from .validation import ValidationError, match_suggestion_message

if TYPE_CHECKING:
    from bokeh.document import Document  # type: ignore


class ConfigDict(dict):
    def __init__(self, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            msg = f"{self.name} with name '{key}' was not found."
            msg = match_suggestion_message(key, list(self), msg)
            raise ValidationError(msg) from e


class SessionCache:

    _session_contexts: ClassVar[WeakKeyDictionary[Document, Any]] = WeakKeyDictionary()

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
        return key in self._curcontext or key in self._global_context

    def __getitem__(self, key):
        return self._curcontext[key]

    def __setitem__(self, key, value):
        self._curcontext[key] = value

    def get(self, key, default=None):
        obj = dict(self._global_context, **self._curcontext).get(key, default)
        if hasattr(obj, "copy"):
            obj = obj.copy()
        return obj

    def keys(self):
        return dict(self._global_context, **self._curcontext).keys()

    def pop(self, key, default=None):
        return self._curcontext.pop(key, default)


_INDICATORS = {k.lower(): v for k, v in param.concrete_descendents(Indicator).items()}
_INDICATORS = ConfigDict("Indicator", **_INDICATORS)

_LAYOUTS: dict[str, type[pn.layout.Panel]] = {
    'accordion': pn.Accordion,
    'column'   : pn.Column,
    'grid'     : pn.GridBox,
    'row'      : pn.Row,
    'tabs'     : pn.Tabs
}

try:
    _DEFAULT_LAYOUT = _LAYOUTS['flex'] = pn.FlexBox
except Exception:
    _DEFAULT_LAYOUT = pn.GridBox

_LAYOUTS = ConfigDict("Layout", **_LAYOUTS)


_TEMPLATES = {
    k[:-8].lower(): v for k, v in param.concrete_descendents(BasicTemplate).items()
}
_TEMPLATES = ConfigDict("Template", **_TEMPLATES)

_THEMES = {'default': DefaultTheme, 'dark': DarkTheme}
_THEMES = ConfigDict("Theme", **_THEMES)


class Template(param.Parameter):
    """
    A Parameter type to validate template types,
    including dict, str, and Template classes.
    """

    def __init__(self, default=_TEMPLATES['material'], **params):
        super().__init__(default=default, **params)
        self._validate(default)

    def _validate(self, val):
        self._validate_value(val, self.allow_None)

    def _validate_value(self, val, allow_None):
        if isinstance(val, (dict, str)) or issubclass(val, BasicTemplate):
            return
        raise ValueError(f"Template type {type(val).__name__} not recognized.")


class _config(param.Parameterized):
    """
    Stores shared configuration for the entire Lumen application.
    """

    _root = param.String(default=None, doc="""
        Root directory relative to which data and modules are loaded.""")

    _dev = param.Boolean(default=None, doc="""
        Whether Lumen should run in development mode.""")

    serializer = param.String(default='session', doc="""
        The default serializer to use when ephemeral data has to be persisted.""")

    template_vars = param.Dict(default={}, doc="""
        Template variables which may be referenced in a dashboard yaml
        specification.""")

    yamls = param.List(default=[], doc="""
        List of yaml files currently being served.""")

    _modules: ClassVar[dict[str, ModuleType]] = {}

    _session_config: ClassVar[weakref.WeakKeyDictionary[Document, dict[str, Any]]] = weakref.WeakKeyDictionary()

    @property
    def root(self):
        if pn.state.curdoc and pn.state.curdoc in self._session_config:
            session_config = self._session_config[pn.state.curdoc]
            if 'root' in session_config:
                return session_config['root']
        if self._root:
            return self._root
        return os.getcwd()

    @root.setter
    def root(self, root):
        if pn.state.curdoc:
            if pn.state.curdoc in self._session_config:
                session_config = self._session_config[pn.state.curdoc]
            else:
                self._session_config[pn.state.curdoc] = session_config = {}
            session_config['root'] = root
        else:
            self._root = root

    def load_local_modules(self):
        """
        Loads local modules containing custom components and templates.
        """
        for imp in ('filters', 'sources', 'transforms', 'template', 'views'):
            path = os.path.join(self.root, imp+'.py')
            if not os.path.isfile(path):
                continue
            spec = importlib.util.spec_from_file_location(f"local_lumen.{imp}", path)
            if path in self._modules:
                continue
            self._modules[path] = module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        templates = param.concrete_descendents(BasicTemplate)
        _TEMPLATES.update({
            k[:-8].lower(): v for k, v in templates.items()
        })

    @property
    def dev(self):
        """
        Whether the application was launched in development mode.
        """
        if self._dev is not None:
            return self._dev
        return '--dev' in sys.argv or '--autoreload' in sys.argv

    @dev.setter
    def dev(self, dev):
        self._dev = dev


config = _config()
