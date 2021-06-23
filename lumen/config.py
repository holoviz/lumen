import importlib
import os
import sys

import param as param
import panel as pn

from panel.widgets.indicators import Indicator
from panel.template.base import BasicTemplate
from panel.template import DefaultTheme, DarkTheme


_INDICATORS = {k.lower(): v for k, v in param.concrete_descendents(Indicator).items()}

_LAYOUTS = {
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

_TEMPLATES = {
    k[:-8].lower(): v for k, v in param.concrete_descendents(BasicTemplate).items()
}

_THEMES = {'default': DefaultTheme, 'dark': DarkTheme}



class _config(param.Parameterized):
    """
    Stores shared configuration for the entire Lumen application.
    """

    _root = param.String(default=None, doc="""
      Root directory relative to which data and modules are loaded.""")

    _dev = param.Boolean(default=None, doc="""
      Whether Lumen should run in development mode.""")

    template_vars = param.Dict(default={}, doc="""
      Template variables which may be referenced in a dashboard yaml
      specification.""")

    yamls = param.List(default=[], doc="""
      List of yaml files currently being served.""")

    _modules = {}

    @property
    def root(self):
        if self._root:
            return self._root
        return os.getcwd()

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
