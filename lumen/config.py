import importlib
import os
import sys

import param as param
import panel as pn

from panel.widgets.indicators import Indicator
from panel.template.base import BasicTemplate
from panel.template import DefaultTheme, DarkTheme

from .filters import Filter
from .sources import Source


_INDICATORS = {k.lower(): v for k, v in param.concrete_descendents(Indicator).items()}

_LAYOUTS = {
    'accordion': pn.Accordion,
    'column'   : pn.Column,
    'grid'     : pn.GridBox,
    'row'      : pn.Row,
    'tabs'     : pn.Tabs
}

_TEMPLATES = {
    k[:-8].lower(): v for k, v in param.concrete_descendents(BasicTemplate).items()
}

_THEMES = {'default': DefaultTheme, 'dark': DarkTheme}



class _config(param.Parameterized):
    """
    Stores shared configuration for the entire Lumen application.
    """

    filters = param.Dict(default={}, doc="""
      A global dictionary of shared Filter objects.""")

    sources = param.Dict(default={}, doc="""
      A global dictionary of shared Source objects.""")

    template_vars = param.Dict(default={}, doc="""
      Template variables which may be referenced in a dashboard yaml
      specification.""")

    yamls = param.List(default=[], doc="""
      List of yaml files currently being served.""")

    def load_global_sources(self, sources, root, clear_cache=True):
        """
        Loads global sources shared across all targets.
        """
        for name, source_spec in sources.items():
            if not source_spec.get('shared'):
                continue
            source_spec = dict(source_spec)
            filter_specs = source_spec.pop('filters', {})
            source = self.sources.get(name)
            source_type = Source._get_type(source_spec['type'])
            if source is not None:
                # Check if old source is the same as the new source
                params = {
                    k: v for k, v in source.param.get_param_values()
                    if k != 'name'
                }
                new_spec = dict(source_spec, **{
                    k: v for k, v in source_type.param.get_param_values()
                    if k != 'name'
                })
                reinitialize = params == new_spec
            else:
                reinitialize = True
            if reinitialize:
                source_spec['name'] = name
                self.sources[name] = source = Source.from_spec(
                    source_spec, self.sources, root=root
                )
            if source.cache_dir and clear_cache:
                source.clear_cache()
            schema = source.get_schema()
            self.filters[name] = {
                fname: Filter.from_spec(filter_spec, schema)
                for fname, filter_spec in filter_specs.items()
            }

    def load_local_modules(self, root):
        """
        Loads local modules containing custom components and templates.

        Arguments
        ---------
        root: str
            The root directory the dashboard is being served from.
        """
        modules = {}
        for imp in ('filters', 'sources', 'transforms', 'template', 'views'):
            path = os.path.join(root, imp+'.py')
            if not os.path.isfile(path):
                continue
            spec = importlib.util.spec_from_file_location(f"local_lumen.{imp}", path)
            modules[imp] = module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        templates = param.concrete_descendents(BasicTemplate)
        _TEMPLATES.update({
            k[:-8].lower(): v for k, v in templates.items()
        })
        return modules

    @property
    def dev(self):
        """
        Whether the application was launched in development mode.
        """
        return '--dev' in sys.argv or '--autoreload' in sys.argv




config = _config()
