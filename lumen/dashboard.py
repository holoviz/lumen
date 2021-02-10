import os
import yaml
import importlib.util

import param
import panel as pn

from panel.template.base import BasicTemplate
from panel.template import DefaultTheme, DarkTheme

from .config import config
from .filters import ConstantFilter, Filter, WidgetFilter # noqa
from .monitor import Monitor # noqa
from .sources import Source, RESTSource # noqa
from .transforms import Transform # noqa
from .util import LAYOUTS, expand_spec
from .views import View # noqa

_templates = {k[:-8].lower(): v for k, v in param.concrete_descendents(BasicTemplate).items()}

_THEMES = {'default': DefaultTheme, 'dark': DarkTheme}

pn.config.raw_css.append("""
.reload .bk-btn {
  background: transparent;
  border: none;
  font-size: 18pt;
}
#header .bk.reload .bk.bk-btn {
  color: white;
}
.reload .bk-btn:hover {
  background: transparent;
}
""")


def load_yaml(yaml_spec, **kwargs):
    expanded = expand_spec(yaml_spec, config.template_vars, **kwargs)
    return yaml.load(expanded, Loader=yaml.Loader)


def apply_global_defaults(defaults):
    """
    Applies a set of global defaults.
    """
    for key, obj in (('filters', Filter), ('sources', Source),
                     ('transforms', Transform), ('views', View)):
        for default in defaults.get(key, []):
            defaults = dict(default)
            obj_type = obj._get_type(defaults.pop('type', None))
            obj_type.param.set_param(**defaults)


def load_global_sources(sources, root, clear_cache=True):
    """
    Loads global sources shared across all targets.
    """
    for name, source_spec in sources.items():
        if source_spec.get('shared'):
            source_spec = dict(source_spec)
            filter_specs = source_spec.pop('filters', {})
            config.sources[name] = source = Source.from_spec(
                source_spec, config.sources, root=root)
            if source.cache_dir and clear_cache:
                source.clear_cache()
            schema = source.get_schema()
            config.filters[name] = {
                fname: Filter.from_spec(filter_spec, schema)
                for fname, filter_spec in filter_specs.items()
            }


class Dashboard(param.Parameterized):

    load_global = param.Boolean(default=True, doc="""
        Whether to initialize the global sources.""")

    specification = param.Filename()

    def __init__(self, specification, **params):
        super(Dashboard, self).__init__(
            specification=specification, **params
        )
        self._load_config(from_file=True)
        self._modules = {}
        self._edited = False
        self._load_local_modules()

        # Construct template
        tmpl = self.config.get('template', 'material')
        theme = self.config.get('theme', 'default').lower()
        kwargs = {'title': self.config.get('title', 'Lumen Dashboard'),
                  'theme': _THEMES[theme]}
        if 'logo' in self.config:
            kwargs['logo'] = self.config['logo']
        self.template = _templates[tmpl](**kwargs)

        # Add editor modal
        self._editor = pn.widgets.Ace(
            value=self._yaml, filename=self.specification,
            sizing_mode='stretch_both', min_height=600,
            theme='monokai'
        )
        self._edit_button = pn.widgets.Button(
            name='✎', width=50, css_classes=['reload'], margin=0,
            align='center'
        )
        self._editor.param.watch(self._edit, 'value')
        self._edit_button.on_click(self._open_modal)
        self._editor_layout = pn.Column(
            f'## Edit {os.path.basename(self.specification)}',
            self._editor, sizing_mode='stretch_both'
        )
        self.template.modal.append(self._editor_layout)

        # Build layouts
        self.filters = pn.Accordion(margin=0, sizing_mode='stretch_width')
        layout = self.config.get('layout', 'grid')
        layout_type = LAYOUTS[layout]
        layout_kwargs = {'sizing_mode': 'stretch_width', 'margin': 10}
        if layout == 'tabs':
            layout_kwargs['dynamic'] = True
        elif layout == 'grid':
            layout_kwargs['ncols'] = self.config.get('ncols', 3)
        self.targets = layout_type(**layout_kwargs)
        if layout == 'tabs':
            self.targets.param.watch(self._activate_filters, 'active')
        self._reload_button = pn.widgets.Button(
            name='↻', width=50, css_classes=['reload'], margin=0,
            align='center'
        )
        self._reload_button.on_click(self._reload)
        menu_items = [os.path.basename(yml).split('.')[0] for yml in config.yamls]
        self._menu_button = pn.widgets.MenuButton(
            name='Select dashboard', items=menu_items, width=200,
            align='center', margin=0
        )
        self._menu_button.on_click(self._navigate)

        self._reload()

        # Populate template
        if len(self.filters):
            self.template.sidebar[:] = [self.filters]
        self.template.main[:] = [self.targets]
        header = pn.Row()
        if len(config.yamls) > 1:
            header.append(self._menu_button)
        header.extend([self._reload_button, self._edit_button])
        if 'auth' in self._spec:
            header.extend([
                pn.layout.HSpacer(),
                f'<b><font size="4.5em">User: {pn.state.user}</font></b>'
            ])
        self.template.header.append(header)

    @property
    def _authorized(self):
        authorized = True
        for k, value in self._spec.get('auth', {}).items():
            if not isinstance(value, list): value = [value]
            if k in pn.state.user_info:
                user_value = pn.state.user_info[k]
                if not isinstance(user_value, list):
                    user_value = [user_value]
                authorized &= any(uv == v for v in value for uv in user_value)
        return authorized

    def _edit(self, event):
        self._yaml = event.new
        self._edited = True

    def _navigate(self, event):
        pn.state.location.pathname = f'/{event.new}'
        pn.state.location.reload = True
        
    def _activate_filters(self, event):
        self.filters.active = [event.new]

    def _open_modal(self, event):
        self.template.open_modal()

    def _load_config(self, from_file=False, reload=False):
        # Load config
        kwargs = {}
        if from_file or self._yaml is None:
            with open(self.specification) as f:
                self._yaml = f.read()
        elif self._edited:
            kwargs = {'getenv': False, 'getshell': False, 'getoauth': False}
        self._spec = load_yaml(self._yaml, **kwargs)
        if not 'targets' in self._spec:
            raise ValueError('Yaml specification did not declare any targets.')
        self.config = self._spec.get('config', {})
        self._root = os.path.abspath(os.path.dirname(self.specification))
        apply_global_defaults(self._spec.get('defaults', {}))
        if reload and self.load_global:
            load_global_sources(self._spec.get('sources', {}), self._root)

    def _load_local_modules(self):
        for imp in ('filters', 'sources', 'transforms', 'views'):
            path = os.path.join(self._root, imp+'.py')
            if not os.path.isfile(path):
                continue
            spec = importlib.util.spec_from_file_location(f"local_lumen.{imp}", path)
            self._modules[imp] = module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    def _reload(self, *events):
        from . import config

        self._load_config(reload=True)
        if not self._authorized:
            auth_keys = list(self._spec.get('auth'))
            if len(auth_keys) == 1:
                auth_keys = repr(auth_keys[0])
            else:
                auth_keys = [repr(k) for k in auth_keys]
            error = ('## Unauthorized User\n'
                     f'{pn.state.user} is not authorized. Ensure '
                     f'{pn.state.user} is permissioned correctly on '
                     f'the {auth_keys} field(s) in OAuth user data.')
            self.targets[:] = [pn.pane.Alert(error, alert_type='danger')]
            self._targets = []
            return

        self._filters = {}
        self._sources = {}
        for name, source_spec in self._spec.get('sources', {}).items():
            source_spec = dict(source_spec)
            filter_specs = source_spec.pop('filters', None)
            if name in config.sources:
                source = config.sources[name]
            else:
                source = Source.from_spec(
                    source_spec, self._sources, root=self._root
                )
            self._sources[name] = source
            if not filter_specs:
                continue
            self._filters[name] = filters = dict(config.filters.get(name, {}))
            schema = source.get_schema()
            for fname, filter_spec in filter_specs.items():
                if fname not in filters:
                    filters[fname] = Filter.from_spec(filter_spec, schema)
        self._targets = self._resolve_targets(self._spec['targets'])
        self._rerender()
        filters = []
        for target in self._targets:
            panel = target.filter_panel
            if panel is not None:
                filters.append(panel)
        self.filters[:] = filters
        self.filters.active = [0]

    def _loading(self, name=''):
        loading = pn.Column(
            pn.indicators.LoadingSpinner(value=True, align='center'),
            f'**Reloading {name}...**'
        )
        if isinstance(self.targets, pn.GridBox):
            items = [pn.pane.HTML(width=self.targets[i].width)
                     for i in range(self.targets.ncols) if i < len(self.targets)]
            index = int(min(self.targets.ncols, (len(self.targets)-1)) / 2)
            if items:
                items[index] = loading
            else:
                items = [loading]
        elif isinstance(self.targets, pn.Tabs):
            items = list(zip(self.targets._names, list(self.targets.objects)))
            items[self.targets.active] = pn.Row(
                pn.layout.HSpacer(), loading, pn.layout.HSpacer()
            )
        else:
            items = [pn.Row(pn.layout.HSpacer(), loading, pn.layout.HSpacer())]
        self.targets[:] = items

    def _rerender(self):
        self.targets[:] = [target.panels for target in self._targets]

    def _resolve_targets(self, target_specs, metadata={}):
        targets = []
        for target_spec in target_specs:
            target_spec = dict(target_spec)

            # Resolve source
            source_spec = target_spec.pop('source', None)
            source = Source.from_spec(source_spec, self._sources, root=self._root)
            schema = source.get_schema()

            # Resolve filters
            filter_specs = target_spec.pop('filters', [])
            if isinstance(source_spec, str):
                source_filters = self._filters.get(source_spec)
            else:
                source_filters = None
            filters = [
                Filter.from_spec(filter_spec, schema, source_filters)
                for filter_spec in filter_specs
            ]

            # Create target
            target = Monitor(
                application=self, filters=filters, source=source,
                schema=schema, **target_spec)
            targets.append(target)
        return targets

    def layout(self):
        """
        Returns a layout rendering the dashboard contents.
        """
        spinner = pn.indicators.LoadingSpinner(width=40, height=40)
        pn.state.sync_busy(spinner)

        title = self.config.get('title', 'Lumen Dashboard')
        title_html = '<font color="white"><h1>'+title+'</h1></font>'
        return pn.Column(
            pn.Row(
                pn.pane.HTML(title_html),
                pn.layout.HSpacer(),
                spinner,
                background='#00aa41'
            ),
            pn.Row(
                pn.Column(self.filters, width=300),
                self.targets
            )
        )

    def show(self):
        """
        Opens the dashboard in a new window.
        """
        title = self.config.get('title', 'Lumen Dashboard')
        self.template.show(title=title)
        for target in self._targets:
            target.start()

    def servable(self):
        """
        Marks the dashboard for serving with `panel serve`.
        """
        title = self.config.get('title', 'Lumen Dashboard')
        self.template.servable(title=title)
        for target in self._targets:
            target.start()
