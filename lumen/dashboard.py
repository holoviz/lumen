import os
import yaml
import importlib.util

import param
import panel as pn

from .config import config
from .filters import ConstantFilter, Filter, WidgetFilter # noqa
from .sources import Source, RESTSource # noqa
from .target import Target
from .transforms import Transform # noqa
from .util import _LAYOUTS, _TEMPLATES, _THEMES, expand_spec
from .views import View # noqa


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


class Config(param.Parameterized):
    """
    High-level configuration options for the Dashboard.
    """

    layout = param.Selector(default=_LAYOUTS['grid'], objects=_LAYOUTS, doc="""
        Overall layout of the dashboard.""")

    logo = param.String(default=None, doc="""
        A logo to add to the theme.""")

    ncols = param.Integer(default=3, bounds=(1, None), doc="""
        Number of columns to lay out targets in.""") 

    title = param.String(default="Lumen Dashboard", doc="""
        The title of the dashboard.""")

    template = param.Selector(default=_TEMPLATES['material'], objects=_TEMPLATES, doc="""
        The Panel template to render the dashboard into.""")

    theme = param.Selector(default=_THEMES['default'], objects=_THEMES, doc="""
        The Panel template theme to style the dashboard with.""")

    @classmethod
    def from_spec(cls, spec):
        params = dict(spec)
        if 'theme' in params:
            params['theme'] = _THEMES[spec['theme']]
        if 'template' in params:
            params['template'] = _TEMPLATES[spec['template']]
        if 'layout' in params:
            params['layout'] = _LAYOUTS[spec['layout']]
        return cls(**params)

    def construct_template(self):
        params = {'title': self.title, 'theme': self.theme}
        if self.logo:
            params['logo'] = self.config.logo
        return self.template(**params)


class Defaults(param.Parameterized):
    """
    Defaults to apply to the component classes.
    """

    filters = param.List(doc="Defaults for Filter objects.", class_=dict)

    sources = param.List(doc="Defaults for Source objects.", class_=dict)

    transforms = param.List(doc="Defaults for Transform objects.", class_=dict)

    views = param.List(doc="Defaults for View objects.", class_=dict)

    @classmethod
    def from_spec(cls, spec):
        return cls(**spec)

    def apply(self):
        for obj, defaults in ((Filter, self.filters), (Source, self.sources),
                              (Transform, self.transforms), (View, self.views)):
            for default in defaults:
                params = dict(default)
                obj_type = obj._get_type(params.pop('type', None))
                obj_type.param.set_param(**params)



class Dashboard(param.Parameterized):

    auth = param.Dict(default={}, doc="""
        Dictionary of keys and values to match the pn.state.user_info
        against.""")

    config = param.ClassSelector(default=Config(), class_=Config, doc="""
        High-level configuration options for the dashboard.""")

    defaults = param.ClassSelector(default=Defaults(), class_=Defaults, doc="""
        Defaults to apply to component classes.""")

    sources = param.Dict(default={}, doc="""
        A dictionary of Source indexed by name.""")

    targets = param.List(default=[], class_=Target, doc="""
        List of targets monitoring some source.""")

    def __init__(self, specification=None, **params):
        self._load_global = params.pop('load_global', True)
        self._yaml_file = specification
        self._root = os.path.abspath(os.path.dirname(self._yaml_file))
        self._modules = {}
        self._filters = {}
        self._edited = False
        super(Dashboard, self).__init__(**params)

        # Initialize from spec
        self._load_local_modules()
        self._load_specification(from_file=True)
        self._materialize_specification()

        # Load and populate template
        self._template = self.config.construct_template()
        self._create_modal()
        self._create_main()
        self._create_sidebar()
        self._create_header()
        self._rerender()
        self._populate_template()

    ##################################################################
    # Load specification
    ##################################################################

    def _load_local_modules(self):
        for imp in ('filters', 'sources', 'transforms', 'views'):
            path = os.path.join(self._root, imp+'.py')
            if not os.path.isfile(path):
                continue
            spec = importlib.util.spec_from_file_location(f"local_lumen.{imp}", path)
            self._modules[imp] = module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    def _load_specification(self, from_file=False):
        kwargs = {}
        if from_file or self._yaml is None:
            with open(self._yaml_file) as f:
                self._yaml = f.read()
        elif self._edited:
            kwargs = {'getenv': False, 'getshell': False, 'getoauth': False}
        self._spec = load_yaml(self._yaml, **kwargs)

    @property
    def _authorized(self):
        authorized = True
        for k, value in self.auth.items():
            if not isinstance(value, list): value = [value]
            if k in pn.state.user_info:
                user_value = pn.state.user_info[k]
                if not isinstance(user_value, list):
                    user_value = [user_value]
                authorized &= any(uv == v for v in value for uv in user_value)
        return authorized

    def _materialize_specification(self, force=False):
        self.auth = self._spec.get('auth', {})
        self.config = Config.from_spec(self._spec.get('config', {}))
        self.defaults = Defaults.from_spec(self._spec.get('defaults', {}))
        self.defaults.apply()
        if force or self._load_global:
            config.load_global_sources(self._spec.get('sources', {}), self._root)
        if not self._authorized:
            self.targets = []
            return
        for name, source_spec in self._spec.get('sources', {}).items():
            source_spec = dict(source_spec)
            filter_specs = source_spec.pop('filters', None)
            if name in config.sources:
                source = config.sources[name]
            else:
                source = Source.from_spec(
                    source_spec, self.sources, root=self._root
                )
            self.sources[name] = source
            if not filter_specs:
                continue
            self._filters[name] = filters = dict(config.filters.get(name, {}))
            schema = source.get_schema()
            for fname, filter_spec in filter_specs.items():
                if fname not in filters:
                    filters[fname] = Filter.from_spec(filter_spec, schema)
        self.targets[:] = [
            Target.from_spec(
                dict(target_spec), self.sources, self._filters,
                self._root, application=self
            )
            for target_spec in self._spec.get('targets', [])
        ]

    ##################################################################
    # Create UI
    ##################################################################

    def _create_header(self):
        self._header = pn.Row()
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
        if len(config.yamls) > 1:
            self._header.append(self._menu_button)
        self._header.extend([self._reload_button, self._edit_button])
        if 'auth' in self._spec:
            self._header.extend([
                pn.layout.HSpacer(),
                f'<b><font size="4.5em">User: {pn.state.user}</font></b>'
            ])

    def _create_main(self):
        layout_kwargs = {'sizing_mode': 'stretch_width', 'margin': 10}
        if self.config.layout is pn.Tabs:
            layout_kwargs['dynamic'] = True
        elif self.config.layout is pn.GridBox:
            layout_kwargs['ncols'] = self.config.ncols
        self._main = self.config.layout(**layout_kwargs)
        if self.config.layout is pn.Tabs:
            self._main.param.watch(self._activate_filters, 'active')

    def _create_modal(self):
        self._editor = pn.widgets.Ace(
            value=self._yaml, filename=self._yaml_file,
            sizing_mode='stretch_both', min_height=600,
            theme='monokai'
        )
        self._edit_button = pn.widgets.Button(
            name='✎', width=50, css_classes=['reload'], margin=0,
            align='center'
        )
        self._editor.param.watch(self._edit, 'value')
        self._edit_button.on_click(self._open_modal)
        self._modal = pn.Column(
            f'## Edit {os.path.basename(self._yaml_file)}',
            self._editor, sizing_mode='stretch_both'
        )

    def _create_sidebar(self):
        self._sidebar = pn.Accordion(margin=0, sizing_mode='stretch_width')

    def _populate_template(self):
        self._template.modal[:] = [self._modal]
        self._template.main[:] = [self._main]
        self._template.header[:] = [self._header]
        if len(self._sidebar):
            self._template.sidebar[:] = [self._sidebar]

    ##################################################################
    # Rendering API
    ##################################################################

    def _activate_filters(self, event):
        self._sidebar.active = [event.new]

    def _edit(self, event):
        self._yaml = event.new
        self._edited = True

    def _navigate(self, event):
        pn.state.location.pathname = f'/{event.new}'
        pn.state.location.reload = True

    def _loading(self, name=''):
        loading = pn.Column(
            pn.indicators.LoadingSpinner(value=True, align='center'),
            f'**Reloading {name}...**'
        )
        if isinstance(self._main, pn.GridBox):
            items = [pn.pane.HTML(width=self._main[i].width)
                     for i in range(self._main.ncols) if i < len(self._main)]
            index = int(min(self._main.ncols, (len(self._main)-1)) / 2)
            if items:
                items[index] = loading
            else:
                items = [loading]
        elif isinstance(self._main, pn.Tabs):
            items = list(zip(self._main._names, list(self._main.objects)))
            items[self._main.active] = pn.Row(
                pn.layout.HSpacer(), loading, pn.layout.HSpacer()
            )
        else:
            items = [pn.Row(pn.layout.HSpacer(), loading, pn.layout.HSpacer())]
        self._main[:] = items

    def _open_modal(self, event):
        self._template.open_modal()

    def _rerender(self):
        if self._authorized:
            filters = []
            for target in self.targets:
                panel = target.filter_panel
                if panel is not None:
                    filters.append(panel)
            self._sidebar[:] = filters
            self._sidebar.active = [0]
            self._main[:] = [target.panels for target in self.targets]
        else:
            auth_keys = list(self.auth)
            if len(auth_keys) == 1:
                auth_keys = repr(auth_keys[0])
            else:
                auth_keys = [repr(k) for k in auth_keys]
            error = ('## Unauthorized User\n'
                     f'{pn.state.user} is not authorized. Ensure '
                     f'{pn.state.user} is permissioned correctly on '
                     f'the {auth_keys} field(s) in OAuth user data.')
            self._main[:] = [pn.pane.Alert(error, alert_type='danger')]

    def _reload(self, *events):
        self._load_specification()
        self._materialize_specification(force=True)
        self._rerender()

    ##################################################################
    # Public API
    ##################################################################

    def layout(self):
        """
        Returns a layout of the dashboard contents.
        """
        spinner = pn.indicators.LoadingSpinner(width=40, height=40)
        pn.state.sync_busy(spinner)

        title_html = f'<font color="white"><h1>{self.config.title}</h1></font>'
        return pn.Column(
            pn.Row(
                pn.pane.HTML(title_html),
                pn.layout.HSpacer(),
                spinner,
                background='#00aa41'
            ),
            pn.Row(
                pn.Column(self._sidebar, width=300),
                self._main
            )
        )

    def show(self):
        """
        Opens the dashboard in a new window.
        """
        self._template.show(title=self.config.title)
        for target in self.targets:
            target.start()

    def servable(self):
        """
        Marks the dashboard for serving with `panel serve`.
        """
        self._template.servable(title=self.config.title)
        for target in self.targets:
            target.start()
