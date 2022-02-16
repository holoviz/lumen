import html
import importlib
import importlib.util
import os
import traceback
import yaml

from concurrent.futures import Future, ThreadPoolExecutor

import param
import panel as pn

from panel.template.base import BasicTemplate

from .auth import AuthPlugin
from .config import config, _DEFAULT_LAYOUT, _LAYOUTS, _TEMPLATES, _THEMES
from .filters import ConstantFilter, Filter, WidgetFilter # noqa
from .panel import IconButton
from .sources import Source, RESTSource # noqa
from .state import state
from .target import Target
from .transforms import Transform # noqa
from .util import expand_spec
from .variables import Variables
from .views import View # noqa

pn.config.css_files.append(
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
)

def load_yaml(yaml_spec, **kwargs):
    expanded = expand_spec(yaml_spec, config.template_vars, **kwargs)
    return yaml.load(expanded, Loader=yaml.Loader)


class Config(param.Parameterized):
    """
    High-level configuration options for the Dashboard.
    """

    background_load = param.Boolean(default=False, doc="""
        Whether to load any targets in the background.""")

    editable = param.Boolean(default=False, doc="""
        Whether the dashboard specification is editable from within
        the deployed dashboard.""")

    layout = param.Selector(default=_DEFAULT_LAYOUT, objects=_LAYOUTS, doc="""
        Overall layout of the dashboard.""")

    loading_spinner = param.Selector(default='dots', objects=[
        'arc', 'arcs', 'bar', 'dots', 'petal'], doc="""
        Loading indicator to use when component loading parameter is set.""")

    loading_color = param.Color(default='#00aa41', doc="""
        Color of the loading indicator.""")

    logo = param.String(default=None, doc="""
        A logo to add to the theme.""")

    ncols = param.Integer(default=3, bounds=(1, None), doc="""
        Number of columns to lay out targets in.""")

    reloadable = param.Boolean(default=True, doc="""
        Whether to allow reloading data from source(s) using a button.""")

    sync_with_url = param.Boolean(default=False, doc="""
        Whether to sync current state of the application.""")

    title = param.String(default="Lumen Dashboard", doc="""
        The title of the dashboard.""")

    template = param.Selector(default=_TEMPLATES['material'], objects=_TEMPLATES,
                              check_on_set=False, doc="""
        The Panel template to render the dashboard into.""")

    theme = param.Selector(default=_THEMES['default'], objects=_THEMES, doc="""
        The Panel template theme to style the dashboard with.""")

    @classmethod
    def from_spec(cls, spec):
        params = dict(spec)
        if 'theme' in params:
            params['theme'] = _THEMES[params['theme']]
        if 'template' in params:
            template = params['template']
            if template in _TEMPLATES:
                params['template'] = _TEMPLATES[template]
            elif '.' not in template:
                raise ValueError(f'Template must be one of {list(_TEMPLATES)} '
                                 'or an absolute import path.')
            else:
                *paths, name = template.split('.')
                path = '.'.join(paths)
                try:
                    module = importlib.import_module(path)
                except Exception as e:
                    raise ImportError(f'Template {path} module could not '
                                      f'be imported and errored with: {e}.')
                if not hasattr(module, name):
                    raise ImportError(f'Template {name} was not found in '
                                      f'module {path}.')
                params['template'] = template = getattr(module, name)
                if not issubclass(template, BasicTemplate):
                    raise ValueError(f'Imported template {path}.{name} '
                                     'is not a valid Panel template.')
        if 'layout' in params:
            params['layout'] = _LAYOUTS[params['layout']]
        return cls(**params)

    def __init__(self, **params):
        super().__init__(**params)
        pn.config.loading_spinner = self.loading_spinner
        pn.config.loading_color = self.loading_color

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


class Auth(param.Parameterized):

    case_sensitive = param.Boolean(default=False, doc="""
        Whether auth validation is case-sensitive or not.""")

    spec = param.Dict({}, doc="""
        Dictionary of keys and values to match the pn.state.user_info
        against.""")

    @classmethod
    def from_spec(cls, spec):
        spec = dict(spec)
        case_sensitive = spec.pop('case_sensitive', cls.case_sensitive)
        plugins = spec.pop('plugins', [])
        for plugin_spec in plugins:
            plugin = AuthPlugin.from_spec(plugin_spec)
            spec = plugin.transform(spec)
        return cls(spec=spec, case_sensitive=case_sensitive)

    @property
    def authorized(self):
        if not self.spec:
            return True
        elif pn.state.user_info is None and self.spec:
            return config.dev
        authorized = True
        for k, value in self.spec.items():
            if not isinstance(value, list): value = [value]
            if not self.case_sensitive:
                value = [v.lower() if isinstance(v, str) else v for v in value]
            value = [v.strip() if isinstance(v, str) else v for v in value]
            if k in pn.state.user_info:
                user_value = pn.state.user_info[k]
                if not isinstance(user_value, list):
                    user_value = [user_value]
                if not self.case_sensitive:
                    user_value = [
                        uv.lower() if isinstance(uv, str) else uv
                        for uv in user_value
                    ]
                authorized &= any(uv == v for v in value for uv in user_value)
        return authorized


class Dashboard(param.Parameterized):

    auth = param.ClassSelector(default=Auth(), class_=Auth, doc="""
        Auth object which validates the auth spec against pn.state.user_info.""")

    config = param.ClassSelector(default=Config(), class_=Config, doc="""
        High-level configuration options for the dashboard.""")

    defaults = param.ClassSelector(default=Defaults(), class_=Defaults, doc="""
        Defaults to apply to component classes.""")

    targets = param.List(default=[], class_=Target, doc="""
        List of targets monitoring some source.""")

    def __init__(self, specification=None, **params):
        self._load_global = params.pop('load_global', True)
        self._yaml_file = specification
        self._root = os.path.abspath(os.path.dirname(self._yaml_file))
        self._edited = False
        super().__init__(**params)

        # Initialize from spec
        config.load_local_modules()
        self._load_specification(from_file=True)
        self._apps = {}

        # Set up background loading
        self._thread_pool = None
        self._rendered = []

        # Initialize high-level settings
        self.auth = Auth.from_spec(state.spec.get('auth', {}))
        self.config = Config.from_spec(state.spec.get('config', {}))
        self.defaults = Defaults.from_spec(state.spec.get('defaults', {}))
        vars = Variables.from_spec(state.spec.get('variables', {}))
        self.variables = state._variables[pn.state.curdoc] = vars
        self.defaults.apply()

        # Load and populate template
        self._template = self.config.construct_template()
        self._create_modal()
        self._create_main()
        self._create_sidebar()
        self._create_header()
        self._populate_template()

        state._apps[pn.state.curdoc] = self
        pn.state.onload(self._render_dashboard)

    ##################################################################
    # Load specification
    ##################################################################

    def _render_dashboard(self):
        try:
            self._materialize_specification()
            self._render()
            self._main[:] = [self._layout]
        except Exception as e:
            self.param.warning(f'Rendering dashboard raised following error:\n\n {type(e).__name__}: {e}') 
            self._main.loading = False
            tb = html.escape(traceback.format_exc())
            alert = pn.pane.HTML(
                f'<b>{type(e).__name__}</b>: {e}</br><pre style="overflow-y: scroll">{tb}</pre>',
                css_classes=['alert', 'alert-danger'], sizing_mode='stretch_width'
            )
            self._main[:] = [alert]
        if isinstance(self._layout, pn.Tabs) and self.config.sync_with_url:
            pn.state.location.sync(self._layout, {'active': 'target'})

    def _load_specification(self, from_file=False):
        kwargs = {}
        if from_file or self._yaml is None:
            with open(self._yaml_file) as f:
                self._yaml = f.read()
        elif self._edited:
            kwargs = {'getenv': False, 'getshell': False, 'getoauth': False}
        state.spec = load_yaml(self._yaml, **kwargs)
        state.resolve_views()

    def _load_target(self, target_spec):
        target_spec = dict(target_spec)
        if 'reloadable' not in target_spec:
            target_spec['reloadable'] = self.config.reloadable
        target = Target.from_spec(target_spec, application=self)
        if isinstance(self._layout, pn.Tabs):
            target.show_title = False
        target.start()
        return target

    def _background_load(self, i, spec):
        self.targets[i] = target = self._load_target(spec)
        return target

    def _materialize_specification(self, force=False):
        if force or self._load_global or not state.global_sources:
            state.load_global_sources(clear_cache=force)
        if not self.auth.authorized:
            self.targets = []
            return
        targets = []
        target_specs = state.spec.get('targets', [])
        ntargets = len(target_specs)
        if isinstance(self._layout, pn.Tabs) and self.config.background_load:
            self._thread_pool = ThreadPoolExecutor(max_workers=ntargets-1)
        for i, target_spec in enumerate(target_specs):
            if state.loading_msg:
                state.loading_msg.object = (
                    f"Loading target {target_spec['title']!r} ({i}/{ntargets})..."
            )
            if isinstance(self._layout, pn.Tabs) and i != self._layout.active:
                if self.config.background_load:
                    item = self._thread_pool.submit(self._background_load, i, target_spec)
                else:
                    item = None
                self._rendered.append(False)
            else:
                item = self._load_target(target_spec)
                self._rendered.append(True)
            targets.append(item)
        self.targets[:] = targets
        if self._thread_pool is not None:
            self._thread_pool.shutdown()

    ##################################################################
    # Create UI
    ##################################################################

    def _create_header(self):
        self._header = pn.Row()
        self._reload_button = IconButton(
            icon='fa-sync', size=18, margin=0
        )
        self._reload_button.on_click(self._reload)
        menu_items = []
        for yml in config.yamls:
            with open(yml) as f:
                spec = load_yaml(f.read())
            endpoint = os.path.basename(yml).split('.')[0]
            title = spec.get('config', {}).get('title', endpoint)
            i = 1
            while title in self._apps:
                i += 1
                title = f'{title} {i}'
            menu_items.append(title)
            self._apps[title] = endpoint
        self._menu_button = pn.widgets.MenuButton(
            name='Select dashboard', items=menu_items, width=200,
            align='center', margin=0
        )
        self._menu_button.on_click(self._navigate)
        if len(config.yamls) > 1:
            self._header.append(self._menu_button)
        if self.config.reloadable:
            self._header.append(self._reload_button)
        if self.config.editable:
            self._header.append(self._edit_button)
        if 'auth' in state.spec:
            logout = IconButton(
                icon='fa-sign-out-alt', size=18, margin=0
            )
            logout.js_on_click(code="""
            window.location.href = '/logout'
            """)
            self._header.extend([
                pn.layout.HSpacer(),
                f'<b><font size="4.5em">User: {pn.state.user}</font></b>',
                logout
            ])

    def _create_main(self):
        layout_kwargs = {'sizing_mode': 'stretch_both', 'margin': 10}
        if self.config.layout is pn.Tabs:
            layout_kwargs['dynamic'] = True
        elif self.config.layout is pn.GridBox:
            layout_kwargs['ncols'] = self.config.ncols
        self._layout = self.config.layout(**layout_kwargs)
        style = {'text-align': 'center', 'font-size': '1.8em', 'font-weight': 'bold'}
        state.loading_msg = pn.pane.HTML(
            'Loading...', align='center', style=style, width=400, height=400
        )
        self._loading = pn.Column(
            pn.layout.HSpacer(), state.loading_msg, pn.layout.HSpacer(),
            sizing_mode='stretch_both'
        )
        self._main = pn.Column(
            self._loading, loading=True, sizing_mode='stretch_both',
        )
        if isinstance(self._layout, pn.Tabs):
            self._layout.param.watch(self._activate_filters, 'active')

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
        self._edit_button.js_on_click(code="""
        var modal = document.getElementById("pn-Modal")
        modal.style.display = "block"
        """)
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
        self._template.sidebar[:] = [self._sidebar]

    ##################################################################
    # Rendering API
    ##################################################################

    def _activate_filters(self, event):
        target = self.targets[event.new]
        rendered = self._rendered[event.new]
        if not rendered:
            spec = state.spec['targets'][event.new]
            self._set_loading(spec['title'])
            if isinstance(target, Future):
                target = target.result()
            else:
                target = self._load_target(spec)
            self.targets[event.new] = target
            self._layout[event.new] = target.panels
            self._rendered[event.new] = True
        self._render_filters()
        self._layout.loading = False

    def _edit(self, event):
        self._yaml = event.new
        self._edited = True

    def _navigate(self, event):
        app = self._apps[event.new]
        pn.state.location.pathname = f'/{app}'
        pn.state.location.reload = True

    def _set_loading(self, name=''):
        state.loading_msg.object = f'<b>Reloading {name}...</b>'
        if isinstance(self._layout, pn.GridBox):
            items = [pn.pane.HTML(width=self._layout[i].width)
                     for i in range(self._layout.ncols) if i < len(self._layout)]
            index = int(min(self._layout.ncols, (len(self._layout)-1)) / 2)
            if items:
                items[index] = self._loading
            else:
                items = [self._loading]
        elif isinstance(self._layout, pn.Tabs):
            items = list(zip(self._layout._names, list(self._layout.objects)))
            tab_name = items[self._layout.active][0]
            if name and tab_name != name:
                return
            items[self._layout.active] = (tab_name, self._loading)
        else:
            items = [self._loading]
        self._layout[:] = items
        self._layout.loading = True

    def _open_modal(self, event):
        self._template.open_modal()

    def _get_global_filters(self):
        views = []
        filters = []
        for target in self.targets:
            if target is None or isinstance(target, Future):
                continue
            for filt in target.filters:
                if ((all(isinstance(target, (type(None), Future)) or filt in target.filters
                         for target in self.targets) and
                    filt.panel is not None) and filt.shared) and filt not in filters:
                    views.append(filt.panel)
                    filters.append(filt)
        if not views or len(self.targets) == 1:
            return None, None
        return filters, pn.Column(*views, name='Filters', sizing_mode='stretch_width')

    def _render_filters(self):
        self._global_filters, global_panel = self._get_global_filters()
        filters = [] if global_panel is None else [global_panel]
        variable_panel = self.variables.panel
        if len(variable_panel):
            filters.append(variable_panel)
        for i, target in enumerate(self.targets):
            if isinstance(self._layout, pn.Tabs) and i != self._layout.active:
                continue
            elif target is None or isinstance(target, Future):
                spec = state.spec['targets'][i]
                panel = pn.Column(name=spec['title'])
            else:
                panel = target.get_filter_panel(skip=self._global_filters)
            if panel is not None:
                filters.append(panel)
        self._sidebar[:] = filters

    def _render_targets(self):
        items = []
        for target, spec in zip(self.targets, state.spec.get('targets', [])):
            if target is None or isinstance(target, Future):
                panel = pn.Column(name=spec['title'])
            else:
                panel = target.panels
            items.append(panel)
        self._layout[:] = items

    def _render_unauthorized(self):
        auth_keys = list(state.spec.get('auth'))
        if len(auth_keys) == 1:
            auth_keys = repr(auth_keys[0])
        else:
            auth_keys = [repr(k) for k in auth_keys]
        if pn.state.user_info is None:
            error = """
            ## Authorization Error

            Cannot verify permissioning since OAuth is not enabled.
            """
        else:
            error = f"""
            ## Unauthorized User

            {pn.state.user} is not authorized. Ensure {pn.state.user}
            is permissioned correctly on the {auth_keys} field(s) in
            OAuth user data.
            """
        alert = pn.pane.Alert(error, alert_type='danger')
        if isinstance(self._layout, pn.Tabs):
            alert = ('Authorization Denied', alert)
        self._layout[:] = [alert]

    def _render(self):
        if self.auth.authorized:
            self._render_filters()
            if isinstance(self._layout, pn.Tabs):
                active = [0, 1] if self._global_filters else [0]
            else:
                active = list(range(len(self._sidebar)))
            self._sidebar.active = active
            self._render_targets()
        else:
            self._render_unauthorized()
        self._main.loading = False

    def _reload(self, *events):
        self._load_specification()
        self._materialize_specification(force=True)
        self._render()

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
                self._layout
            )
        )

    def show(self, **kwargs):
        """
        Opens the dashboard in a new window.
        """
        self._template.show(title=self.config.title, **kwargs)

    def servable(self):
        """
        Marks the dashboard for serving with `panel serve`.
        """
        self._template.servable(title=self.config.title)
