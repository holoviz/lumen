import html
import importlib
import importlib.util
import os
import traceback
import yaml

import param
import panel as pn

from panel.template.base import BasicTemplate

from .config import config, _LAYOUTS, _TEMPLATES, _THEMES
from .filters import ConstantFilter, Filter, WidgetFilter # noqa
from .sources import Source, RESTSource # noqa
from .target import Target
from .transforms import Transform # noqa
from .util import expand_spec
from .views import View # noqa


pn.config.raw_css.append("""
.reload .bk-btn {
  background: transparent;
  border: none;
  font-size: 18pt;
}
.logout .bk-btn {
  background: transparent;
  font-size: 16pt;
  border: none;
  color: white;
}
#header .bk.reload .bk.bk-btn {
  color: white;
}
.reload .bk-btn:hover, .logout .bk-btn:hover {
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

    editable = param.Boolean(default=False, doc="""
        Whether the dashboard specification is editable from within
        the deployed dashboard.""")

    layout = param.Selector(default=_LAYOUTS['grid'], objects=_LAYOUTS, doc="""
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
        self._filters = {}
        self._edited = False
        super().__init__(**params)

        # Initialize from spec
        self._modules = config.load_local_modules(self._root)
        self._load_specification(from_file=True)

        # Initialize high-level settings
        self.auth = self._spec.get('auth', {})
        self.config = Config.from_spec(self._spec.get('config', {}))
        self.defaults = Defaults.from_spec(self._spec.get('defaults', {}))
        self.defaults.apply()

        # Load and populate template
        self._template = self.config.construct_template()
        self._create_modal()
        self._create_main()
        self._create_sidebar()
        self._create_header()
        self._populate_template()

        pn.state.onload(self._render_dashboard)

    ##################################################################
    # Load specification
    ##################################################################

    def _render_dashboard(self):
        try:
            self._materialize_specification()
            self._rerender()
        except Exception as e:
            self.param.warning(f'Rendering dashboard raised following error:\n\n {type(e).__name__}: {e}') 
            self._main.loading = False
            tb = html.escape(traceback.format_exc())
            alert = pn.pane.HTML(
                f'<b>{type(e).__name__}</b>: {e}</br><pre style="overflow-y: scroll">{tb}</pre>',
                css_classes=['alert', 'alert-danger'], sizing_mode='stretch_width'
            )
            self._main[:] = [alert]

    def _load_specification(self, from_file=False):
        kwargs = {}
        if from_file or self._yaml is None:
            with open(self._yaml_file) as f:
                self._yaml = f.read()
        elif self._edited:
            kwargs = {'getenv': False, 'getshell': False, 'getoauth': False}
        self._spec = load_yaml(self._yaml, **kwargs)
        self._resolve_views()

    def _resolve_views(self):
        exts = []
        for target in self._spec.get('targets', []):
            views = target.get('views', [])
            if isinstance(views, dict):
                views = list(views.values())
            for view in views:
                view_type = View._get_type(view.get('type'))
                if view_type and view_type._extension:
                    exts.append(view_type._extension)
        for ext in exts:
            __import__(pn.extension._imports[ext])

    @property
    def _authorized(self):
        if pn.state.user_info is None and self.auth:
            return False
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
        sources = self._spec.get('sources', {})
        if force or self._load_global or not config.sources:
            clear_cache = config.sources or config.dev
            config.load_global_sources(sources, self._root, clear_cache=clear_cache)
        if not self._authorized:
            self.targets = []
            return
        for name, source_spec in sources.items():
            source_spec = dict(source_spec)
            filter_specs = source_spec.pop('filters', None)
            if name in config.sources:
                source = config.sources[name]
            else:
                source_spec['name'] = name
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
        self._header.append(self._reload_button)
        if self.config.editable:
            self._header.append(self._edit_button)
        if 'auth' in self._spec:
            logout = pn.widgets.Button(
                name='✖', width=40, css_classes=['logout'], margin=0,
                align='center'
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
        self._main = pn.Column(self._layout, loading=True, **layout_kwargs)
        if self.config.layout is pn.Tabs:
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
        if self._global_filters:
            active = [0, event.new+1]
        else:
            active = [event.new]
        self._sidebar.active = active

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
        if isinstance(self._layout, pn.GridBox):
            items = [pn.pane.HTML(width=self._layout[i].width)
                     for i in range(self._layout.ncols) if i < len(self._layout)]
            index = int(min(self._layout.ncols, (len(self._layout)-1)) / 2)
            if items:
                items[index] = loading
            else:
                items = [loading]
        elif isinstance(self._layout, pn.Tabs):
            items = list(zip(self._layout._names, list(self._layout.objects)))
            tab_name = items[self._layout.active][0]
            if name and tab_name != name:
                return
            items[self._layout.active] = pn.Row(
                pn.layout.HSpacer(), loading, pn.layout.HSpacer()
            )
        else:
            items = [pn.Row(pn.layout.HSpacer(), loading, pn.layout.HSpacer())]
        self._layout[:] = items

    def _open_modal(self, event):
        self._template.open_modal()

    def _get_global_filters(self):
        views = []
        filters = []
        for target in self.targets:
            for filt in target.filters:
                if ((all(filt in target.filters for target in self.targets) and
                    filt.panel is not None) or filt.shared) and filt not in filters:
                    views.append(filt.panel)
                    filters.append(filt)
        if not views or len(self.targets) == 1:
            return None, None
        return filters, pn.Column(*views, name='Filters', sizing_mode='stretch_width')

    def _rerender(self):
        if self._authorized:
            self._global_filters, global_panel = self._get_global_filters()
            filters = [] if global_panel is None else [global_panel]
            for target in self.targets:
                panel = target.get_filter_panel(skip=self._global_filters)
                if panel is not None:
                    filters.append(panel)
            self._sidebar[:] = filters
            self._sidebar.active = [0, 1] if self._global_filters else [0]
            self._layout[:] = [target.panels for target in self.targets]
        else:
            auth_keys = list(self.auth)
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
        self._main.loading = False

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
                self._layout
            )
        )

    def show(self, **kwargs):
        """
        Opens the dashboard in a new window.
        """
        self._template.show(title=self.config.title, **kwargs)
        for target in self.targets:
            target.start()

    def servable(self):
        """
        Marks the dashboard for serving with `panel serve`.
        """
        self._template.servable(title=self.config.title)
        for target in self.targets:
            target.start()
