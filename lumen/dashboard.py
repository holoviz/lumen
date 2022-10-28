import html
import importlib
import importlib.util
import os
import traceback

from concurrent.futures import Future, ThreadPoolExecutor

import panel as pn
import param
import yaml

from panel.io.resources import CSS_URLS
from panel.template.base import BasicTemplate

from .auth import AuthPlugin
from .base import Component
from .config import (
    _DEFAULT_LAYOUT, _LAYOUTS, _TEMPLATES, _THEMES, config,
)
from .filters import ConstantFilter, Filter, WidgetFilter  # noqa
from .panel import IconButton
from .pipeline import Pipeline
from .sources import RESTSource, Source  # noqa
from .state import state
from .target import Target
from .transforms import Transform  # noqa
from .util import catch_and_notify, expand_spec, resolve_module_reference
from .validation import ValidationError, match_suggestion_message
from .variables import Variable, Variables
from .views import View  # noqa


def load_yaml(yaml_spec, **kwargs):
    expanded = expand_spec(yaml_spec, config.template_vars, **kwargs)
    return yaml.load(expanded, Loader=yaml.Loader)


class Config(Component):
    """
    `Config` provides high-level configuration options for the Dashboard.
    """

    auto_update = param.Boolean(default=True, constant=True, doc="""
        Whether changes in filters, transforms and references automatically
        trigger updates in the data or whether an update has to be triggered
        manually using the update event or the update button in the UI."""
    )

    background_load = param.Boolean(default=False, constant=True, doc="""
        Whether to load any targets in the background.""")

    editable = param.Boolean(default=False, constant=True, doc="""
        Whether the dashboard specification is editable from within
        the deployed dashboard.""")

    layout = param.Selector(default=_DEFAULT_LAYOUT, constant=True, objects=_LAYOUTS,
                            check_on_set=False, doc="""
        Overall layout of the dashboard.""")

    loading_spinner = param.Selector(default='dots', constant=True, objects=[
        'arc', 'arcs', 'bar', 'dots', 'petal'], doc="""
        Loading indicator to use when component loading parameter is set.""")

    loading_color = param.Color(default='#00aa41', constant=True, doc="""
        Color of the loading indicator.""")

    logo = param.String(default=None, constant=True, doc="""
        A logo to add to the theme.""")

    ncols = param.Integer(default=3, bounds=(1, None), constant=True, doc="""
        Number of columns to lay out targets in.""")

    reloadable = param.Boolean(default=True, constant=True, doc="""
        Whether to allow reloading data from source(s) using a button.""")

    sync_with_url = param.Boolean(default=False, constant=True, doc="""
        Whether to sync current state of the application.""")

    title = param.String(default="Lumen Dashboard", constant=True, doc="""
        The title of the dashboard.""")

    template = param.Selector(default=_TEMPLATES['material'], constant=True, objects=_TEMPLATES,
                              check_on_set=False, doc="""
        The Panel template to render the dashboard into.""")

    theme = param.Selector(default=_THEMES['default'], objects=_THEMES, constant=True,
                           check_on_set=False, doc="""
        The Panel template theme to style the dashboard with.""")

    _valid_keys = 'params'
    _validate_params = True

    @classmethod
    def _validate_layout(cls, layout, spec, context):
        if layout not in _LAYOUTS:
            msg = f'Config layout {layout!r} could not be found. Layout must be one of {list(_LAYOUTS)}.'
            raise ValidationError(msg, spec, 'layout')
        return layout

    @classmethod
    def _validate_template(cls, template, spec, context):
        if template in _TEMPLATES:
            return template
        elif '.' not in template:
            raise ValidationError(
                f'Config template {template!r} not found. Template must be one '
                'of {list(_TEMPLATES)} or an absolute import path.', spec, template
            )
        *paths, name = template.split('.')
        path = '.'.join(paths)
        try:
            module = importlib.import_module(path)
        except Exception as e:
            raise ValidationError(
                f'Config template {path!r} module could not be imported '
                f'errored with: {e}.', spec, path
            )
        if not hasattr(module, name):
            raise ValidationError(
                f'Config template {name!r} was not found in {path!r} module.',
                spec, name
            )
        template_cls = getattr(module, name)
        if not issubclass(template_cls, BasicTemplate):
            raise ValidationError(
                f'Config template \'{path}.{name}\' is not a valid Panel template.',
                spec, 'template'
            )
        return template

    @classmethod
    def _validate_theme(cls, theme, spec, context):
        if theme not in _THEMES:
            msg = f'Config theme {theme!r} could not be found. Theme must be one of {list(_THEMES)}.'
            raise ValidationError(msg, spec, 'theme')
        return theme

    @classmethod
    def from_spec(cls, spec):
        if 'template' in spec:
            template = spec['template']
            if template in _TEMPLATES:
                template_cls = _TEMPLATES[template]
            else:
                template_cls = resolve_module_reference(template, BasicTemplate)
            spec['template'] = template_cls
        if 'theme' in spec:
            spec['theme'] = _THEMES[spec['theme']]
        if 'layout' in spec:
            spec['layout'] = _LAYOUTS[spec['layout']]
        return cls(**spec)

    def to_spec(self, context=None):
        spec = super().to_spec(context=context)
        if 'layout' in spec:
            spec['layout'] = {v: k for k, v in _LAYOUTS.items()}[spec['layout']]
        if 'template' in spec:
            tmpl = spec['template']
            spec['template'] = {v: k for k, v in _TEMPLATES.items()}.get(tmpl, f'{tmpl.__module__}.{tmpl.__name__}')
        if 'theme' in spec:
            spec['theme'] = {v: k for k, v in _THEMES.items()}[spec['theme']]
        return spec

    def __init__(self, **params):
        super().__init__(**params)
        pn.config.param.update(
            loading_spinner=self.loading_spinner,
            loading_color=self.loading_color
        )

    def construct_template(self):
        params = {'title': self.title, 'theme': self.theme}
        if self.logo:
            params['logo'] = self.config.logo
        return self.template(**params)


class Defaults(Component):
    """
    `Defaults` to apply to the component classes.
    """

    filters = param.List(doc="Defaults for Filter objects.", class_=dict)

    sources = param.List(doc="Defaults for Source objects.", class_=dict)

    transforms = param.List(doc="Defaults for Transform objects.", class_=dict)

    views = param.List(doc="Defaults for View objects.", class_=dict)

    _valid_keys = 'params'
    _validate_params = True

    @classmethod
    def _validate_defaults(cls, defaults_type, defaults, spec):
        highlight = f'{defaults_type.__name__.lower()}s'
        if not isinstance(defaults, list):
            raise ValidationError(
                f'Defaults must be defined as a list not as {type(defaults)}.',
                spec, highlight
            )
        for default in defaults:
            if 'type' not in default:
                raise ValidationError(
                    'Defaults must declare the type the defaults apply to.',
                    spec, highlight
                )
            default = dict(default)
            dtype = default.pop('type')
            obj_type = defaults_type._get_type(dtype)
            for p in default:
                if p in obj_type.param:
                    pobj = obj_type.param[p]
                    try:
                        pobj._validate(default[p])
                    except Exception as e:
                        msg = f"The default for {obj_type.__name__} {p!r} parameter failed validation: {str(e)}"
                        raise ValidationError(msg, default, p)
                else:
                    msg = (
                        f'Default for {obj_type.__name__} {p!r} parameter cannot be set '
                        'as there is no such parameter.'
                    )
                    msg = match_suggestion_message(p, list(obj_type.param), msg)
                    raise ValidationError(msg, default, p)
        return defaults

    @classmethod
    def _validate_filters(cls, filter_defaults, spec, context):
        return cls._validate_defaults(Filter, filter_defaults, spec)

    @classmethod
    def _validate_sources(cls, source_defaults, spec, context):
        return cls._validate_defaults(Source, source_defaults, spec)

    @classmethod
    def _validate_transforms(cls, transform_defaults, spec, context):
        return cls._validate_defaults(Transform, transform_defaults, spec)

    @classmethod
    def _validate_views(cls, view_defaults, spec, context):
        return cls._validate_defaults(View, view_defaults, spec)

    def apply(self):
        for obj, defaults in ((Filter, self.filters), (Source, self.sources),
                              (Transform, self.transforms), (View, self.views)):
            for default in defaults:
                params = dict(default)
                obj_type = obj._get_type(params.pop('type'))
                obj_type.param.set_param(**params)


class Auth(Component):
    """
    `Auth` allows specifying a spec that is validated against panel.state.user_info.

    To enable `Auth` you must configure an OAuth provider when deploying
    an application with `panel.serve`.
    """

    case_sensitive = param.Boolean(default=False, constant=True, doc="""
        Whether auth validation is case-sensitive or not.""")

    spec = param.Dict({}, constant=True, doc="""
        Dictionary of keys and values to match the pn.state.user_info
        against.""")

    _allows_refs = False

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


class Dashboard(Component):

    auth = param.ClassSelector(default=Auth(), class_=Auth, doc="""
        Auth object which validates the auth spec against pn.state.user_info.""")

    config = param.ClassSelector(default=Config(), class_=Config, doc="""
        High-level configuration options for the dashboard.""")

    defaults = param.ClassSelector(default=Defaults(), class_=Defaults, doc="""
        Defaults to apply to component classes.""")

    targets = param.List(default=[], item_type=Target, doc="""
        List of targets monitoring some source.""")

    # Specification configuration
    _allows_refs = False

    _valid_keys = ['variables', 'auth', 'defaults', 'config', 'sources', 'pipelines', 'targets']

    def __init__(self, specification=None, **params):
        self._load_global = params.pop('load_global', True)
        if isinstance(specification, dict):
            state.spec = self.validate(specification)
            self._yaml = yaml.dump(specification)
            self._yaml_file = 'local'
            root = params.pop('root', os.getcwd())
        else:
            self._yaml_file = specification
            root = os.path.abspath(os.path.dirname(self._yaml_file))
        self._root = config.root = root
        self._edited = False
        self._debug = params.pop('debug', False)
        self._rerender_watchers = {}
        super().__init__(**params)
        self._init_config()

        # Initialize from spec
        config.load_local_modules()
        if isinstance(specification, dict):
            state.resolve_views()
        else:
            self._load_specification(from_file=True)
        self._apps = {}

        # Set up background loading
        self._thread_pool = None
        self._rendered = []

        # Initialize high-level settings
        self.auth = Auth.from_spec(state.spec.get('auth', {}))
        self.config = Config.from_spec(state.spec.get('config', {}))
        self.defaults = Defaults.from_spec(state.spec.get('defaults', {}))
        self.variables = Variables.from_spec(state.spec.get('variables', {}))
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

    def _init_config(self):
        pn.config.notifications = True
        if CSS_URLS['font-awesome'] not in pn.config.css_files:
            pn.config.css_files.append(CSS_URLS['font-awesome'])

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
        if self._yaml_file == 'local':
            return
        kwargs = {}
        if from_file or self._yaml is None:
            with open(self._yaml_file) as f:
                self._yaml = f.read()
        elif self._edited:
            kwargs = {'getenv': False, 'getshell': False, 'getoauth': False}
        state.spec = self.validate(load_yaml(self._yaml, **kwargs))
        state.resolve_views()

    def _load_target(self, target_spec):
        target_spec = dict(target_spec)
        if 'reloadable' not in target_spec:
            target_spec['reloadable'] = self.config.reloadable
        if 'auto_update' not in target_spec:
            target_spec['auto_update'] = self.config.auto_update
        target = Target.from_spec(target_spec)
        self._rerender_watchers[target] = target.param.watch(
            self._render_targets, 'rerender'
        )
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
        if force or self._load_global or not state.pipelines:
            state.load_pipelines(auto_update=self.config.auto_update)
        if not self.auth.authorized:
            self.targets = []
            return

        # Clean up old targets
        for target in self.targets:
            if isinstance(target, Target) and target in self._rerender_watchers:
                target.param.unwatch(self._rerender_watchers[target])
                del self._rerender_watchers[target]

        targets = []
        target_specs = state.spec.get('targets', [])
        ntargets = len(target_specs)
        if isinstance(self._layout, pn.Tabs) and self.config.background_load:
            self._thread_pool = ThreadPoolExecutor(max_workers=ntargets-1)
        for i, target_spec in enumerate(target_specs):
            if state.loading_msg:
                state.loading_msg.object = (
                    f"Loading target {target_spec['title']!r} ({i + 1}/{ntargets})..."
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
        if len(config.yamls) > 1:
            self._reload_button.margin = (7, 0, 0, 0)
            self._menu_button = pn.widgets.MenuButton(
                name='Select dashboard', items=menu_items, width=200,
                align='center', margin=(0, 25, 0, 0)
            )
            self._menu_button.on_click(self._navigate)
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

    @catch_and_notify
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
            for pipeline in target._pipelines.values():
                for filt in pipeline.filters:
                    if ((all(isinstance(target, (type(None), Future)) or any(filt in p.filters for p in target._pipelines.values())
                             for target in self.targets) and
                         filt.panel is not None) and filt.shared) and filt not in filters:
                        views.append(filt.panel)
                        filters.append(filt)
        if not self.config.auto_update:
            button = pn.widgets.Button(name='Apply Update')
            def update_pipelines(event):
                for target in self.targets:
                    if target is None or isinstance(target, Future):
                        continue
                    for pipeline in target._pipelines.values():
                        pipeline.param.trigger('update')
            button.on_click(update_pipelines)
            views.append(button)
        if not views or len(self.targets) == 1:
            return None, None
        return filters, pn.Column(*views, name='Filters', sizing_mode='stretch_width')

    def _render_filters(self):
        self._global_filters, global_panel = self._get_global_filters()
        filters = [] if global_panel is None else [global_panel]
        global_refs = [ref.split('.')[1] for ref in state.global_refs]
        self._variable_panel = self.variables.panel(global_refs)
        apply_button = not self.config.auto_update or len(self.targets) == 1
        if self._variable_panel is not None:
            filters.append(self._variable_panel)
        for i, target in enumerate(self.targets):
            if isinstance(self._layout, pn.Tabs) and i != self._layout.active:
                continue
            elif target is None or isinstance(target, Future):
                spec = state.spec['targets'][i]
                panel = pn.Column(name=spec['title'])
            else:
                panel = target.get_filter_panel(skip=self._global_filters, apply_button=apply_button)
            if panel is not None:
                filters.append(panel)
        self._sidebar[:] = filters

    def _render_targets(self, event=None):
        if event is not None:
            self._set_loading(event.obj.title)
        items = []
        for target, spec in zip(self.targets, state.spec.get('targets', [])):
            if target is None or isinstance(target, Future):
                panel = pn.Column(name=spec['title'])
            else:
                panel = target.panels
            items.append(panel)
        self._layout[:] = items
        self._layout.loading = False

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
                count = 1 + bool(self._global_filters) + bool(self._variable_panel)
                active = list(range(count))
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
    # Validation API
    ##################################################################

    @classmethod
    def _validate_pipelines(cls, pipeline_specs, spec, context):
        if 'pipelines' not in context:
            context['pipelines'] = {}
        return cls._validate_dict_subtypes('pipelines', Pipeline, pipeline_specs, spec, context, context['pipelines'])

    @classmethod
    def _validate_sources(cls, source_specs, spec, context):
        if 'sources' not in context:
            context['sources'] = {}
        return cls._validate_dict_subtypes('sources', Source, source_specs, spec, context, context['sources'])

    @classmethod
    def _validate_targets(cls, target_specs, spec, context):
        if 'targets' not in context:
            context['targets'] = []
        return cls._validate_list_subtypes('targets', Target, target_specs, spec, context, context['targets'])

    @classmethod
    def _validate_variables(cls, variable_specs, spec, context):
        if 'variables' not in context:
            context['variables'] = {}
        return cls._validate_dict_subtypes('variables', Variable, variable_specs, spec, context, context['variables'])

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def validate(cls, spec):
        """
        Validates the component specification given the validation context.

        Arguments
        -----------
        spec: dict
          The specification for the component being validated.

        Returns
        --------
        Validated specification.
        """
        cls._validate_keys(spec)
        cls._validate_required(spec)
        return cls._validate_spec(spec)

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

    def to_spec(self):
        """
        Exports the full specification to reconstruct this component.

        Returns
        -------
        Declarative specification of this component.
        """
        spec = {}
        spec['variables'] = {
            name: variable.to_spec(context=spec)
            for name, variable in state.variables._vars.items()
        }
        spec['sources'] = {
            name: source.to_spec(context=spec)
            for name, source in state.sources.items()
        }
        spec['pipelines'] = {
            name: pipeline.to_spec(context=spec)
            for name, pipeline in state.pipelines.items()
        }
        spec.update(super().to_spec(context=spec))
        spec['targets'] = [
            tspec if target is None else target
            for target, tspec in zip(spec['targets'], state.spec['targets'])
        ]
        return {k: v for k, v in spec.items() if v != {}}

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
