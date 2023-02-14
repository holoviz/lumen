from __future__ import annotations

import html
import importlib
import importlib.util
import os
import traceback

from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Tuple, Type,
)

import panel as pn
import param  # type: ignore
import yaml

from panel.io.resources import CSS_URLS
from panel.template.base import BasicTemplate
from panel.viewable import Viewable, Viewer
from typing_extensions import Literal

from .auth import Auth
from .base import Component, MultiTypeComponent
from .config import (
    _DEFAULT_LAYOUT, _LAYOUTS, _TEMPLATES, _THEMES, config,
)
from .filters.base import ConstantFilter, Filter, WidgetFilter  # noqa
from .layout import Layout
from .panel import IconButton
from .pipeline import Pipeline
from .sources.base import RESTSource, Source  # noqa
from .state import state
from .transforms.base import Transform  # noqa
from .util import (
    bokeh3, catch_and_notify, expand_spec, resolve_module_reference,
)
from .validation import (
    ValidationError, match_suggestion_message, validate_callback,
)
from .variables.base import Variable, Variables
from .views.base import Download, View  # noqa

if TYPE_CHECKING:
    from bokeh.server.contexts import BokehSessionContext


def load_yaml(yaml_spec: str, **kwargs) -> Dict[str, Any]:
    expanded = expand_spec(yaml_spec, config.template_vars, **kwargs)
    return yaml.load(expanded, Loader=yaml.Loader)


class Config(Component):
    """
    `Config` provides high-level configuration options for the
    :class:`lumen.dashboard.Dashboard`.
    """

    auto_update = param.Boolean(default=None, constant=True, doc="""
        Whether changes in filters, transforms and references automatically
        trigger updates in the data or whether an update has to be triggered
        manually using the update event or the update button in the UI."""
    )

    background_load = param.Boolean(default=False, constant=True, doc="""
        Whether to load any layouts in the background.""")

    editable = param.Boolean(default=False, constant=True, doc="""
        Whether the dashboard specification is editable from within
        the deployed dashboard.""")

    layout = param.Selector(default=_DEFAULT_LAYOUT, constant=True, objects=_LAYOUTS,
                            check_on_set=False, doc="""
        Overall layout of the dashboard.""")

    loading_spinner = param.Selector(default='arc', constant=True, objects=[
        'arc', 'arcs', 'bar', 'dots', 'petal'], doc="""
        Loading indicator to use when component loading parameter is set.""")

    loading_color = param.Color(default='#0072B5', constant=True, doc="""
        Color of the loading indicator.""")

    logo = param.String(default=None, constant=True, doc="""
        A logo to add to the theme.""")

    ncols = param.Integer(default=3, bounds=(1, None), constant=True, doc="""
        Number of columns to lay out layouts in.""")

    on_session_created = param.Callable(constant=True, doc="""
        Callback that fires when a user session is created.""")

    on_session_destroyed = param.Callable(constant=True, doc="""
        Callback that fires when a user session is destroyed.""")

    on_loaded = param.Callable(constant=True, doc="""
        Callback that fires when a user frontend session is fully loaded.""")

    on_error = param.Callable(constant=True, doc="""
        Callback that fires if an error occurs in a dashboard callback.
        The exception is passed as the first argument.""")

    on_update = param.Callable(constant=True, doc="""
        Callback that fires when a pipeline is updated. The updated
        pipeline is passed as the first argument.""")

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

    _valid_keys: ClassVar[Literal['params']] = 'params'
    _validate_params: ClassVar[bool] = True

    @classmethod
    def _validate_layout(cls, layout, spec, context):
        if layout not in _LAYOUTS:
            msg = f'Config layout {layout!r} could not be found. Layout must be one of {list(_LAYOUTS)}.'
            raise ValidationError(msg, spec, 'layout')
        return layout

    @classmethod
    def _validate_template(
        cls, template: str, spec: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        if template in _TEMPLATES:
            return template
        elif '.' not in template:
            raise ValidationError(
                f'Config template {template!r} not found. Template must be one '
                f'of {list(_TEMPLATES)} or an absolute import path.', spec, template
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
    def _validate_callback(cls, callback: Callable[..., Any] | str) -> Callable[Any, None]:
        if isinstance(callback, str):
            return resolve_module_reference(callback)
        return callback

    @classmethod
    def _validate_on_error(
        cls, on_error: Callable[[Type[Exception]], None] | str,
        spec: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        cb = cls._validate_callback(on_error)
        validate_callback(cb, ('exception',), what='on_error callback')
        return cb

    @classmethod
    def _validate_on_loaded(
        cls, on_loaded: Callable[[], None] | str, spec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        cb = cls._validate_callback(on_loaded)
        validate_callback(cb, (), what='on_loaded callback')
        return cb

    @classmethod
    def _validate_on_update(
        cls, on_update: Callable[[Pipeline], None] | str, spec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        cb = cls._validate_callback(on_update)
        validate_callback(cb, ('pipeline',), what='on_update callback')
        return cb

    @classmethod
    def _validate_on_session_created(
        cls, on_session_created: Callable[[], None] | str, spec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        cb = cls._validate_callback(on_session_created)
        validate_callback(cb, (), what='on_session_created callback')
        return cb

    @classmethod
    def _validate_on_session_destroyed(
        cls, on_session_destroyed: Callable[[BokehSessionContext], None] | str,
        spec: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        cb = cls._validate_callback(on_session_destroyed)
        validate_callback(cb, ('session_context',), what='on_session_destroyed callback')
        return cb

    @classmethod
    def _validate_theme(
        cls, theme: str, spec: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        if theme not in _THEMES:
            msg = f'Config theme {theme!r} could not be found. Theme must be one of {list(_THEMES)}.'
            raise ValidationError(msg, spec, 'theme')
        return theme

    @classmethod
    def from_spec(cls, spec: Dict[str, Any] | str) -> 'Dashboard':
        if isinstance(spec, str):
            raise ValueError(
                "Config cannot be materialized by reference. Please pass "
                "full specification for the Config."
            )
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
        for key in list(spec):
            if key.startswith('on_'):
                spec[key] = getattr(cls, f'_validate_{key}')(spec[key], spec, {})
        return cls(**spec)

    def to_spec(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        spec = super().to_spec(context=context)
        if 'layout' in spec:
            spec['layout'] = {v: k for k, v in _LAYOUTS.items()}[spec['layout']]
        if 'template' in spec:
            tmpl = spec['template']
            spec['template'] = {v: k for k, v in _TEMPLATES.items()}.get(tmpl, f'{tmpl.__module__}.{tmpl.__name__}')
        if 'theme' in spec:
            spec['theme'] = {v: k for k, v in _THEMES.items()}[spec['theme']]
        for key in list(spec):
            if key.startswith('on_'):
                cb = spec[key]
                ref = f'{cb.__module__}.{cb.__name__}'
                if ref.startswith('__main__'):
                    raise ValueError(
                        f"Cannot serialize 'config.{key}' callback because "
                        "it was defined in the __main__ scope. Please move the "
                        "callback into an importable module."
                    )
                spec[key] = ref
        return spec

    def __init__(self, **params):
        super().__init__(**params)
        pn.config.param.update(
            loading_spinner=self.loading_spinner,
            loading_color=self.loading_color
        )
        if self.on_session_destroyed:
            try:
                pn.state.on_session_destroyed(self.on_session_destroyed)
            except RuntimeError:
                pass

    def construct_template(self):
        params = {'title': self.title, 'theme': self.theme}
        if self.logo:
            params['logo'] = self.config.logo
        return self.template(**params)


class Defaults(Component):
    """
    `Defaults` to apply to the component classes.
    """

    download = param.Dict(default={}, doc="Defaults for the Download object")

    filters = param.List(doc="Defaults for Filter objects.", class_=dict)

    sources = param.List(doc="Defaults for Source objects.", class_=dict)

    transforms = param.List(doc="Defaults for Transform objects.", class_=dict)

    views = param.List(doc="Defaults for View objects.", class_=dict)

    _valid_keys: ClassVar[Literal['params']] = 'params'
    _validate_params: ClassVar[bool] = True

    @classmethod
    def _validate_defaults(
        cls, defaults_type: Type[MultiTypeComponent], defaults: List[Dict[str, Any]], spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
    def _validate_download(
        cls, download_defaults: Dict[str, Any], spec: Dict[str, Any], context: Dict[str, Any]
    ):
        if not isinstance(download_defaults, dict):
            msg = f'Defaults for Download component must be declared as a dictionary, not as a {type(download_defaults)}.'
            raise ValidationError(msg, spec, 'download:')
        for p in download_defaults:
            if p in Download.param:
                pobj = Download.param[p]
                try:
                    pobj._validate(download_defaults[p])
                except Exception as e:
                    msg = f"The default for Download {p!r} parameter failed validation: {str(e)}"
                    raise ValidationError(msg, download_defaults, p)
                continue
            msg = (
                f'Default for Download {p!r} parameter cannot be set as there '
                'is no such parameter.'
            )
            msg = match_suggestion_message(p, list(Download.param), msg)
            raise ValidationError(msg, download_defaults, p)
        return download_defaults

    @classmethod
    def _validate_filters(
        cls, filter_defaults: List[Dict[str, Any]], spec: Dict[str, Any], context: Dict[str, Any]
    ):
        return cls._validate_defaults(Filter, filter_defaults, spec)

    @classmethod
    def _validate_sources(
        cls, source_defaults: List[Dict[str, Any]], spec: Dict[str, Any], context: Dict[str, Any]
    ):
        return cls._validate_defaults(Source, source_defaults, spec)

    @classmethod
    def _validate_transforms(
        cls, transform_defaults: List[Dict[str, Any]], spec: Dict[str, Any], context: Dict[str, Any]
    ):
        return cls._validate_defaults(Transform, transform_defaults, spec)

    @classmethod
    def _validate_views(
        cls, view_defaults: List[Dict[str, Any]], spec: Dict[str, Any], context: Dict[str, Any]
    ):
        return cls._validate_defaults(View, view_defaults, spec)

    def apply(self):
        Download.param.update(self.download)
        for obj, defaults in ((Filter, self.filters), (Source, self.sources),
                              (Transform, self.transforms), (View, self.views)):
            for default in defaults:
                params = dict(default)
                obj_type = obj._get_type(params.pop('type'))
                obj_type.param.set_param(**params)


class AuthSpec(Component):
    """
    `AuthSpec` allows specifying a spec that is validated against panel.state.user_info.

    To enable `AuthSpec` you must configure an OAuth provider when deploying
    an application with `panel.serve`.
    """

    case_sensitive = param.Boolean(default=False, constant=True, doc="""
        Whether auth validation is case-sensitive or not.""")

    spec = param.Dict({}, constant=True, doc="""
        Dictionary of keys and values to match the pn.state.user_info
        against.""")

    _allows_refs: ClassVar[bool] = False

    @classmethod
    def from_spec(cls, spec: Dict[str, Any] | str) -> 'AuthSpec':
        if isinstance(spec, str):
            raise ValueError(
                "AuthSpec cannot be materialized by reference. Please pass "
                "full specification for AuthSpec."
            )
        spec = dict(spec)
        case_sensitive = spec.pop('case_sensitive', cls.case_sensitive)
        plugins = spec.pop('plugins', [])
        for plugin_spec in plugins:
            plugin = Auth.from_spec(plugin_spec)
            spec = plugin.transform(spec)
        return cls(spec=spec, case_sensitive=case_sensitive)

    @property
    def authorized(self) -> bool:
        if not self.spec:
            return True
        elif pn.state.user_info is None:
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

    @classmethod
    def validate(
        cls, spec: Dict[str, Any] | str, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any] | str:
        """
        Validates the component specification given the validation context.

        Arguments
        -----------
        spec: dict | str
          The specification for the component being validated (or a referene to the component)
        context: dict
          Validation context contains the specification of all previously validated components,
          e.g. to allow resolving of references.

        Returns
        --------
        Validated specification.
        """
        spec = super().validate(spec, context)
        plugins = [
            Auth.validate(plugin, context) for plugin in spec.get('plugins', [])
        ]
        if plugins:
            spec['plugins'] = plugins
        return spec


class Dashboard(Component, Viewer):

    auth = param.ClassSelector(default=AuthSpec(), class_=AuthSpec, doc="""
        AuthSpec object which validates the auth spec against pn.state.user_info.""")

    config = param.ClassSelector(default=Config(), class_=Config, doc="""
        High-level configuration options for the dashboard.""")

    defaults = param.ClassSelector(default=Defaults(), class_=Defaults, doc="""
        Defaults to apply to component classes.""")

    layouts = param.List(default=[], item_type=Layout, doc="""
        List of layouts monitoring some source.""")

    # Specification configuration
    _allows_refs = False

    _valid_keys = [
        'variables', 'auth', 'defaults', 'config', 'sources',
        'pipelines', 'layouts'
    ]

    def __init__(self, specification=None, **params):
        self._load_global = params.pop('load_global', True)
        for subobj in (Auth, Config, Defaults):
            pname = subobj.__name__.lower()
            if pname in params and isinstance(params[pname], dict):
                params[pname] = subobj.from_spec(params[pname])
        if specification is None:
            specification = state.to_spec(**params)
            load_vars = False
        else:
            load_vars = True
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
        self.auth = AuthSpec.from_spec(state.spec.get('auth', {}))
        self.config = Config.from_spec(state.spec.get('config', {}))
        self.defaults = Defaults.from_spec(state.spec.get('defaults', {}))
        if load_vars:
            self.variables = Variables.from_spec(state.spec.get('variables', {}))
        else:
            self.variables = state.variables
        self.defaults.apply()

        # Load and populate template
        self._template = self.config.construct_template()
        self._create_modal()
        self._create_main()
        self._create_sidebar()
        self._create_header()
        self._populate_template()

        state.config = self.config
        if pn.state._is_pyodide:
            self._render_dashboard()
            if self.config.on_loaded:
                pn.state.execute(self.config.on_loaded)
        else:
            pn.state.onload(self._render_dashboard)
            if self.config.on_loaded:
                pn.state.onload(self.config.on_loaded)
        if self.config.on_session_created:
            pn.state.execute(self.config.on_session_created)

    def _init_config(self):
        pn.config.notifications = True
        if CSS_URLS['font-awesome'] not in pn.config.css_files:
            pn.config.css_files.append(CSS_URLS['font-awesome'])

    def __panel__(self) -> Viewable:
        return self.layout()

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
            pn.state.location.sync(self._layout, {'active': 'layout'})

    def _load_specification(self, from_file: bool = False):
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

    def _load_layout(self, layout_spec: Dict[str, Any]) -> Layout:
        layout_spec = dict(layout_spec)
        if 'reloadable' not in layout_spec:
            layout_spec['reloadable'] = self.config.reloadable
        if 'auto_update' not in layout_spec and self.config.auto_update is not None:
            layout_spec['auto_update'] = self.config.auto_update
        layout = Layout.from_spec(layout_spec)
        self._rerender_watchers[layout] = layout.param.watch(
            self._render_layouts, 'rerender'
        )
        if isinstance(self._layout, pn.Tabs):
            layout.show_title = False
        layout.start()
        return layout

    def _background_load(self, i: int, spec: Dict[str, Any]) -> Layout:
        self.layouts[i] = layout = self._load_layout(spec)
        return layout

    def _materialize_specification(self, force: bool = False):
        if force or self._load_global or not state.global_sources:
            state.load_global_sources(clear_cache=force)
        if force or self._load_global or not state.pipelines:
            kwargs = {}
            if self.config.auto_update is not None:
                kwargs['auto_update'] = self.config.auto_update
            state.load_pipelines(**kwargs)
        if not self.auth.authorized:
            self.layouts = []
            return

        # Clean up old layouts
        for layout in self.layouts:
            if isinstance(layout, Layout) and layout in self._rerender_watchers:
                layout.param.unwatch(self._rerender_watchers[layout])
                del self._rerender_watchers[layout]

        layouts = []
        layout_specs = state.spec.get('layouts', [])
        nlayouts = len(layout_specs)
        if isinstance(self._layout, pn.Tabs) and self.config.background_load:
            self._thread_pool = ThreadPoolExecutor(max_workers=nlayouts-1)
        for i, layout_spec in enumerate(layout_specs):
            if state.loading_msg:
                state.loading_msg.object = (
                    f"Loading layout {layout_spec['title']!r} ({i + 1}/{nlayouts})..."
            )
            if isinstance(self._layout, pn.Tabs) and i != self._layout.active:
                if self.config.background_load:
                    item = self._thread_pool.submit(self._background_load, i, layout_spec)
                else:
                    item = None
                self._rendered.append(False)
            else:
                item = self._load_layout(layout_spec)
                self._rendered.append(True)
            layouts.append(item)
        self.layouts[:] = layouts
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
        if False:#self.config.editable:
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
        styles = {'text-align': 'center', 'font-size': '1.8em', 'font-weight': 'bold'}
        style_params = {'styles': styles} if bokeh3 else {'style': styles}
        state.loading_msg = pn.pane.HTML(
            'Loading...', align='center', width=400, height=400, **style_params
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
        self._editor = pn.panel('Editor')
        self._edit_button = pn.widgets.Button(
            name='✎', width=50, css_classes=['reload'], margin=0,
            align='center'
        )
        #self._editor.param.watch(self._edit, 'value')
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
        self._update_button = pn.widgets.Button(name='Apply Update')
        self._update_button.on_click(self._update_pipelines)

    def _populate_template(self):
        self._template.modal[:] = [self._modal]
        self._template.main[:] = [self._main]
        self._template.header[:] = [self._header]
        self._template.sidebar[:] = [self._sidebar]

    ##################################################################
    # Rendering API
    ##################################################################

    @catch_and_notify
    def _activate_filters(self, event: param.parameterized.Event):
        layout = self.layouts[event.new]
        rendered = self._rendered[event.new]
        if not rendered:
            spec = state.spec['layouts'][event.new]
            self._set_loading(spec['title'])
            try:
                if isinstance(layout, Future):
                    layout = layout.result()
                else:
                    layout = self._load_layout(spec)
            except Exception as e:
                self._main.loading = False
                raise e
            self.layouts[event.new] = layout
            self._layout[event.new] = layout.panels
            self._rendered[event.new] = True
        self._render_filters()
        self._main.loading = False

    def _edit(self, event: param.parameterized.Event):
        self._yaml = event.new
        self._edited = True

    def _navigate(self, event: param.parameterized.Event):
        if pn.state.location is None:
            return
        app = self._apps[event.new]
        pn.state.location.pathname = f'/{app}'
        pn.state.location.reload = True

    def _set_loading(self, name: str = ''):
        state.loading_msg.object = f'<b>Reloading {name}...</b>'
        if isinstance(self._layout, pn.GridBox):
            items = [pn.pane.HTML(width=self._layout[i].width) # type: ignore
                     for i in range(self._layout.ncols) if i < len(self._layout)]
            index = int(min(self._layout.ncols, (len(self._layout)-1)) / 2)
            if items:
                items[index] = self._loading
            else:
                items = [self._loading]
        elif isinstance(self._layout, pn.Tabs):
            items = list(zip(self._layout._names, list(self._layout.objects))) # type: ignore
            tab_name = items[self._layout.active][0]
            if name and tab_name != name:
                return
            items[self._layout.active] = (tab_name, self._loading)
        else:
            items = [self._loading]
        self._layout[:] = items
        self._main.loading = True

    def _open_modal(self, event: param.parameterized.Event):
        self._template.open_modal()

    def _get_global_filters(self) -> Tuple[List[Filter] | None, pn.Column | None]:
        views = []
        filters = []
        for layout in self.layouts:
            if layout is None or isinstance(layout, Future):
                continue
            for pipeline in layout._pipelines.values():
                for filt in pipeline.filters:
                    if ((all(isinstance(layout, (type(None), Future)) or any(filt in p.filters for p in layout._pipelines.values())
                             for layout in self.layouts) and
                         filt.panel is not None) and filt.shared) and filt not in filters:
                        views.append(filt.panel)
                        filters.append(filt)
        if not views or len(self.layouts) == 1:
            return None, None
        if not self.config.auto_update:
            views.append(self._update_button)
        return filters, pn.Column(*views, name='Filters', sizing_mode='stretch_width')

    def _update_pipelines(self, event: param.parameterized.Event | None = None):
        self._update_button.loading = True
        try:
            for layout in self.layouts:
                if layout is None or isinstance(layout, Future):
                    continue
                for pipeline in layout._pipelines.values():
                    pipeline.param.trigger('update')
        finally:
            self._update_button.loading = False

    def _render_filters(self):
        self._global_filters, global_panel = self._get_global_filters()
        filters = [] if global_panel is None else [global_panel]
        global_refs = [ref.split('.')[1] for ref in state.global_refs]
        self._variable_panel = self.variables.panel(global_refs)
        apply_button = not self.config.auto_update or len(self.layouts) == 1
        if self._variable_panel is not None:
            filters.append(self._variable_panel)
        for i, layout in enumerate(self.layouts):
            if isinstance(self._layout, pn.Tabs) and i != self._layout.active:
                continue
            elif layout is None or isinstance(layout, Future):
                spec = state.spec['layouts'][i]
                panel = pn.Column(name=spec['title'])
            else:
                panel = layout.get_filter_panel(skip=self._global_filters, apply_button=apply_button)
            if panel is not None:
                filters.append(panel)
        self._sidebar[:] = filters

    def _render_layouts(self, event: param.parameterized.Event | None = None):
        if event is not None:
            self._set_loading(event.obj.title)
        items = []
        for layout, spec in zip(self.layouts, state.spec.get('layouts', [])):
            if layout is None or isinstance(layout, Future):
                panel = pn.layout.VSpacer(name=spec['title'])
            else:
                panel = layout.panels
            items.append(panel)
        self._layout[:] = items
        self._main.loading = False

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
            self._render_layouts()
        else:
            self._render_unauthorized()
        self._main.loading = False

    def _reload(self, *events: param.parameterized.Event):
        self._load_specification()
        self._materialize_specification(force=True)
        self._render()

    ##################################################################
    # Validation API
    ##################################################################

    @classmethod
    def _validate_pipelines(
        cls, pipeline_specs: Dict[str, Any], spec: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if 'pipelines' not in context:
            context['pipelines'] = {}
        return cls._validate_dict_subtypes('pipelines', Pipeline, pipeline_specs, spec, context, context['pipelines'])

    @classmethod
    def _validate_sources(
        cls, source_specs: Dict[str, Any], spec: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if 'sources' not in context:
            context['sources'] = {}
        return cls._validate_dict_subtypes('sources', Source, source_specs, spec, context, context['sources'])

    @classmethod
    def _validate_layouts(
        cls, layout_specs: List[Dict[str, Any] | str], spec: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if 'layouts' not in context:
            context['layouts'] = []
        return cls._validate_list_subtypes('layouts', Layout, layout_specs, spec, context, context['layouts'])

    @classmethod
    def _validate_variables(
        cls, variable_specs: Dict[str, Any], spec: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if 'variables' not in context:
            context['variables'] = {}
        return cls._validate_dict_subtypes('variables', Variable, variable_specs, spec, context, context['variables'])

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def validate(
        cls, spec: Dict[str, Any] | str, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
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
        if isinstance(spec, str):
            raise ValueError(
                'Dashboard must be materialized with a full specification. '
                'Materializing a Dashboard from a string is not supported.'
            )
        if context is not None:
            raise ValueError(
                'Dashboard.validate is already the top-level context '
                'you may not pass a context argument.'
            )
        # Backward compatibility for old naming (targets -> layouts)
        if 'targets' in spec:
            spec['layouts'] = spec.pop('targets')

        cls._validate_keys_(spec)
        cls._validate_required_(spec)
        return cls._validate_spec_(spec)

    def layout(self):
        """
        Returns a layout of the dashboard contents.
        """
        spinner = pn.indicators.LoadingSpinner(width=30, height=30, align=('end', 'center'))
        pn.state.sync_busy(spinner)

        title_html = f'<div style="color: white; font-size: 2em;">{self.config.title}</div>'
        return pn.Column(
            pn.Row(
                pn.pane.HTML(title_html, margin=(20, 20), align='center'),
                pn.layout.HSpacer(),
                spinner,
                background='#00aa41', sizing_mode='stretch_width'
            ),
            pn.Row(
                pn.Column(self._sidebar, width=300),
                self._layout,
                sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_both'
        )

    def to_spec(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Exports the full specification to reconstruct this component.

        Returns
        -------
        Declarative specification of this component.
        """
        if context is not None:
            raise ValueError(
                'Dashboard.to_spec is already the top-level context '
                'you may not pass a context argument.'
            )
        spec: Dict[str, Any] = {}
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
        spec['layouts'] = [
            tspec if layout is None else layout
            for layout, tspec in zip(spec['layouts'], state.spec['layouts'])
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
