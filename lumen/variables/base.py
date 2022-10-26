import os

from functools import partial

import panel as pn
import param

from panel.widgets import Widget as _PnWidget

from ..base import MultiTypeComponent
from ..state import state
from ..util import is_ref, resolve_module_reference

_PARAM_MAP = {
    dict : param.Dict,
    float: param.Number,
    list : param.List,
    str  : param.String
}


class Variables(param.Parameterized):
    """
    `Variables` stores a number Variable types and mirrors their
    values onto dynamically created parameters allowing other
    components to easily watch changes in a variable.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._vars = {}
        self._watchers = {}

    @classmethod
    def from_spec(cls, spec):
        variables = cls()
        if pn.state.curdoc:
            state._variables[pn.state.curdoc] = variables
        for name, var_spec in spec.items():
            if not isinstance(var_spec, dict):
                var_spec = {
                    'type': 'constant',
                    'default': var_spec
                }
            var = Variable.from_spec(dict(var_spec, name=name), variables)
            variables.add_variable(var)
        return variables

    def add_variable(self, var):
        """
        Adds a new variable to the Variables instance and sets up
        a parameter that can be watched.
        """
        if var.name in self._vars:
            variable = self._vars.pop(var.name)
            variable.param.unwatch(self._watchers.pop(var.name))
        self._vars[var.name] = var
        self.param.add_parameter(var.name, param.Parameter(default=var.value))
        self._watchers[var.name] = var.param.watch(partial(self._update_value, var.name), 'value')

    def _update_value(self, name, event):
        self.param.update({name: event.new})

    def __getitem__(self, key):
        if key in self.param:
            return getattr(self, key)
        else:
            raise KeyError(f'No variable named {key!r} has been defined.')

    def panel(self, variables=None):
        if variables == []:
            return None
        column = pn.Column(name='Variables', sizing_mode='stretch_width')
        for key, var in self._vars.items():
            if variables is None or key in variables:
                var_panel = var.panel
                if var_panel is not None:
                    column.append(var_panel)
        return column


class Variable(MultiTypeComponent):
    """
    `Variable` components declare values that can be referenced from other components.

    The source of the `Variable` value can be anything from an
    environment variable to a widget or URL parameter. `Variable`
    components allow a concise way to configure other components and
    make it possible to orchestrate actions across multiple
    components.
    """

    default = param.Parameter(doc="""
       Default value to use if no other value is defined""")

    label = param.String(default=None, doc="""
       Optional label override for variable. Used wherever the variable
       shows up in the UI.""")

    materialize = param.Boolean(default=False, constant=True, doc="""
       Whether the variable should be inlined as a constant variable
       (in the Lumen Builder).""")

    required = param.Boolean(default=False, constant=True, doc="""
       Whether the variable should be treated as required.""")

    secure = param.Boolean(default=False, constant=True, doc="""
       Whether the variable should be treated as secure.""")

    value = param.Parameter(doc="""
       The materialized value of the variable.""")

    variable_type = None

    __abstract = True

    _valid_keys = 'params'
    _validate_params = True

    def __init__(self, **params):
        if 'value' not in params and 'default' in params:
            params['value'] = params['default']
        if 'label' not in params and 'name' in params:
            params['label'] = params['name']
        super().__init__(**params)
        self.param.watch(self._update_value_from_default, 'default')

    def _update_value_from_default(self, event):
        if event.old is self.value or self.value is self.param.value.default:
            self.value = event.new

    @classmethod
    def from_spec(cls, spec, variables=None):
        if isinstance(spec, dict):
            spec = dict(spec)
            var_type = spec.pop('type')
        else:
            var_type = 'constant'
            spec = {'default': spec}
        var_type = cls._get_type(var_type)
        resolved_spec, refs = {}, {}
        for k, val in spec.items():
            if is_ref(val):
                refs[k] = val
                val = state.resolve_reference(val, variables)
                if isinstance(val, Variable):
                    val = val.value
            resolved_spec[k] = val
        return var_type(refs=refs, **resolved_spec)

    def as_materialized(self):
        """
        If the variable is to be materialized by the builder this
        implements the conversion from a variable that references
        some external value to a materialized value.
        """
        return Constant(default=self.value)

    @property
    def panel(self):
        """
        Optionally returns a renderable component that allows
        controlling the variable.
        """
        return None


class Constant(Variable):
    """
    `Constant` declares a constant value that can be referenced.
    """

    variable_type = 'constant'


class EnvVariable(Variable):
    """
    `EnvVariable` fetches an environment variable that can be referenced.
    """

    key = param.String(default=None, doc="""
        The name of the environment variable to look up.""")

    variable_type = 'env'

    def __init__(self, **params):
        super().__init__(**params)
        key = self.key or self.name
        if key in os.environ:
            self.value = os.environ[key]


class Widget(Variable):
    """
    `Widget` variables dynamically reflect the current widget value.
    """

    kind = param.String(default=None, doc="""
       The type of widget to instantiate.""")

    throttled = param.Boolean(default=True, doc="""
       If the widget has a value_throttled parameter use that instead,
       ensuring that no intermediate events are generated, e.g. when
       dragging a slider.""")

    variable_type = 'widget'

    _valid_keys = None

    _required_keys = ["kind"]

    def __init__(self, **params):
        default = params.pop('default', None)
        refs = params.pop('refs', {})
        throttled = params.pop('throttled', True)
        label = params.pop('label', None)
        kind = params.pop('kind', None)
        widget = params.pop('widget', None)
        super().__init__(
            default=default, refs=refs, name=params.get('name'), label=label,
            throttled=throttled, kind=kind,
        )
        if '.' in self.kind:
            widget_type = resolve_module_reference(self.kind, _PnWidget)
        else:
            widget_type = getattr(pn.widgets, self.kind)
        if 'value' not in params and default is not None:
            params['value'] = default
        if self.label:
            params['name'] = self.label
        deserialized = {}
        for k, v in params.items():
            if k in widget_type.param:
                try:
                    v = widget_type.param[k].deserialize(v)
                except Exception:
                    pass
            deserialized[k] = v
        if widget:
            self._widget = widget
        else:
            self._widget = widget_type(**deserialized)
        if self.throttled and 'value_throttled' in self._widget.param:
            self._widget.link(self, value_throttled='value')
            self.param.watch(lambda e: self._widget.param.update({'value': e.new}), 'value')
        else:
            self.value = self._widget.value
            self._widget.link(self, value='value', bidirectional=True)

    @property
    def panel(self):
        return self._widget


class Parameter(Variable):
    """
    `Parameter` variables reflect the current value of a parameter.
    """

    parameter = param.ClassSelector(class_=param.Parameter, doc="""
        A parameter instance whose current value will be reflected
        on this variable.""")

    def __init__(self, **params):
        super().__init__(**params)
        self.param.parameter.owner.watch(lambda e: self.param.update(value=e.new), self.param.parameter.name)


class URLQuery(Variable):
    """
    `URLQuery` variables reflect the value of a URL query parameter.
    """

    key = param.String(default=None)

    variable_type = 'url'

    def __init__(self, **params):
        super().__init__(**params)
        key = self.key or self.name
        if pn.state.location:
            pn.state.location.param.watch(self._update_value, 'search')
            self._update_value()
        if pn.state.session_args and key in pn.state.session_args:
            self.value = pn.state.session_args[key][0].decode('utf-8')

    def _update_value(self, *args):
        key = self.key or self.name
        self.value = pn.state.location.query_params.get(key, self.default)


class Cookie(Variable):
    """
    `Cookie` variables reflect the value of a cookie in the request.
    """

    key = param.String(default=None)

    variable_type = 'cookie'

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.cookies:
            key = self.key or self.name
            self.value = pn.state.cookies.get(key, self.default)


class UserInfo(Variable):
    """
    `UserInfo` variables reflect a value in the user info returned by an OAuth provider.
    """

    key = param.String(default=None)

    variable_type = 'user'

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.user_info:
            key = self.key or self.name
            self.value = pn.state.user_info.get(key, self.default)


class Header(Variable):
    """
    `Header` variables reflect the value of a request header.
    """

    key = param.String(default=None)

    variable_type = 'header'

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.headers:
            key = self.key or self.name
            self.value = pn.state.headers.get(key, self.default)
