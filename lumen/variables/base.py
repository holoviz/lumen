from __future__ import annotations

import os

from functools import partial
from typing import TYPE_CHECKING, ClassVar, List

import panel as pn
import param  # type: ignore

from panel.widgets import Widget as _PnWidget
from typing_extensions import Literal

from ..base import MultiTypeComponent
from ..state import state
from ..util import is_ref, resolve_module_reference

if TYPE_CHECKING:
    from panel.viewable import Viewable

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

    _counter: ClassVar[int] = 0

    def __init__(self, **params):
        super().__init__(**params)
        self._vars = {}
        self._watchers = {}

    def keys(self):
        return self._vars.keys()

    @classmethod
    def create_variables(cls):
        new_cls = type(f'Variables{cls._counter}', (cls,), {})()
        cls._counter += 1
        return new_cls

    @classmethod
    def from_spec(cls, spec):
        variables = cls.create_variables()
        doc = pn.state.curdoc
        if doc and doc._session_context:
            state._variables[doc] = variables
        else:
            state._variable = variables
        for name, var_spec in spec.items():
            if not isinstance(var_spec, dict):
                var_spec = {
                    'type': 'constant',
                    'default': var_spec
                }
            var = Variable.from_spec(dict(var_spec, name=name), variables)
            variables.add_variable(var)
        return variables

    def _convert_to_variable(self, var: param.Parameter | _PnWidget) -> 'Variable':
        throttled = False
        if isinstance(var, param.Parameter):
            if isinstance(var.owner, pn.widgets.Widget) and var.name in ('value', 'value_throttled'):
                throttled = var.name == 'value_throttled'
                var = var.owner
            else:
                var_type = Parameter
                var_name = var.name
                var_kwargs = {'parameter': var, 'value': getattr(var.owner, var.name)}
        else:
            var_type = None

        if isinstance(var, _PnWidget):
            var_name = var.name
            widget_type = type(var).__name__
            if not var_name:
                raise ValueError(
                    f'Cannot use {widget_type} widget without an explicit '
                    'name as a reference. Ensure any Widget passed in as '
                    'a reference has been given a name.'
                )
            var_type = Widget
            extras = {k: v for k, v in var.param.values().items() if k != 'name'}
            var_kwargs = dict(
                kind=f'{var.__module__}.{widget_type}', throttled=throttled,
                widget=var, **extras
            )

        if var_type is None:
            raise ValueError(
                f'Could not convert object of type {type(var).__name__} '
                'to a Variable. Only panel.widgets.Widget and param.Parameter '
                'types may be automatically promoted to Variables.'
            )
        var_name = var_name.replace(' ', '_')
        old_var = self._vars.get(var_name)
        if (
            isinstance(old_var, var_type) and
            ((var_type is Widget and var is old_var._widget) or
             (var_type is Parameter and var is old_var.parameter))
        ):
            return old_var
        return var_type(name=var_name, **var_kwargs)

    def add_variable(self, var: 'Variable' | _PnWidget | param.Parameter) -> 'Variable':
        """
        Adds a new variable to the Variables instance and sets up
        a parameter that can be watched.

        Arguments
        ---------
        var: Variable | panel.widgets.Widget | param.Parameter
            The Variable to add to the Variables store.
        """
        if not isinstance(var, Variable):
            var = self._convert_to_variable(var)
        old_var = self._vars.get(var.name)
        if old_var:
            if var is old_var:
                return var
            elif old_var and type(var) is type(old_var):
                vtname = type(var).__name__
                self.param.warning(
                    f'A {vtname} named {var.name!r} has already been defined and '
                    'is linked to a different source. Overriding the existing '
                    'variable and unlinking any attached components.'
                )
            else:
                self.param.warning(
                    f'A Variable named {var.name!r} of a different type has '
                    'already been defined. Overriding the existing variable '
                    'and unlinking any attached components.'
                )
            variable = self._vars.pop(var.name)
            variable.param.unwatch(self._watchers.pop(var.name))
        self._vars[var.name] = var
        self.param.add_parameter(var.name, param.Parameter(default=var.value))
        self.param.update(**{var.name: var.value})
        self._watchers[var.name] = var.param.watch(
            partial(self._update_value, var.name), 'value'
        )
        return var

    def _update_value(self, name: str, event: param.parameterized.Event):
        self.param.update({name: event.new})

    def __getitem__(self, key: str) -> 'Variable':
        if key in self.param:
            return getattr(self, key)
        else:
            raise KeyError(f'No variable named {key!r} has been defined.')

    def panel(self, variables: List[str] | None = None) -> pn.Column | None:
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

    default = param.Parameter(constant=True, doc="""
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

    variable_type: ClassVar[str | None] = None

    __abstract = True

    _valid_keys: ClassVar[List[str] | Literal['params'] | None] = 'params'
    _validate_params: ClassVar[bool] = True

    def __init__(self, **params):
        if 'value' not in params and 'default' in params:
            params['value'] = params['default']
        if 'label' not in params and 'name' in params:
            params['label'] = params['name']
        super().__init__(**params)
        self.param.watch(self._update_value_from_default, 'default')

    def _update_value_from_default(self, event: param.parameterized.Event):
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

    def as_materialized(self) -> 'Constant':
        """
        If the variable is to be materialized by the builder this
        implements the conversion from a variable that references
        some external value to a materialized value.
        """
        return Constant(default=self.value)

    @property
    def panel(self) -> Viewable | None:
        """
        Optionally returns a renderable component that allows
        controlling the variable.
        """
        return None


class Constant(Variable):
    """
    `Constant` declares a constant value that can be referenced.
    """

    variable_type: ClassVar[str] = 'constant'


class EnvVariable(Variable):
    """
    `EnvVariable` fetches an environment variable that can be referenced.
    """

    key = param.String(default=None, constant=True, doc="""
        The name of the environment variable to observe.""")

    variable_type: ClassVar[str] = 'env'

    def __init__(self, **params):
        super().__init__(**params)
        key = self.key or self.name
        if key in os.environ:
            self.value = os.environ[key]


class Widget(Variable):
    """
    `Widget` variables dynamically reflect the current widget value.
    """

    kind = param.String(default=None, constant=True, doc="""
        The type of widget to instantiate.""")

    throttled = param.Boolean(default=True, constant=True, doc="""
        If the widget has a value_throttled parameter use that instead,
        ensuring that no intermediate events are generated, e.g. when
        dragging a slider.""")

    variable_type: ClassVar[str] = 'widget'

    _valid_keys: ClassVar[None] = None

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
        self.value = self._widget.value
        if self.throttled and 'value_throttled' in self._widget.param:
            self._widget.link(self, value_throttled='value')
            self.param.watch(lambda e: self._widget.param.update({'value': e.new}), 'value')
        else:
            self._widget.link(self, value='value', bidirectional=True)

    def to_spec(self, context=None):
        """
        Exports the full specification to reconstruct this component.

        Returns
        -------
        Resolved and instantiated Component object
        """
        spec = super().to_spec(context=context)
        pvals = self._widget.param.values()
        for pname, pobj in self._widget.param.objects().items():
            pval = pvals[pname]
            if (
                pval is pobj.default or pname in spec or
                (pname == 'name' and not pval) or pname == 'value_throttled'
                or pname == 'design'  # Design is Panel 1.0 only
            ):
                continue
            try:
                # Gracefully handle failed equality checks
                if pval == pobj.default:
                    continue
            except Exception:
                pass
            spec[pname] = pvals[pname]
        return spec

    @property
    def panel(self):
        return self._widget


class Parameter(Variable):
    """
    `Parameter` variables reflect the current value of a parameter.
    """

    parameter = param.ClassSelector(class_=param.Parameter, constant=True, doc="""
        A parameter instance whose current value will be reflected
        on this variable.""")

    variable_type: ClassVar[str] = 'param'

    _allows_refs: ClassVar[bool] = False

    def __init__(self, **params):
        super().__init__(**params)
        self.parameter.owner.param.watch(
            lambda e: self.param.update(value=e.new),
            self.parameter.name
        )


class URLQuery(Variable):
    """
    `URLQuery` variables reflect the value of a URL query parameter.
    """

    key = param.String(default=None, constant=True, doc="""
        The URL query parameter to observe.""")

    variable_type: ClassVar[str] = 'url'

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

    key = param.String(default=None, constant=True, doc="""
        The cookie to observe.""")

    variable_type: ClassVar[str] = 'cookie'

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.cookies:
            key = self.key or self.name
            self.value = pn.state.cookies.get(key, self.default)


class UserInfo(Variable):
    """
    `UserInfo` variables reflect a value in the user info returned by an OAuth provider.
    """

    key = param.String(default=None, constant=True, doc="""
        The key in the OAuth pn.state.user_info dictionary to observe.""")

    variable_type: ClassVar[str] = 'user'

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.user_info:
            key = self.key or self.name
            self.value = pn.state.user_info.get(key, self.default)


class Header(Variable):
    """
    `Header` variables reflect the value of a request header.
    """

    key = param.String(default=None, constant=True, doc="""
        The request header to observe.""")

    variable_type: ClassVar[str] = 'header'

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.headers:
            key = self.key or self.name
            self.value = pn.state.headers.get(key, self.default)
