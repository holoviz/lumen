import os

import panel as pn
import param

from ..base import Component
from ..util import resolve_module_reference


class Variables(param.Parameterized):

    def __init__(self, **vars):
        super().__init__()
        for p, var in vars.items():
            self.param.add_parameter(p, param.Parameter)
            var.param.watch(self._update_value, 'value')
            var.param.trigger('value')
        self._vars = vars

    def _update_value(self, event):
        self.param.update({event.name: event.value})

    @classmethod
    def from_spec(cls, spec):
        vars = {
            name: Variable.from_spec(var_spec)
            for name, var_spec in spec.items()
        }
        return cls(**vars)


class Variable(Component):

    default = param.Parameter(doc="""
       Default value to use if no other value is defined""")

    value = param.Parameter()

    def __init__(self, **params):
        if 'value' not in params:
            params['value'] = self.default
        super().__init__(**params)

    @classmethod
    def from_spec(cls, spec):
        if isinstance(spec, dict):
            var_type = spec.pop('type')
            spec = dict(spec)
        else:
            var_type = 'constant'
            spec = {'default': spec}
        var_type = cls._get_type(var_type)
        return var_type(**spec)


class Constant(Variable):
    """
    A constant value variable.
    """


class EnvVariable(Variable):
    """
    An environment variable.
    """

    key = param.String(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        if self.key in os.env:
            self.value = os.env[self.key]


class URLQuery(Variable):
    """
    A variable obtained from the URL query parameters.
    """

    key = param.String(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.location:
            pn.state.location.param.watch(self._update_value, 'search')
            self._update_value()

    def _update_value(self):
        self.value = pn.state.location.query_params.get(self.key, self.default)


class Cookie(Variable):
    """
    A variable obtained from the cookies in the request.
    """

    key = param.String(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.cookies:
            self.value = pn.state.cookies.get(self.key, self.default)


class UserInfo(Variable):
    """
    A variable obtained from OAuth provider's user info.
    """

    key = param.String(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        if pn.state.user_info:
            self.value = pn.state.user_info.get(self.key, self.default)
