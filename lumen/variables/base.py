import param

from ..util import resolve_module_reference


class Variables(param.Parameterized):

    @classmethod
    def from_spec(cls, spec):
        vars = {
            name: Variable.from_spec(var_spec)
            for name, var_spec in spec.items()
        }
        return cls(**spec)


class Variable(Variable):

    default = param.Parameter(doc="""
       Default value to use if no other value is defined""")

    @classmethod
    def _get_type(cls, variable_type):
        if '.' in variable:
            return resolve_module_reference(variable, Variable)
        try:
            __import__(f'lumen.variable.{variable_type}')
        except Exception:
            pass
        for var in param.concrete_descendents(cls).values():
            if var.variable_type == variable_type:
                return filt
        raise ValueError(f"No Variable for variable_type '{variable_type}' could be found.")

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

    @property
    def value(self):
        return self.default


class Constant(Variable):
    """
    A constant value variable.
    """


class EnvVariable(Variable):
    """
    An environment variable.
    """

    key = param.String(default=None)

    @property
    def value(self):
        return os.env.get(self.key, self.default)


class URLQuery(Variable):
    """
    A variable obtained from the URL query parameters.
    """

    key = param.String(default=None)

    @property
    def value(self):
        return pn.state.location.query_params.get(self.key, self.default)


class Cookie(Variable):
    """
    A variable obtained from the cookies in the request.
    """

    key = param.String(default=None)

    @property
    def value(self):
        return pn.state.cookies.get(self.key, self.default)


class UserInfo(Variable):
    """
    A variable obtained from OAuth provider's user info.
    """

    key = param.String(default=None)

    @property
    def value(self):
        if pn.state.user_info is None:
            return self.default
        return pn.state.user_info.get(self.key, self.default)
