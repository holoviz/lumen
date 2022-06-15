from collections import defaultdict

import param
import yaml

from .util import resolve_module_reference


class AuthPlugin(param.Parameterized):
    """
    An AuthPlugin is given the auth specfication and can apply arbitrary
    transforms to it.
    """

    auth_type = None

    @classmethod
    def _get_type(cls, auth_type):
        if '.' in auth_type:
            return resolve_module_reference(auth_type, AuthPlugin)
        try:
            __import__(f'lumen.auth.{auth_type}')
        except Exception:
            pass
        for auth in param.concrete_descendents(cls).values():
            if auth.auth_type == auth_type:
                return auth
        raise ValueError(f"No AuthPlugin for auth_type '{auth_type}' could be found.")

    @classmethod
    def from_spec(cls, spec):
        spec = dict(spec)
        auth_plugin = cls._get_type(spec.pop('type'))
        return auth_plugin(**spec)

    def transform(self, spec):
        return dict(spec)


class YamlAuthMapperPlugin(AuthPlugin):
    """
    The YamlAuthMapperPlugin uses a Yaml file to map auth keys
    allowing more concise declarations for individual dashboards, e.g.
    the following yaml specification:

        group:
          admins:
            email:
              - lumen@holoviz.org
              - panel@holoviz.org
              - param@holoviz.org
          users:
            email:
              - philipp@holoviz.org

    will expand this auth spec:

        auth:
          group:
            - admins

    to this expanded spec:

        auth:
          email:
            - lumen@holoviz.org
            - panel@holoviz.org
            - param@holoviz.org
            - philipp@holoviz.org
    """

    yaml_file = param.Filename()

    auth_type = 'yaml'

    def transform(self, spec):
        spec = dict(spec)
        with open(self.yaml_file) as f:
            text = f.read()
        mapping = yaml.load(text, Loader=yaml.Loader)
        new = defaultdict(list)
        for origin, replacements in mapping.items():
            values = spec.pop(origin)
            for val in values:
                for key, new_vals in replacements.get(val, {}).items():
                    new[key] += new_vals
        spec.update(new)
        return spec
