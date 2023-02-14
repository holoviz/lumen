from __future__ import annotations

import os

from collections import defaultdict
from typing import (
    Any, ClassVar, Dict, List,
)

import param  # type: ignore
import yaml

from .base import MultiTypeComponent
from .config import config


class Auth(MultiTypeComponent):
    """
    An AuthPlugin is given the auth specfication and can apply arbitrary
    transforms to it.
    """

    auth_type: ClassVar[str | None] = None

    @classmethod
    def _import_module(cls, component_type: str):
        pass

    def transform(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return dict(spec)


class YamlAuthMapperPlugin(Auth):
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

    auth_type: ClassVar[str] = 'yaml'

    @classmethod
    def _validate_yaml_file(cls, yaml_file, spec, context):
        if config.root:
            yaml_file = os.path.join(config.root, yaml_file)
        return yaml_file

    def transform(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec = dict(spec)
        with open(self.yaml_file) as f:
            text = f.read()
        mapping = yaml.load(text, Loader=yaml.Loader)
        new: Dict[str, List[str]] = defaultdict(list)
        for origin, replacements in mapping.items():
            values = spec.pop(origin)
            for val in values:
                for key, new_vals in replacements.get(val, {}).items():
                    new[key] += new_vals
        spec.update(new)
        return spec
