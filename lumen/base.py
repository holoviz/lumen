from difflib import get_close_matches
from functools import partial

import param

from .state import state
from .util import resolve_module_reference, validate_parameters


class Component(param.Parameterized):
    """
    Baseclass for all Lumen component types including Source, Filter,
    Transform, Variable and View types.
    """

    __abstract = True

    def __init__(self, **params):
        self._refs = params.pop('refs', {})
        expected = list(self.param.params())
        validate_parameters(params, expected, type(self).name)
        super().__init__(**params)
        for p, ref in self._refs.items():
            if isinstance(state.variables, dict):
                continue
            elif isinstance(ref, str) and ref.startswith('$variables.'):
                ref = ref.split('$variables.')[1]
                state.variables.param.watch(partial(self._update_ref, p), ref)

    def _update_ref(self, pname, event):
        """
        Component should implement appropriate downstream events
        following a change in a variable.
        """
        self.param.update({pname: event.new})

    @property
    def refs(self):
        return [v for k, v in self._refs.items() if v.startswith('$variables.')]

    @classmethod
    def _get_type(cls, component_type):
        clsname = cls.__name__
        clslower = clsname.lower()
        if component_type is None:
            raise ValueError(f"'type' for '{clslower}' is not available.")
        if '.' in component_type:
            return resolve_module_reference(component_type, cls)
        try:
            __import__(f'lumen.{clslower}s.{component_type}')
        except Exception:
            pass

        cls_types = set()
        for component in param.concrete_descendents(cls).values():
            cls_type = getattr(component, f'{clsname.lower()}_type')
            if cls_type is None:
                continue
            cls_types.add(cls_type)
            if cls_type == component_type:
                return component

        msg = f"No '{clslower}' for {clslower}_type '{component_type}' could be found."
        matches = "', '".join(get_close_matches(component_type, cls_types))
        if matches:
            msg += f" Did you mean '{matches}'?"
        raise ValueError(msg)
