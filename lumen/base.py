"""
Implements the baseclasses for all Component types in Lumen.

The base classes implement the core validation logic and the ability
to dynamically resolve Source and Variable references.
"""
from __future__ import annotations

import warnings

from functools import partial
from typing import Any, ClassVar, Literal

import pandas as pd
import panel as pn
import param  # type: ignore

from panel.io.cache import _container_hash, _hash_funcs
from panel.util import classproperty
from panel.widgets.base import WidgetBase

from .state import state
from .util import (
    VARIABLE_RE, cleanup_expr, is_ref, resolve_module_reference,
)
from .validation import (
    ValidationError, match_suggestion_message, reverse_match_suggestion,
    validate_parameters,
)


class Component(param.Parameterized):
    """
    Baseclass for all Lumen component types including Source, Filter,
    Transform, Variable and View types. Components must implement
    serialization and deserialization into a specification dictionary
    via the `from_spec` and `to_spec` protocol. Additonally they
    should implement validation.
    """

    __abstract = True

    # Whether the component allows references
    _allows_refs: ClassVar[bool] = True

    # Parameters that are computed internally and are not part of the
    # component specification
    _internal_params: ClassVar[list[str]] = ['name']

    # Deprecated parameters that are still allowed as spec keys
    _legacy_params: ClassVar[list[str]] = []

    # Keys that must be declared declared as a list of strings or
    # tuples of strings if one of multiple must be defined.
    _required_keys: ClassVar[list[str | tuple[str, ...]]] = []

    # Keys that are valid to define
    _valid_keys: ClassVar[list[str] | Literal['params'] | None] = None

    # Whether to enforce parameter validation in specification
    _validate_params: ClassVar[bool] = False

    def __init__(self, **params):
        self._refs = params.pop('refs', {})
        expected = list(self.param)
        validate_parameters(params, expected, type(self).name)
        if self._allows_refs:
            params = self._extract_refs(params, self._refs)
        super().__init__(**params)
        for p, ref in self._refs.items():
            if isinstance(state.variables, dict):
                continue
            elif isinstance(ref, str) and '$variables' in ref:
                ref_vars = VARIABLE_RE.findall(ref)
                state.variables.param.watch(partial(self._update_ref, p, ref), ref_vars)
                if '.' not in p and p not in params:
                    self._update_ref(p, ref)

    def _extract_refs(self, params: dict[str, Any], refs: dict[str, Any]):
        from .variables import Variable
        processed = {}
        for pname, pval in params.items():
            if is_ref(pval):
                refs[pname] = pval
            elif isinstance(pval, (pn.widgets.Widget | param.Parameter | Variable)):
                var = state.variables.add_variable(pval)
                processed[pname] = var.value
                refs[pname] = f'$variables.{var.name}'
                continue
            elif isinstance(pval, dict):
                subrefs = {}
                processed[pname] = self._extract_refs(pval, subrefs)
                for subkey, subref in subrefs.items():
                    refs[f'{pname}.{subkey}'] = subref
                continue
            else:
                processed[pname] = pval
        return processed

    def _update_ref(self, pname: str, ref: str, *events: param.parameterized.Event):
        """
        Component should implement appropriate downstream events
        following a change in a variable.
        """
        expr = cleanup_expr(ref)
        new_value = pd.eval(expr, local_dict=dict(state.variables), engine='python')
        if '.' in pname:
            pname, *keys = pname.split('.')
            old = getattr(self, pname)
            current = new = old.copy()
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = new_value
        else:
            new = new_value
        self.param.update({pname: new})

    def _sync_refs(self, trigger: bool = True):
        updates = []
        for p, ref in self._refs.items():
            if isinstance(state.variables, dict):
                continue
            elif isinstance(ref, str) and '$variables' in ref:
                with param.discard_events(self):
                    self._update_ref(p, ref)
                    pname, *_ = p.split('.')
                    updates.append(pname)
        if trigger:
            self.param.trigger(*updates)

    def _serialize_param_ref(self, obj, objects=None):
        if isinstance(obj, WidgetBase):
            obj = obj.param.value
        if isinstance(obj, param.Parameter):
            for key, ref_obj in objects.items():
                if obj.owner is ref_obj:
                    owner = key
                    break
            else:
                owner = obj.owner.name
            return {'type': 'param', 'owner': owner, 'name': obj.name}

        objects = objects or {}
        input_obj = None
        prev = obj._prev
        if isinstance(prev, param.reactive.rx):
            prev = self._serialize_param_ref(prev, objects)
        elif prev is None:
            prev = obj._fn
            # Hacky way of checking for rx wrapper around parameter
            func = getattr(prev, '__bound_function__', None)
            if (
                str(func).startswith('<function rx.__new__.<locals>.<lambda>') and
                isinstance(prev._dinfo['kw']['__arg0'], param.Parameter)
            ):
                prev = prev._dinfo['kw']['__arg0']
            else:
                raise ValueError("Bound functions and methods cannot be serialized.")
            prev = self._serialize_param_ref(prev, objects)
        elif obj._shared_obj:
            input_obj = obj._shared_obj[0]

        if obj._operation:
            op = dict(obj._operation)
            if not isinstance(op['fn'], str):
                op['fn'] = f"{op['fn'].__module__}.{op['fn'].__name__}"
            op['args'] = list(op['args'])
        else:
            op = None
        return {
            'type': 'rx',
            'prev': prev,
            'input_obj': input_obj,
            'kwargs': obj._kwargs,
            'method': obj._method,
            'operation': op
        }

    @classmethod
    def _finalize_param_ref(cls, ref, objects=None):
        if ref['type'] == 'param' and ref['owner'] is None:
            if ref['owner'] in objects:
                ref['owner'] = objects[ref['owner']]
            else:
                raise ValueError(
                    f"Parameter {ref['name']!r} could not be serialized since "
                    f"the owning Parameterized {ref['owner']!r} could not be found."
                )
        else:
            current = ref
            while isinstance(current, dict) and current.get('prev') is not None:
                current = current.get('prev')
                if current['type'] == 'param':
                    cls._finalize_param_ref(current, objects)

    @classmethod
    def _materialize_param_ref(cls, spec, objects=None):
        current = spec
        chain = [current]
        while isinstance(current, dict) and current.get('prev') is not None:
            current = current.get('prev')
            chain.append(current)

        input_spec = chain[-1]
        if input_spec.get('input_obj') is not None:
            obj = input_spec['input_obj']
        elif isinstance(input_spec, dict) and input_spec.get('type') == 'param':
            obj = objects[input_spec['owner']].param[input_spec['name']]
            chain = chain[:-1]
        expr = param.rx(obj)
        for op_spec in chain[:-1][::-1]:
            op = op_spec['operation']
            if '.' in op['fn']:
                fn = resolve_module_reference(op['fn'])
                expr = expr._apply_operator(fn, *op['args'], reverse=op['reverse'], **op['kwargs'])
            else:
                expr = getattr(expr, op['fn'])(*op['args'], **op['kwargs'])
        return expr

    ##################################################################
    # Validation API
    ##################################################################

    @classproperty
    def _valid_keys_(cls) -> list[str] | None:
        if cls._valid_keys == 'params':
            valid = [p for p in cls.param if p not in cls._internal_params]
        else:
            valid = cls._valid_keys
        return valid if valid is None else valid + cls._legacy_params

    @classmethod
    def _validate_keys_(cls, spec: dict[str, Any]):
        valid_keys = cls._valid_keys_
        for key in spec:
            if valid_keys is None or key in valid_keys:
                continue
            msg = f'{cls.__name__} component specification contained unknown key {key!r}.'
            msg = match_suggestion_message(key, cls._valid_keys_ or [], msg)
            raise ValidationError(msg, spec, key)

    @classmethod
    def _validate_required_(
        cls, spec: dict[str, Any], required: list[str | tuple[str, ...]] | None = None
    ):
        if required is None:
            required_keys = cls._required_keys
        else:
            required_keys = required
        for key in required_keys:
            if isinstance(key, str):
                if key in spec:
                    continue
                msg = f'The {cls.__name__} component requires {key!r} parameter to be defined.'
                msg, attr = reverse_match_suggestion(key, list(spec), msg)
                raise ValidationError(msg, spec, attr)
            elif isinstance(key, tuple):
                if any(f in spec for f in key):
                    continue
                skey = sorted(key)
                key_str = "', '".join(skey[:-1]) + f"' or '{skey[-1]}"
                msg = f"{cls.__name__} component requires one of '{key_str}' to be defined."
                for f in key:
                    msg, attr = reverse_match_suggestion(f, list(spec), msg)
                    if attr:
                        break
                raise ValidationError(msg, spec, attr)

    @classmethod
    def _validate_list_subtypes(
        cls, key: str, subtype: type[Component], subtype_specs: list[dict[str, Any] | str],
        spec: dict[str, Any], context: dict[str, Any], subcontext: list[dict[str, Any] | str] | None = None
    ):
        if not isinstance(subtype_specs, list):
            raise ValidationError(
                f'{cls.__name__} component {key!r} key expected list type but got {type(subtype_specs).__name__}. '
                "This could be because of a missing dash in the yaml file.",
                spec, key
            )
        subtypes = []
        for subtype_spec in subtype_specs:
            subtype_spec = subtype.validate(subtype_spec, context)
            subtypes.append(subtype_spec)
            if subcontext is not None:
                subcontext.append(subtype_spec)
        return subtypes

    @classmethod
    def _validate_dict_subtypes(
        cls, key: str, subtype: type[Component], subtype_specs: dict[str, dict[str, Any] | str],
        spec: dict[str, Any], context: dict[str, Any], subcontext: dict[str, Any] | None = None
    ):
        if not isinstance(subtype_specs, dict):
            raise ValidationError(
                f'{cls.__name__} component {key!r} key expected dict type but got {type(subtype_specs).__name__}.',
                spec, key
            )
        subtypes = {}
        for subtype_name, subtype_spec in subtype_specs.items():
            subtypes[subtype_name] = subtype.validate(subtype_spec, context)
            if subcontext is not None:
                subcontext[subtype_name] = subtypes[subtype_name]
        return subtypes

    @classmethod
    def _validate_str_or_spec(
        cls, key: str, subtype: type[Component], subtype_spec: dict[str, Any] | str,
        spec: dict[str, Any], context: dict[str, Any]
    ):
        if isinstance(subtype_spec, str):
            if subtype_spec not in context[f'{key}s']:
                msg = f'{cls.__name__} component specified non-existent {key} {subtype_spec!r}.'
                msg = match_suggestion_message(subtype_spec, list(context[key]), msg)
                raise ValidationError(msg, spec, subtype_spec)
            return subtype_spec
        return subtype.validate(subtype_spec, context)

    @classmethod
    def _validate_dict_or_list_subtypes(
        cls, key: str, subtype: type[Component], subtype_specs: dict[str, dict[str, Any] | str] | list[dict[str, Any] | str],
        spec: dict[str, Any], context: dict[str, Any], subcontext: dict[str, Any] | list[dict[str, Any] | str] | None = None
    ):
        if isinstance(subtype_specs, list):
            assert subcontext is None or isinstance(subcontext, list)
            return cls._validate_list_subtypes(key, subtype, subtype_specs, spec, context, subcontext)
        else:
            assert subcontext is None or isinstance(subcontext, dict)
            return cls._validate_dict_subtypes(key, subtype, subtype_specs, spec, context, subcontext)

    @classmethod
    def _deprecation(
        cls, msg: str, key: str, spec: dict[str, Any], update: dict[str, Any]
    ):
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        if key not in spec:
            spec[key] = {}
        spec[key].update(update)

    @classmethod
    def _validate_ref(
        cls, key: str, value: Any, spec: dict[str, Any], context: dict[str, Any]
    ):
        refs = value[1:].split('.')
        if refs[0] == 'variables':
            if refs[1] not in context.get('variables', {}):
                msg = f'{cls.__name__} component {key!r} references undeclared variable {value!r}.'
                msg = match_suggestion_message(refs[1], list(context.get('variables', {})), msg)
                raise ValidationError(msg, spec, refs[1])
        elif refs[0] not in context.get('sources', {}):
            msg = f'{cls.__name__} component {key!r} references undeclared source {value!r}.'
            msg = match_suggestion_message(refs[1], list(context.get('sources', {})), msg)
            raise ValidationError(msg, spec, refs[1])

    @classmethod
    def _validate_param(cls, key: str, value: Any, spec: dict[str, Any]):
        pobj = cls.param[key]
        try:
            if isinstance(pobj, param.Selector) and pobj.names and value in pobj.names:
                return
            pobj._validate(value)
        except Exception as e:
            msg = f"{cls.__name__} component {key!r} value failed validation: {e!s}"
            raise ValidationError(msg, spec, key) from e

    @classmethod
    def _is_component_key(cls, key: str) -> bool:
        if key not in cls.param:
            return False
        pobj = cls.param[key]
        return (
            isinstance(pobj, param.ClassSelector) and
            isinstance(pobj.class_, type) and
            issubclass(pobj.class_, Component)
        )

    @classmethod
    def _is_list_component_key(cls, key: str) -> bool:
        if key not in cls.param:
            return False
        pobj = cls.param[key]
        return (
            isinstance(pobj, param.List) and
            isinstance(pobj.item_type, type) and
            issubclass(pobj.item_type, Component)
        )

    @classmethod
    def _is_param_function(cls, p):
        pobj = cls.param[p]
        if not isinstance(pobj, param.ClassSelector):
            return False
        if isinstance(pobj.class_, tuple):
            return all(issubclass(c, param.ParameterizedFunction) for c in pobj.class_)
        else:
            return issubclass(pobj.class_, param.ParameterizedFunction)

    @classmethod
    def _is_list_param_function(cls, p):
        pobj = cls.param[p]
        return (
            isinstance(pobj, param.List) and
            isinstance(pobj.item_type, type) and
            issubclass(pobj.item_type, param.ParameterizedFunction)
        )

    @classmethod
    def _validate_spec_(
        cls, spec: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        if context is None:
            context = validated

        for key in (cls._valid_keys_ or list(spec)):
            if key not in spec:
                continue
            val = spec[key]
            if is_ref(val) and not cls._allows_refs:
                raise ValidationError(
                    f'{cls.__name__} component does not allow references but {key} '
                    f'value ({val!r}) is a reference.', spec, val
                )
            if hasattr(cls, f'_validate_{key}'):
                val = getattr(cls, f'_validate_{key}')(val, spec, context)
            elif cls._is_component_key(key):
                val = cls.param[key].class_.validate(val, context)
            elif cls._is_list_component_key(key):
                val = cls._validate_list_subtypes(
                    key, cls.param[key].item_type, val, spec, context
                )
            elif is_ref(val):
                cls._validate_ref(key, val, spec, context)
            elif key in cls.param:
                if isinstance(val, str) and val.startswith('@'):
                    continue
                cls._validate_param(key, val, spec)
            validated[key] = val
        return validated

    ##################################################################
    # Public API
    ##################################################################

    @property
    def refs(self) -> list[str]:
        return [v for k, v in self._refs.items() if v.startswith('$variables.')]

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | str) -> Component:
        """
        Creates a Component instance from a specification.

        Parameters
        ----------
        spec : dict or str
            Specification declared as a dictionary of parameter values
            or a string referencing a source in the sources dictionary.

        Returns
        -------
        Resolved and instantiated Component object
        """
        if isinstance(spec, str):
            raise ValueError(
                "Component cannot be materialized by reference. Please pass "
                "full specification for the component."
            )
        return cls(**spec)

    def to_spec(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Exports the full specification to reconstruct this component.

        Parameters
        ----------
        context: Dict[str, Any]
          Context contains the specification of all previously serialized components,
          e.g. to allow resolving of references.

        Returns
        -------
        Declarative specification of this component.
        """
        spec = {}
        for p, value in self.param.values().items():
            pobj = self.param[p]
            try:
                is_equal = value is pobj.default or value == pobj.default
            except Exception:
                is_equal = False
            if p in self._internal_params or is_equal:
                continue
            elif self._is_component_key(p):
                if value is None:
                    continue
                pspec = value.to_spec(context=context)
                if not pspec:
                    continue
                value = pspec
            elif self._is_list_component_key(p):
                value = [
                    None if v is None else v.to_spec(context=context)
                    for v in value
                ]
            elif self._is_param_function(p):
                func_type = type(value)
                module_spec = f'{func_type.__module__}.{func_type.__name__}'
                func_params = {
                    vp: v for vp, v in value.param.values().items()
                    if vp != 'name' and v != value.param[vp].default
                }
                value = dict(func_params, type=module_spec)
            elif self._is_list_param_function(p):
                vs = []
                for val in value:
                    func_type = type(val)
                    module_spec = f'{func_type.__module__}.{func_type.__name__}'
                    func_params = {
                        vp: v for vp, v in val.param.values().items()
                        if vp != 'name' and v != val.param[vp].default
                    }
                    vs.append(dict(func_params, type=module_spec))
                value = vs
            spec[p] = value
        if context is not None:
            spec.update(self._refs)
        return spec

    @classmethod
    def validate(
        cls, spec: dict[str, Any] | str, context: dict[str, Any] | None = None
    ) -> dict[str, Any] | str:
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
        if isinstance(spec, str):
            return spec
        context = {} if context is None else context
        cls._validate_keys_(spec)
        cls._validate_required_(spec)
        return cls._validate_spec_(spec, context)


class MultiTypeComponent(Component):
    """
    MultiComponentType is the baseclass for extensible Lumen components.

    A `MultiTypeComponent` can be resolved using the `_get_type` method
    either by the name declared in the `<component>_type` attribute,
    where the name of the component represent the immediate subclasses
    of MultiTypeComponent. For example `class View(MultiTypeComponent)`
    should define `view_type` for all its descendants to override.

    Just as any other Component, `MultiTypeComponent` implements
    methods to construct an instance from a specification, export the
    specification of a component and the ability to validate a
    component.
    """

    __abstract = True

    _required_keys: ClassVar[list[str | tuple[str, ...]]] = ['type']

    @classproperty
    def _valid_keys_(cls) -> list[str | tuple[str, ...]] | None:
        if cls._valid_keys is None:
            valid = None
        elif cls._valid_keys == 'params':
            valid = list(cls.param)
        elif 'params' in cls._valid_keys:
            valid = cls._valid_keys.copy()
            valid.extend(list(cls.param))
        else:
            valid = cls._valid_keys.copy()

        if valid is not None and 'type' not in valid:
            valid.append('type')
        return valid if valid is None else valid + cls._legacy_params

    @classproperty
    def _base_type(cls):
        if cls is MultiTypeComponent:
            return
        return cls.__mro__[cls.__mro__.index(MultiTypeComponent)-1]

    @classproperty
    def _component_type(cls) -> str:
        component_type = getattr(cls, f'{cls._base_type.__name__.lower()}_type')
        if component_type is not None:
            return component_type
        return f'{cls.__module__}.{cls.__name__}'

    @classmethod
    def _import_module(cls, component_type: str):
        base_type = cls._base_type
        try:
            import_name = f'lumen.{base_type.__name__.lower()}s.{component_type}'
            __import__(import_name)
        except ImportError as e:
            if e.name != import_name:
                msg = (
                    f"In order to use the {base_type.__name__.lower()} "
                    f"component '{component_type}', the '{e.name}' package "
                    "must be installed."
                )
                raise ImportError(msg) from e

    @classmethod
    def _get_type(
        cls, component_type: str, spec: dict[str, Any] | None = None
    ) -> type[MultiTypeComponent]:
        base_type = cls._base_type
        if component_type is None:
            raise ValidationError(
                f"No 'type' was provided during instantiation of {base_type.__name__} component.",
                spec
            )
        if '.' in component_type:
            return resolve_module_reference(component_type, base_type)

        cls._import_module(component_type)

        subcls_types = set()
        for subcls in param.concrete_descendents(cls).values():
            subcls_type = subcls._component_type
            if subcls_type is None:
                continue
            subcls_types.add(subcls_type)
            if subcls_type == component_type:
                return subcls

        msg = f"{base_type.__name__} component specification declared unknown type '{component_type}'."
        msg = match_suggestion_message(component_type, list(subcls_types), msg)
        raise ValidationError(msg, spec, component_type)

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | str) -> MultiTypeComponent:
        if isinstance(spec, str):
            raise ValueError(
                "MultiTypeComponent cannot be materialized by reference. Please pass "
                "full specification for the MultiTypeComponent."
            )
        spec = dict(spec)
        component_cls = cls._get_type(spec.pop('type'), spec)
        return component_cls(**spec)

    def to_spec(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Exports the full specification to reconstruct this component.

        Returns
        -------
        Resolved and instantiated Component object
        """
        spec = super().to_spec(context=context)
        spec['type'] = self._component_type
        return spec

    @classmethod
    def validate(
        cls, spec: dict[str, Any] | str, context: dict[str, Any] | None = None
    ) -> dict[str, Any] | str:
        """
        Validates the component specification given the validation context and the path.

        Arguments
        -----------
        spec: dict | str
          The specification for the component being validated or a reference to the component.
        context: dict
          Validation context contains the specification of all previously validated components,
          e.g. to allow resolving of references.

        Returns
        --------
        Validated specification.
        """
        if isinstance(spec, str):
            return spec
        context = {} if context is None else context
        if 'type' not in spec:
            msg = f'{cls.__name__} component specification did not declare a type.'
            msg, attr = reverse_match_suggestion('type', list(spec), msg)
            raise ValidationError(msg, spec, attr)
        component_cls = cls._get_type(spec['type'], spec)
        component_cls._validate_keys_(spec)
        component_cls._validate_required_(spec)
        return component_cls._validate_spec_(spec, context)


_hash_funcs[Component] = lambda component: _container_hash(component.to_spec())
