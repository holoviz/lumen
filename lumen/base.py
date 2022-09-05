import warnings

from functools import partial

import param

from panel.util import classproperty

from .state import state
from .util import is_ref, resolve_module_reference
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
    _allows_refs = True

    # Keys that must be declared declared as a list of strings or
    # tuples of strings if one of multiple must be defined.
    _required_keys = []

    # Keys that are valid to define
    _valid_keys = None

    # Whether to enforce parameter validation in specification
    _validate_params = False

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

    ##################################################################
    # Validation API
    ##################################################################

    @classproperty
    def _valid_keys_(cls):
        return list(cls.param) if cls._valid_keys == 'params' else cls._valid_keys

    @classmethod
    def _validate_keys(cls, spec):
        valid_keys = cls._valid_keys_
        for key in spec:
            if valid_keys is None or key in valid_keys:
                continue
            msg = f'{cls.__name__} component specification contained unknown key {key!r}.'
            msg = match_suggestion_message(key, cls._valid_keys_ or [], msg)
            raise ValidationError(msg, spec, key)

    @classmethod
    def _validate_required(cls, spec, required=None):
        required = required or cls._required_keys
        for key in required:
            if isinstance(key, str):
                if key in spec:
                    continue
                msg = f'The {cls.__name__} component requires {key!r} parameter to be defined.'
                msg, attr = reverse_match_suggestion(key, spec, msg)
                raise ValidationError(msg, spec, attr)
            elif isinstance(key, tuple):
                if any(f in spec for f in key):
                    continue
                key_str = "', '".join(key[:-1]) + f" or '{key[-1]}"
                msg = f'{cls.__name__} component requires one of {key_str} to be defined.'
                for f in key:
                    msg, attr = reverse_match_suggestion(f, spec, msg)
                    if attr:
                        break
                raise ValidationError(msg, spec, attr)

    @classmethod
    def _validate_list_subtypes(cls, key, subtype, subtype_specs, spec, context, subcontext=None):
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
    def _validate_dict_subtypes(cls, key, subtype, subtype_specs, spec, context, subcontext=None):
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
    def _validate_str_or_spec(cls, key, subtype, subtype_spec, spec, context):
        if isinstance(subtype_spec, str):
            if subtype_spec not in context[f'{key}s']:
                msg = f'{cls.__name__} component specified non-existent {key} {subtype_spec!r}.'
                msg = match_suggestion_message(subtype_spec, list(context[key]), msg)
                raise ValidationError(msg, spec, subtype_spec)
            return subtype_spec
        return subtype.validate(subtype_spec, context)

    @classmethod
    def _validate_dict_or_list_subtypes(cls, key, subtype, subtype_specs, spec, context, subcontext=None):
        if isinstance(subtype_specs, list):
            return cls._validate_list_subtypes(key, subtype, subtype_specs, spec, context, subcontext)
        else:
            return cls._validate_dict_subtypes(key, subtype, subtype_specs, spec, context, subcontext)

    @classmethod
    def _deprecation(cls, msg, key, spec, update):
        warnings.warn(msg, DeprecationWarning)
        if key not in spec:
            spec[key] = {}
        spec[key].update(update)

    @classmethod
    def _validate_ref(cls, key, value, spec, context):
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
    def _validate_param(cls, key, value, spec):
        pobj = cls.param[key]
        try:
            pobj._validate(value)
        except Exception as e:
            msg = f"{cls.__name__} component {key!r} value failed validation: {str(e)}"
            raise ValidationError(msg, spec, key)

    @classmethod
    def _is_component_key(cls, key):
        if key not in cls.param:
            return False
        pobj = cls.param[key]
        return (
            isinstance(pobj, param.ClassSelector) and
            isinstance(pobj.class_, type) and
            issubclass(pobj.class_, Component)
        )

    @classmethod
    def _is_list_component_key(cls, key):
        if key not in cls.param:
            return False
        pobj = cls.param[key]
        return (
            isinstance(pobj, param.List) and
            isinstance(pobj.item_type, type) and
            issubclass(pobj.item_type, Component)
        )

    @classmethod
    def _validate_spec(cls, spec, context=None):
        validated = {}
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
                cls._validate_param(key, val, spec)
            validated[key] = val
        return validated

    ##################################################################
    # Public API
    ##################################################################

    @property
    def refs(self):
        return [v for k, v in self._refs.items() if v.startswith('$variables.')]

    @classmethod
    def from_spec(cls, spec):
        """
        Creates a Component instanceobject from a specification.

        Parameters
        ----------
        spec : dict or str
            Specification declared as a dictionary of parameter values
            or a string referencing a source in the sources dictionary.

        Returns
        -------
        Resolved and instantiated Component object
        """
        return cls(**spec)

    def to_spec(self):
        """
        Exports the full specification to reconstruct this component.

        Returns
        -------
        Declarative specification of this component.
        """
        spec = {}
        for p, value in self.param.items():
            if self._is_component_key(p):
                value = value.to_spec()
            elif self._is_list_component_key(p):
                value = [v.to_spec() for v in value]
            spec[p] = value
        spec.update(self._refs)
        return spec

    @classmethod
    def validate(cls, spec, context=None):
        """
        Validates the component specification given the validation context.

        Arguments
        -----------
        spec: dict
          The specification for the component being validated.
        context: dict
          Validation context contains the specification of all previously validated components,
          e.g. to allow resolving of references.

        Returns
        --------
        Validated specification.
        """
        context = {} if context is None else context
        cls._validate_keys(spec)
        cls._validate_required(spec)
        return cls._validate_spec(spec, context)


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

    _required_keys = ['type']

    @classproperty
    def _valid_keys_(cls):
        valid = list(cls.param) if cls._valid_keys == 'params' else cls._valid_keys
        if valid:
            valid.append('type')
        return valid

    @classproperty
    def _base_type(cls):
        if cls is MultiTypeComponent:
            return
        return cls.__mro__[cls.__mro__.index(MultiTypeComponent)-1]

    @classproperty
    def _component_type(cls):
        component_type = getattr(cls, f'{cls._base_type.__name__.lower()}_type')
        if component_type is not None:
            return component_type
        return f'{cls.__module__}.{cls.__name__}'

    @classmethod
    def _get_type(cls, component_type, spec=None):
        base_type = cls._base_type
        if component_type is None:
            raise ValidationError(
                f"No 'type' was provided during instantiation of {base_type.__name__} component.",
                spec
            )
        if '.' in component_type:
            return resolve_module_reference(component_type, base_type)
        try:
            __import__(f'lumen.{base_type.__name__.lower()}s.{component_type}')
        except Exception:
            pass

        subcls_types = set()
        for subcls in param.concrete_descendents(cls).values():
            subcls_type = subcls._component_type
            if subcls_type is None:
                continue
            subcls_types.add(subcls_type)
            if subcls_type == component_type:
                return subcls

        msg = f"{base_type.__name__} component specification declared unknown type '{component_type}'."
        msg = match_suggestion_message(component_type, subcls_types, msg)
        raise ValidationError(msg, spec, component_type)

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def from_spec(cls, spec):
        component_cls = cls._get_type(spec['type'], spec)
        return component_cls(**spec)

    def to_spec(self):
        """
        Exports the full specification to reconstruct this component.

        Returns
        -------
        Resolved and instantiated Component object
        """
        spec = super().to_spec()
        spec['type'] = self._component_type
        return spec

    @classmethod
    def validate(cls, spec, context=None):
        """
        Validates the component specification given the validation context and the path.

        Arguments
        -----------
        spec: dict
          The specification for the component being validated.
        context: dict
          Validation context contains the specification of all previously validated components,
          e.g. to allow resolving of references.

        Returns
        --------
        Validated specification.
        """
        context = {} if context is None else context
        if 'type' not in spec:
            msg = f'{cls.__name__} component specification did not declare a type.'
            msg, attr = reverse_match_suggestion('type', spec, msg)
            raise ValidationError(msg, spec, attr)
        component_cls = cls._get_type(spec['type'], spec)
        component_cls._validate_keys(spec)
        component_cls._validate_required(spec)
        return component_cls._validate_spec(spec, context)
