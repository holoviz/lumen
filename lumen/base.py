from functools import partial

import param

from .state import state
from .util import resolve_module_reference
from .validation import (
    ValidationError, match_suggestion_message, reverse_match_suggestion,
    validate_parameters,
)


class Component(param.Parameterized):
    """
    Baseclass for all Lumen component types including Source, Filter,
    Transform, Variable and View types.
    """

    __abstract = True

    # Fields that are allowed to be declared
    _allowed_fields = []

    # Fields that must be declared declared as a list of strings
    # If one of multiple fields must be declared use a tuple.
    _required_fields = []

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

    @property
    def refs(self):
        return [v for k, v in self._refs.items() if v.startswith('$variables.')]

    @classmethod
    def _validate_allowed(cls, spec):
        if not cls._allowed_fields:
            return
        for field in spec:
            if field in cls._allowed_fields:
                continue
            msg = f'{cls.__name__} specification contained unknown field {field!r}'
            new_msg, attr = reverse_match_suggestion(field, spec, msg)
            if attr not in spec:
                msg = new_msg
            raise ValidationError(msg, spec, field)

    @classmethod
    def _validate_required(cls, spec, required=None):
        required = required or cls._required_fields
        for field in required:
            if isinstance(field, str):
                if field in spec:
                    continue
                msg = f'The {cls.__name__} component requires {field!r} parameter to be defined.'
                msg, attr = reverse_match_suggestion(field, spec, msg)
                raise ValidationError(msg, spec, attr)
            elif isinstance(field, tuple):
                if any(f in spec for f in field):
                    continue
                field_str = "', '".join(field[:-1]) + f" or '{field[-1]}"
                msg = f'The {cls.__name__} component requires one of {field_str} to be defined.'
                for f in field:
                    msg, attr = reverse_match_suggestion(f, spec, msg)
                    if attr:
                        break
                raise ValidationError(msg, spec, attr)

    @classmethod
    def _validate_fields(cls, spec, context):
        validated = {}
        for field in (cls._allowed_fields or list(spec)):
            if field not in spec:
                continue
            val = spec[field]
            if hasattr(cls, f'_validate_{field}'):
                validated[field] = val = getattr(cls, f'_validate_{field}')(val, spec, context)
            if field not in cls.param or (isinstance(field, param.Selector) and not cls.param.check_on_set):
                continue
            if field in cls.param and cls._validate_params:
                pobj = cls.param[field]
                try:
                    pobj._validate(val)
                except Exception as e:
                    msg = f"The {cls.__name__}.{field!r} field failed validation: {str(e)}"
                    raise ValidationError(msg, spec, field)
            validated[field] = val
        return validated

    @classmethod
    def from_spec(cls, spec):
        spec = cls.validate(spec)
        return cls(**spec)

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
        cls._validate_allowed(spec)
        cls._validate_required(spec)
        return cls._validate_fields(spec, context)


class MultiTypeComponent(Component):

    _required_fields = ['type']

    @classmethod
    def _get_type(cls, component_type, spec=None):
        clsname = cls.__name__
        clslower = clsname.lower()
        if component_type is None:
            raise ValidationError(f"No 'type' was provided during instantiation of '{clsname}' component.", spec)
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

        msg = f"The specification of a {clsname} component declared unknown type '{component_type}'."
        msg = match_suggestion_message(component_type, cls_types, msg)
        raise ValidationError(msg, spec=spec, attr=component_type)

    @classmethod
    def _missing_type(cls, spec):
        msg = f'The specification of a {cls.__name__} component did not declare a type.'
        msg, attr = reverse_match_suggestion('type', spec, msg)
        raise ValidationError(msg, spec=spec, attr=attr)

    @classmethod
    def from_spec(cls, spec):
        spec = cls.validate(spec)
        component_cls = cls._get_type(spec['type'], spec)
        return component_cls(**spec)

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
        if 'type' not in spec:
            cls._missing_type(spec)
        component_cls = cls._get_type(spec['type'], spec)
        component_cls._validate_allowed(spec)
        component_cls._validate_required(spec)
        component_cls._validate_fields(spec, context)
        return spec
