import warnings

from functools import partial

import param

from .state import state
from .util import is_ref, resolve_module_reference
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

    # Whether the component allows references
    _allows_refs = True

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
    def _allowed(cls):
        return list(cls.param) if cls._allowed_fields == 'params' else cls._allowed_fields

    @classmethod
    def _validate_allowed(cls, spec):
        if not cls._allowed_fields:
            return
        allowed = cls._allowed()
        for field in spec:
            if field in allowed:
                continue
            msg = f'{cls.__name__} specification contained unknown field {field!r}.'
            msg = match_suggestion_message(field, allowed, msg)
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
                msg = f'{cls.__name__} component requires one of {field_str} to be defined.'
                for f in field:
                    msg, attr = reverse_match_suggestion(f, spec, msg)
                    if attr:
                        break
                raise ValidationError(msg, spec, attr)

    @classmethod
    def _validate_list_subtypes(cls, key, subtype, subtype_specs, spec, context, subcontext=None):
        if not isinstance(subtype_specs, list):
            raise ValidationError(
                f'{cls.__name__} {key} field expected list type but got {type(subtype_specs)}.',
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
                f'{cls.__name__} {key} field expected dict type but got {type(subtype_specs)}.',
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
                msg = f'{cls.__name__} specified non-existent {key} {subtype_spec!r}.'
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
    def _validate_fields(cls, spec, context=None):
        validated = {}
        if context is None:
            context = validated
        allowed = cls._allowed()
        for field in (allowed or list(spec)):
            if field not in spec:
                continue
            val = spec[field]
            if hasattr(cls, f'_validate_{field}'):
                val = getattr(cls, f'_validate_{field}')(val, spec, context)
            elif (field in cls.param and isinstance(cls.param[field], param.ClassSelector) and
                  isinstance(cls.param[field].class_, type) and issubclass(cls.param[field].class_, Component)):
                val = cls.param[field].class_.validate(val, context)
            elif (field in cls.param and isinstance(cls.param[field], param.List) and
                  isinstance(cls.param[field].item_type, type) and issubclass(cls.param[field].item_type, Component)):
                val = cls._validate_list_subtypes(field, cls.param[field].item_type, val, spec, context)
            elif is_ref(val):
                refs = val[1:].split('.')
                if refs[0] == 'variables':
                    if refs[1] not in context.get('variables', {}):
                        msg = f'{cls.__name__} component {field!r} references undeclared variable {val!r}.'
                        msg = match_suggestion_message(refs[1], list(context.get('variables', {})), msg)
                        raise ValidationError(msg, spec, refs[1])
                elif refs[0] not in context.get('sources', {}):
                    msg = f'{cls.__name__} component {field!r} references undeclared source {val!r}.'
                    msg = match_suggestion_message(refs[1], list(context.get('sources', {})), msg)
                    raise ValidationError(msg, spec, refs[1])
            elif field in cls.param and cls._validate_params:
                if (isinstance(field, param.Selector) and not cls.param.check_on_set):
                    validated[field] = val
                    continue
                pobj = cls.param[field]
                try:
                    pobj._validate(val)
                except Exception as e:
                    msg = f"{cls.__name__} component {field!r} value failed validation: {str(e)}"
                    raise ValidationError(msg, spec, field)
            validated[field] = val
        return validated

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
        cls._validate_allowed(spec)
        cls._validate_required(spec)
        return cls._validate_fields(spec, context)


class MultiTypeComponent(Component):

    _required_fields = ['type']

    @classmethod
    def _get_type(cls, component_type, spec=None):
        for scls in cls.__mro__:
            if hasattr(scls, f'{scls.__name__.lower()}_type'):
                break
        cls = scls
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

        msg = f"{clsname} component declared unknown type '{component_type}'."
        msg = match_suggestion_message(component_type, cls_types, msg)
        raise ValidationError(msg, spec=spec, attr=component_type)

    @classmethod
    def _allowed(cls):
        allowed = super()._allowed()
        if allowed:
            allowed.append('type')
        return allowed

    @classmethod
    def _missing_type(cls, spec):
        msg = f'{cls.__name__} component specification did not declare a type.'
        msg, attr = reverse_match_suggestion('type', spec, msg)
        raise ValidationError(msg, spec=spec, attr=attr)

    @classmethod
    def from_spec(cls, spec):
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
        context = {} if context is None else context
        print(spec, context)
        if 'type' not in spec:
            cls._missing_type(spec)
        component_cls = cls._get_type(spec['type'], spec)
        component_cls._validate_allowed(spec)
        component_cls._validate_required(spec)
        component_cls._validate_fields(spec, context)
        return spec
