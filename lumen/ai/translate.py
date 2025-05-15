from __future__ import annotations

import datetime
import inspect
import re
import warnings

from collections.abc import Callable
from inspect import Signature, signature
from types import FunctionType
from typing import (
    Any, Literal, TypeVar, Union, get_origin,
)

import param

from griffe import Docstring, DocstringSectionKind
from instructor.dsl.partial import PartialLiteralMixin
from pydantic import (
    BaseModel, ConfigDict, Field, PrivateAttr, create_model,
)
from pydantic._internal import _typing_extra
from pydantic.fields import FieldInfo, PydanticUndefined
from pydantic.json_schema import SkipJsonSchema
from pydantic_extra_types.color import Color

DATE_TYPE = datetime.datetime | datetime.date
PARAM_TYPE_MAPPING: dict[param.Parameter, type] = {
    param.String: str,
    param.Integer: int,
    param.Number: float,
    param.Boolean: bool,
    param.Event: bool,
    param.Tuple: tuple,
    param.NumericTuple: tuple,
    param.Date: DATE_TYPE,
    param.DateRange: tuple[DATE_TYPE],
    param.CalendarDate: DATE_TYPE,
    param.CalendarDateRange: tuple[DATE_TYPE],
    param.Parameter: object,
    param.Color: Color,
    param.Callable: Callable,
}
PandasDataFrame = TypeVar("PandasDataFrame")
DocstringStyle = Literal["google", "numpy", "sphinx"]

_docstring_style_patterns: list[tuple[str, list[str], DocstringStyle]] = [
    (
        r"\n[ \t]*:{0}([ \t]+\w+)*:([ \t]+.+)?\n",
        [
            "param",
            "parameter",
            "arg",
            "argument",
            "key",
            "keyword",
            "type",
            "var",
            "ivar",
            "cvar",
            "vartype",
            "returns",
            "return",
            "rtype",
            "raises",
            "raise",
            "except",
            "exception",
        ],
        "sphinx",
    ),
    (
        r"\n[ \t]*{0}:([ \t]+.+)?\n[ \t]+.+",
        [
            "args",
            "arguments",
            "params",
            "parameters",
            "keyword args",
            "keyword arguments",
            "other args",
            "other arguments",
            "other params",
            "other parameters",
            "raises",
            "exceptions",
            "returns",
            "yields",
            "receives",
            "examples",
            "attributes",
            "functions",
            "methods",
            "classes",
            "modules",
            "warns",
            "warnings",
        ],
        "google",
    ),
    (
        r"\n[ \t]*{0}\n[ \t]*---+\n",
        [
            "deprecated",
            "parameters",
            "other parameters",
            "returns",
            "yields",
            "receives",
            "raises",
            "warns",
            "attributes",
            "functions",
            "methods",
            "classes",
            "modules",
        ],
        "numpy",
    ),
]


class ArbitraryTypesModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _create_literal(obj: list[str | type]) -> type:
    enum = []
    for item in obj:
        if item is None:
            continue
        elif isinstance(item, str):
            enum.append(item)
        else:
            enum.append(item.__name__)
    if enum:
        return Literal[tuple(enum)]
    else:
        return str


def _get_model(type_, created_models: dict[str, type[BaseModel]]) -> Any:
    if isinstance(type_, tuple):
        return tuple(_get_model(t, created_models) for t in type_)
    try:
        if issubclass(type_, param.Parameterized):
            if type_.__name__ not in created_models and hasattr(type_, "param"):
                # Pass the default base_model (ArbitraryTypesModel) for parent conversion
                # if not already part of a specific conversion chain.
                param_to_pydantic(type_, created_models=created_models, process_subclasses=True)
            type_ = created_models.get(type_.__name__, type_.__name__)
    except TypeError:
        pass
    return type_


def _infer_docstring_style(doc: str) -> DocstringStyle:
    for pattern, replacements, style in _docstring_style_patterns:
        matches = (re.search(pattern.format(replacement), doc, re.IGNORECASE | re.MULTILINE) for replacement in replacements)
        if any(matches):
            return style
    return "google"


def parameter_to_field(parameter: param.Parameter, created_models: dict[str, type[BaseModel]], literals: list[str] | None) -> tuple[type, FieldInfo]:
    param_type = parameter.__class__
    field_kwargs = {}

    if parameter.doc:
        field_kwargs["description"] = " ".join(parameter.doc.split())
    if not literals and hasattr(parameter, "get_range"):
        literals = list(parameter.get_range())

    if param_type in PARAM_TYPE_MAPPING:
        type_ = PARAM_TYPE_MAPPING[param_type]
        if parameter.default is not None and parameter.default is not PydanticUndefined:
            field_kwargs["default"] = parameter.default
    elif param_type is param.ClassSelector:
        if hasattr(parameter, "instantiate") and not parameter.instantiate:
            if hasattr(parameter.class_, "param"):
                param_to_pydantic(parameter.class_, created_models=created_models, process_subclasses=True)
            if parameter.default is not None and inspect.isclass(parameter.default):
                if hasattr(parameter.default, "param"):
                    param_to_pydantic(parameter.default, created_models=created_models, process_subclasses=True)

            base_class_name = parameter.class_.__name__
            # Type should be the Pydantic model class itself, not an instance
            type_ = created_models.get(base_class_name, parameter.class_)

            if parameter.default is not None:
                if inspect.isclass(parameter.default):
                    default_class_name = parameter.default.__name__
                    # Default should also be the Pydantic model class
                    field_kwargs["default"] = created_models.get(default_class_name, parameter.default)
                else:  # Potentially an instance, or other non-class default.
                    field_kwargs["default"] = parameter.default

        elif isinstance(parameter.class_, tuple):
            for cls in parameter.class_:
                if hasattr(cls, "param"):
                    param_to_pydantic(cls, created_models=created_models, process_subclasses=True)
            mapped_types = []
            for cls in parameter.class_:
                model_name = cls.__name__
                if hasattr(cls, "param") and model_name in created_models:
                    mapped_types.append(created_models[model_name])
                else:  # Not a param class or not converted, use raw type or mapping
                    mapped_types.append(PARAM_TYPE_MAPPING.get(cls, cls))
            type_ = Union.__getitem__(tuple(mapped_types))
            if parameter.default is not None:  # Add default handling for tuple class types
                field_kwargs["default"] = parameter.default

        else:
            type_ = _get_model(parameter.class_, created_models)
            if parameter.default is not None:
                if callable(parameter.default) and not inspect.isclass(parameter.default):
                    default_factory = parameter.default
                    field_kwargs["default_factory"] = default_factory
                elif isinstance(parameter.default, param.Parameterized):
                    pass  # Leave default undefined for pydantic to handle via type_
                else:
                    field_kwargs["default"] = parameter.default
    elif param_type in [param.List, param.ListSelector]:
        type_ = list
        if parameter.default is not None:
            if callable(parameter.default):
                field_kwargs["default_factory"] = parameter.default
            else:
                field_kwargs["default"] = parameter.default
        if param_type is param.List and parameter.item_type:
            if isinstance(parameter.item_type, tuple):
                for cls in parameter.item_type:
                    if hasattr(cls, "param"):
                        param_to_pydantic(cls, created_models=created_models)
                mapped_types = []
                for cls in parameter.item_type:
                    model_name = cls.__name__
                    if hasattr(cls, "param") and model_name in created_models:
                        mapped_types.append(created_models[model_name])
                    else:
                        mapped_types.append(PARAM_TYPE_MAPPING.get(cls, cls))
                item_type = Union.__getitem__(tuple(mapped_types))
                type_ = list[item_type]
            else:
                type_ = list[_get_model(parameter.item_type, created_models)]
        elif param_type is param.ListSelector:
            # Ensure literals is not None; if it's None, this might error or behave unexpectedly
            type_ = list[_create_literal(literals if literals is not None else [])]
    elif param_type is param.Dict:
        type_ = dict
        if parameter.default == {}:  # Check for empty dict default specifically
            field_kwargs["default_factory"] = dict
        elif parameter.default is not None:
            if callable(parameter.default):  # Typically for default_factory, but param.Dict doesn't have it
                pass  # Let Pydantic handle callable default if it's not default_factory
            else:
                field_kwargs["default"] = parameter.default
        elif parameter.default is None and parameter.allow_None:  # Explicitly handle None default with allow_None
            field_kwargs["default"] = None

    elif param_type in [param.Selector, param.ObjectSelector]:
        if parameter.default is not None:
            field_kwargs["default"] = parameter.default

        current_literals = literals
        if not current_literals and hasattr(parameter, "objects"):  # Use objects if literals not provided
            current_literals = parameter.objects

        if current_literals:
            type_ = _create_literal(current_literals)
        else:  # Fallback if no literals or objects
            type_ = Any if parameter.allow_None else str  # Or object, or more specific based on context
    elif issubclass(param_type, param.DataFrame):
        type_ = PandasDataFrame
    elif parameter.name == "align":
        type_ = _create_literal(["auto", "start", "center", "end"])
    elif parameter.name == "aspect_ratio":
        type_ = Literal["auto"] | float
    elif parameter.name == "margin":
        type_ = float | tuple[float, float] | tuple[float, float, float, float]
    else:
        raise NotImplementedError(f"Parameter {parameter.name!r} of {param_type.__name__!r} not supported")

    if hasattr(parameter, "bounds") and parameter.bounds and type_ in [int, float]:
        try:
            field_kwargs["ge"] = parameter.bounds[0]
            field_kwargs["le"] = parameter.bounds[1]
        except (TypeError, IndexError):  # Handle if bounds is not as expected
            pass

    is_none_default = parameter.default is None

    if parameter.allow_None:
        type_ = Union[type_, None]  # type: ignore # noqa: UP007
        if is_none_default and "default" not in field_kwargs and "default_factory" not in field_kwargs:
            field_kwargs["default"] = None

    field_info = Field(**field_kwargs)
    return type_, field_info


def param_to_pydantic(
    parameterized: type[param.Parameterized] | list[type[param.Parameterized]],
    base_model: type[BaseModel] = ArbitraryTypesModel,
    created_models: dict[str, type[BaseModel]] | None = None,  # MODIFIED type hint
    schema: dict[str, Any] | None = None,
    excluded: str | list[str] = "_internal_params",
    extra_fields: dict[str, tuple[type, FieldInfo]] | None = None,
    process_subclasses: bool = True,  # ADDED parameter
) -> dict[str, type[BaseModel]]:  # MODIFIED return type hint
    if created_models is None:
        created_models = {}

    if isinstance(parameterized, list):
        for param_class in parameterized:
            # Propagate process_subclasses when handling a list
            param_to_pydantic(
                param_class,
                base_model=base_model,
                created_models=created_models,
                schema=schema,
                excluded=excluded,
                extra_fields=extra_fields,
                process_subclasses=process_subclasses,  # PROPAGATE
            )
        return created_models

    parameterized_name = parameterized.__name__
    if parameterized_name in created_models:
        return created_models

    pydantic_model_bases = []
    # Iterate over direct base classes of `parameterized`
    for param_parent_cls in parameterized.__bases__:
        # Check if the parent is a param.Parameterized class itself,
        # but not the root param.Parameterized (to avoid issues if param.Parameterized is in created_models)
        if issubclass(param_parent_cls, param.Parameterized) and param_parent_cls is not param.Parameterized:
            # Ensure parent Pydantic model is created if it doesn't exist
            if param_parent_cls.__name__ not in created_models:
                # Recursively call param_to_pydantic for the parent.
                # Propagate `base_model` for consistent ultimate ancestor.
                # Propagate `process_subclasses`.
                param_to_pydantic(
                    param_parent_cls,
                    base_model=base_model,
                    created_models=created_models,
                    # schema, excluded, extra_fields are generally not propagated
                    # to parents unless explicitly needed, as they are often
                    # specific to the current class conversion.
                    process_subclasses=process_subclasses,  # PROPAGATE
                )

            # If the parent Pydantic model was created or already existed, add it to bases
            if param_parent_cls.__name__ in created_models:
                parent_pydantic_cls = created_models[param_parent_cls.__name__]
                if parent_pydantic_cls not in pydantic_model_bases:  # Avoid duplicates
                    pydantic_model_bases.append(parent_pydantic_cls)

    # If no direct param parent Pydantic model was added (e.g., `parameterized`
    # inherits directly from param.Parameterized, or its param parents weren't
    # converted for some reason), then use the `base_model` argument as the base.
    if not pydantic_model_bases:
        if isinstance(base_model, tuple):
            # If base_model is a tuple of bases, add them all
            for b_item in base_model:
                if b_item not in pydantic_model_bases:  # Should be redundant if list is empty
                    pydantic_model_bases.append(b_item)
        elif base_model not in pydantic_model_bases:  # Should be redundant if list is empty
            pydantic_model_bases.append(base_model)

    current_excluded: list[str]
    if isinstance(excluded, str) and hasattr(parameterized, excluded):
        current_excluded = getattr(parameterized, excluded, [])
    elif isinstance(excluded, list):  # If excluded is already a list
        current_excluded = excluded
    else:  # Fallback: excluded is a string but not an attribute, or other unexpected type
        current_excluded = []

    field_params = list(getattr(parameterized, "_field_params", []))

    fields = {}
    private_attrs: dict[str, PrivateAttr] = {}  # Store PrivateAttr objects
    use_literal_mixin = False

    for parameter_name in parameterized.param:
        if parameter_name in current_excluded:
            continue
        parameter = parameterized.param[parameter_name]

        # Handle nested Parameterized classes in ClassSelector or List item_type
        classes_to_process_for_field = []
        if hasattr(parameter, "class_"):  # For param.ClassSelector
            if isinstance(parameter.class_, tuple):
                classes_to_process_for_field.extend(c for c in parameter.class_ if inspect.isclass(c))
            elif inspect.isclass(parameter.class_):
                classes_to_process_for_field.append(parameter.class_)
        if hasattr(parameter, "item_type"):  # For param.List
            if isinstance(parameter.item_type, tuple):
                classes_to_process_for_field.extend(c for c in parameter.item_type if inspect.isclass(c))
            elif inspect.isclass(parameter.item_type):
                classes_to_process_for_field.append(parameter.item_type)

        for cls_to_check in classes_to_process_for_field:
            if hasattr(cls_to_check, "param") and cls_to_check.__name__ != parameterized_name:
                if cls_to_check.__name__ not in created_models:
                    param_to_pydantic(  # Recursive call for nested class types
                        cls_to_check,
                        base_model=base_model,
                        created_models=created_models,
                        process_subclasses=process_subclasses,  # PROPAGATE
                    )

        literals = list(schema) if schema and parameter_name in field_params else None
        type_, field_info = parameter_to_field(parameter, created_models, literals)

        if not use_literal_mixin and get_origin(type_) is Literal:
            use_literal_mixin = True

        if parameter_name == "schema":
            field_info.alias = "schema"  # type: ignore
            parameter_name = "schema_"
        elif parameter_name == "copy":
            field_info.alias = "copy"  # type: ignore
            parameter_name = "copy_"
        elif parameter_name.startswith("_"):
            # Ensure default is wrapped in PrivateAttr for private fields
            private_attrs[parameter_name] = PrivateAttr(default=parameter.default if parameter.default is not None else None)
            continue  # Skip adding to regular fields

        # Pydantic's FieldInfo.from_annotation handles default Undefined for required params.
        # We only override if param has a specific default that Pydantic might miss.
        if field_info.default is PydanticUndefined and parameter.default is not None:
            field_info.default = parameter.default
        # If allow_None is True, Pydantic will make it Optional if not already.
        # Explicit `default=None` is set by parameter_to_field if param.default is None and allow_None.
        fields[parameter_name] = (type_, field_info)

    if use_literal_mixin:
        if PartialLiteralMixin not in pydantic_model_bases:
            pydantic_model_bases.append(PartialLiteralMixin)

    # Deduplicate bases while preserving order (important for MRO)
    final_model_creation_base: type[BaseModel] | tuple[type[BaseModel], ...]
    unique_final_bases = []
    seen_bases = set()
    for b_item in pydantic_model_bases:
        if b_item not in seen_bases:
            unique_final_bases.append(b_item)
            seen_bases.add(b_item)

    if not unique_final_bases:  # Should ideally not happen if ArbitraryTypesModel is a fallback
        final_model_creation_base = ArbitraryTypesModel
    elif len(unique_final_bases) == 1:
        final_model_creation_base = unique_final_bases[0]
    else:
        final_model_creation_base = tuple(unique_final_bases)

    fields["_parameterized"] = (type, parameterized)  # type: ignore

    if extra_fields:
        fields.update(extra_fields)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # For `__parameterized__` field
        pydantic_model = create_model(
            parameterized.__name__,
            __base__=final_model_creation_base,
            **fields,  # type: ignore
        )

    # Add private attributes to the model class after creation
    for attr_name, private_attr_obj in private_attrs.items():
        setattr(pydantic_model, attr_name, private_attr_obj)

    created_models[pydantic_model.__name__] = pydantic_model  # type: ignore

    if process_subclasses:
        try:
            # Use a set to avoid processing the same subclass multiple times
            subclasses_to_process = set(parameterized.__subclasses__())
        except TypeError:  # Should not happen for classes, but defensive
            subclasses_to_process = set()

        for sub_cls in subclasses_to_process:
            # Check if it's a param.Parameterized class and not yet processed
            if (
                inspect.isclass(sub_cls)
                and issubclass(sub_cls, param.Parameterized)
                and sub_cls is not param.Parameterized
                and sub_cls.__name__ not in created_models
            ):
                # Recursively call for subclasses
                param_to_pydantic(
                    sub_cls,
                    base_model=base_model,  # Propagate original base_model for consistency
                    created_models=created_models,  # Pass the same dict to accumulate models
                    schema=None,  # Schemas are specific, don't propagate by default
                    excluded=excluded,  # Propagate exclusion rules
                    extra_fields=None,  # Extra fields are specific
                    process_subclasses=True,  # Continue processing for sub-subclasses
                )
    return created_models


def pydantic_to_param_instance(model: BaseModel) -> param.Parameterized:
    """
    Tries to convert a Pydantic model instance back to a param.Parameterized instance.
    Only valid if the model was initially translated using `param_to_pydantic`.
    """
    try:
        parameterized_class = model._parameterized  # type: ignore
        valid_param_names = set(parameterized_class.param)
    except AttributeError:
        raise ValueError("The provided model does not have a _parameterized attribute, indicating it was not created from a param.Parameterized class.")

    kwargs = {}
    for key, value in model:
        original_key = key
        # Handle aliased fields (schema_ -> schema, copy_ -> copy)
        if key == "schema_" and "schema" in valid_param_names:
            original_key = "schema"
        elif key == "copy_" and "copy" in valid_param_names:
            original_key = "copy"

        if original_key not in valid_param_names:
            continue  # Skip fields not present in the param.Parameterized class

        if isinstance(value, BaseModel):
            # Recursively convert nested Pydantic models to param.Parameterized instances
            kwargs[original_key] = pydantic_to_param_instance(value)
        elif isinstance(value, Color):  # pydantic_extra_types.color.Color
            kwargs[original_key] = value.as_hex()
        elif isinstance(value, list):
            processed_list = []
            for item in value:
                if isinstance(item, BaseModel):
                    processed_list.append(pydantic_to_param_instance(item))
                else:
                    # Otherwise, keep the item as is (e.g., str, int, dict if not a model)
                    processed_list.append(item)
            kwargs[original_key] = processed_list
        elif isinstance(value, dict):  # For general dictionaries
            # Recursively process dictionaries that might contain BaseModel instances
            # This is important if a dict field itself can contain Pydantic models
            # (though less common for param.Dict which usually expects simple types or
            # specific structures not auto-converted to Pydantic models by param_to_pydantic).
            sub_kwargs = {}
            for k, v_item in value.items():
                if isinstance(v_item, BaseModel):
                    sub_kwargs[k] = pydantic_to_param_instance(v_item)
                else:
                    sub_kwargs[k] = v_item
            kwargs[original_key] = sub_kwargs
        elif original_key == "table" and isinstance(value, str) and "." in value:
            # Specific handling for 'table' field as per original code
            kwargs[original_key] = value.split(".")[0]
        else:
            # For all other types, assign the value directly
            kwargs[original_key] = value

    parameterized_instance = parameterized_class(**kwargs)
    return parameterized_instance


def doc_descriptions(function: FunctionType, sig: Signature | None = None) -> tuple[str, dict[str, str]]:
    doc = function.__doc__
    if doc is None:
        return "", {}
    if sig is None:
        sig = signature(function)

    parser = _infer_docstring_style(doc)
    docstring = Docstring(doc, lineno=1, parser=parser, parent=sig)  # type: ignore
    sections = docstring.parse()
    params = {}
    if parameters_section := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters_section.value}  # type: ignore

    main_desc = ""
    if main_section := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main_section.value  # type: ignore

    return main_desc, params


def function_to_model(function: FunctionType, skipped: list[str] | None = None) -> type[BaseModel]:
    skipped = skipped or []
    sig = signature(function)
    type_hints = _typing_extra.get_function_type_hints(function)
    fields: dict[str, tuple[Any, FieldInfo]] = {}  # Changed core_schema.TypedDictField to tuple[Any, FieldInfo] for create_model
    description, field_descriptions = doc_descriptions(function, sig)

    for index, (name, p) in enumerate(sig.parameters.items()):
        if p.annotation is sig.empty:
            annotation = Any
        else:
            annotation = type_hints.get(name, Any)  # Use .get for safety

        field_name = p.name
        # Create FieldInfo from annotation, then update description
        field_info_kwargs = {}
        if p.default is not inspect.Parameter.empty:
            field_info_kwargs["default"] = p.default

        # Get description from docstring
        doc_desc = field_descriptions.get(field_name)
        if doc_desc:
            field_info_kwargs["description"] = doc_desc

        field_info_obj = Field(**field_info_kwargs)

        final_annotation = annotation
        if name in skipped:
            # Ensure Union with None if original annotation didn't allow it, for SkipJsonSchema
            if get_origin(annotation) is Union:
                args = _typing_extra.get_args(annotation)
                if type(None) not in args:  # type: ignore
                    final_annotation = Union[annotation, None]  # type: ignore # noqa: UP007
            elif annotation is not Any and annotation is not type(None):  # type: ignore
                final_annotation = Union[annotation, None]  # type: ignore # noqa: UP007
            final_annotation = SkipJsonSchema[final_annotation]  # type: ignore

        fields[field_name] = (final_annotation, field_info_obj)

    model = create_model(function.__name__, __doc__=description, **fields)  # type: ignore
    return model
