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
    param.Magnitude: float,
    param.HookList: list[Callable],
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
                param_to_pydantic(type_, created_models=created_models, process_subclasses=True)
            type_ = created_models.get(type_.__name__, type_.__name__)
    except TypeError: # issubclass can raise TypeError if type_ is not a class (e.g. an instance or specific type obj)
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
        try:
            literals = list(parameter.get_range())
        except TypeError:
            literals = None

    if param_type in PARAM_TYPE_MAPPING:
        type_ = PARAM_TYPE_MAPPING[param_type]
        if parameter.default is not None and parameter.default is not PydanticUndefined:
            field_kwargs["default"] = parameter.default
    elif param_type is param.ClassSelector:
        # Handle instantiate=False: field should accept the class itself.
        if hasattr(parameter, "instantiate") and not parameter.instantiate:
            pydantic_model_classes_for_type = []
            param_classes_to_process = parameter.class_
            if not isinstance(param_classes_to_process, tuple):
                param_classes_to_process = (param_classes_to_process,)

            for cls_item in param_classes_to_process:
                if hasattr(cls_item, "param"): # If it's a param.Parameterized subclass
                    # Ensure its Pydantic model is created
                    param_to_pydantic(cls_item, created_models=created_models, process_subclasses=True)
                    pydantic_model_classes_for_type.append(created_models.get(cls_item.__name__, cls_item))
                else: # If it's a regular class
                    pydantic_model_classes_for_type.append(cls_item)

            if len(pydantic_model_classes_for_type) == 1:
                # Pydantic type will be type[PydanticModel] or type[RegularClass]
                type_ = type[pydantic_model_classes_for_type[0]]
            else:
                # Pydantic type will be Union[type[PydanticModelA], type[PydanticModelB], ...]
                union_args = tuple(type[m_cls] for m_cls in pydantic_model_classes_for_type)
                type_ = Union.__getitem__(union_args)

            if parameter.default is not None:
                if inspect.isclass(parameter.default):
                    # Default value for Pydantic field should be the Pydantic model class (if applicable)
                    # or the regular class itself.
                    default_value_for_pydantic = created_models.get(parameter.default.__name__, parameter.default)
                    field_kwargs["default"] = default_value_for_pydantic
                else:
                    # A non-class default for instantiate=False ClassSelector is unusual and likely problematic.
                    warnings.warn(
                        f"Parameter '{parameter.name}' is a ClassSelector with instantiate=False, "
                        f"but its default value '{parameter.default!r}' is not a class. "
                        "This default will not be translated to the Pydantic model.", UserWarning
                    )
        # Handle instantiate=True (or default behavior)
        elif isinstance(parameter.class_, tuple): # Union of classes, instantiate instances
            for cls in parameter.class_:
                if hasattr(cls, "param"):
                    param_to_pydantic(cls, created_models=created_models, process_subclasses=True)
            mapped_types = []
            for cls in parameter.class_:
                model_name = cls.__name__
                if hasattr(cls, "param") and model_name in created_models:
                    mapped_types.append(created_models[model_name])
                else: # Not a param class or not converted, use raw type or mapping
                    mapped_types.append(PARAM_TYPE_MAPPING.get(cls, cls)) # Should just be cls if not in mapping
            type_ = Union.__getitem__(tuple(mapped_types))
            if parameter.default is not None:
                field_kwargs["default"] = parameter.default
        else: # Single class, instantiate instance
            type_ = _get_model(parameter.class_, created_models) # Gets Pydantic model class
            if parameter.default is not None:
                if callable(parameter.default) and not inspect.isclass(parameter.default):
                    default_factory = parameter.default
                    field_kwargs["default_factory"] = default_factory
                elif isinstance(parameter.default, param.Parameterized): # Default is an instance
                    # Pydantic will initialize from type if default is an instance of compatible param class
                    # Or, if we want to pass the raw instance: field_kwargs["default"] = parameter.default
                    # For now, let Pydantic handle instantiation from type.
                    pass
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
            type_ = list[_create_literal(literals if literals is not None else [])]
    elif param_type is param.Dict:
        type_ = dict
        if parameter.default == {}:
            field_kwargs["default_factory"] = dict
        elif parameter.default is not None:
            if callable(parameter.default):
                pass
            else:
                field_kwargs["default"] = parameter.default
        elif parameter.default is None and parameter.allow_None:
            field_kwargs["default"] = None

    elif param_type in [param.Selector, param.ObjectSelector]:
        if parameter.default is not None:
            field_kwargs["default"] = parameter.default

        current_literals = literals
        if not current_literals and hasattr(parameter, "objects"):
            current_literals = parameter.objects

        if current_literals:
            type_ = _create_literal(current_literals)
        else:
            type_ = Any if parameter.allow_None else str
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
        except (TypeError, IndexError):
            pass

    is_none_default = parameter.default is None

    if parameter.allow_None:
        type_ = Union[type_, None]  # noqa: UP007
        # using Union instead of | because of https://github.com/567-labs/instructor/issues/1523
        if is_none_default and "default" not in field_kwargs and "default_factory" not in field_kwargs:
            field_kwargs["default"] = None

    field_info = Field(**field_kwargs)
    return type_, field_info


def param_to_pydantic(
    parameterized: type[param.Parameterized] | list[type[param.Parameterized]],
    base_model: type[BaseModel] = ArbitraryTypesModel,
    created_models: dict[str, type[BaseModel]] | None = None,
    schema: dict[str, Any] | None = None,
    excluded: str | list[str] = "_internal_params",
    extra_fields: dict[str, tuple[type, FieldInfo]] | None = None,
    process_subclasses: bool = True,
) -> dict[str, type[BaseModel]]:
    if created_models is None:
        created_models = {}

    if isinstance(parameterized, list):
        for param_class in parameterized:
            param_to_pydantic(
                param_class,
                base_model=base_model,
                created_models=created_models,
                schema=schema,
                excluded=excluded,
                extra_fields=extra_fields,
                process_subclasses=process_subclasses,
            )
        return created_models

    parameterized_name = parameterized.__name__
    if parameterized_name in created_models:
        return created_models

    pydantic_model_bases = []
    for param_parent_cls in parameterized.__bases__:
        if issubclass(param_parent_cls, param.Parameterized) and param_parent_cls is not param.Parameterized:
            if param_parent_cls.__name__ not in created_models:
                param_to_pydantic(
                    param_parent_cls,
                    base_model=base_model,
                    created_models=created_models,
                    process_subclasses=process_subclasses,
                )
            if param_parent_cls.__name__ in created_models:
                parent_pydantic_cls = created_models[param_parent_cls.__name__]
                if parent_pydantic_cls not in pydantic_model_bases:
                    pydantic_model_bases.append(parent_pydantic_cls)

    if not pydantic_model_bases:
        if isinstance(base_model, tuple):
            for b_item in base_model:
                if b_item not in pydantic_model_bases:
                    pydantic_model_bases.append(b_item)
        elif base_model not in pydantic_model_bases:
            pydantic_model_bases.append(base_model)

    current_excluded: list[str]
    if isinstance(excluded, str) and hasattr(parameterized, excluded):
        current_excluded = getattr(parameterized, excluded, [])
    elif isinstance(excluded, list):
        current_excluded = excluded
    else:
        current_excluded = []

    field_params = list(getattr(parameterized, "_field_params", []))

    fields = {}
    private_attrs: dict[str, PrivateAttr] = {}
    use_literal_mixin = False

    for parameter_name in parameterized.param:
        if parameter_name in current_excluded:
            continue
        parameter = parameterized.param[parameter_name]

        classes_to_process_for_field = []
        if hasattr(parameter, "class_"):
            param_cls_attr = parameter.class_
            if isinstance(param_cls_attr, tuple):
                classes_to_process_for_field.extend(c for c in param_cls_attr if inspect.isclass(c))
            elif inspect.isclass(param_cls_attr):
                classes_to_process_for_field.append(param_cls_attr)
        if hasattr(parameter, "item_type"):
            item_type_attr = parameter.item_type
            if isinstance(item_type_attr, tuple):
                classes_to_process_for_field.extend(c for c in item_type_attr if inspect.isclass(c))
            elif inspect.isclass(item_type_attr):
                classes_to_process_for_field.append(item_type_attr)

        for cls_to_check in classes_to_process_for_field:
            if hasattr(cls_to_check, "param") and cls_to_check.__name__ != parameterized_name:
                if cls_to_check.__name__ not in created_models:
                    param_to_pydantic(
                        cls_to_check,
                        base_model=base_model,
                        created_models=created_models,
                        process_subclasses=process_subclasses,
                    )

        literals = list(schema) if schema and parameter_name in field_params else None
        type_, field_info = parameter_to_field(parameter, created_models, literals)

        if not use_literal_mixin and get_origin(type_) is Literal:
            use_literal_mixin = True

        if parameter_name == "schema":
            field_info.alias = "schema"
            parameter_name = "schema_"
        elif parameter_name == "copy":
            field_info.alias = "copy"
            parameter_name = "copy_"
        elif parameter_name.startswith("_"):
            private_attrs[parameter_name] = PrivateAttr(default=parameter.default if parameter.default is not None else None)
            continue

        if field_info.default is PydanticUndefined and parameter.default is not None:
             # Check if default was already handled by parameter_to_field (e.g. for instantiate=False)
            if not (isinstance(parameter, param.ClassSelector) and \
                    hasattr(parameter, "instantiate") and not parameter.instantiate and \
                    inspect.isclass(parameter.default)):
                field_info.default = parameter.default

        fields[parameter_name] = (type_, field_info)

    if use_literal_mixin:
        if PartialLiteralMixin not in pydantic_model_bases:
            pydantic_model_bases.append(PartialLiteralMixin)

    final_model_creation_base: type[BaseModel] | tuple[type[BaseModel], ...]
    unique_final_bases = []
    seen_bases = set()
    for b_item in pydantic_model_bases:
        if b_item not in seen_bases:
            unique_final_bases.append(b_item)
            seen_bases.add(b_item)

    if not unique_final_bases:
        final_model_creation_base = ArbitraryTypesModel
    elif len(unique_final_bases) == 1:
        final_model_creation_base = unique_final_bases[0]
    else:
        final_model_creation_base = tuple(unique_final_bases)

    if extra_fields:
        fields.update(extra_fields)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pydantic_model = create_model(
            parameterized.__name__,
            __base__=final_model_creation_base,
            **fields,
        )

    setattr(pydantic_model, "_param_source_class", parameterized)

    for attr_name, private_attr_obj in private_attrs.items():
        setattr(pydantic_model, attr_name, private_attr_obj)

    created_models[pydantic_model.__name__] = pydantic_model


    if process_subclasses:
        try:
            subclasses_to_process = set(parameterized.__subclasses__())
        except TypeError:
            subclasses_to_process = set()

        for sub_cls in subclasses_to_process:
            if (
                inspect.isclass(sub_cls)
                and issubclass(sub_cls, param.Parameterized)
                and sub_cls is not param.Parameterized
                and sub_cls.__name__ not in created_models
            ):
                param_to_pydantic(
                    sub_cls,
                    base_model=base_model,
                    created_models=created_models,
                    schema=None,
                    excluded=excluded,
                    extra_fields=None,
                    process_subclasses=True,
                )
    return created_models


def pydantic_to_param_instance(model: BaseModel) -> param.Parameterized:
    try:
        parameterized_class = model.__class__._param_source_class
        valid_param_names = set(parameterized_class.param)
        param_definition_map = parameterized_class.param
    except AttributeError:
        raise ValueError(
            "The provided model does not have a _param_source_class attribute, "
            "indicating it was not correctly created from a param.Parameterized class or the attribute is missing."
        )

    kwargs = {}
    model_dict = model.model_dump()

    for key, _ in model_dict.items(): # Iterate over keys from model_dump
        original_key = key
        if key == "schema_" and "schema" in valid_param_names:
            original_key = "schema"
        elif key == "copy_" and "copy" in valid_param_names:
            original_key = "copy"

        if original_key not in valid_param_names:
            continue

        actual_value = getattr(model, key) # Get actual value from model instance
        param_obj = param_definition_map[original_key]

        if (inspect.isclass(actual_value) and
            issubclass(actual_value, BaseModel) and
            hasattr(actual_value, "_param_source_class") and
            isinstance(param_obj, param.ClassSelector) and
            hasattr(param_obj, "instantiate") and
            not param_obj.instantiate):
            kwargs[original_key] = actual_value._param_source_class
        elif isinstance(actual_value, BaseModel):
            kwargs[original_key] = pydantic_to_param_instance(actual_value)
        elif isinstance(actual_value, Color):
            kwargs[original_key] = actual_value.as_hex()
        elif isinstance(actual_value, list):
            processed_list = []
            original_list_items = getattr(model, key)
            for item_idx, item_in_dump in enumerate(actual_value):
                original_item = original_list_items[item_idx]
                if isinstance(original_item, BaseModel):
                    processed_list.append(pydantic_to_param_instance(original_item))
                else:
                    processed_list.append(original_item)
            kwargs[original_key] = processed_list
        elif isinstance(actual_value, dict) and not isinstance(param_obj, param.Dict):
            # This condition is to handle general dicts that might contain BaseModels,
            # but specifically not for param.Dict fields which are handled by direct assignment of 'value'
            # (which would be the dict from model_dump).
            # This block might need re-evaluation if param.Dict itself can contain Pydantic models
            # that need recursive conversion. For now, assume param.Dict values are simple.
            sub_kwargs = {}
            original_dict_items = getattr(model, key) # Get the actual dict from the model
            for k, v_item_in_dump in actual_value.items():
                original_v_item = original_dict_items[k]
                if isinstance(original_v_item, BaseModel):
                    sub_kwargs[k] = pydantic_to_param_instance(original_v_item)
                else:
                    sub_kwargs[k] = original_v_item
            kwargs[original_key] = sub_kwargs
        elif original_key == "table" and isinstance(actual_value, str) and "." in actual_value:
            kwargs[original_key] = actual_value.split(".")[0]
        else:
            # For all other types (including simple ones from model_dump, or param.Dict value)
            kwargs[original_key] = actual_value # Use the value from model_dump (or actual_value from model)

    try:
        parameterized_instance = parameterized_class(**kwargs)
    except TypeError as e:
        # Attempt with .instance() if direct instantiation fails, common for some param classes
        # This might occur if __init__ is not standard.
        try:
            parameterized_instance = parameterized_class.instance(**kwargs)
        except Exception as e_instance:
            raise TypeError(f"Failed to instantiate {parameterized_class.__name__} with direct __init__: {e}. "
                            f"Also failed with .instance(): {e_instance}. Kwargs: {kwargs}") from e_instance

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
