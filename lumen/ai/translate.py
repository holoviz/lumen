from __future__ import annotations

import ast
import datetime
import inspect
import re
import warnings

from collections.abc import Callable
from functools import partial
from inspect import Signature, signature
from types import FunctionType
from typing import (
    Any, ForwardRef, Literal, TypeVar, Union, get_origin,
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
    param.Range: tuple[float, float],
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


def _infer_docstring_style(doc: str) -> DocstringStyle:
    for pattern, replacements, style in _docstring_style_patterns:
        matches = (re.search(pattern.format(replacement), doc, re.IGNORECASE | re.MULTILINE) for replacement in replacements)
        if any(matches):
            return style
    return "google"


def _get_model(
    type_: Any,
    created_models: dict[str, type[BaseModel] | str],
    current_depth: int,
    max_depth: int | None,
    excluded: str | list[str] = "_internal_params",  # Add excluded parameter
) -> Any:
    if isinstance(type_, tuple):
        return tuple(_get_model(t, created_models, current_depth, max_depth, excluded) for t in type_)  # Propagate depth
    try:
        if issubclass(type_, param.Parameterized):  # type: ignore
            model_name = type_.__name__
            if max_depth is not None and current_depth >= max_depth:
                # Depth limit reached for resolving this parameterized type as a field.
                # Return its name as a forward reference, or Any if it's too complex.
                # If it's not yet in created_models, it means we haven't even started processing it.
                # We might want to just return Any in this case to prevent further processing.
                if model_name not in created_models:
                    # print(f"Depth limit reached for {model_name} at depth {current_depth}. Returning Any.") # For debugging
                    return Any  # Or ForwardRef(model_name) if Pydantic can handle unresolved forward refs better
                return created_models.get(model_name, ForwardRef(model_name))

            if model_name not in created_models:
                param_to_pydantic(
                    type_,
                    created_models=created_models,
                    process_subclasses=True,  # Or False depending on desired behavior at depth
                    base_model=ArbitraryTypesModel,
                    current_depth=current_depth,  # Pass current_depth, param_to_pydantic will increment for its own children
                    max_depth=max_depth,
                    excluded=excluded,  # Pass excluded parameter
                )
            return created_models[model_name]
    except TypeError:
        pass
    return type_


def parameter_to_field(
    parameter: param.Parameter,
    created_models: dict[str, type[BaseModel] | str],
    literals: list[str] | None,
    current_depth: int,
    max_depth: int | None,
    excluded: str | list[str] = "_internal_params",  # Add excluded parameter
) -> tuple[type, FieldInfo]:
    param_type = parameter.__class__
    field_kwargs = {}

    if parameter.doc:
        field_kwargs["description"] = " ".join(parameter.doc.split())
    if not literals and hasattr(parameter, "get_range"):
        try:
            literals = list(parameter.get_range())
        except TypeError:
            literals = None

    type_: Any = object

    if param_type in PARAM_TYPE_MAPPING:
        type_ = PARAM_TYPE_MAPPING[param_type]
        if parameter.default is not None and parameter.default is not PydanticUndefined:
            field_kwargs["default"] = parameter.default
    elif param_type is param.ClassSelector:
        if hasattr(parameter, "instantiate") and not parameter.instantiate:
            pydantic_types_for_class_selector = []
            param_classes_to_process = parameter.class_
            if not isinstance(param_classes_to_process, tuple):
                param_classes_to_process = (param_classes_to_process,)

            for cls_item in param_classes_to_process:
                # Pass depth to _get_model
                model_or_ref = _get_model(cls_item, created_models, current_depth, max_depth, excluded)
                pydantic_types_for_class_selector.append(model_or_ref)

            type_args = []
            for item in pydantic_types_for_class_selector:
                if isinstance(item, str):
                    type_args.append(type[ForwardRef(item)])
                elif inspect.isclass(item):
                    type_args.append(type[item])
                else:
                    type_args.append(type[Any])  # Fallback for items that couldn't be resolved (e.g. due to depth)

            if len(type_args) == 1:
                type_ = type_args[0]
            else:
                type_ = Union.__getitem__(tuple(type_args))

            if parameter.default is not None:
                if inspect.isclass(parameter.default):
                    field_kwargs["default"] = parameter.default
                else:
                    warnings.warn(
                        f"Parameter '{parameter.name}' is a ClassSelector with instantiate=False, but its default value '{parameter.default!r}' is not a class. ",
                        UserWarning,
                    )
        elif isinstance(parameter.class_, tuple):
            # Pass depth to _get_model
            mapped_types = [_get_model(cls, created_models, current_depth, max_depth, excluded) for cls in parameter.class_]
            type_ = Union.__getitem__(tuple(mapped_types))
            if parameter.default is not None:
                field_kwargs["default"] = parameter.default
        else:
            # Pass depth to _get_model
            type_ = _get_model(parameter.class_, created_models, current_depth, max_depth, excluded)
            if parameter.default is not None:
                if callable(parameter.default) and not inspect.isclass(parameter.default):
                    field_kwargs["default_factory"] = parameter.default
                elif isinstance(parameter.default, param.Parameterized):
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

        item_type_annotation = Any
        if param_type is param.List and parameter.item_type:
            if isinstance(parameter.item_type, tuple):
                # Pass depth to _get_model for item types
                mapped_item_types = [_get_model(it, created_models, current_depth, max_depth, excluded) for it in parameter.item_type]
                item_type_annotation = Union.__getitem__(tuple(mapped_item_types))
            else:
                # Pass depth to _get_model for item type
                item_type_annotation = _get_model(parameter.item_type, created_models, current_depth, max_depth, excluded)
            type_ = list[item_type_annotation]
        elif param_type is param.ListSelector:
            item_type_annotation = _create_literal(literals if literals is not None else [])
            type_ = list[item_type_annotation]

    elif param_type is param.Dict:
        type_ = dict
        if parameter.default is not None:
            if callable(parameter.default) and not inspect.isclass(parameter.default):
                field_kwargs["default_factory"] = parameter.default
            else:
                field_kwargs["default"] = parameter.default
        elif parameter.default == {}:
            field_kwargs["default_factory"] = dict
        elif parameter.allow_None:
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
            type_ = Any if parameter.allow_None else object
    elif issubclass(param_type, param.DataFrame):
        type_ = PandasDataFrame
    elif parameter.name == "align":
        type_ = _create_literal(["auto", "start", "center", "end"])
    elif parameter.name == "aspect_ratio":
        type_ = Literal["auto"] | float
    elif parameter.name == "margin":
        type_ = float | tuple[float, float] | tuple[float, float, float, float]
    elif param_type.__name__ == "ChildDict":  # Handle by name as it might not be directly importable/typed
        type_ = dict[str, Any]  # If ChildDict values can be Parameterized, this needs depth control too
        if parameter.default is not None:
            if callable(parameter.default) and not inspect.isclass(parameter.default):
                field_kwargs["default_factory"] = parameter.default
            else:
                field_kwargs["default"] = parameter.default
    else:
        # print(f"Warning: Parameter {parameter.name!r} of type {param_type.__name__!r} is not explicitly supported, defaulting to 'object'.")
        type_ = object
        if parameter.default is not None:
            field_kwargs["default"] = parameter.default

    if hasattr(parameter, "bounds") and parameter.bounds:
        actual_type_for_bounds = type_
        if get_origin(actual_type_for_bounds) is Union:
            args = [arg for arg in _typing_extra.get_args(actual_type_for_bounds) if arg is not type(None)]
            if len(args) == 1:
                actual_type_for_bounds = args[0]
        if actual_type_for_bounds in [int, float]:
            try:
                field_kwargs["ge"] = parameter.bounds[0]
                field_kwargs["le"] = parameter.bounds[1]
            except (TypeError, IndexError):
                pass

    is_none_default = parameter.default is None
    if parameter.allow_None:
        current_origin = get_origin(type_)
        current_args = _typing_extra.get_args(type_)
        if not (current_origin is Union and type(None) in current_args) and type_ is not Any:
            type_ = Union[type_, None]  # noqa: UP007
        if is_none_default and "default" not in field_kwargs and "default_factory" not in field_kwargs:
            field_kwargs["default"] = None

    field_info = Field(**field_kwargs)
    return type_, field_info


def param_to_pydantic(
    parameterized: type[param.Parameterized] | list[type[param.Parameterized]],
    base_model: type[BaseModel] | tuple[type[BaseModel], ...] = ArbitraryTypesModel,
    created_models: dict[str, type[BaseModel] | str] | None = None,
    schema: dict[str, Any] | None = None,
    excluded: str | list[str] = "_internal_params",
    extra_fields: dict[str, tuple[type, FieldInfo]] | None = None,
    process_subclasses: bool = True,
    max_depth: int | None = 6,
    current_depth: int = 0,  # Current recursion depth for this instance
) -> dict[str, type[BaseModel] | str]:
    if created_models is None:
        created_models = {}

    if isinstance(parameterized, list):
        for param_class in parameterized:
            param_to_pydantic(  # Propagate depth controls
                param_class, base_model, created_models, schema, excluded, extra_fields, process_subclasses, max_depth, current_depth
            )
        return created_models

    parameterized_name = parameterized.__name__

    # If depth limit is reached for *this class itself*, stop processing it further.
    # We might still want to create a placeholder if it's a field type for another class.
    if max_depth is not None and current_depth >= max_depth:
        if parameterized_name not in created_models:
            created_models[parameterized_name] = parameterized_name  # Placeholder for forward ref
        return created_models

    if parameterized_name in created_models and not isinstance(created_models[parameterized_name], str):
        return created_models

    if parameterized_name not in created_models:
        created_models[parameterized_name] = parameterized_name

    # Increment depth for children (parents, fields, subclasses) of *this* parameterized class
    child_depth = current_depth + 1

    pydantic_model_bases_candidates = []
    for param_parent_cls in parameterized.__bases__:
        if issubclass(param_parent_cls, param.Parameterized) and param_parent_cls is not param.Parameterized:
            if param_parent_cls.__name__ not in created_models or isinstance(created_models[param_parent_cls.__name__], str):
                param_to_pydantic(  # Recursive call for parent
                    param_parent_cls,
                    ArbitraryTypesModel,
                    created_models,
                    process_subclasses=process_subclasses,  # If true, parents' subclasses also processed
                    max_depth=max_depth,
                    current_depth=child_depth,  # Use child_depth
                    excluded=excluded,  # Add excluded parameter
                )
            parent_model_or_ref = created_models.get(param_parent_cls.__name__)
            if isinstance(parent_model_or_ref, type) and issubclass(parent_model_or_ref, BaseModel):
                pydantic_model_bases_candidates.append(parent_model_or_ref)

    if isinstance(base_model, tuple):
        for b_item in base_model:
            if isinstance(b_item, type) and issubclass(b_item, BaseModel):
                pydantic_model_bases_candidates.append(b_item)
    elif isinstance(base_model, type) and issubclass(base_model, BaseModel):
        pydantic_model_bases_candidates.append(base_model)

    fields = {}
    private_attrs: dict[str, PrivateAttr] = {}
    use_literal_mixin = False

    current_excluded_list: list[str]
    if isinstance(excluded, str) and hasattr(parameterized, excluded):
        current_excluded_list = getattr(parameterized, excluded, [])
    elif isinstance(excluded, list):
        current_excluded_list = excluded
    else:
        current_excluded_list = []

    # Create a set of excluded parameters for efficient lookup
    excluded_set = set(current_excluded_list)
    field_params_list = list(getattr(parameterized, "_field_params", []))

    for parameter_name_orig_loop in parameterized.param:
        # Skip excluded parameters
        if parameter_name_orig_loop in excluded_set:
            continue
        parameter_obj = parameterized.param[parameter_name_orig_loop]

        def ensure_deps_processed_loop(attr_val):
            items_to_check = attr_val if isinstance(attr_val, tuple) else (attr_val,)
            for item_check in items_to_check:
                if inspect.isclass(item_check) and issubclass(item_check, param.Parameterized) and item_check.__name__ != parameterized_name:
                    # Use child_depth when resolving field types via _get_model
                    _get_model(item_check, created_models, child_depth, max_depth, excluded)

        if hasattr(parameter_obj, "class_"):
            ensure_deps_processed_loop(parameter_obj.class_)
        if hasattr(parameter_obj, "item_type"):
            ensure_deps_processed_loop(parameter_obj.item_type)

        literals_for_field = list(schema) if schema and parameter_name_orig_loop in field_params_list else None
        # Pass child_depth to parameter_to_field, as fields are "children" in terms of depth
        type_for_field, field_info_obj = parameter_to_field(parameter_obj, created_models, literals_for_field, child_depth, max_depth, excluded)

        if not use_literal_mixin and get_origin(type_for_field) is Literal:
            use_literal_mixin = True

        effective_field_name_loop = parameter_name_orig_loop
        if parameter_name_orig_loop == "schema":
            field_info_obj.alias = "schema"
            effective_field_name_loop = "schema_"
        elif parameter_name_orig_loop == "copy":
            field_info_obj.alias = "copy"
            effective_field_name_loop = "copy_"
        elif parameter_name_orig_loop.startswith("_"):
            private_attrs[parameter_name_orig_loop] = PrivateAttr(default=parameter_obj.default if parameter_obj.default is not None else None)
            continue

        if field_info_obj.default is PydanticUndefined and field_info_obj.default_factory is None and parameter_obj.default is not None:
            is_cs_inst_false_class_default = (
                isinstance(parameter_obj, param.ClassSelector)
                and hasattr(parameter_obj, "instantiate")
                and not parameter_obj.instantiate
                and inspect.isclass(parameter_obj.default)
            )
            if not is_cs_inst_false_class_default:
                field_info_obj.default = parameter_obj.default

        fields[effective_field_name_loop] = (type_for_field, field_info_obj)

    if use_literal_mixin:
        pydantic_model_bases_candidates.append(PartialLiteralMixin)

    ordered_dedup_bases_list = list(dict.fromkeys(b for b in pydantic_model_bases_candidates if isinstance(b, type) and issubclass(b, BaseModel)))

    final_model_creation_base_val: type[BaseModel] | tuple[type[BaseModel], ...]
    if not ordered_dedup_bases_list:
        final_model_creation_base_val = ArbitraryTypesModel
    else:
        minimal_bases_list = []
        for new_base_item in ordered_dedup_bases_list:
            is_redundant_superclass_item = False
            for existing_base_item in minimal_bases_list:
                if issubclass(existing_base_item, new_base_item):
                    is_redundant_superclass_item = True
                    break
            if is_redundant_superclass_item:
                continue
            minimal_bases_list = [eb_item for eb_item in minimal_bases_list if not issubclass(new_base_item, eb_item)]
            minimal_bases_list.append(new_base_item)

        if not minimal_bases_list:
            final_model_creation_base_val = ArbitraryTypesModel
        elif len(minimal_bases_list) == 1:
            final_model_creation_base_val = minimal_bases_list[0]
        else:
            final_model_creation_base_val = tuple(minimal_bases_list)

    # Before creating the model, filter out any fields that match our exclusion list
    # This additional check helps prevent fields like 'download' from being included
    fields_filtered = (extra_fields or {}).copy()
    for field_name, field_info in fields.items():
        # Skip fields that match our excluded list
        skip = False
        if field_name in excluded_set:
            skip = True
        if field_name.endswith("_") and field_name[:-1] in excluded_set:
            skip = True

        if not skip:
            fields_filtered[field_name] = field_info

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pydantic_model_instance = create_model(
            parameterized_name,
            __base__=final_model_creation_base_val,
            **fields_filtered,
        )

    created_models[parameterized_name] = pydantic_model_instance
    setattr(pydantic_model_instance, "_param_source_class", parameterized)
    for attr_name, private_attr_obj_val in private_attrs.items():
        setattr(pydantic_model_instance, attr_name, private_attr_obj_val)

    # This is a partial rebuild that might not fully resolve all ForwardRefs yet
    # Full resolutions are attempted at the end of the initial call
    # Also, ensure we handle exclusions correctly after rebuilding the model
    for model_name_in_cache_val, model_or_ref_in_cache_val in [(parameterized_name, pydantic_model_instance)]:
        if isinstance(model_or_ref_in_cache_val, type) and hasattr(model_or_ref_in_cache_val, "model_rebuild"):
            try:
                model_or_ref_in_cache_val.model_rebuild(force=True, _types_namespace=created_models)  # type: ignore

                # After rebuilding, double-check if any excluded fields need to be removed
                # This helps catch fields that might have been inherited from parent classes
                if excluded:
                    fields_to_exclude = set(current_excluded_list)
                    for field_name in list(model_or_ref_in_cache_val.model_fields.keys()):
                        # If the field name matches any in our exclusion list, we need to handle it
                        if field_name in fields_to_exclude or (field_name.endswith("_") and field_name[:-1] in fields_to_exclude):
                            # Mark this field so we can filter it during instance creation
                            if not hasattr(model_or_ref_in_cache_val, "_excluded_fields"):
                                setattr(model_or_ref_in_cache_val, "_excluded_fields", set())
                            getattr(model_or_ref_in_cache_val, "_excluded_fields").add(field_name)
            except Exception as e_rebuild:
                warnings.warn(f"Failed to rebuild model {model_name_in_cache_val}: {e_rebuild}", UserWarning)

    if process_subclasses:
        if max_depth is None or child_depth < max_depth:  # Check depth before processing subclasses
            try:
                subclasses_to_process_list = set(parameterized.__subclasses__())
            except TypeError:
                subclasses_to_process_list = set()

            for sub_cls_item in subclasses_to_process_list:
                if (
                    inspect.isclass(sub_cls_item)
                    and issubclass(sub_cls_item, param.Parameterized)
                    and sub_cls_item is not param.Parameterized
                    and (sub_cls_item.__name__ not in created_models or isinstance(created_models[sub_cls_item.__name__], str))
                ):
                    param_to_pydantic(  # Recursive call for subclass
                        sub_cls_item,
                        pydantic_model_instance,
                        created_models,
                        excluded=excluded,
                        process_subclasses=True,
                        max_depth=max_depth,
                        current_depth=child_depth,  # Use child_depth
                    )

    if current_depth == 0:  # Only do this global rebuild at the end of the top-level call
        # print("Attempting final model rebuild for all created models...")
        all_pydantic_models_in_cache = {
            name: model_obj for name, model_obj in created_models.items() if isinstance(model_obj, type) and issubclass(model_obj, BaseModel)
        }

        # It might be necessary to rebuild multiple times to resolve chained ForwardRefs
        for _ in range(len(all_pydantic_models_in_cache) + 1):  # Heuristic: rebuild N+1 times
            updated_any = False
            for model_name, model_obj in all_pydantic_models_in_cache.items():
                try:
                    # Crucial: Pydantic needs the _actual_ model objects in the namespace,
                    # not just their names, for resolving ForwardRefs between them.
                    model_obj.model_rebuild(force=True, _types_namespace=all_pydantic_models_in_cache)
                    updated_any = True  # If rebuild doesn't raise, assume it might have done something
                except Exception as e_rebuild_final:
                    # Non-critical warning if some still fail, but main ones should resolve
                    warnings.warn(f"Final rebuild attempt for model {model_name} failed: {e_rebuild_final}", UserWarning, stacklevel=2)
            if not updated_any and _ > 0:  # Optimization: if a pass makes no updates, further passes might not either.
                break
    return created_models


def pydantic_to_param_instance(model: BaseModel, excluded: list[str] | None = None) -> param.Parameterized:
    try:
        parameterized_class = model.__class__._param_source_class  # type: ignore
        valid_param_names = set(parameterized_class.param)
        param_definition_map = parameterized_class.param
    except AttributeError:
        raise ValueError("The provided model does not have a _param_source_class attribute.")

    # Create a set of all excluded field names for efficient lookup
    excluded_set = set(excluded or [])

    # Also check for any fields that were marked as excluded during model creation
    model_excluded_fields = getattr(model.__class__, "_excluded_fields", set())
    excluded_set.update(model_excluded_fields)

    kwargs = {}
    for field_name_in_model in model.model_fields_set:
        original_param_name = field_name_in_model
        field_info = type(model).model_fields.get(field_name_in_model)
        if field_info and field_info.alias and field_info.alias in valid_param_names:
            original_param_name = field_info.alias

        # Skip excluded parameters - check both the field name and its original name
        if original_param_name in excluded_set or field_name_in_model in excluded_set:
            continue

        # Also check for aliased versions with underscores (schema_ for schema, etc.)
        if original_param_name.endswith("_") and original_param_name[:-1] in excluded_set:
            continue

        if original_param_name not in valid_param_names:
            continue

        actual_value = getattr(model, field_name_in_model)
        param_obj = param_definition_map[original_param_name]

        if (
            inspect.isclass(actual_value)
            and hasattr(actual_value, "_param_source_class")
            and isinstance(param_obj, param.ClassSelector)
            and hasattr(param_obj, "instantiate")
            and not param_obj.instantiate
        ):
            kwargs[original_param_name] = actual_value._param_source_class  # type: ignore
        elif isinstance(actual_value, BaseModel):
            kwargs[original_param_name] = pydantic_to_param_instance(actual_value)
        elif isinstance(actual_value, Color):
            kwargs[original_param_name] = actual_value.as_hex()
        elif isinstance(actual_value, list):
            processed_list = []
            for item in actual_value:
                if isinstance(item, BaseModel):
                    processed_list.append(pydantic_to_param_instance(item))
                else:
                    processed_list.append(item)
            kwargs[original_param_name] = processed_list
        elif isinstance(actual_value, dict) and not isinstance(param_obj, param.Dict):
            sub_kwargs = {}
            for k, v_item in actual_value.items():
                if isinstance(v_item, BaseModel):
                    sub_kwargs[k] = pydantic_to_param_instance(v_item)
                else:
                    sub_kwargs[k] = v_item
            kwargs[original_param_name] = sub_kwargs
        elif original_param_name == "table" and isinstance(actual_value, str) and "." in actual_value:
            kwargs[original_param_name] = actual_value.split(".")[0]
        else:
            kwargs[original_param_name] = actual_value

    for k, v in kwargs.items():
        if isinstance(v, str) and v in {"True", "False", "None"}:
            kwargs[k] = ast.literal_eval(v)

    if hasattr(parameterized_class, "instance"):
        return parameterized_class.instance(**kwargs)
    else:
        try:
            return parameterized_class(**kwargs)
        except Exception:
            return partial(parameterized_class, **kwargs)  # type: ignore


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
    if ps := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in ps.value}  # type: ignore
    main_desc = ""
    if ms := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = ms.value  # type: ignore
    return main_desc, params


def function_to_model(function: FunctionType, skipped: list[str] | None = None) -> type[BaseModel]:
    skipped_list = skipped or []
    sig_obj = signature(function)
    type_hints_dict = _typing_extra.get_function_type_hints(function)
    fields_dict: dict[str, tuple[Any, FieldInfo]] = {}
    description_str, field_descriptions_dict = doc_descriptions(function, sig_obj)

    for _, p_obj in enumerate(sig_obj.parameters.items()):
        name_str, param_obj_loop = p_obj
        annotation_val = Any if param_obj_loop.annotation is sig_obj.empty else type_hints_dict.get(name_str, Any)

        field_info_kwargs_dict = {}
        if param_obj_loop.default is not inspect.Parameter.empty:
            field_info_kwargs_dict["default"] = param_obj_loop.default
        if doc_desc_val := field_descriptions_dict.get(name_str):
            field_info_kwargs_dict["description"] = doc_desc_val
        field_info_obj_val = Field(**field_info_kwargs_dict)

        final_annotation_val = annotation_val
        if name_str in skipped_list:
            curr_orig = get_origin(annotation_val)
            curr_args = _typing_extra.get_args(annotation_val)
            if not (curr_orig is Union and type(None) in curr_args) and annotation_val is not Any and annotation_val is not type(None):
                final_annotation_val = Union[annotation_val, None]  # noqa: UP007
            final_annotation_val = SkipJsonSchema[final_annotation_val]  # type: ignore
        fields_dict[name_str] = (final_annotation_val, field_info_obj_val)

    model_instance = create_model(function.__name__, __doc__=description_str, **fields_dict)  # type: ignore
    return model_instance
