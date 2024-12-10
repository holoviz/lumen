from __future__ import annotations

import datetime
import inspect
import re
import warnings

from collections.abc import Callable
from inspect import Signature, signature
from types import FunctionType
from typing import (
    Any, Literal, TypeVar, Union,
)

import param

from _griffe.enumerations import DocstringSectionKind
from _griffe.models import Docstring
from pydantic import BaseModel, ConfigDict, create_model
from pydantic._internal import _typing_extra
from pydantic.fields import FieldInfo, PydanticUndefined
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import core_schema
from pydantic_extra_types.color import Color

DATE_TYPE = datetime.datetime | datetime.date
PARAM_TYPE_MAPPING: dict[param.Parameter, type] = {
    param.String: str,
    param.Integer: int,
    param.Number: float,
    param.Boolean: bool,
    param.Event: bool,
    param.Date: DATE_TYPE,
    param.DateRange: tuple[DATE_TYPE],
    param.CalendarDate: DATE_TYPE,
    param.CalendarDateRange: tuple[DATE_TYPE],
    param.Parameter: object,
    param.Color: Color,
    param.Callable: Callable,
}
PandasDataFrame = TypeVar("PandasDataFrame")
DocstringStyle = Literal['google', 'numpy', 'sphinx']

# See https://github.com/mkdocstrings/griffe/issues/329#issuecomment-2425017804
_docstring_style_patterns: list[tuple[str, list[str], DocstringStyle]] = [
    (
        r'\n[ \t]*:{0}([ \t]+\w+)*:([ \t]+.+)?\n',
        [
            'param',
            'parameter',
            'arg',
            'argument',
            'key',
            'keyword',
            'type',
            'var',
            'ivar',
            'cvar',
            'vartype',
            'returns',
            'return',
            'rtype',
            'raises',
            'raise',
            'except',
            'exception',
        ],
        'sphinx',
    ),
    (
        r'\n[ \t]*{0}:([ \t]+.+)?\n[ \t]+.+',
        [
            'args',
            'arguments',
            'params',
            'parameters',
            'keyword args',
            'keyword arguments',
            'other args',
            'other arguments',
            'other params',
            'other parameters',
            'raises',
            'exceptions',
            'returns',
            'yields',
            'receives',
            'examples',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
            'warns',
            'warnings',
        ],
        'google',
    ),
    (
        r'\n[ \t]*{0}\n[ \t]*---+\n',
        [
            'deprecated',
            'parameters',
            'other parameters',
            'returns',
            'yields',
            'receives',
            'raises',
            'warns',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
        ],
        'numpy',
    ),
]

class ArbitraryTypesModel(BaseModel):
    """
    A Pydantic model that allows arbitrary types.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _create_literal(obj: list[str | type]) -> type:
    """
    Create a literal type from a list of objects.
    """
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


def _get_model(type_, created_models: dict[str, BaseModel]) -> Any:
    try:
        if issubclass(type_, param.Parameterized):
            type_ = created_models.get(type_.__name__, type_.__name__)
    except TypeError:
        pass
    return type_


def _infer_docstring_style(doc: str) -> DocstringStyle:
    """Simplistic docstring style inference."""
    for pattern, replacements, style in _docstring_style_patterns:
        matches = (
            re.search(pattern.format(replacement), doc, re.IGNORECASE | re.MULTILINE) for replacement in replacements
        )
        if any(matches):
            return style
    # fallback to google style
    return 'google'


def parameter_to_field(
    parameter: param.Parameter, created_models: dict[str, BaseModel],
    literals: list[str] | None
) -> (type, FieldInfo):
    """
    Translate a parameter to a pydantic field.
    """
    param_type = parameter.__class__
    description = " ".join(parameter.doc.split()) if parameter.doc else None
    field_info = FieldInfo(description=description)
    if not literals and hasattr(parameter, 'get_range'):
        literals = list(parameter.get_range())

    if param_type in PARAM_TYPE_MAPPING:
        type_ = PARAM_TYPE_MAPPING[param_type]
        field_info.default = parameter.default
    elif param_type is param.ClassSelector:
        type_ = _get_model(parameter.class_, created_models)
        if isinstance(type_, tuple):
            type_ = Union[tuple([PARAM_TYPE_MAPPING.get(t, t) for t in type_])]  # noqa
        if parameter.default is not None:
            default_factory = parameter.default
            if not callable(default_factory):
                default_factory = type(default_factory)
            field_info.default_factory = default_factory
    elif param_type in [param.List, param.ListSelector]:
        type_ = list
        if parameter.default is not None:
            field_info.default_factory = parameter.default
        if param_type is param.List and parameter.item_type:
            type_ = list[_get_model(parameter.item_type, created_models)]
        elif param_type is param.ListSelector:
            type_ = list[_create_literal(literals)]
    elif param_type is param.Dict:
        type_ = dict
        if parameter.default == {}:
            field_info.default_factory = dict
        elif parameter.default is not None:
            field_info.default_factory = parameter.default
    elif param_type in [param.Selector, param.ObjectSelector]:  # ObjectSelector is deprecated
        field_info.default = parameter.default
        if literals:
            type_ = _create_literal(literals)
        elif parameter.objects:
            type_ = _create_literal(parameter.objects)
        else:
            type_ = str
    elif issubclass(param_type, param.DataFrame):
        type_ = PandasDataFrame
    elif parameter.name == "align":
        type_ = _create_literal(["auto", "start", "center", "end"])
    elif parameter.name == "aspect_ratio":
        type_ = Literal["auto"] | float
    elif parameter.name == "margin":
        type_ = float | tuple[float, float] | tuple[float, float, float, float]
    else:
        raise NotImplementedError(
            f"Parameter {parameter.name!r} of {param_type.__name__!r} not supported"
        )

    if hasattr(parameter, "bounds") and parameter.bounds and type_ in [int, float]:
        try:
            field_info.ge = parameter.bounds[0]
            field_info.le = parameter.bounds[1]
        except Exception:
            pass

    if parameter.allow_None:
        type_ = type_ | None

    return type_, field_info


def param_to_pydantic(
    parameterized: type[param.Parameterized],
    base_model: type[BaseModel] = ArbitraryTypesModel,
    created_models: dict[str, BaseModel] | None = None,
    schema: dict[str, Any] | None = None,
    excluded: str | list[str] = "_internal_params",
    extra_fields: dict[str, tuple[type, FieldInfo]] | None = None,
) -> dict[str, BaseModel]:
    """
    Translate a param Parameterized to a Pydantic BaseModel.

    Parameters
    ----------
    parameterized : Type[param.Parameterized]
        The parameterized class to translate.
    base_model : Type[BaseModel], optional
        The base model to use, by default ArbitraryTypesModel
    created_models : Optional[Dict[str, BaseModel]], optional
        A dictionary of already created models, by default None
    excluded : Union[str, List[str]], optional
        A list of parameters to exclude. If a string is provided,
        it looks up the parameter on the parameterized class and
        excludes the parameter's value, by default "_internal_params".
    extra_fields : Optional[Dict[str, Tuple[Type, FieldInfo]]], optional
        Extra fields to add to the model, by default None
    """
    parameterized_name = parameterized.__name__
    if created_models is None:
        created_models = {}
    if parameterized_name in created_models:
        return created_models

    if isinstance(excluded, str) and hasattr(parameterized, excluded):
        excluded = getattr(parameterized, excluded)

    parameterized_signature = signature(parameterized.__init__)
    required_args = [
        arg.name
        for arg in parameterized_signature.parameters.values()
        if arg.name not in ["self", "params"] and arg.default == inspect._empty
    ]
    field_params = list(getattr(parameterized, '_field_params', []))

    fields = {}
    for parameter_name in parameterized.param:
        if parameter_name in excluded or parameterized_name.lower() == parameter_name.lower():
            continue
        parameter = parameterized.param[parameter_name]
        if hasattr(parameter, "class_") and hasattr(parameter.class_, "param"):
            parameter_class_name = parameter.class_.__name__
            if parameterized_name != parameter_class_name:
                param_to_pydantic(parameter.class_, created_models=created_models)

        literals = list(schema) if schema and parameter_name in field_params else None

        type_, field_info = parameter_to_field(parameter, created_models, literals)
        if parameter_name == "schema":
            field_info.alias = "schema"
            parameter_name = "schema_"
        elif parameter_name == "copy":
            field_info.alias = "copy"
            parameter_name = "copy_"
        elif parameter_name[0] == "_":
            field_info.alias = parameter_name
            parameter_name = parameter_name.lstrip("_")

        if parameter_name in required_args:
            field_info.default = PydanticUndefined
        elif field_info.default == PydanticUndefined and not field_info.default_factory:
            field_info.default = None
        fields[parameter_name] = (type_, field_info)

    fields["__parameterized__"] = (type, parameterized)

    if extra_fields:
        fields.update(extra_fields)

    # ignore RuntimeWarning: fields may not start with an underscore
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pydantic_model = create_model(
            parameterized.__name__, __base__=base_model, **fields
        )
    created_models[pydantic_model.__name__] = pydantic_model
    return created_models


def pydantic_to_param(model: BaseModel) -> param.Parameterized:
    """
    Serialize a Pydantic model to a Parameterized instance.

    Parameters
    ----------
    model : BaseModel
        The model to serialize.
    """
    kwargs = {}
    for key, value in dict(model).items():
        if isinstance(value, BaseModel):
            kwargs[key] = pydantic_to_param(value)
        elif isinstance(value, Color):
            kwargs[key] = value.as_hex()
        elif isinstance(value, dict):
            sub_kwargs = {}
            for k, v in value.items():
                if isinstance(v, BaseModel):
                    sub_kwargs[k] = pydantic_to_param(v)
                else:
                    sub_kwargs[k] = v
            kwargs[key] = sub_kwargs
        elif key == "table":
            # couldn't convince AI to provide path to file for `tables`
            # but remove the extension for `table`
            kwargs[key] = value.split(".")[0]
        else:
            kwargs[key] = value

    parameterized = model.__parameterized__(**kwargs)
    return parameterized

def doc_descriptions(function: FunctionType, sig: Signature | None = None) -> tuple[str, dict[str, str]]:
    """
    Parses the docstring and returns the function description and descriptions for each argument.

    Arguments
    ---------
    function : FunctionType
        Function to extract argument descriptions from.
    sig: inspect.Signature | None
        Optional Signature of the function

    Returns
    -------
    tuple[str, dict[str, str]]
        Tuple of the main description of the function and dictionary of descriptions per argument
    """
    doc = function.__doc__
    if doc is None:
        return '', {}
    if sig is None:
        sig = signature(function)

    parser = _infer_docstring_style(doc)
    docstring = Docstring(doc, lineno=1, parser=parser, parent=sig)
    sections = docstring.parse()
    params = {}
    if parameters := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters.value}

    main_desc = ''
    if main := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main.value

    return main_desc, params

def function_to_model(function: FunctionType, skipped: list[str] | None = None) -> type[BaseModel]:
    """
    Translates the arguments to a function into a pydantic Model.

    Arguments
    ---------
    function : FunctionType
        The function to translate.
    skipped : list[str] | None
        Arguments that will be skipped when exporting the JSON schema.

    Returns
    -------
    pydantic.BaseModel
        The translated model with fields corresponding to the arguments of the function.
    """
    skipped = skipped or []
    sig = signature(function)
    type_hints = _typing_extra.get_function_type_hints(function)
    fields: dict[str, core_schema.TypedDictField] = {}
    description, field_descriptions = doc_descriptions(function, sig)
    for index, (name, p) in enumerate(sig.parameters.items()):
        if p.annotation is sig.empty:
            annotation = Any
        else:
            annotation = type_hints[name]
        field_name = p.name
        field_info = FieldInfo.from_annotation(annotation)
        if field_info.description is None:
            field_info.description = field_descriptions.get(field_name)
        if name in skipped:
            annotation = SkipJsonSchema[annotation | None]
        fields[field_name] = (annotation, field_info)
    model = create_model(function.__name__, __doc__=description, **fields)
    return model
