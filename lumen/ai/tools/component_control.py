"""
Hand Lumen AI the controls of an application.

:class:`ComponentController` takes a set of Panel widgets, ``Parameterized``
objects or an entire layout (e.g. a ``panel_material_ui.Page``) and turns them
into LLM tools:

* ``list_<namespace>_components`` -- an overview of every controllable
  component including the current value of each widget.
* ``describe_<namespace>_component`` -- every parameter of one component,
  with types, allowed values, current values and their ``doc`` strings.
* ``set_<component>`` / ``click_<component>`` -- one tool per component that
  writes the exposed parameters (or triggers a button).

The controller is re-resolved every time the tools are requested, so a layout
whose contents change over the lifetime of a session stays in sync and the
schemas the LLM sees always reflect the live state of the application.

Usage::

    controller = ComponentController(
        components=page, purpose="Controls for the wind turbine dashboard."
    )
    ui = ExplorerUI(llm_tools=[controller])
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import re

from typing import Annotated, Any, Literal

import pandas as pd
import param

from panel.layout.base import ListLike, NamedListLike
from panel.viewable import Layoutable, Viewable, Viewer
from panel.widgets.base import WidgetBase
from pydantic import Field

from ..utils import truncate_string
from .base import FunctionTool

# Number of allowed values that are inlined into a schema before truncating
MAX_OPTIONS = 50

# Components from these modules are never picked up when walking a layout;
# they belong to the chat UI itself rather than to the application being
# driven, which matters because the assistant is usually mounted inside the
# very layout it is being handed.
EXCLUDED_MODULES = ("lumen.ai.", "panel.chat", "panel_material_ui.chat")

# Parameters that are only ever exposed when explicitly requested
ALWAYS_SKIPPED = frozenset({"name", "value_throttled"})

_UNSET = object()


def _chrome_parameters() -> frozenset[str]:
    """
    Parameters contributed by framework baseclasses (layout, styling, chrome).

    These are excluded when the exposed parameters of a component are derived
    automatically; they can still be exposed by requesting them explicitly.
    """
    names: set[str] = set(param.Parameterized.param) | set(Layoutable.param) | set(Viewable.param)
    for module, cls_name in (
        ("panel.widgets.base", "WidgetBase"),
        ("panel_material_ui.base", "MaterialComponent"),
        ("panel_material_ui.widgets.base", "MaterialWidget"),
    ):
        try:
            cls = getattr(importlib.import_module(module), cls_name, None)
        except ImportError:
            continue
        if cls is not None:
            names |= set(cls.param)
    return frozenset(names - {"value", "options"})


CHROME_PARAMETERS = _chrome_parameters()


def _slugify(name: str) -> str:
    """Convert a label into a valid, lowercase Python identifier."""
    slug = re.sub(r"\W+", "_", (name or "").strip()).strip("_").lower()
    if not slug or slug[0].isdigit():
        slug = f"c_{slug}" if slug else "component"
    return slug[:48]


def _label(component: param.Parameterized) -> str:
    """The human readable label of a component, if it has a meaningful one."""
    for attr in ("label", "name"):
        if attr not in component.param:
            continue
        value = getattr(component, attr, None)
        if not isinstance(value, str) or not value:
            continue
        # param auto-generates names of the form ``ClassName00001``
        if re.fullmatch(rf"{re.escape(type(component).__name__)}\d*", value):
            continue
        return value
    return ""


def _serializer(parameter: param.Parameter):
    """
    The parameter's own JSON serializer, if it declares one.

    ``param.Parameter`` defines identity ``serialize``/``deserialize``
    classmethods that subclasses override to describe how their values cross a
    JSON boundary (dates as ISO strings, tuples as lists). Honouring them means
    the values shown to the LLM are exactly the values it has to send back.
    """
    return _custom_classmethod(parameter, "serialize")


def _deserializer(parameter: param.Parameter):
    """The parameter's own JSON deserializer, if it declares one."""
    return _custom_classmethod(parameter, "deserialize")


def _custom_classmethod(parameter: param.Parameter, name: str):
    method = getattr(type(parameter), name, None)
    base = getattr(param.Parameter, name)
    if method is None or getattr(method, "__func__", method) is getattr(base, "__func__", base):
        return None
    return method


def _format_value(value: Any, options: dict[str, Any] | None = None) -> str:
    """Render a parameter value for display to the LLM."""
    if options is not None:
        if isinstance(value, list):
            labels = [_option_label(v, options) for v in value]
            return "[" + ", ".join(labels) + "]"
        return _option_label(value, options)
    if getattr(value, "shape", None) == () and hasattr(value, "item"):
        # unwrap numpy scalars, whose repr is noise to an LLM
        value = value.item()
    if isinstance(value, str):
        return repr(truncate_string(value, max_length=200))
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (bool, int, float, type(None))):
        return repr(value)
    if isinstance(value, (list, tuple, dict, set)):
        return truncate_string(repr(value), max_length=200)
    if isinstance(value, param.Parameterized):
        return _label(value) or type(value).__name__
    return truncate_string(repr(value), max_length=100)


def _option_label(value: Any, options: dict[str, Any]) -> str:
    """The label an option value is listed under."""
    for label, option in options.items():
        try:
            if option is value or option == value:
                return repr(label)
        except Exception:
            continue
    return _format_value(value)


class ParameterInfo:
    """
    Everything needed to expose a single ``param.Parameter`` to the LLM.

    Bundles the parameter itself with the constraints that Panel widgets
    declare on sibling parameters (a ``Select``'s ``options``, a slider's
    ``start``/``end``/``step``) so both the generated schema and the coercion
    of incoming values can take them into account.
    """

    def __init__(self, component: param.Parameterized, name: str):
        self.component = component
        self.name = name
        self.parameter = component.param[name]
        self.options = _options(component, self.parameter)
        # param.List declares bounds on the number of items rather than on the
        # values themselves, so the two are tracked separately
        self.length_bounds = _bounds(component, self.parameter) if isinstance(self.parameter, param.List) else None
        self.bounds = None if self.options or self.length_bounds else _bounds(component, self.parameter)
        self.step = getattr(component, "step", None) if self.name == "value" else None
        self.annotation = _annotation(self.parameter, self.options, self.bounds)

    @property
    def value(self) -> Any:
        return getattr(self.component, self.name)

    @property
    def settable(self) -> bool:
        parameter = self.parameter
        return (
            self.annotation is not None
            and not parameter.readonly
            and not parameter.constant
        )

    @property
    def doc(self) -> str:
        return " ".join((self.parameter.doc or "").split())

    def display(self, value: Any = _UNSET) -> str:
        """Render a value in the form the LLM is expected to supply it in."""
        if value is _UNSET:
            value = self.value
        if self.options is None:
            serialize = _serializer(self.parameter)
            if serialize is not None:
                try:
                    return _format_value(serialize(value))
                except Exception:
                    pass
        return _format_value(value, self.options)

    def constraints(self) -> str:
        """A description of the values this parameter accepts."""
        parts = []
        if self.options is not None:
            labels = list(self.options)
            listed = ", ".join(labels[:MAX_OPTIONS])
            if len(labels) > MAX_OPTIONS:
                listed += f", ... ({len(labels)} options in total)"
            multiple = isinstance(self.parameter, (param.List, param.ListSelector))
            parts.append(f"{'any of' if multiple else 'one of'}: {listed}")
        elif self.bounds:
            low, high = self.bounds
            if low is not None or high is not None:
                low_str = "unbounded" if low is None else self.display(low)
                high_str = "unbounded" if high is None else self.display(high)
                parts.append(f"between {low_str} and {high_str}")
            if self.step:
                parts.append(f"step {_format_value(self.step)}")
        if self.length_bounds:
            low, high = self.length_bounds
            if low:
                parts.append(f"at least {low} items")
            if high is not None:
                parts.append(f"at most {high} items")
        return "; ".join(parts)

    def nullable(self) -> bool:
        """Whether the LLM may pass null to clear the parameter."""
        return bool(self.parameter.allow_None) and self.options is None

    def summary(self) -> str:
        """One line description of the parameter and its current value."""
        summary = f"{self.name}: {self.display()}"
        notes = [note for note in (self.constraints(), "may be null" if self.nullable() else "") if note]
        if notes:
            summary += " (" + "; ".join(notes) + ")"
        return summary

    def describe(self) -> str:
        """Multi-line description including the parameter's doc string."""
        type_name = type(self.parameter).__name__
        lines = [f"- `{self.name}` ({type_name}) = {self.display()}"]
        constraints = self.constraints()
        if constraints:
            lines.append(f"  Accepts: {constraints}" + (" (or null)" if self.nullable() else ""))
        elif self.nullable():
            lines.append("  Accepts: null to clear it")
        if self.doc:
            lines.append(f"  Doc: {truncate_string(self.doc, max_length=500)}")
        if not self.settable:
            reason = "read-only" if (self.parameter.readonly or self.parameter.constant) else "not settable via this API"
            lines.append(f"  ({reason})")
        return "\n".join(lines)

    def argument_doc(self) -> str:
        """The docstring entry generated for this parameter in a setter tool."""
        doc = self.doc or f"The {self.name} of the component."
        constraints = self.constraints()
        if constraints:
            doc += f" Accepts {constraints}."
        if self.nullable():
            doc += " May be null."
        return f"{doc} Currently {self.display()}."


def _options(component: param.Parameterized, parameter: param.Parameter) -> dict[str, Any] | None:
    """
    The allowed values of a parameter as a ``{label: value}`` mapping.

    Covers both ``param.Selector`` parameters and Panel widgets, which declare
    the allowed values of their ``value`` on a sibling ``options`` parameter.
    """
    objects: Any = None
    if isinstance(parameter, param.Selector):
        try:
            objects = parameter.get_range()
        except Exception:
            objects = None
    elif parameter.name == "value" and "options" in component.param:
        objects = getattr(component, "options", None)
    if not objects:
        return None
    if isinstance(objects, dict):
        return {str(label): value for label, value in objects.items()}
    return {str(value): value for value in objects}


def _bounds(component: param.Parameterized, parameter: param.Parameter) -> tuple[Any, Any] | None:
    """
    The bounds of a parameter.

    Panel sliders declare the range of their ``value`` on sibling ``start`` and
    ``end`` parameters rather than on the value parameter itself.
    """
    bounds = getattr(parameter, "bounds", None) or getattr(parameter, "softbounds", None)
    if bounds is None and parameter.name == "value":
        start = getattr(component, "start", None)
        end = getattr(component, "end", None)
        if start is not None or end is not None:
            bounds = (start, end)
    if bounds and len(bounds) == 2:
        return (bounds[0], bounds[1])
    return None


def _numeric_annotation(base: type, bounds: tuple[Any, Any] | None) -> Any:
    """Annotate a numeric type with its bounds so they show up in the schema."""
    if not bounds:
        return base
    low, high = bounds
    constraints = {}
    if isinstance(low, (int, float)) and not isinstance(low, bool):
        constraints["ge"] = low
    if isinstance(high, (int, float)) and not isinstance(high, bool):
        constraints["le"] = high
    if not constraints:
        return base
    return Annotated[base, Field(**constraints)]


def _annotation(
    parameter: param.Parameter,
    options: dict[str, Any] | None,
    bounds: tuple[Any, Any] | None,
) -> Any:
    """
    The type annotation to expose a parameter under, or None if it cannot be
    represented in a JSON schema.
    """
    if options is not None:
        literal = Literal[tuple(list(options)[:MAX_OPTIONS])]
        if isinstance(parameter, (param.List, param.ListSelector)):
            return list[literal]
        return literal
    if isinstance(parameter, param.Boolean):  # includes param.Event
        return bool
    if isinstance(parameter, param.Integer):
        return _numeric_annotation(int, bounds)
    if isinstance(parameter, param.CalendarDate):
        return dt.date
    if isinstance(parameter, param.Date):
        return dt.datetime
    if isinstance(parameter, param.Number):
        return _numeric_annotation(float, bounds)
    if isinstance(parameter, param.CalendarDateRange):
        return tuple[dt.date, dt.date]
    if isinstance(parameter, param.DateRange):
        return tuple[dt.datetime, dt.datetime]
    if isinstance(parameter, param.Range):
        return tuple[float, float]
    if isinstance(parameter, param.NumericTuple):
        return tuple[float, ...]
    if isinstance(parameter, param.Tuple):
        return tuple
    if isinstance(parameter, param.Color):
        return str
    if isinstance(parameter, (param.String, param.Path)):
        return str
    if isinstance(parameter, param.List):
        item_type = getattr(parameter, "item_type", None)
        if item_type in (str, int, float, bool):
            return list[item_type]
        return list
    if isinstance(parameter, param.Dict):
        return dict[str, Any]
    if isinstance(parameter, param.ClassSelector):
        classes = parameter.class_ if isinstance(parameter.class_, tuple) else (parameter.class_,)
        if all(cls in (str, int, float, bool, list, dict) for cls in classes):
            return classes[0] if len(classes) == 1 else Any
        return None
    if type(parameter) is param.Parameter:
        return Any
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return True
        if lowered in ("false", "no", "0", "off", ""):
            return False
    return bool(value)


def _to_datetime(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            pass
    timestamp = pd.Timestamp(value)
    if not isinstance(timestamp, pd.Timestamp):  # i.e. NaT
        raise ValueError(f"{value!r} could not be interpreted as a date")
    return dt.datetime.fromisoformat(timestamp.isoformat())


def _to_date(value: Any) -> dt.date:
    converted = _to_datetime(value)
    return converted.date()


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None or isinstance(value, (list, tuple, dict)):
        raise ValueError(f"{value!r} is not a number")
    return float(value)


def _to_int(value: Any) -> int:
    return round(_to_number(value))


def _coerce_option(info: ParameterInfo, value: Any) -> Any:
    """Map a label (or raw value) supplied by the LLM onto an allowed value."""
    options = info.options or {}
    if isinstance(value, str) and value in options:
        return options[value]
    lowered = {label.lower(): label for label in options}
    if isinstance(value, str) and value.strip().lower() in lowered:
        return options[lowered[value.strip().lower()]]
    for option in options.values():
        try:
            if option is value or option == value:
                return option
        except Exception:
            continue
    labels = ", ".join(list(options)[:MAX_OPTIONS]) or "(none)"
    raise ValueError(f"{value!r} is not one of the allowed values: {labels}")


def _coerce(info: ParameterInfo, value: Any) -> Any:
    """Coerce a raw JSON value from the LLM into a valid parameter value."""
    parameter = info.parameter
    if value is None and parameter.allow_None:
        return None
    if info.options is not None:
        if isinstance(parameter, (param.List, param.ListSelector)):
            values = value if isinstance(value, (list, tuple)) else [value]
            return [_coerce_option(info, item) for item in values]
        return _coerce_option(info, value)
    # Prefer the parameter's own JSON contract, but only if it yields a value
    # param actually accepts; the declared formats are strict (param.Date
    # insists on microseconds) and an LLM will happily send "2020-01-01".
    deserialize = _deserializer(parameter)
    if deserialize is not None:
        try:
            deserialized = deserialize(value)
        except Exception:
            deserialized = _UNSET
        if deserialized is not _UNSET and _validation_error(info, deserialized) is None:
            return deserialized
    if isinstance(parameter, param.Boolean):
        return _to_bool(value)
    if isinstance(parameter, param.Integer):
        return _to_int(value)
    if isinstance(parameter, param.CalendarDate):
        return _to_date(value)
    if isinstance(parameter, param.Date):
        return _to_datetime(value)
    if isinstance(parameter, param.Number):
        return _to_number(value)
    if isinstance(parameter, (param.CalendarDateRange, param.DateRange, param.Range, param.Tuple)):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{value!r} is not a list of values")
        if isinstance(parameter, param.CalendarDateRange):
            return tuple(_to_date(item) for item in value)
        elif isinstance(parameter, param.DateRange):
            return tuple(_to_datetime(item) for item in value)
        elif isinstance(parameter, param.NumericTuple):  # includes param.Range
            return tuple(_to_number(item) for item in value)
        return tuple(value)
    if isinstance(parameter, (param.String, param.Path, param.Color)):
        return value if isinstance(value, str) else str(value)
    if isinstance(parameter, param.List):
        items = list(value) if isinstance(value, (list, tuple, set)) else [value]
        item_type = getattr(parameter, "item_type", None)
        converter = {bool: _to_bool, int: _to_int, float: _to_number, str: str}.get(item_type)
        return [converter(item) for item in items] if converter else items
    return value


def _validation_error(info: ParameterInfo, value: Any) -> str | None:
    """
    Validate a coerced value, returning an error message if it is rejected.

    Bounds that a Panel widget declares on sibling parameters (a slider's
    ``start`` and ``end``) are not enforced by the value parameter itself, so
    they are checked here rather than silently accepting an invalid value.
    """
    try:
        info.parameter._validate(value)
    except Exception as e:
        return str(e)
    if info.bounds and not isinstance(value, bool):
        low, high = info.bounds
        # A range widget bounds every element of its value tuple
        values = value if isinstance(value, tuple) else (value,)
        for item in values:
            try:
                if low is not None and item < low:
                    return f"{info.display(item)} is below the lower bound {info.display(low)}"
                if high is not None and item > high:
                    return f"{info.display(item)} is above the upper bound {info.display(high)}"
            except TypeError:
                continue
    return None


def _describe_function(function, name: str, doc: str, arguments: list[inspect.Parameter], docs: list[str]):
    """
    Attach the metadata :func:`~lumen.ai.translate.function_to_model`
    introspects to a dynamically generated tool function.
    """
    if docs:
        doc = f"{doc}\n\nParameters\n----------\n" + "\n".join(docs)
    function.__name__ = name
    function.__qualname__ = name
    function.__doc__ = doc
    function.__signature__ = inspect.Signature(arguments)
    function.__annotations__ = {
        argument.name: argument.annotation for argument in arguments
    }
    return function


def _dress(function, name: str, doc: str, arguments: list[tuple[ParameterInfo, bool]]):
    """
    Give a ``**kwargs`` function the signature, annotations and docstring
    of the parameters it writes.

    Arguments are declared as ``(info, required)`` pairs; optional arguments
    default to None which is interpreted as "leave unchanged".
    """
    signature, docs = [], []
    for info, required in arguments:
        annotation = info.annotation
        default = inspect.Parameter.empty
        if not required:
            annotation = annotation | None
            default = None
        signature.append(
            inspect.Parameter(info.name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        )
        docs.append(f"{info.name}\n    {info.argument_doc()}")
    return _describe_function(function, name, doc, signature, docs)


class ComponentSpec:
    """
    A single component the LLM may inspect and control.

    Resolved by :class:`ComponentController`; holds the tool-safe key the
    component is addressed by, the description the LLM is given and the names
    of the parameters that are exposed for writing.
    """

    def __init__(
        self,
        key: str,
        component: param.Parameterized,
        description: str = "",
        parameters: list[str] | None = None,
    ):
        self.key = key
        self.component = component
        self.description = " ".join((description or "").split())
        self.parameters = list(parameters) if parameters else _exposed_parameters(component)

    @property
    def label(self) -> str:
        return _label(self.component)

    @property
    def type_name(self) -> str:
        return type(self.component).__name__

    @property
    def infos(self) -> list[ParameterInfo]:
        """Information about every exposed parameter."""
        infos = []
        for name in self.parameters:
            if name in self.component.param:
                infos.append(ParameterInfo(self.component, name))
        return infos

    @property
    def settable(self) -> list[ParameterInfo]:
        return [info for info in self.infos if info.settable]

    @property
    def primary(self) -> ParameterInfo | None:
        """The parameter that holds the component's value, if it has one."""
        settable = self.settable
        for info in settable:
            if info.name == "value":
                return info
        return settable[0] if len(settable) == 1 else None

    @property
    def is_trigger(self) -> bool:
        """Whether controlling this component means firing an event (a button)."""
        primary = self.primary
        return (
            primary is not None
            and len(self.settable) == 1
            and isinstance(primary.parameter, param.Event)
        )

    def _headline(self) -> str:
        headline = f"`{self.key}` — {self.type_name}"
        label = self.label
        if label and label != self.key:
            headline += f' labelled "{label}"'
        return headline

    def _auto_description(self) -> str:
        """Description derived from the component itself."""
        if self.description:
            return self.description
        for attr in ("description", "tooltip"):
            value = getattr(self.component, attr, None)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        return ""

    def summary(self) -> str:
        """Overview of the component and its current state."""
        lines = [f"- {self._headline()}"]
        description = self._auto_description()
        if description:
            lines.append(f"  {truncate_string(description, max_length=300)}")
        if not self.is_trigger:
            for info in self.infos:
                lines.append(f"  {info.summary()}")
        tool_name = self.tool_name()
        if tool_name:
            lines.append(f"  Control with: {tool_name}")
        return "\n".join(lines)

    def describe(self) -> str:
        """Full description of every parameter of the component."""
        lines = [f"### {self._headline()}"]
        description = self._auto_description()
        if description:
            lines.append(description)
        class_doc = " ".join((type(self.component).__doc__ or "").split())
        if class_doc:
            lines.append(f"{self.type_name}: {truncate_string(class_doc, max_length=400)}")
        exposed = self.parameters
        lines.append(f"\nParameters exposed for control ({len(exposed)}):")
        lines += [info.describe() for info in self.infos] or ["(none)"]
        others = [
            name for name in sorted(self.component.param)
            if name not in exposed and name not in ALWAYS_SKIPPED
        ]
        if others:
            lines.append(
                "\nOther parameters (not exposed): " + ", ".join(
                    f"`{name}`={_format_value(getattr(self.component, name, None))}"
                    for name in others[:40]
                )
            )
        tool_name = self.tool_name()
        if tool_name:
            lines.append(f"\nControl with: {tool_name}")
        return "\n".join(lines)

    def tool_name(self) -> str | None:
        """Name of the tool that controls this component."""
        if not self.settable:
            return None
        if self.is_trigger:
            return f"click_{self.key}" if "clicks" in self.component.param else f"trigger_{self.key}"
        return f"set_{self.key}"

    def apply(self, values: dict[str, Any]) -> str:
        """Write *values* onto the component, reporting what changed."""
        infos = {info.name: info for info in self.settable}
        updates, messages, errors = {}, [], []
        for name, value in values.items():
            info = infos.get(name)
            if info is None:
                errors.append(f"`{name}` cannot be set on `{self.key}`")
                continue
            try:
                updates[name] = _coerce(info, value)
            except (ValueError, TypeError) as e:
                errors.append(f"`{name}`: {e}")
        for name, value in list(updates.items()):
            info = infos[name]
            error = _validation_error(info, value)
            if error:
                del updates[name]
                errors.append(f"`{name}`: {error}")
                continue
            messages.append(f"`{name}` {info.display()} → {info.display(value)}")
        if updates:
            try:
                self.component.param.update(**updates)
            except Exception as e:
                return f"Failed to update `{self.key}`: {e}"
        label = self.label or self.key
        result = f"Updated {label}: " + ", ".join(messages) if messages else f"No changes applied to {label}."
        if errors:
            result += "\nRejected: " + "; ".join(errors)
        return result

    def trigger(self) -> str:
        """Fire the component's event parameter, i.e. click a button."""
        primary = self.primary
        if primary is None:
            return f"`{self.key}` cannot be triggered."
        if "clicks" in self.component.param:
            # Panel buttons dispatch on_click callbacks off the clicks parameter
            self.component.param.update(clicks=getattr(self.component, "clicks", 0) + 1)
        self.component.param.trigger(primary.name)
        return f"Clicked {self.label or self.key}."

    def tool(self) -> FunctionTool | None:
        """Build the tool that controls this component."""
        settable = self.settable
        tool_name = self.tool_name()
        if not settable or tool_name is None:
            return None
        label = self.label or self.key
        description = self._auto_description()
        if self.is_trigger:
            async def control(**values) -> str:
                return self.trigger()
            _dress(control, tool_name, f'Click the "{label}" {self.type_name}.', [])
            purpose = f'Click the "{label}" {self.type_name} in the application UI.'
        else:
            primary = self.primary
            arguments = [(info, info is primary) for info in settable]

            async def control(**values) -> str:
                return self.apply({k: v for k, v in values.items() if v is not None})
            _dress(
                control,
                tool_name,
                f'Set parameters of the "{label}" {self.type_name}. '
                'Arguments that are left out (or null) are not modified.',
                arguments,
            )
            purpose = f'Change the "{label}" {self.type_name} in the application UI.'
        if description:
            purpose += f" {truncate_string(description, max_length=300)}"
        return FunctionTool(control, purpose=purpose)


def _exposed_parameters(component: param.Parameterized) -> list[str]:
    """
    Derive the parameters of a component to expose to the LLM.

    Widgets are controlled through their ``value``; for any other
    ``Parameterized`` object every parameter it declares itself is exposed,
    i.e. everything but layout, styling and other framework chrome.
    """
    if "value" in component.param and isinstance(component, WidgetBase):
        return ["value"]
    exposed = []
    for name, parameter in component.param.objects("existing").items():
        if name in ALWAYS_SKIPPED or name in CHROME_PARAMETERS:
            continue
        if parameter.readonly or parameter.constant:
            continue
        if parameter.precedence is not None and parameter.precedence < 0:
            continue
        if _annotation(parameter, _options(component, parameter), _bounds(component, parameter)) is None:
            continue
        exposed.append(name)
    return exposed


def _is_excluded(component: Any, exclude: list[Any]) -> bool:
    for excluded in exclude:
        if component is excluded:
            return True
        if isinstance(excluded, type) and isinstance(component, excluded):
            return True
    module = type(component).__module__ or ""
    return module.startswith(EXCLUDED_MODULES)


def _children(component: Any):
    """Yield the children of a container, layout, pane or ``Viewer``."""
    if isinstance(component, (ListLike, NamedListLike)):
        yield from list(component.objects)
    if isinstance(component, Viewer):
        view = getattr(component, "_view__", None)
        if view is None:
            try:
                view = component.__panel__()
            except Exception:
                view = None
        if isinstance(view, Viewable):
            yield view
    if not isinstance(component, param.Parameterized):
        return
    for name, parameter in component.param.objects("existing").items():
        if name == "name" or isinstance(parameter, param.Callable):
            continue
        try:
            value = getattr(component, name)
        except Exception:
            continue
        if isinstance(value, (Viewable, Viewer)):
            yield value
        elif isinstance(value, (list, tuple)):
            yield from (item for item in value if isinstance(item, (Viewable, Viewer)))


def _walk(component: Any, exclude: list[Any], seen: set[int], depth: int = 0):
    """Recursively collect the widgets nested inside a layout."""
    if depth > 25 or id(component) in seen:
        return
    seen.add(id(component))
    if not isinstance(component, (Viewable, Viewer)) or _is_excluded(component, exclude):
        return
    if isinstance(component, WidgetBase):
        infos = (ParameterInfo(component, name) for name in _exposed_parameters(component))
        if any(info.settable for info in infos):
            yield component
        return
    for child in _children(component):
        yield from _walk(child, exclude, seen, depth + 1)


class ComponentController(param.Parameterized):
    """
    Exposes a set of components to an LLM so it can drive an application.

    Accepts individual widgets, ``Parameterized`` objects, layouts or a whole
    ``panel_material_ui.Page`` (which is walked for the widgets it contains)
    and generates one control tool per component alongside two discovery
    tools. Pass the controller anywhere ``llm_tools`` are accepted, e.g.
    ``ExplorerUI(llm_tools=[controller])``, or hand it to a
    ``ComponentControlAgent``.

    The components are resolved lazily, every time :attr:`tools` is accessed,
    so containers whose contents change stay in sync and the schemas handed to
    the LLM always describe the current state of the application.
    """

    components = param.Parameter(default=None, doc="""
        The components to expose. Either a single component, a list of
        components or a dictionary mapping from the name the LLM addresses a
        component by to the component. Layouts, templates and ``Page``
        objects are walked to collect the widgets they contain, while
        components declared with an explicit name are always treated as a
        single component.""")

    descriptions = param.Dict(default={}, doc="""
        Optional mapping from component name to a description of what the
        component does, for cases where the label and the ``description`` of
        the component itself are not enough.""")

    exclude = param.List(default=[], doc="""
        Components or component types to ignore when walking layouts.""")

    namespace = param.String(default="ui", doc="""
        Namespace inserted into the names of the discovery tools, e.g.
        ``list_ui_components``. Set it to distinguish multiple controllers.""")

    parameters = param.Dict(default={}, doc="""
        Optional mapping from component name to the list of parameters to
        expose for that component. By default widgets expose their ``value``
        and other ``Parameterized`` objects expose every parameter they
        declare themselves.""")

    purpose = param.String(default="", doc="""
        Description of what the set of components does as a whole, e.g.
        "Controls for the wind turbine dashboard". Shown to the LLM alongside
        the list of components.""")

    def __call__(self, context: Any = None) -> list[FunctionTool]:
        """Allows passing the controller directly as an ``llm_tools`` entry."""
        return self.tools

    @property
    def specs(self) -> list[ComponentSpec]:
        """
        Resolve the components into :class:`ComponentSpec` objects.

        Re-resolved on every access so that components added to or removed
        from a layout are picked up immediately.
        """
        components = self.components
        if components is None:
            entries: list[tuple[str | None, Any]] = []
        elif isinstance(components, dict):
            entries = list(components.items())
        elif isinstance(components, (list, tuple)):
            entries = [(None, component) for component in components]
        else:
            entries = [(None, components)]

        specs: list[ComponentSpec] = []
        keys: set[str] = set()
        seen: set[int] = set()
        for key, component in entries:
            if isinstance(component, ComponentSpec):
                resolved = [(key or component.key, component.component, component)]
            elif key is None and not isinstance(component, WidgetBase) and isinstance(component, (Viewable, Viewer)):
                resolved = [(None, found, None) for found in _walk(component, self.exclude, seen)]
            else:
                resolved = [(key, component, None)]
            for resolved_key, resolved_component, spec in resolved:
                if not isinstance(resolved_component, param.Parameterized):
                    self.param.warning(
                        f"Cannot control {resolved_component!r}; components must be "
                        "Parameterized objects such as Panel widgets."
                    )
                    continue
                spec_key = self._unique_key(resolved_key, resolved_component, keys)
                keys.add(spec_key)
                specs.append(ComponentSpec(
                    spec_key,
                    resolved_component,
                    description=self._lookup(self.descriptions, spec_key, resolved_key) or (spec.description if spec else ""),
                    parameters=self._lookup(self.parameters, spec_key, resolved_key) or (spec.parameters if spec else None),
                ))
        return specs

    @staticmethod
    def _lookup(mapping: dict, key: str, original: str | None) -> Any:
        """Look up per-component overrides by either the resolved or original key."""
        if key in mapping:
            return mapping[key]
        if original is not None and original in mapping:
            return mapping[original]
        return None

    def _unique_key(self, key: str | None, component: param.Parameterized, taken: set[str]) -> str:
        base = _slugify(key or _label(component) or type(component).__name__)
        if base not in taken:
            return base
        index = 2
        while f"{base}_{index}" in taken:
            index += 1
        return f"{base}_{index}"

    @property
    def tools(self) -> list[FunctionTool]:
        """The discovery and control tools for the current set of components."""
        specs = self.specs
        tools = [self._list_tool(), self._describe_tool(specs)]
        for spec in specs:
            tool = spec.tool()
            if tool is not None:
                tools.append(tool)
        return tools

    def summary(self) -> str:
        """An overview of every component and its current state."""
        specs = self.specs
        if not specs:
            return "No controllable components are currently available."
        header = f"Controllable UI components ({len(specs)}):"
        if self.purpose:
            header = f"{' '.join(self.purpose.split())}\n\n{header}"
        listing = "\n".join(spec.summary() for spec in specs)
        return (
            f"{header}\n{listing}\n\n"
            f"Call {self._tool_name('describe')} for the full parameter list of one component."
        )

    def _tool_name(self, kind: Literal["list", "describe"]) -> str:
        namespace = _slugify(self.namespace)
        if kind == "list":
            return f"list_{namespace}_components" if namespace else "list_components"
        return f"describe_{namespace}_component" if namespace else "describe_component"

    def _list_tool(self) -> FunctionTool:
        async def list_components() -> str:
            return self.summary()

        _describe_function(
            list_components,
            self._tool_name("list"),
            "List the interactive components of the application that can be controlled, "
            "including the current value of each one.",
            [], [],
        )
        purpose = (
            "Discover the interactive components of the application UI that can be "
            "inspected and controlled, and their current values. Call this before "
            "changing the UI when unsure which components exist."
        )
        if self.purpose:
            purpose += f" {' '.join(self.purpose.split())}"
        return FunctionTool(list_components, purpose=purpose)

    def _describe_tool(self, specs: list[ComponentSpec]) -> FunctionTool:
        lookup = {spec.key: spec for spec in specs}

        async def describe_component(component: str) -> str:
            spec = lookup.get(component)
            if spec is None:
                keys = ", ".join(lookup) or "(none)"
                return f"Unknown component {component!r}. Available components: {keys}."
            return spec.describe()

        annotation = Literal[tuple(lookup)] if lookup else str
        _describe_function(
            describe_component,
            self._tool_name("describe"),
            "Describe every parameter of one component of the application UI.",
            [inspect.Parameter("component", inspect.Parameter.KEYWORD_ONLY, annotation=annotation)],
            [f"component\n    Name of the component, one of: {', '.join(lookup) or '(none)'}."],
        )
        return FunctionTool(describe_component, purpose=(
            "Inspect one component of the application UI in detail: all of its "
            "parameters, their types, allowed values, current values and documentation. "
            f"Use {self._tool_name('list')} first to discover the component names."
        ))
