"""
Hand Lumen AI the controls of an application.

:class:`ComponentController` takes a set of Panel widgets, ``Parameterized``
objects or an entire layout (e.g. a ``panel_material_ui.Page``) and turns them
into a fixed set of LLM tools:

* ``list_<namespace>_components`` -- an overview of every component and its
  current state.
* ``describe_<namespace>_component`` -- every parameter of one component, with
  types, allowed values, current values and their ``doc`` strings.
* ``set_<namespace>_components`` -- writes any number of components in one
  call and reports the values read back from the application afterwards,
  including components that changed as a consequence.
* ``click_<namespace>_component`` -- clicks one of the buttons passed in
  ``components`` or listed in ``actions``.

The components are resolved every time a tool runs, so a layout whose contents
change over the lifetime of a session stays in sync. Hand the controller to a
:class:`~lumen.ai.agents.ComponentControlAgent`, which keeps these tools to
itself rather than sharing them with every agent of a coordinator::

    controller = ComponentController(
        components=page, actions=[reset], purpose="Controls for the turbine dashboard."
    )
    agent = ComponentControlAgent(controller=controller)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
import keyword
import re

from typing import Annotated, Any, Literal

import param

from panel.io.document import hold
from panel.layout.base import ListLike, NamedListLike
from panel.viewable import Layoutable, Viewable, Viewer
from panel.widgets import PasswordInput
from panel.widgets.base import WidgetBase
from panel_material_ui.base import MaterialComponent
from panel_material_ui.widgets import PasswordInput as MaterialPasswordInput
from panel_material_ui.widgets.base import MaterialWidget
from pydantic import TypeAdapter, ValidationError, WithJsonSchema

from ..translate import parameter_to_json_type
from ..utils import truncate_string
from .base import FunctionTool

# Above this many allowed values the schema falls back to a plain string and
# the values are only enumerated by the describe tool.
MAX_OPTIONS = 50

# The assistant is usually mounted inside the very layout it is handed, so
# its own components must never be picked up as application controls.
EXCLUDED_MODULES = ("lumen.ai.", "panel.chat", "panel_material_ui.chat")

ALWAYS_SKIPPED = frozenset({"name", "value_input", "value_throttled"})

SECRET_TYPES = (PasswordInput, MaterialPasswordInput)

CHROME_BASES = (param.Parameterized, Layoutable, Viewable, WidgetBase, MaterialComponent, MaterialWidget)


def _chrome_parameters(component: param.Parameterized) -> set[str]:
    """
    Parameters contributed by the framework base classes of a component.

    Only the bases the component actually inherits count, so a plain
    Parameterized keeps a ``label`` or ``width`` of its own.
    """
    names = set()
    for base in CHROME_BASES:
        if isinstance(component, base):
            names |= set(base.param)
    return names - {"value", "options"}

_UNSET = object()


def _slugify(name: str) -> str:
    """Convert a label into a valid, lowercase Python identifier."""
    slug = re.sub(r"\W+", "_", (name or "").strip()).strip("_").lower()
    if not slug or slug[0].isdigit():
        slug = f"c_{slug}" if slug else "component"
    slug = slug[:48]
    return f"{slug}_" if keyword.iskeyword(slug) else slug


def _label(component: param.Parameterized) -> str:
    """The human readable label of a component, if it has a meaningful one."""
    # Only Panel components use ``label`` as a display name; on a plain
    # Parameterized it may well be data.
    attrs = ("label", "name") if isinstance(component, Viewable) else ("name",)
    for attr in attrs:
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
    The parameter's own JSON serializer, if it overrides the identity one.

    Displaying values in this form (dates as ISO strings, tuples as lists)
    shows the LLM exactly the form it is expected to send back.
    """
    method = getattr(type(parameter), "serialize", None)
    base = param.Parameter.serialize
    if method is None or getattr(method, "__func__", method) is getattr(base, "__func__", base):
        return None
    return method


def _format_value(value: Any, options: dict[str, Any] | None = None) -> str:
    """Render a parameter value for display to the LLM."""
    if options is not None:
        if isinstance(value, list):
            return "[" + ", ".join(_option_label(v, options) for v in value) + "]"
        return _option_label(value, options)
    if getattr(value, "shape", None) == () and hasattr(value, "item"):
        # numpy scalars have a noisy repr
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
    for label, option in options.items():
        try:
            if option is value or option == value:
                return repr(label)
        except Exception:
            continue
    return _format_value(value)


def _options(component: param.Parameterized, parameter: param.Parameter) -> dict[str, Any] | None:
    """
    The allowed values of a parameter as a ``{label: value}`` mapping.

    Panel widgets declare the allowed values of their ``value`` on a sibling
    ``options`` parameter rather than on the parameter itself.
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

    Panel sliders declare the range of their ``value`` on sibling ``start``
    and ``end`` parameters rather than on the value parameter itself.
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


def _validation_messages(error: ValidationError) -> str:
    return "; ".join(item["msg"] for item in error.errors())


class ParameterInfo:
    """
    Everything needed to expose a single ``param.Parameter`` to the LLM.

    Bundles the parameter with the constraints Panel widgets declare on
    sibling parameters (a ``Select``'s ``options``, a slider's
    ``start``/``end``/``step``), which both the schema and the validation of
    incoming values need.
    """

    def __init__(self, component: param.Parameterized, name: str):
        self.component = component
        self.name = name
        self.parameter = component.param[name]
        self.options = _options(component, self.parameter)
        self.multiple = isinstance(self.parameter, (param.List, param.ListSelector))
        # param.List bounds the number of items rather than the items
        self.length_bounds = _bounds(component, self.parameter) if isinstance(self.parameter, param.List) else None
        self.bounds = None if self.options or self.length_bounds else _bounds(component, self.parameter)
        self.step = getattr(component, "step", None) if name == "value" else None
        self.json_type = parameter_to_json_type(self.parameter, self.options, self.bounds, MAX_OPTIONS)
        self.secret = isinstance(component, SECRET_TYPES) and name == "value"
        self._adapter = None

    @property
    def value(self) -> Any:
        return getattr(self.component, self.name)

    @property
    def readonly(self) -> bool:
        return bool(self.parameter.readonly or self.parameter.constant)

    @property
    def disabled(self) -> bool:
        return bool(getattr(self.component, "disabled", False)) if isinstance(self.component, WidgetBase) else False

    @property
    def settable(self) -> bool:
        return self.json_type is not None and not self.readonly and not self.disabled

    @property
    def doc(self) -> str:
        # The generic doc of a widget's value says nothing; its meaning comes
        # from the label and description of the widget.
        if self.name == "value" and isinstance(self.component, WidgetBase):
            return ""
        return " ".join((self.parameter.doc or "").split())

    @property
    def nullable(self) -> bool:
        """Whether the LLM may send null to clear the parameter."""
        return bool(self.parameter.allow_None) and self.options is None

    def display(self, value: Any = _UNSET) -> str:
        """Render a value in the form the LLM is expected to supply it in."""
        if self.secret:
            return "<hidden>"
        if value is _UNSET:
            value = self.value
        if self.options is None and (serialize := _serializer(self.parameter)) is not None:
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
            # A model told "one of" will not send a list
            parts.append(f"{'any of' if self.multiple else 'one of'}: {listed}")
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

    def _status(self) -> str:
        if self.readonly:
            return "read-only"
        if self.disabled:
            return "disabled"
        if self.json_type is None:
            return "not settable"
        return ""

    def summary(self) -> str:
        """One line description of the parameter and its current value."""
        notes = [
            note for note in (self._status(), self.constraints(), "may be null" if self.nullable and self.settable else "")
            if note
        ]
        summary = f"{self.name}: {self.display()}"
        return summary + (" (" + "; ".join(notes) + ")" if notes else "")

    def describe(self) -> str:
        """Multi-line description including the parameter's doc string."""
        lines = [f"- `{self.name}` ({type(self.parameter).__name__}) = {self.display()}"]
        constraints = self.constraints()
        if constraints:
            lines.append(f"  Accepts: {constraints}" + (" (or null)" if self.nullable else ""))
        elif self.nullable:
            lines.append("  Accepts: null to clear it")
        if self.doc:
            lines.append(f"  Doc: {truncate_string(self.doc, max_length=500)}")
        if status := self._status():
            lines.append(f"  ({status})")
        return "\n".join(lines)

    def argument_doc(self) -> str:
        """Description of the parameter in the schema of the setter tool."""
        parts = [self.doc] if self.doc else []
        if constraints := self.constraints():
            parts.append(f"Accepts {constraints}.")
        if self.nullable:
            parts.append("Send null to clear it.")
        return " ".join(parts)

    def json_schema(self) -> dict[str, Any]:
        schema = TypeAdapter(self.json_type | None if self.nullable else self.json_type).json_schema()
        if doc := self.argument_doc():
            schema["description"] = doc
        return schema

    def coerce(self, value: Any) -> Any:
        """
        Convert a JSON value sent by the LLM into a value for the parameter.

        Options are matched by label (case-insensitively) or by value, since
        a model happily sends either; everything else is validated against
        the parameter's JSON type in pydantic's lax mode.
        """
        if self.options is not None:
            if self.multiple:
                values = value if isinstance(value, (list, tuple)) else [value]
                return [self._coerce_option(item) for item in values]
            return self._coerce_option(value)
        if self._adapter is None:
            self._adapter = TypeAdapter(self.json_type)
        try:
            value = self._adapter.validate_python(value)
        except ValidationError as e:
            raise ValueError(_validation_messages(e)) from None
        # Tuples travel as JSON arrays
        return tuple(value) if isinstance(self.parameter, param.Tuple) and isinstance(value, list) else value

    def _coerce_option(self, value: Any) -> Any:
        options = self.options or {}
        if isinstance(value, str):
            if value in options:
                return options[value]
            lowered = {label.lower(): label for label in options}
            if value.strip().lower() in lowered:
                return options[lowered[value.strip().lower()]]
        for option in options.values():
            try:
                if option is value or option == value:
                    return option
            except Exception:
                continue
        labels = ", ".join(list(options)[:MAX_OPTIONS]) or "(none)"
        raise ValueError(f"{value!r} is not one of the allowed values: {labels}")

    def validation_error(self, value: Any) -> str | None:
        """
        Validate a coerced value, returning an error message if it is rejected.

        Bounds a widget declares on sibling parameters (a slider's ``start``
        and ``end``) are not enforced by param, so they are checked here.
        """
        try:
            self.parameter._validate(value)
        except Exception as e:
            return str(e)
        if self.bounds and not isinstance(value, bool):
            low, high = self.bounds
            for item in value if isinstance(value, tuple) else (value,):
                try:
                    if low is not None and item < low:
                        return f"{self.display(item)} is below the lower bound {self.display(low)}"
                    if high is not None and item > high:
                        return f"{self.display(item)} is above the upper bound {self.display(high)}"
                except TypeError:
                    continue
        return None


def _exposed_parameters(component: param.Parameterized) -> list[str]:
    """
    Derive the parameters of a component to expose to the LLM.

    Widgets are controlled through their ``value``. Other Panel components
    (layouts, panes, templates) expose nothing by default, since their
    parameters describe the structure of the application rather than its
    state. Any other ``Parameterized`` exposes every public parameter it
    declares itself, including constant and read-only ones so the LLM can
    read them back.
    """
    if isinstance(component, WidgetBase):
        return ["value"] if "value" in component.param else []
    if isinstance(component, Viewable):
        return []
    chrome = _chrome_parameters(component)
    exposed = []
    for name, parameter in component.param.objects("existing").items():
        if name in ALWAYS_SKIPPED or name in chrome or name.startswith("_"):
            continue
        if parameter.precedence is not None and parameter.precedence < 0:
            continue
        options, bounds = _options(component, parameter), _bounds(component, parameter)
        if parameter_to_json_type(parameter, options, bounds) is None:
            continue
        exposed.append(name)
    return exposed


class ComponentSpec:
    """
    A single component resolved by :class:`ComponentController`.

    Holds the key the LLM addresses the component by, its description and
    the parameters that are exposed. The parameter information is built once
    per resolution, while the values are always read live.
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
        self.refresh()

    def refresh(self):
        """Rebuild the parameter information, e.g. after the options of a widget changed."""
        self.infos = [ParameterInfo(self.component, name) for name in self.parameters if name in self.component.param]
        self.settable = [info for info in self.infos if info.settable]

    @property
    def label(self) -> str:
        return _label(self.component)

    @property
    def type_name(self) -> str:
        return type(self.component).__name__

    @property
    def is_action(self) -> bool:
        """Whether controlling this component means firing an event (a button)."""
        return len(self.settable) == 1 and isinstance(self.settable[0].parameter, param.Event)

    @property
    def takes_value(self) -> bool:
        """Whether the setter accepts the value itself rather than an object of parameters."""
        return len(self.settable) == 1 and self.settable[0].name == "value"

    @property
    def writable(self) -> bool:
        return bool(self.settable) and not self.is_action

    def state(self) -> dict[str, str]:
        return {info.name: info.display() for info in self.infos}

    def _headline(self) -> str:
        headline = f"`{self.key}` — {self.type_name}"
        label = self.label
        if label and label != self.key:
            headline += f' labelled "{label}"'
        return headline

    def full_description(self) -> str:
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
        if description := self.full_description():
            lines.append(f"  {truncate_string(description, max_length=300)}")
        if not self.is_action:
            lines += [f"  {info.summary()}" for info in self.infos]
        return "\n".join(lines)

    def describe(self) -> str:
        """Full description of every parameter of the component."""
        lines = [f"### {self._headline()}"]
        if description := self.full_description():
            lines.append(description)
        class_doc = " ".join((type(self.component).__doc__ or "").split())
        if class_doc:
            lines.append(f"{self.type_name}: {truncate_string(class_doc, max_length=400)}")
        lines.append(f"\nExposed parameters ({len(self.infos)}):")
        lines += [info.describe() for info in self.infos] or ["(none)"]
        exposed = {info.name for info in self.infos}
        chrome = _chrome_parameters(self.component)
        # Names only: values of unexposed parameters may hold anything,
        # including the text of a password input.
        others = [
            name for name in sorted(self.component.param)
            if name not in exposed and name not in ALWAYS_SKIPPED and name not in chrome
            and not name.startswith("_")
        ]
        if others:
            lines.append("\nOther parameters (not exposed): " + ", ".join(f"`{name}`" for name in others[:40]))
        return "\n".join(lines)

    def argument_schema(self) -> dict[str, Any]:
        """JSON schema of the argument the setter tool takes for this component."""
        if self.takes_value:
            return self.settable[0].json_schema()
        return {
            "type": "object",
            "properties": {info.name: info.json_schema() for info in self.settable},
            "additionalProperties": False,
        }

    def argument_doc(self) -> str:
        doc = f"{self.type_name}" + (f' "{self.label}"' if self.label else "")
        if description := self.full_description():
            doc += f": {truncate_string(description, max_length=300).rstrip('.')}"
        detail = self.settable[0].argument_doc() if self.takes_value else "An object of the parameters to change."
        return f"{doc}. {detail}".strip()

    def prepare(self, raw: Any) -> tuple[dict[str, Any], list[str]]:
        """Convert the raw argument sent by the LLM into validated parameter updates."""
        if self.takes_value:
            raw = {"value": raw}
        elif not isinstance(raw, dict):
            return {}, [f"`{self.key}`: expected an object of parameter values, got {raw!r}"]
        infos = {info.name: info for info in self.settable}
        updates, errors = {}, []
        for name, value in raw.items():
            info = infos.get(name)
            if info is None:
                errors.append(f"`{self.key}.{name}` cannot be set")
                continue
            if value is None:
                # Providers that send every key send null for the ones they do
                # not mean to change, so null only clears what may be cleared.
                if info.nullable:
                    updates[name] = None
                continue
            try:
                value = info.coerce(value)
            except (ValueError, TypeError) as e:
                errors.append(f"`{self.key}.{name}`: {e}")
                continue
            if error := info.validation_error(value):
                errors.append(f"`{self.key}.{name}`: {error}")
                continue
            updates[name] = value
        return updates, errors

    def trigger(self):
        """Fire the component's event parameter, i.e. click a button."""
        name = self.settable[0].name
        if "clicks" in self.component.param:
            # Panel buttons dispatch on_click callbacks off the clicks parameter
            self.component.param.update(clicks=getattr(self.component, "clicks", 0) + 1)
        self.component.param.trigger(name)


def _is_excluded(component: Any, exclude: list[Any]) -> bool:
    for excluded in exclude:
        if component is excluded or (isinstance(excluded, type) and isinstance(component, excluded)):
            return True
    return (type(component).__module__ or "").startswith(EXCLUDED_MODULES)


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
    # Templates such as a Page hold their contents on parameters
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
    if getattr(component, "visible", True) is False:
        return
    if isinstance(component, WidgetBase):
        if not isinstance(component, SECRET_TYPES) and _exposed_parameters(component):
            yield component
        return
    for child in _children(component):
        yield from _walk(child, exclude, seen, depth + 1)


def _describe_function(function, name: str, doc: str, arguments: list[tuple[str, Any, Any, str]]):
    """
    Attach the signature and docstring :func:`~lumen.ai.translate.function_to_model`
    introspects to a generated ``**kwargs`` tool function.

    Arguments are given as ``(name, annotation, default, doc)``.
    """
    if arguments:
        doc += "\n\nParameters\n----------\n" + "\n".join(f"{arg}\n    {arg_doc}" for arg, _, _, arg_doc in arguments)
    signature = [
        inspect.Parameter(arg, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        for arg, annotation, default, _ in arguments
    ]
    function.__name__ = function.__qualname__ = name
    function.__doc__ = doc
    function.__signature__ = inspect.Signature(signature)
    function.__annotations__ = {p.name: p.annotation for p in signature}
    return function


class ComponentController(param.Parameterized):
    """
    Exposes a set of components to an LLM so it can drive an application.

    Accepts individual widgets, ``Parameterized`` objects, layouts or a whole
    ``panel_material_ui.Page`` (which is walked for the widgets it contains)
    and generates a fixed set of discovery, write and click tools from them.
    Hand it to a :class:`~lumen.ai.agents.ComponentControlAgent`, or pass
    :meth:`as_llm_tools` to the ``llm_tools`` of a single actor.

    Every parameter of a ``Parameterized`` object handed over becomes writable
    by the LLM unless it is constant, read-only or has a negative precedence;
    use ``parameters`` to narrow that down. Buttons found by walking a layout
    are only clickable when listed in ``actions``, and password inputs are
    never picked up.
    """

    actions = param.List(default=[], doc="""
        Buttons (or other components whose only settable parameter is a
        ``param.Event``) the LLM may click. Buttons found by walking a
        layout are ignored unless listed here, since their callbacks may
        do anything; buttons passed explicitly in ``components`` are always
        clickable.""")

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
        Namespace inserted into the tool names, e.g. ``set_ui_components``.
        Set it to distinguish multiple controllers.""")

    parameters = param.Dict(default={}, doc="""
        Optional mapping from component name to the list of parameters to
        expose for that component. By default widgets expose their ``value``
        and other ``Parameterized`` objects expose every parameter they
        declare themselves.""")

    purpose = param.String(default="", doc="""
        Description of what the set of components does as a whole, e.g.
        "Controls for the wind turbine dashboard". Shown to the LLM alongside
        the list of components.""")

    settle = param.Callable(default=None, allow_refs=False, doc="""
        Optional (async) callable awaited after every write or click, before
        the state of the application is read back. Use it when the
        application updates asynchronously, e.g. through async watchers or
        bound coroutines.""")

    def __init__(self, **params):
        super().__init__(**params)
        # Parallel tool calls would otherwise interleave their before and
        # after snapshots of the application state.
        self._lock = asyncio.Lock()

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
        entries += [(None, action) for action in self.actions]

        resolved: list[tuple[str | None, param.Parameterized, bool]] = []
        seen: set[int] = set()
        for key, component in entries:
            if key is None and not isinstance(component, WidgetBase) and isinstance(component, (Viewable, Viewer)):
                resolved += [(None, found, True) for found in _walk(component, self.exclude, seen)]
            elif not isinstance(component, param.Parameterized):
                self.param.warning(
                    f"Cannot control {component!r}; components must be "
                    "Parameterized objects such as Panel widgets."
                )
            elif id(component) not in seen:
                seen.add(id(component))
                resolved.append((key, component, False))

        # Reserved first so a label can never take a key the user chose
        taken: set[str] = set()
        for key, _, _ in resolved:
            if key is None:
                continue
            slug = _slugify(key)
            if slug in taken:
                raise ValueError(f"Component key {key!r} collides with another component named {slug!r}.")
            taken.add(slug)

        actions = {id(action) for action in self.actions}
        specs = []
        for key, component, walked in resolved:
            if key is None:
                spec_key = self._unique_key(_label(component) or type(component).__name__, taken)
                taken.add(spec_key)
            else:
                spec_key = _slugify(key)
            spec = ComponentSpec(
                spec_key,
                component,
                description=self._lookup(self.descriptions, spec_key, key) or "",
                parameters=self._lookup(self.parameters, spec_key, key),
            )
            if walked and spec.is_action and id(component) not in actions:
                continue
            if not spec.infos:
                if not walked:
                    self.param.warning(
                        f"{type(component).__name__} {spec_key!r} exposes no parameters; "
                        "list the ones to expose in `parameters`."
                    )
                continue
            specs.append(spec)
        return specs

    @staticmethod
    def _lookup(mapping: dict, key: str, original: str | None) -> Any:
        """Look up per-component overrides by either the resolved or original key."""
        if key in mapping:
            return mapping[key]
        if original is not None and original in mapping:
            return mapping[original]
        return None

    @staticmethod
    def _unique_key(name: str, taken: set[str]) -> str:
        base = _slugify(name)
        if base not in taken:
            return base
        index = 2
        while f"{base}_{index}" in taken:
            index += 1
        return f"{base}_{index}"

    def tool_name(self, kind: Literal["list", "describe", "set", "click"]) -> str:
        # OpenAI limits tool names to 64 characters
        namespace = _slugify(self.namespace)[:40] if self.namespace else ""
        noun = "components" if kind in ("list", "set") else "component"
        return f"{kind}_{namespace}_{noun}" if namespace else f"{kind}_{noun}"

    def summary(self, specs: list[ComponentSpec] | None = None) -> str:
        """An overview of every component and its current state."""
        specs = self.specs if specs is None else specs
        if not specs:
            return "No controllable components are currently available."
        controls = [spec for spec in specs if not spec.is_action]
        actions = [spec for spec in specs if spec.is_action]
        lines = []
        if self.purpose:
            lines += [" ".join(self.purpose.split()), ""]
        if controls:
            lines.append(f"Components ({len(controls)}), set with {self.tool_name('set')}:")
            lines += [spec.summary() for spec in controls]
        if actions:
            lines.append(f"\nActions, click with {self.tool_name('click')}:")
            lines += [spec.summary() for spec in actions]
        lines.append(f"\nCall {self.tool_name('describe')} for the full parameter list of one component.")
        return "\n".join(lines)

    def routing_summary(self) -> str:
        """Compact list of the controls, for a coordinator choosing an agent."""
        controls = []
        for spec in self.specs:
            names = [info.name for info in spec.settable]
            if spec.is_action:
                controls.append(f"{spec.key} (button)")
            elif spec.takes_value or not names:
                controls.append(spec.key)
            else:
                controls.append(f"{spec.key} ({', '.join(names)})")
        return "; ".join(controls)

    def as_llm_tools(self, context: Any = None) -> list[FunctionTool]:
        """
        The tools for the current set of components.

        Accepted by ``llm_tools``, which calls it every time an LLM is
        invoked, so the tools always describe the current layout.
        """
        specs = self.specs
        tools = [self._list_tool(), self._describe_tool()]
        writable = [spec for spec in specs if spec.writable]
        if writable:
            tools.append(self._set_tool(writable))
        actions = [spec for spec in specs if spec.is_action]
        if actions:
            tools.append(self._click_tool(actions))
        return tools

    async def _settle(self) -> str | None:
        """Await the settle hook, returning an error message if it fails."""
        if self.settle is None:
            return None
        try:
            result = self.settle()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            return f"Waiting for the application to settle failed: {e}"
        return None

    def _changes(self, specs: list[ComponentSpec], before: dict[str, dict[str, str]], written: set[tuple[str, str]]) -> list[str]:
        """Changes the application made on its own in response to a write."""
        changes = []
        for spec in specs:
            for name, value in spec.state().items():
                old = before.get(spec.key, {}).get(name, value)
                if old != value and (spec.key, name) not in written:
                    changes.append(f"- `{spec.key}.{name}` {old} → {value}")
        return changes

    async def apply(self, values: dict[str, Any]) -> str:
        """Write the values onto the components, reporting the resulting state."""
        async with self._lock:
            return await self._apply(values)

    async def _apply(self, values: dict[str, Any]) -> str:
        specs = self.specs
        lookup = {spec.key: spec for spec in specs if spec.writable}
        before = {spec.key: spec.state() for spec in specs}
        attempted, errors = [], []
        with hold():
            # Components are written one by one, in the order they were sent,
            # so each is validated against the effects of the previous ones,
            # e.g. options that depend on another selection.
            for key, raw in values.items():
                spec = lookup.get(key)
                if spec is None:
                    errors.append(f"`{key}` is not a component that can be set; available: {', '.join(lookup) or '(none)'}")
                    continue
                spec.refresh()
                updates, spec_errors = spec.prepare(raw)
                errors += spec_errors
                if not updates:
                    continue
                try:
                    spec.component.param.update(**updates)
                except Exception as e:
                    errors.append(f"`{key}`: the application raised an error while applying the change: {e}")
                attempted.append((spec, updates))
        if error := await self._settle():
            errors.append(error)

        lines, written = [], set()
        for spec, updates in attempted:
            infos = {info.name: info for info in spec.infos}
            for name, requested in updates.items():
                info = infos[name]
                written.add((spec.key, name))
                old, new = before[spec.key][name], info.display()
                if old == new and info.display(requested) == new:
                    lines.append(f"- `{spec.key}.{name}` already {new}")
                    continue
                line = f"- `{spec.key}.{name}` {old} → {new}"
                if info.display(requested) != new:
                    line += f" (requested {info.display(requested)}, the application changed it)"
                lines.append(line)
        result = ["Updated:", *lines] if lines else ["No changes applied."]
        if changes := self._changes(specs, before, written):
            result += ["Also changed as a result:", *changes]
        if errors:
            result += ["Rejected:", *(f"- {error}" for error in errors)]
        return "\n".join(result)

    async def click(self, component: str) -> str:
        """Click an action, reporting how the state of the application changed."""
        async with self._lock:
            specs = self.specs
            action = next((spec for spec in specs if spec.is_action and spec.key == component), None)
            if action is None:
                available = ", ".join(spec.key for spec in specs if spec.is_action) or "(none)"
                return f"Unknown action {component!r}. Available actions: {available}."
            before = {spec.key: spec.state() for spec in specs}
            result = [f"Clicked {action.label or action.key}."]
            try:
                with hold():
                    action.trigger()
            except Exception as e:
                result.append(f"The application raised an error in response: {e}")
            if error := await self._settle():
                result.append(error)
            if changes := self._changes(specs, before, set()):
                result += ["Changed as a result:", *changes]
            return "\n".join(result)

    def _list_tool(self) -> FunctionTool:
        async def list_components() -> str:
            return self.summary()

        _describe_function(
            list_components, self.tool_name("list"),
            "List the interactive components of the application and their current state.", [],
        )
        purpose = (
            "Discover the components of the application UI that can be inspected and "
            "controlled, and their current values."
        )
        if self.purpose:
            purpose += f" {' '.join(self.purpose.split())}"
        return FunctionTool(list_components, purpose=purpose)

    def _describe_tool(self) -> FunctionTool:
        async def describe_component(component: str) -> str:
            specs = {spec.key: spec for spec in self.specs}
            spec = specs.get(component)
            if spec is None:
                return f"Unknown component {component!r}. Available components: {', '.join(specs) or '(none)'}."
            return spec.describe()

        _describe_function(
            describe_component, self.tool_name("describe"),
            "Describe every parameter of one component of the application UI.",
            [("component", str, inspect.Parameter.empty, f"Name of the component as listed by {self.tool_name('list')}.")],
        )
        return FunctionTool(describe_component, purpose=(
            "Inspect one component of the application UI in detail: all of its parameters, "
            "their types, allowed values, current values and documentation."
        ))

    def _set_tool(self, specs: list[ComponentSpec]) -> FunctionTool:
        async def set_components(**values) -> str:
            return await self.apply(values)

        arguments = [
            (spec.key, Annotated[Any, WithJsonSchema(spec.argument_schema())], None, spec.argument_doc())
            for spec in specs
        ]
        _describe_function(
            set_components, self.tool_name("set"),
            "Change any number of components of the application UI in one call. Components "
            "that are left out are not modified. Returns the values read back from the "
            "application, including other components that changed as a result.",
            arguments,
        )
        return FunctionTool(set_components, purpose="Change the components of the application UI.")

    def _click_tool(self, specs: list[ComponentSpec]) -> FunctionTool:
        async def click_component(component: str) -> str:
            return await self.click(component)

        listing = "; ".join(
            f"{spec.key}" + (f' ("{spec.label}")' if spec.label and spec.label != spec.key else "")
            + (f": {truncate_string(spec.full_description(), max_length=200)}" if spec.full_description() else "")
            for spec in specs
        )
        _describe_function(
            click_component, self.tool_name("click"),
            "Click a button of the application UI. Returns how the state of the application changed.",
            [("component", Literal[tuple(spec.key for spec in specs)], inspect.Parameter.empty, f"The button to click, one of: {listing}.")],
        )
        return FunctionTool(click_component, purpose="Click a button of the application UI.")
