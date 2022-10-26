"""
The Filter components supply query parameters used to filter the
tables returned by a Source.
"""
import types

import bokeh
import pandas as pd
import panel as pn
import param

from packaging.version import Version
from panel.util import classproperty

from ..base import MultiTypeComponent
from ..schema import JSONSchema
from ..state import state
from ..util import resolve_module_reference
from ..validation import ValidationError


class Filter(MultiTypeComponent):
    """
    `Filter` components supply the filter values used by `Source` components to query data. .
    """

    field = param.String(doc="The field being filtered.")

    label = param.String(doc="A label for the Filter.")

    schema = param.Dict(doc="""
        The JSON schema provided by the Source declaring information
        about the data to be filtered.""")

    shared = param.Boolean(default=False, doc="""
        Whether the filter is shared across all targets.""")

    sync_with_url = param.Boolean(default=True, doc="""
        Whether to sync the filter state with the URL parameters.""")

    table = param.String(default=None, doc="""
        The table being filtered. If None applies to all tables.""")

    value = param.Parameter(doc="The current filter value.")

    filter_type = None

    __abstract = True

    # Specification configuration
    _internal_params = ['name', 'schema']
    _requires_field = True

    def __init__(self, **params):
        super().__init__(**params)
        if state.app and state.app.config.sync_with_url and self.sync_with_url and pn.state.location:
            pn.state.location.sync(self, {'value': self.field}, on_error=self._url_sync_error)

    @classproperty
    def _required_keys(cls):
        return ['field'] if cls._requires_field else []

    def _url_sync_error(self, values):
        """
        Called when URL syncing errors.
        """

    ##################################################################
    # Public API
    ##################################################################

    @classmethod
    def from_spec(cls, spec, source_schema, source_filters=None):
        """
        Resolves a Filter specification given the schema of the Source
        (and optionally the table) it will be filtering on.

        Parameters
        ----------
        spec: dict or str
            Specification declared as a dictionary of parameter values.
        source_schema: dict
            A dictionary containing the JSON schema of the Source to
            be filtered on.
        source_filters: dict
            A dictionary of filters associated with the Source

        Returns
        -------
        The resolved Filter object.
        """
        if isinstance(spec, str):
            if source_filters is None:
                raise ValueError(f"Filter spec {repr(spec)} could not be resolved "
                                 "because source has not declared filters.")
            elif spec not in source_filters:
                raise ValueError(f"Filter spec {repr(spec)} could not be resolved, "
                                 f"available filters on the source include {list(source_filters)}.")
            return source_filters[spec]
        spec = dict(spec)
        filter_type = Filter._get_type(spec.pop('type', None))
        if not filter_type._requires_field:
            return filter_type(**spec)
        elif not 'field' in spec:
            raise ValueError('Filter specification must declare field to filter on.')
        field = spec['field']
        if 'label' not in spec:
            spec['label'] = field.title()
        table = spec.get('table')
        schema = None
        fields = []
        for key, table_schema in source_schema.items():
            if table is not None and key != table:
                continue
            schema = table_schema.get(field)
            fields.extend(list(table_schema))
            if schema is not None:
                break
        if not schema:
            raise ValueError(
                f"Source did not declare a schema for {field!r} filter. "
                f"Available fields include: {fields!r}."
            )
        return filter_type(schema={field: schema}, **spec)

    def to_spec(self, context=None):
        spec = super().to_spec(context=context)
        if spec.get('label', '').lower() == spec.get('field'):
            del spec['label']
        return spec

    @property
    def panel(self):
        """
        Returns
        -------
        panel.Viewable or None
            A Panel Viewable object representing the filter.
        """
        return None

    @property
    def query(self):
        """
        Returns
        -------
        object
            The current filter query which will be used by the
            Source to filter the data.
        """

    @classmethod
    def validate(cls, spec, context=None):
        if isinstance(spec, str):
            return spec
        return super().validate(spec, context)


class ConstantFilter(Filter):
    """
    The ConstantFilter allows requesting a constant value from the
    Source.
    """

    filter_type = 'constant'

    @property
    def query(self):
        return self.value

    @property
    def panel(self):
        return None


class FacetFilter(Filter):
    """
    The FacetFilter allows faceting the data along some dimension to
    allow a single View to be exploded into multiple by grouping over
    one of the indexes.
    """

    filter_type = 'facet'

    @property
    def filters(self):
        field_schema = self.schema[self.field]
        if 'enum' in field_schema:
            values = field_schema['enum']
        elif field_schema['type'] == 'integer':
            values = list(range(field_schema['inclusiveMinimum'],
                                field_schema['inclusiveMaximum']+1))
        else:
            raise ValueError(f'Could not facet on field {field_schema!r}, '
                             'currently only enum and integer fields are '
                             'supported.')
        return [
            ConstantFilter(field=self.field, value=value, label=self.label)
            for value in values
        ]


class BaseWidgetFilter(Filter):

    default = param.Parameter(doc="""
        The default value to use on the widget.""")

    disabled = param.Boolean(default=False, doc="""
        Whether the filter should be disabled.""")

    throttled = param.Boolean(default=True, doc="""
       If the widget has a value_throttled parameter use that instead,
       ensuring that no intermediate events are generated, e.g. when
       dragging a slider.""")

    visible = param.Boolean(default=True, doc="""
        Whether the filter should be visible.""")

    __abstract = True

    def _url_sync_error(self, values):
        value = values['value']
        if value is self.widget.value:
            return
        elif isinstance(self.widget.param.value, param.Tuple):
            self.widget.value = tuple(value)
        else:
            raise ValueError(f'URL syncing failed, value {value!r} could not be applied.')

    @property
    def panel(self):
        widget = self.widget.clone()
        if self.throttled and 'value_throttled' in self.widget.param:
            widget.link(self.widget, value_throttled='value')
            self.widget.link(widget, value='value')
            self.widget.link(widget, visible='visible', disabled='disabled', bidirectional=True)
        else:
            self.widget.link(widget, value='value', visible='visible', disabled='disabled', bidirectional=True)
        return widget



class WidgetFilter(BaseWidgetFilter):
    """
    `WidgetFilter` generates a Widget from the table schema provided by a Source.

    By default the widget type will be inferred from the data and
    depending on whether `multi` value selection is enabled.
    """

    empty_select = param.Boolean(default=True, doc="""
        Add an option to Select widgets to indicate no filtering.""")

    multi = param.Boolean(default=True, doc="""
        Whether to use a single-value or multi-value selection widget,
        e.g. for a numeric value this could be a regular slider or a
        range slider.""")

    widget = param.ClassSelector(class_=pn.widgets.Widget, doc="""
        The widget instance. When declared in a specification a full
        module path to a Panel widget class can be provided to override
        the default inferred widget.""")

    filter_type = 'widget'

    _internal_params = ['name', 'schema']

    def __init__(self, **params):
        wtype = params.pop('widget', None)
        self._inferred_widget = wtype is None
        super().__init__(**params)
        self.widget = JSONSchema(
            schema=self.schema, sizing_mode='stretch_width', multi=self.multi,
            widgets={self.field: wtype} if wtype else {}
        )._widgets[self.field]
        if isinstance(self.widget, pn.widgets.Select) and self.empty_select:
            if self.widget.options[0] != ' ':
                self.widget.options.insert(0, ' ')
            self.widget.value = ' '
        self.widget.name = self.label
        self.widget.visible = self.visible
        self.widget.disabled = self.disabled
        self.widget.link(self, bidirectional=True, value='value', visible='visible', disabled='disabled')
        if self.default is not None:
            self.widget.value = self.default

    @classmethod
    def _validate_widget(cls, widget, spec, context):
        try:
            resolve_module_reference(widget, pn.widgets.Widget)
        except Exception:
            raise ValidationError(
                f'{cls.__name__} could not resolve widget module reference {widget!r}.', spec
            )
        else:
            return widget

    def to_spec(self, context=None):
        spec = super().to_spec(context=context)
        if self._inferred_widget:
            del spec['widget']
        else:
            wtype = type(spec['widget'])
            spec['widget'] = f'{wtype.__module__}.{wtype.__name__}'
        return spec

    @property
    def query(self):
        if self.widget.value == ' ' and self.empty_select:
            return None
        if not hasattr(self.widget.param.value, 'serialize') or self.widget.value is None:
            return self.widget.value
        else:
            value = self.widget.param.value.serialize(self.widget.value)
            if isinstance(value, list) and isinstance(self.widget.value, tuple):
                return self.widget.value
            return value


class BinFilter(BaseWidgetFilter):
    """
    `BinFilter` is a special `WidgetFilter` that allows selecting from a set of bins.

    The `bins` must be declared from a list of tuples declaring the
    lower- and upper-bound of each bin and an optional set of `labels`
    of the same length.
    """

    bins = param.List(default=[], constant=True, doc="""
        A list of bins expressed as length two tuples.""")

    empty_select = param.Boolean(default=True, doc="""
        Add an option to Select widgets to indicate no filtering.""")

    labels = param.List(default=None, constant=True, doc="""
        A list of labels for each bin (optional).""")

    multi = param.Boolean(default=True, doc="""
        Whether to use a single-value or multi-value selection widget.""")

    filter_type = 'bins'

    def __init__(self, **params):
        super().__init__(**params)
        if self.multi:
            widget = pn.widgets.MultiSelect
        else:
            widget = pn.widgets.Select
        if self.labels:
            options = dict(zip(self.labels, [tuple(b) for b in self.bins]))
        else:
            options = {f'{l} - {u}': (l, u) for l, u in self.bins}
        options[' '] = None
        if self.default is None:
            value = [] if self.multi else None
        else:
            value = tuple(self.default)
        self.widget = widget(name=self.label, options=options, value=value)
        self.widget.link(self, value='value', visible='visible', disabled='disabled', bidirectional=True)

    @property
    def query(self):
        return self.widget.value


class BaseDateFilter(BaseWidgetFilter):

    mode = param.Selector(default='slider', objects=['slider', 'picker'], doc="""
        Whether to use a slider or a picker.""")

    multi = param.Boolean(default=True, doc="""
        Whether to use a single-value or multi-value/range selection widget.""")

    throttled = param.Boolean(default=True, doc="""
        Whether to throttle slider value changes.""")

    _as_date = False

    # Mapping from mode to a Panel widget, or to a function
    # that must return a tuple of a Panel widget and a dictionnary
    # of parameter values that will override the current ones.
    _widget_mode_mapping = {}

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        widget_type = self._widget_mode_mapping[self.mode]
        param_overrides = {}
        if isinstance(widget_type, types.FunctionType):
            widget_type, param_overrides = widget_type(self)
        if param_overrides:
            self.param.set_param(**param_overrides)
        self.widget = widget_type(**self._widget_kwargs(as_date=self._as_date))
        self.widget.link(self, value='value', visible='visible', disabled='disabled', bidirectional=True)

    def _widget_kwargs(self, as_date):
        field_schema = self.schema.get(self.field, {})
        kwargs = {
            'name': self.label,
            'start': pd.to_datetime(field_schema.get('inclusiveMinimum', None)),
            'end': pd.to_datetime(field_schema.get('inclusiveMaximum', None)),
        }
        if self.default is not None:
            kwargs['value'] = pd.to_datetime(self.default)
        if as_date:
            for key in ['value', 'start', 'end']:
                if key in kwargs and kwargs[key] is not None:
                    kwargs[key] = kwargs[key].date()
        return kwargs

    @property
    def panel(self):
        widget = self.widget.clone()
        if self.throttled and self.mode == 'slider':
            self.widget.link(widget, value='value')
            widget.link(self.widget, value_throttled='value')
        else:
            self.widget.link(widget, value='value', visible='visible', disabled='disabled', bidirectional=True)
        return widget

    @property
    def query(self):
        return self.widget.value


# DateFilter and DatetimeFilter both implement __new__ to return
# a Parameterized class that has the correct `value` Parameter type,
# which depends on the filter type (calendar date vs. datetime) and mode
# (slider vs picker).

class DateFilter(BaseDateFilter):
    """
    `DateFilter` is a `WidgetFilter` specialized to select calendar dates.

    Depending on whether `multi` selection is enabled a single date or
    range of dates can be selected with a `DatePicker` or
    `DateRangeSlider` respectively.
    """

    filter_type = 'date'

    def __new__(cls, **params):
        if params.get('multi', cls.param.multi.default):
            return _MultiCalendarDateFilter(**params)
        else:
            return _SingleCalendarDateFilter(**params)


class _SingleCalendarDateFilter(BaseDateFilter):

    _as_date = True

    _widget_mode_mapping = {
        'slider': pn.widgets.DateSlider,
        'picker': pn.widgets.DatePicker,
    }

    value = param.CalendarDate()


class _MultiCalendarDateFilter(BaseDateFilter):

    _as_date = True

    _widget_mode_mapping = {
        'slider': pn.widgets.DateRangeSlider,
        #'picker': pn.widgets.DateRangePicker,
    }

    value = param.CalendarDateRange()


class DatetimeFilter(BaseDateFilter):
    """
    `DatetimeFilter` is a `WidgetFilter` specialized to filter by datetimes.

    Depending on whether `multi` selection is enabled a single date or
    range of dates can be selected with a `DatetimePicker` or
    `DatetimeRangeSlider` respectively.
    """

    filter_type = 'datetime'

    def __new__(cls, **params):
        if params.get('multi', cls.param.multi.default):
            return _MultiDatetimeFilter(**params)
        else:
            return _SingleDatetimeFilter(**params)


def _fallback_to_datetimepicker(inst):
    inst.param.warning(
        'Datetime multi/range slider filter not yet available, '
        'fallback to using a picker filter.'
    )
    return pn.widgets.DatetimePicker, {'mode': 'picker'}


class _SingleDatetimeFilter(BaseDateFilter):

    _as_date = False

    _widget_mode_mapping = {
        'slider': _fallback_to_datetimepicker,
        'picker': pn.widgets.DatetimePicker,
    }

    value = param.Date()


def _handle_datetimerangeslider(inst):
    if Version(bokeh.__version__) <= Version('2.4.3'):
        inst.param.warning(
            'Datetime multi/range slider filter requires Bokeh >= 2.4.3, '
            'fallback to using a picker filter.'
        )
        return pn.widgets.DatetimeRangePicker, {'mode': 'picker'}
    else:
        return pn.widgets.DatetimeRangeSlider, {}


class _MultiDatetimeFilter(BaseDateFilter):

    _as_date = False

    _widget_mode_mapping = {
        'slider': _handle_datetimerangeslider,
        'picker': pn.widgets.DatetimeRangePicker,
    }

    value = param.DateRange()


class ParamFilter(Filter):
    """
    `ParamFilter` reflects the value of a parameter declared on a `View`.

    The `ParamFilter` can be used to implement cross-filtering between
    different views.
    """

    parameter = param.ClassSelector(default=None, class_=(param.Parameter, str), doc="""
        Reference to a Parameter on an existing View.""")

    filter_type = 'param'

    _requires_field = False

    def __init__(self, **params):
        super().__init__(**params)
        self.param.watch(self._attach_watcher, 'parameter')
        self._attach_watcher()

    def _update_value(self, event):
        self.value = event.new

    def _attach_watcher(self, *args):
        if isinstance(self.parameter, param.Parameter):
            self.parameter.owner.param.watch(self._update_value, self.parameter.name)
