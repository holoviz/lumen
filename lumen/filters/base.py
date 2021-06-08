"""
The Filter components supply query parameters used to filter the
tables returned by a Source.
"""

import pandas as pd
import panel as pn
import param

from ..schema import JSONSchema
from ..util import resolve_module_reference


class Filter(param.Parameterized):
    """
    A Filter provides a query which will be used to filter the data
    returned by a Source.
    """

    field = param.String(doc="The field being filtered.")

    label = param.String(doc="A label for the Filter.")

    schema = param.Dict(doc="""
      The JSON schema provided by the Source declaring information
      about the data to be filtered.""")

    shared = param.Boolean(default=False, doc="""
      Whether the filter is shared across all targets.""")

    table = param.String(default=None, doc="""
      The table being filtered. If None applies to all tables.""")

    value = param.Parameter(doc="The current filter value.")

    filter_type = None

    _requires_field = True

    __abstract = True

    @classmethod
    def _get_type(cls, filter_type):
        if '.' in filter_type:
            return resolve_module_reference(filter_type, Filter)
        try:
            __import__(f'lumen.filters.{filter_type}')
        except Exception:
            pass
        for filt in param.concrete_descendents(cls).values():
            if filt.filter_type == filter_type:
                return filt
        raise ValueError(f"No Filter for filter_type '{filter_type}' could be found.")

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
        filter_type = Filter._get_type(spec.pop('type'))
        if not filter_type._requires_field:
            return filter_type(**spec)
        elif not 'field' in spec:
            raise ValueError('Filter specification must declare field to filter on.')
        field = spec['field']
        if 'label' not in spec:
            spec['label'] = field.title()
        table = spec.get('table')
        schema = None
        for key, table_schema in source_schema.items():
            if table is not None and key != table:
                continue
            schema = table_schema.get(field)
            if schema is not None:
                break
        if not schema:
            raise ValueError("Source did not declare a schema for "
                             f"'{field}' filter.")
        return filter_type(schema={field: schema}, **spec)

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
        return [
            ConstantFilter(field=self.field, value=value, label=self.label)
            for value in self.schema[self.field]['enum']
        ]


class BaseWidgetFilter(Filter):

    default = param.Parameter(doc="""
        The default value to use on the widget.""")

    __abstract__ = True

    @property
    def panel(self):
        widget = self.widget.clone()
        self.widget.link(widget, value='value', bidirectional=True)
        return widget



class WidgetFilter(BaseWidgetFilter):
    """
    The WidgetFilter generates a Widget from the table schema provided
    by a Source and returns the current widget value to query the data
    returned by the Source.
    """

    empty_select = param.Boolean(default=True, doc="""
        Add an option to Select widgets to indicate no filtering.""")

    multi = param.Boolean(default=True, doc="""
        Whether to use a single-value or multi-value selection widget,
        e.g. for a numeric value this could be a regular slider or a
        range slider.""")

    widget = param.ClassSelector(class_=pn.widgets.Widget)

    filter_type = 'widget'

    def __init__(self, **params):
        super().__init__(**params)
        self.widget = JSONSchema(
            schema=self.schema, sizing_mode='stretch_width', multi=self.multi
        )._widgets[self.field]
        if isinstance(self.widget, pn.widgets.Select) and self.empty_select:
            self.widget.options.insert(0, ' ')
            self.widget.value = ' '
        self.widget.name = self.label
        self.widget.link(self, value='value')
        if self.default is not None:
            self.widget.value = self.default

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
    The BinFilter allows declaring a set of bins as a list of tuples
    and an optional set of labels of the same length.
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
        self.widget.link(self, value='value')

    @property
    def query(self):
        return self.widget.value


class DateFilter(BaseWidgetFilter):

    mode = param.Selector(default='slider', objects=['slider', 'picker'], doc="""
        Whether to use a slider or a picker.""")

    throttled = param.Boolean(default=True, doc="""
        Whether to throttle slider value changes.""")

    filter_type = 'date'

    def __init__(self, **params):
        super().__init__(**params)
        if self.mode == 'slider':
            widget = pn.widgets.DateRangeSlider
        else:
            widget = pn.widget.DatePicker
        field_schema = self.schema.get(self.field, {})
        start = pd.to_datetime(field_schema.get('inclusiveMinimum', None))
        end = pd.to_datetime(field_schema.get('inclusiveMaximum', None))
        kwargs = {'name': self.label, 'start': start, 'end': end}
        if self.default is not None:
            kwargs['value'] = pd.to_datetime(self.default)
        self.widget = widget(**kwargs)
        self.widget.link(self, value='value')

    @property
    def panel(self):
        widget = self.widget.clone()
        if self.throttled and self.mode == 'slider':
            self.widget.link(widget, value='value')
            widget.link(self.widget, value_throttled='value')
        else:
            self.widget.link(widget, value='value', bidirectional=True)
        return widget

    @property
    def query(self):
        return self.widget.value


class ParamFilter(Filter):

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
