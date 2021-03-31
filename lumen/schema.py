import pandas as pd
import panel as pn
import param


class JSONSchema(pn.pane.PaneBase):

    default_layout = param.ClassSelector(default=pn.Column, class_=(pn.layout.Panel),
                                         is_instance=False, doc="""
        Defines the layout the model(s) returned by the pane will
        be placed in.""")

    multi = param.Boolean(default=True, doc="""
        Whether to pick multi-select and range widgets by default.""")

    properties = param.List(default=[], doc="""
        If set this serves as a whitelist of properties to display on
        the supplied schema object.""")

    schema = param.Dict(doc="""
        A valid jsonschema""")

    widgets = param.Dict(doc="""
        Dictionary of widget overrides, mapping from parameter name
        to widget class.""")

    _precedence = ['enum', 'type']

    _unpack = True

    _select_widget = pn.widgets.Select
    _multi_select_widget = pn.widgets.MultiSelect
    _bounded_number_widget = pn.widgets.FloatSlider
    _bounded_number_range_widget = pn.widgets.RangeSlider
    _unbounded_number_widget = pn.widgets.FloatInput
    _list_select_widget = pn.widgets.MultiSelect
    _bounded_int_widget = pn.widgets.IntSlider
    _bounded_int_range_widget = pn.widgets.IntRangeSlider
    _unbounded_int_widget = pn.widgets.IntInput
    _string_widget = pn.widgets.TextInput
    _boolean_widget = pn.widgets.Checkbox
    _literal_widget = pn.widgets.LiteralInput
    _unbounded_date_widget = pn.widgets.DatePicker
    _date_range_widget = pn.widgets.DateRangeSlider

    # Not available until Panel 0.11
    try:
        _datetime_range_widget = pn.widgets.DatetimeRangeInput
    except Exception:
        _datetime_range_widget = pn.widgets.DateRangeSlider

    # Not available until Panel 0.12
    try:
        _unbounded_datetime_widget = pn.widgets.DatetimePicker
        _datetime_range_widget = pn.widgets.DatetimeRangePicker
    except Exception:
        _unbounded_datetime_widget = pn.widgets.DatetimeInput

    def _array_type(self, schema):
        if 'items' in schema and not schema.get('additionalItems', True):
            items = schema['items']
            if len(items) == 2:
                if all(s['type'] == 'number' for s in items):
                    return pn.widgets.RangeSlider, {}
                elif all(s['type'] == 'integer' for s in items):
                    return pn.widgets.IntRangeSlider, {}
            elif 'enum' in items:
                return self._list_select_type, {'options': items['enum']}
        else:
            return self._literal_widget, {'type': (list, tuple)}

    def _number_type(self, schema):
        if 'inclusiveMaximum' in schema and 'inclusiveMinimum' in schema:
            if self.multi:
                wtype = self._bounded_number_range_widget
            else:
                wtype = self._bounded_number_widget
            return wtype, {'start': schema['inclusiveMinimum'],
                           'end': schema['inclusiveMaximum']}
        else:
            return self._unbounded_number_widget, {}

    def _integer_type(self, schema):
        if 'inclusiveMaximum' in schema and 'inclusiveMinimum' in schema:
            if self.multi:
                wtype = self._bounded_int_range_widget
            else:
                wtype = self._bounded_int_widget
            return wtype, {'start': schema['inclusiveMinimum'],
                           'end': schema['inclusiveMaximum']}
        else:
            return self._unbounded_int_widget, {'step': 1}

    def _boolean_type(self, schema):
        return self._boolean_widget, {}

    def _string_type(self, schema):
        if schema.get('format') in ('date', 'date-time', 'datetime'):
            if 'formatMaximum' in schema and 'formatMinimum' in schema:
                dt_min = pd.to_datetime(schema['formatMinimum'])
                if dt_min.tz:
                    dt_min = dt_min.astimezone('UTC').tz_localize(None)
                dt_max = pd.to_datetime(schema['formatMaximum'])
                if dt_max.tz:
                    dt_max = dt_max.astimezone('UTC').tz_localize(None)
                dt_range = {'start': dt_min.to_pydatetime(),
                            'end': dt_max.to_pydatetime()}
                if schema.get('format') == 'date':
                    wtype = self._date_range_widget
                else:
                    wtype = self._datetime_range_widget
                return wtype, dt_range
            if 'inclusiveMaximum' in schema and 'inclusiveMinimum' in schema:
                dt_min = pd.to_datetime(schema['inclusiveMinimum'])
                if dt_min.tz:
                    dt_min = dt_min.astimezone('UTC').tz_localize(None)
                dt_max = pd.to_datetime(schema['inclusiveMaximum'])
                if dt_max.tz:
                    dt_max = dt_max.astimezone('UTC').tz_localize(None)
                dt_range = {'start': dt_min.to_pydatetime(),
                            'end': dt_max.to_pydatetime()}
                if schema.get('format') == 'date':
                    wtype = self._date_range_widget
                else:
                    wtype = self._datetime_range_widget
                return wtype, dt_range
            elif schema.get('format') == 'date':
                return self._unbounded_date_widget, {}
            else:
                return self._unbounded_datetime_widget, {}
        return self._string_widget, {}

    def _enum(self, schema):
        wtype = self._multi_select_widget if self.multi else self._select_widget
        return wtype, {'options': schema['enum']}

    def __init__(self, object=None, schema=None, **params):
        if schema is not None:
            params['schema'] = schema
        super().__init__(object, **params)
        self._widgets = {}
        self._update_widgets_from_schema()

    @param.depends('schema', watch=True)
    def _update_widgets_from_schema(self):
        if not self.schema:
            self.layout[:] = []
            return

        values = {} if self.object is None else self.object
        widgets = []
        for p, schema in self.schema.items():
            if self.properties and p not in self.properties:
                continue
            for prop in self._precedence:
                if prop in schema:
                    break

            if self.widgets is None or p not in self.widgets:
                wtype, kwargs = self._widget_type(prop, schema)
            elif isinstance(self.widgets[p], dict):
                kwargs = dict(self.widgets[p])
                if 'type' in kwargs:
                    wtype = kwargs.pop('type')
                else:
                    wtype = self._widget_type(prop, schema)
            else:
                wtype = self.widgets[p]

            if isinstance(wtype, pn.widgets.Widget):
                widget = wtype
            else:
                if p in values:
                    kwargs['value'] = values[p]
                widget = wtype(name=schema.get('title', p), **kwargs)
            self._widgets[p] = widget
            widgets.append(widget)
        self.layout[:] = widgets

    def _widget_type(self, prop, schema):
        if prop == 'enum':
            wtype, kwargs = self._enum(schema)
        else:
            wtype, kwargs = getattr(self, '_%s_%s' % (schema[prop], prop))(schema)
        return wtype, kwargs

    @param.depends('object', watch=True)
    def _update_widget_values(self):
        if not self.object:
            return
        for p, value in self.object.items():
            self._widgets[p].value = value

    def _get_model(self, doc, root=None, parent=None, comm=None):
        model = self.layout._get_model(doc, root, parent, comm)
        self._models[root.ref['id']] = (model, parent)
        return model

    def _cleanup(self, root):
        self.layout._cleanup(root)
        super(JSONSchema, self)._cleanup(root)
