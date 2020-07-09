import panel as pn
import param


class JSONSchema(pn.pane.PaneBase):

    default_layout = param.ClassSelector(default=pn.Column, class_=(pn.layout.Panel),
                                         is_instance=False, doc="""
        Defines the layout the model(s) returned by the pane will
        be placed in.""")

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
    _bounded_number_widget = pn.widgets.FloatSlider
    _unbounded_number_widget = pn.widgets.Spinner
    _list_select_widget = pn.widgets.MultiSelect
    _bounded_int_widget = pn.widgets.IntSlider
    _string_widget = pn.widgets.TextInput
    _boolean_widget = pn.widgets.Checkbox
    _literal_widget = pn.widgets.LiteralInput
    _date_widget = pn.widgets.DatetimeInput

    @classmethod
    def _array_type(cls, schema):
        if 'items' in schema and not schema.get('additionalItems', True):
            items = schema['items']
            if len(items) == 2:
                if all(s['type'] == 'number' for s in items):
                    return pn.widgets.RangeSlider, {}
                elif all(s['type'] == 'integer' for s in items):
                    return pn.widgets.IntRangeSlider, {}
            elif 'enum' in items:
                return cls._list_select_type, {'options': items['enum']}
        else:
            return cls._literal_widget, {'type': (list, tuple)}

    @classmethod
    def _number_type(cls, schema):
        if 'maximum' in schema and 'minimum' in schema:
            return cls._bounded_number_widget, {'start': schema['minimum'], 'end': schema['maximum']}
        else:
            return cls._bounded_number_widget, {}

    @classmethod
    def _integer_type(cls, schema):
        if 'maximum' in schema and 'minimum' in schema:
            return cls._bounded_int_widget, {'start': schema['minimum'], 'end': schema['maximum']}
        else:
            return cls._bounded_number_widget, {'step': 1}

    @classmethod
    def _boolean_type(cls, schema):
        return cls._boolean_widget, {}

    @classmethod
    def _string_type(cls, schema):
        if schema.get('format') == 'date-time':
            return cls._date_widget, {}
        return cls._string_widget, {}

    @classmethod
    def _enum(cls, schema):
        return cls._select_widget, {'options': schema['enum']}

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
