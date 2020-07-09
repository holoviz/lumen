import param
import panel as pn

from .schema import JSONSchema


class Filter(param.Parameterized):

    value = param.Parameter()

    schema = param.Dict()

    filter_type = None

    __abstract = True

    @classmethod
    def get(cls, filter_type):
        for filt in param.concrete_descendents(cls).values():
            if filt.filter_type == filter_type:
                return filt
        raise ValueError("No Filter for filter_type '%s' could be found."
                         % filter_type)
    @property
    def panel(self):
        raise NotImplementedError


class ConstantFilter(Filter):
    
    filter_type = 'constant'

    @property
    def query(self):
        return self.value


class WidgetFilter(Filter):

    widget = param.ClassSelector(class_=pn.widgets.Widget)

    filter_type = 'widget'

    def __init__(self, **params):
        super(WidgetFilter, self).__init__(**params)
        self.widget = JSONSchema(schema=self.schema, sizing_mode='stretch_width')._widgets[self.name]
        self.widget.link(self, value='value')

    @property
    def query(self):
        return self.widget.param.value.ir_serialize(self.widget.value)

    @property
    def panel(self):
        return self.widget
