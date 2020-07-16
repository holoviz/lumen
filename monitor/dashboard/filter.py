"""
The Filter components of the monitor provide the ability to filter
the data returned by a QueryAdaptor.
"""

import param
import panel as pn

from .schema import JSONSchema


class Filter(param.Parameterized):
    """
    A filter provides the ability to return a query which will be used
    to filter the data returned by a QueryAdaptor.
    """

    schema = param.Dict(doc="""
      The JSON schema provided by the QueryAdaptor declaring
      information about the data.""" )

    label = param.String(doc="")

    value = param.Parameter(doc="The current filter value")

    filter_type = None

    __abstract = True

    @classmethod
    def get(cls, filter_type):
        for filt in param.concrete_descendents(cls).values():
            if filt.filter_type == filter_type:
                return filt
        raise ValueError(f"No Filter for filter_type '{filter_type}' could be found.")

    @property
    def panel(self):
        """
        Should return an object that can be displayed by Panel 
        representing the Filter value. 
        """
        raise NotImplementedError


class ConstantFilter(Filter):
    """
    The ConstantFilter allows requesting a constant value from the
    QueryAdaptor.
    """
    
    filter_type = 'constant'

    @property
    def query(self):
        return self.value

    @property
    def panel(self):
        None


class FacetFilter(Filter):

    filter_type = 'facet'

    @property
    def query(self):
        None
        
    @property
    def panel(self):
        None

    @property
    def filters(self):
        return [
            ConstantFilter(name=self.name, value=value, label=self.label)
            for value in self.schema[self.name]['enum']
        ]


class WidgetFilter(Filter):
    """
    The WidgetFilter generates a Widget from the schema used for
    generate the query to the QueryAdaptor.
    """

    widget = param.ClassSelector(class_=pn.widgets.Widget)

    filter_type = 'widget'

    def __init__(self, **params):
        super(WidgetFilter, self).__init__(**params)
        self.widget = JSONSchema(schema=self.schema, sizing_mode='stretch_width')._widgets[self.name]
        self.widget.link(self, value='value')

    @property
    def query(self):
        return self.widget.param.value.serialize(self.widget.value)

    @property
    def panel(self):
        return self.widget
