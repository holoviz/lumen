"""
The Filter components supply query parameters used to filter the
tables returned by a Source.
"""

import param
import panel as pn

from ..schema import JSONSchema


class Filter(param.Parameterized):
    """
    A Filter provides a query which will be used to filter the data
    returned by a Source.
    """

    field = param.String(doc="The field being filtered.")

    label = param.String(doc="A label for the Filter.")

    schema = param.Dict(doc="""
        The JSON schema provided by the Source declaring information
        about the data to be filtered.""" )

    value = param.Parameter(doc="The current filter value.")

    filter_type = None

    __abstract = True

    @classmethod
    def _get_type(cls, filter_type):
        try:
            __import__(f'lumen.filters.{filter_type}')
        except Exception:
            pass
        for filt in param.concrete_descendents(cls).values():
            if filt.filter_type == filter_type:
                return filt
        raise ValueError(f"No Filter for filter_type '{filter_type}' could be found.")

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


class WidgetFilter(Filter):
    """
    The WidgetFilter generates a Widget from the table schema provided
    by a Source and returns the current widget value to query the data
    returned by the Source.
    """

    widget = param.ClassSelector(class_=pn.widgets.Widget)

    filter_type = 'widget'

    def __init__(self, **params):
        super(WidgetFilter, self).__init__(**params)
        self.widget = JSONSchema(
            schema=self.schema, sizing_mode='stretch_width'
        )._widgets[self.field]
        if isinstance(self.widget, pn.widgets.Select):
            self.widget.options.insert(0, ' ')
            self.widget.value = ' '
        self.widget.link(self, value='value')

    @property
    def query(self):
        if self.widget.value == ' ':
            return None
        return self.widget.param.value.serialize(self.widget.value)

    @property
    def panel(self):
        return self.widget
