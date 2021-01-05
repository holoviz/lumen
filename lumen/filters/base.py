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
      about the data to be filtered.""")

    table = param.String(default=None, doc="""
      The table being filtered. If None "applies to all tables.""")

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

    @classmethod
    def from_spec(cls, spec, source_schema):
        """
        Resolves a Filter specification given the schema of the Source
        (and optionally the table) it will be filtering on.

        Parameters
        ----------
        spec: dict
            Specification declared as a dictionary of parameter values.
        source_schema: dict
            A dictionary containing the JSON schema of the Source to
            be filtered on.

        Returns
        -------
        The resolved Filter object.
        """
        spec = dict(spec)
        if not 'field' in spec:
            raise ValueError('Filter specification must declare field to filter on.')
        field = spec['field']
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
        filter_type = Filter._get_type(spec.pop('type'))
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


class WidgetFilter(Filter):
    """
    The WidgetFilter generates a Widget from the table schema provided
    by a Source and returns the current widget value to query the data
    returned by the Source.
    """

    default = param.Parameter(doc="""
        The default value to use on the widget.""")

    empty_select = param.Boolean(default=True, doc="""
        Add an option to Select widgets to indicate no filtering.""")

    multi = param.Boolean(default=True, doc="""
        Whether to use a single-value or multi-value selection widget,
        e.g. for a numeric value this could be a regular slider or a
        range slider.""")

    widget = param.ClassSelector(class_=pn.widgets.Widget)

    filter_type = 'widget'

    def __init__(self, **params):
        super(WidgetFilter, self).__init__(**params)
        self.widget = JSONSchema(
            schema=self.schema, sizing_mode='stretch_width',
            multi=self.multi
        )._widgets[self.field]
        if isinstance(self.widget, pn.widgets.Select) and self.empty_select:
            self.widget.options.insert(0, ' ')
            self.widget.value = ' '
        self.widget.link(self, value='value')
        if self.default is not None:
            self.widget.value = self.default

    @property
    def query(self):
        if self.widget.value == ' ' and self.empty_select:
            return None
        if not hasattr(self.widget.param.value, 'serialize'):
            return self.widget.value
        else:
            return self.widget.param.value.serialize(self.widget.value)

    @property
    def panel(self):
        return self.widget
