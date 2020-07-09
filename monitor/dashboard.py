import yaml
import param
import requests

import panel as pn
import pandas as pd

from .schema import JSONSchema


class QueryAdaptor(param.Parameterized):
    """
    Makes some query to get the metrics and their data
    """

    adaptor_type = None

    __abstract = True
    @classmethod
    def get(cls, adaptor_type):
        for adaptor in param.concrete_descendents(cls).values():
            if adaptor.adaptor_type == adaptor_type:
                return adaptor
        return QueryAdaptor


class RESTAdaptor(QueryAdaptor):

    url = param.String()

    adaptor_type = 'rest'

    def get_metrics(self):
        return requests.get(self.url+'/metrics').json()

    def get_metric(self, metric, **query):
        return pd.DataFrame(requests.get(self.url+'/metric', params=dict(metric=metric, **query)).json())


class Metric(param.Parameterized):

    adaptor = param.ClassSelector(class_=QueryAdaptor)

    filters = param.List()

    metric_type = None

    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._layout = pn.Row()
        for filt in self.filters:
            filt.param.watch(self._update_view, 'value')
        self._update_view()

    @classmethod
    def get(cls, metric_type):
        for metric in param.concrete_descendents(cls).values():
            if metric.metric_type == metric_type:
                return metric
        return DefaultMetric

    def get_data(self):
        query = {filt.name: filt.query for filt in self.filters}
        return self.adaptor.get_metric(self.name, **query)

    def get_view(self):
        return pn.panel(self.get_data())

    def _update_view(self, *events):
        self._layout[:] = [self.get_view()]

    def view(self):
        return self._layout


class DefaultMetric(Metric):

    metric_type = None



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
        return self.widget.param.value.serialize(self.widget.value)

    @property
    def panel(self):
        return self.widget


class View(param.Parameterized):

    adaptor = param.ClassSelector(class_=QueryAdaptor)

    title = param.String()

    current = param.List()

    schema = param.Dict()

    metrics = param.List()

    filters = param.List()

    def __init__(self, **params):
        super(View, self).__init__(**params)
        self._card = pn.Card(height=400, width=400, title=self.title)
        self._update_metrics()

    @pn.depends('current')
    def _update_metrics(self):
        metrics = []
        for metric in self.metrics:
            metric_type = metric.get('type')
            metric = Metric.get(metric_type)(
                adaptor=self.adaptor, filters=self.filters, **metric
            )
            metrics.append(metric)
        self._metrics = metrics
        self._card[:] = [m.view() for m in metrics]

    @property
    def panel(self):
        return self._card


class Dashboard(param.Parameterized):

    specification = param.Filename()

    title = param.String(default='Monitoring Dashboard')

    template = param.ClassSelector(class_=pn.template.base.BasicTemplate)

    reload = param.Action(default=lambda x: x._reload())

    def __init__(self, specification, **params):
        if 'template' not in params:
            params['template'] = pn.template.MaterialTemplate()
        super(Dashboard, self).__init__(
            specification=specification, **params
        )
        self.template.title = self.title
        with open(self.specification) as f:
            self._spec = yaml.load(f.read(), Loader=yaml.BaseLoader)
        if not 'endpoints' in self._spec:
            raise ValueError('%s did not specify any endpoints.'
                             % self.specification)
        self.filters = pn.Column(sizing_mode='stretch_width')
        self.views = pn.GridBox(ncols=5, margin=10)
        self.template.sidebar[:] = [self.filters]
        self.template.main[:] = [self.views]
        self._reload()

    def _reload(self):
        self._views, self._filters = self._resolve_endpoints(self._spec['endpoints'])
        self._rerender()

    def _rerender(self):
        self.views[:] = [view.panel for view in self._views]
        self.filters[:] = [filt.panel for filt in self._filters.values()]

    @classmethod
    def _resolve_endpoints(cls, endpoints, metadata={}):
        views, filters = [], {}
        for endpoint in endpoints:
            view_spec = dict(endpoint)

            # Create adaptor
            endpoint_spec = endpoint.pop('endpoint')
            endpoint_type = endpoint_spec.pop('type', 'rest')
            adaptor = QueryAdaptor.get(endpoint_type)(**endpoint_spec)
            schema = adaptor.get_metrics()

            # Initialize filters
            endpoint_filters = [] 
            for filt in endpoint.get('filters', []):
                fname = filt['name']
                if fname not in filters:
                    for s in schema.values():
                        metric_schema = s['items']['properties']
                        if fname in metric_schema:
                            filt_schema = metric_schema[fname]
                    filter_type = filt.pop('type')
                    filters[fname] = Filter.get(filter_type)(
                        schema={fname: filt_schema}, **filt
                    )
                endpoint_filters.append(filters[fname])

            # Create view
            view_spec['filters'] = endpoint_filters
            view = View(adaptor=adaptor, schema=schema, **view_spec)
            views.append(view)

        return views, filters

    def view(self):
        pass
