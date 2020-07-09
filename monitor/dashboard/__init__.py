import yaml

import param
import panel as pn

from .adaptor import QueryAdaptor, RESTAdaptor # noqa
from .filter import ConstantFilter, Filter, WidgetFilter # noqa
from .metric import DefaultMetric, Metric # noqa
from .view import View # noqa


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
            self._spec = yaml.load(f.read(), Loader=yaml.CLoader)
        if not 'endpoints' in self._spec:
            raise ValueError('%s did not specify any endpoints.'
                             % self.specification)
        self.filters = pn.Column(sizing_mode='stretch_width')
        self.views = pn.GridBox(ncols=5, margin=10)
        self._reload()
        if self._filters:
            self.template.sidebar[:] = [self.filters]
        self.template.main[:] = [self.views]

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
            endpoint_spec = view_spec.pop('endpoint')
            endpoint_type = endpoint_spec.pop('type', 'rest')
            adaptor = QueryAdaptor.get(endpoint_type)(**endpoint_spec)
            schema = adaptor.get_metrics()

            # Initialize filters
            endpoint_filters = [] 
            for filt in endpoint.get('filters', []):
                fname = filt['name']
                if fname not in filters:
                    filt_schema = None
                    for s in schema.values():
                        metric_schema = s['items']['properties']
                        if fname in metric_schema:
                            filt_schema = metric_schema[fname]
                    if not filt_schema:
                        continue
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

