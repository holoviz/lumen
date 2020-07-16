import yaml

import param
import panel as pn

from panel.template.base import BasicTemplate

from .adaptor import QueryAdaptor, RESTAdaptor # noqa
from .filter import ConstantFilter, Filter, WidgetFilter # noqa
from .metric import DefaultMetricView, MetricView # noqa
from .view import View # noqa


_templates = {k[:-8].lower(): v for k, v in param.concrete_descendents(BasicTemplate).items()}


class Dashboard(param.Parameterized):

    specification = param.Filename()

    reload = param.Action(default=lambda x: x._reload())

    def __init__(self, specification, **params):
        with open(specification) as f:
            self._spec = yaml.load(f.read(), Loader=yaml.CLoader)
        config = self._spec.get('config', {}) 
        super(Dashboard, self).__init__(
            specification=specification, **params
        )
        tmpl = config.get('template', 'material')
        self.template = _templates[tmpl](title=config.get('title', 'Monitoring Dashboard'))
        if not 'endpoints' in self._spec:
            raise ValueError('%s did not specify any endpoints.'
                             % self.specification)
        self.filters = pn.Accordion(margin=0, sizing_mode='stretch_width')
        self.views = pn.GridBox(ncols=config.get('ncols', 5), margin=10)
        self._reload()
        if len(self.filters):
            self.template.sidebar[:] = [self.filters]
        self.template.main[:] = [self.views]

    def _reload(self):
        self._views = self._resolve_endpoints(self._spec['endpoints'])
        self._rerender()

    def _rerender(self):
        self.views[:] = [p for view in self._views for p in view.panels]
        filters = []
        for view in self._views:
            view_filters = [filt.panel for filt in view.filters
                            if filt.panel is not None]
            if view_filters:
                filter_col = pn.Column(*view_filters, sizing_mode='stretch_width')
                filters.append((view.title, filter_col))
        self.filters[:] = filters

    @classmethod
    def _resolve_endpoints(cls, endpoints, metadata={}):
        views = []
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
                filt_schema = None
                for s in schema.values():
                    metric_schema = s['items']['properties']
                    if fname in metric_schema:
                        filt_schema = metric_schema[fname]
                if not filt_schema:
                    continue
                filter_type = filt.pop('type')
                filter_obj = Filter.get(filter_type)(
                    schema={fname: filt_schema}, **filt
                )
                endpoint_filters.append(filter_obj)

            # Create view
            view_spec['filters'] = endpoint_filters
            view = View(adaptor=adaptor, schema=schema, **view_spec)
            views.append(view)
        return views
