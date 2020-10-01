import yaml

import param
import panel as pn

from panel.template.base import BasicTemplate

from .adaptors import QueryAdaptor, RESTAdaptor # noqa
from .filters import ConstantFilter, Filter, WidgetFilter # noqa
from .monitor import Monitor # noqa
from .views import DefaultView, MetricView # noqa

_templates = {k[:-8].lower(): v for k, v in param.concrete_descendents(BasicTemplate).items()}

pn.config.raw_css.append("""
.reload .bk-btn {
  background: transparent;
  border: none;
  font-size: 18pt;
}
#header .reload .bk-btn {
  color: white;
}
.reload .bk-btn:hover {
  background: transparent;
}
""")

class Dashboard(param.Parameterized):

    specification = param.Filename()

    reload = param.Action(default=lambda x: x._reload())

    def __init__(self, specification, **params):
        with open(specification) as f:
            self._spec = yaml.load(f.read(), Loader=yaml.CLoader)
        self.config = self._spec.get('config', {}) 
        super(Dashboard, self).__init__(
            specification=specification, **params
        )
        tmpl = self.config.get('template', 'material')
        self.template = _templates[tmpl](title=self.config.get('title', 'Monitoring Dashboard'))
        if not 'endpoints' in self._spec:
            raise ValueError('%s did not specify any endpoints.'
                             % self.specification)
        self.filters = pn.Column(margin=0, sizing_mode='stretch_width')
        self.monitors = pn.GridBox(ncols=self.config.get('ncols', 5), margin=10)
        self._reload()
        if len(self.filters):
            self.template.sidebar[:] = [self.filters]
        self.template.main[:] = [self.monitors]
        self._reload_button = pn.widgets.Button(
            name='↻', width=50, css_classes=['reload'], margin=0
        )
        self._reload_button.on_click(self._reload_data)
        self.template.header.append(self._reload_button)

    def _reload_data(self, *events):
        for monitor in self._monitors:
            monitor.update()

    def _reload(self, *events):
        self._monitors = self._resolve_endpoints(self._spec['endpoints'])
        self._rerender()
        filters = []
        for monitor in self._monitors:
            panel = monitor.filter_panel
            if panel is not None:
                filters.append(panel)
        self.filters[:] = filters

    def _rerender(self):
        self.monitors[:] = [p for monitor in self._monitors for p in monitor.panels]

    def _resolve_endpoints(self, endpoints, metadata={}):
        monitors = []
        for endpoint in endpoints:
            monitor_spec = dict(endpoint)

            # Create adaptor
            endpoint_spec = dict(monitor_spec.pop('endpoint'))
            endpoint_type = endpoint_spec.pop('type', 'rest')
            adaptor = QueryAdaptor.get(endpoint_type)(**endpoint_spec)
            schema = adaptor.get_metrics()

            # Initialize filters
            endpoint_filters = []
            for filt in endpoint.get('filters', []):
                filt = dict(filt)
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

            # Create monitor
            monitor_spec['filters'] = endpoint_filters
            monitor = Monitor(adaptor=adaptor, application=self, schema=schema, **monitor_spec)
            monitors.append(monitor)
        return monitors

    def show(self):
        """
        Opens the monitoring dashboard in a new window.
        """
        self.template.show()
        for monitor in self._monitors:
            monitor.start()

    def servable(self):
        """
        Marks the monitoring dashboard for serving with `panel serve`.
        """
        self.template.servable(title=self.config.get('title', 'Monitoring Dashboard'))
        for monitor in self._monitors:
            monitor.start()
