import yaml

import param
import panel as pn

from panel.template.base import BasicTemplate

from .filters import ConstantFilter, Filter, WidgetFilter # noqa
from .monitor import Monitor # noqa
from .sources import Source, RESTSource # noqa
from .views import View # noqa

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

    def __init__(self, specification, **params):
        # Load config
        with open(specification) as f:
            self._spec = yaml.load(f.read(), Loader=yaml.CLoader)
        if not 'targets' in self._spec:
            raise ValueError('Yaml specification did not declare any targets.')
        self.config = self._spec.get('config', {})
        super(Dashboard, self).__init__(
            specification=specification, **params
        )

        # Construct template
        tmpl = self.config.get('template', 'material')
        kwargs = {'title': self.config.get('title', 'Lumen Dashboard')}
        if 'logo' in self.config:
            kwargs['logo'] = self.config['logo']
        self.template = _templates[tmpl](**kwargs)

        # Build layouts
        self.filters = pn.Column(margin=0, sizing_mode='stretch_width')
        layout = self.config.get('layout', 'grid')
        if layout == 'grid':
            self.targets = pn.GridBox(ncols=self.config.get('ncols', 5),
                                      margin=10, sizing_mode='stretch_width')
        elif layout == 'tabs':
            self.targets = pn.Tabs(sizing_mode='stretch_width')
        self._reload_button = pn.widgets.Button(
            name='↻', width=50, css_classes=['reload'], margin=0
        )
        self._reload_button.on_click(self._reload_data)
        self._reload()

        # Populate template
        if len(self.filters):
            self.template.sidebar[:] = [self.filters]
        self.template.main[:] = [self.targets]
        self.template.header.append(self._reload_button)

    def _reload_data(self, *events):
        for target in self._targets:
            target.update()

    def _reload(self, *events):
        self._targets = self._resolve_sources(self._spec['targets'])
        self._rerender()
        filters = []
        for target in self._targets:
            panel = target.filter_panel
            if panel is not None:
                filters.append(panel)
        self.filters[:] = filters

    def _rerender(self):
        self.targets[:] = [p for target in self._targets for p in target.panels]

    def _resolve_sources(self, sources, metadata={}):
        targets = []
        for target_spec in sources:
            target_spec = dict(target_spec)

            # Create source
            source_spec = dict(target_spec.pop('source'))
            source_type = source_spec.pop('type', 'rest')
            source = Source._get_type(source_type)(**source_spec)
            schema = source.get_schema()

            # Initialize filters
            source_filters = []
            for filter_spec in target_spec.get('filters', []):
                filter_spec = dict(filter_spec)
                filter_name = filter_spec['field']
                filter_schema = None
                for view_schema in schema.values():
                    if filter_name in view_schema:
                        filter_schema = view_schema[filter_name]
                if not filter_schema:
                    continue
                filter_type = filter_spec.pop('type')
                filter_obj = Filter._get_type(filter_type)(
                    schema={filter_name: filter_schema}, **filter_spec
                )
                source_filters.append(filter_obj)

            # Create target
            target_spec['filters'] = source_filters
            target = Monitor(source=source, application=self, schema=schema, **target_spec)
            targets.append(target)
        return targets

    def show(self):
        """
        Opens the dashboard in a new window.
        """
        title = self.config.get('title', 'Lumen Dashboard')
        self.template.show(title=title)
        for target in self._targets:
            target.start()

    def servable(self):
        """
        Marks the dashboard for serving with `panel serve`.
        """
        title = self.config.get('title', 'Lumen Dashboard')
        self.template.servable(title=title)
        for target in self._targets:
            target.start()
