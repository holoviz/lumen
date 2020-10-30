import os
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
        super(Dashboard, self).__init__(
            specification=specification, **params
        )
        self._load_config(from_file=True)

        # Construct template
        tmpl = self.config.get('template', 'material')
        kwargs = {'title': self.config.get('title', 'Lumen Dashboard')}
        if 'logo' in self.config:
            kwargs['logo'] = self.config['logo']
        self.template = _templates[tmpl](**kwargs)

        # Add editor modal
        self._editor = pn.widgets.Ace(
            value=self._yaml, filename=self.specification,
            sizing_mode='stretch_both', min_height=600,
            theme='monokai'
        )
        self._edit_button = pn.widgets.Button(
            name='✎', width=50, css_classes=['reload'], margin=0
        )
        self._editor.link(self, value='_yaml')
        self._edit_button.on_click(self._open_modal)
        self._editor_layout = pn.Column(
            f'## Edit {os.path.basename(self.specification)}',
            self._editor, sizing_mode='stretch_both'
        )
        self.template.modal.append(self._editor_layout)

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
        self._reload_button.on_click(self._reload)
        self._reload()

        # Populate template
        if len(self.filters):
            self.template.sidebar[:] = [self.filters]
        self.template.main[:] = [self.targets]
        self.template.header.append(pn.Row(
            self._reload_button,
            self._edit_button
        ))

    def _open_modal(self, event):
        # Workaround until Panel https://github.com/holoviz/panel/pull/1719
        # is released
        self.template.close_modal()
        self.template.open_modal()

    def _reload_data(self, *events):
        for target in self._targets:
            target.update()

    def _load_config(self, from_file=False):
        # Load config
        if from_file or self._yaml is None:
            with open(self.specification) as f:
                self._yaml = f.read()
        self._spec = yaml.load(self._yaml, Loader=yaml.CLoader)
        if not 'targets' in self._spec:
            raise ValueError('Yaml specification did not declare any targets.')
        self.config = self._spec.get('config', {})

    def _reload(self, *events):
        self._load_config()
        self._sources = {
            name: Source.from_spec(source_spec)
            for name, source_spec in source_specs.items()
        }
        self._targets = self._resolve_targets(self._spec['targets'])
        self._rerender()
        filters = []
        for target in self._targets:
            panel = target.filter_panel
            if panel is not None:
                filters.append(panel)
        self.filters[:] = filters

    def _rerender(self):
        self.targets[:] = [p for target in self._targets for p in target.panels]

    def _resolve_targets(self, target_specs, metadata={}):
        targets = []
        for target_spec in target_specs:
            target_spec = dict(target_spec)

            # Resolve source
            source_spec = target_spec.pop('source', None)
            source = Source.from_spec(source_spec, self._sources)
            schema = source.get_schema()

            # Resolve filters
            filters = [
                Filter.from_spec(filter_spec, schema)
                for filter_spec in target_spec.get('filters', [])
            ]

            # Create target
            target = Monitor(
                application=self, filters=filters, source=source,
                schema=schema, **target_spec)
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
