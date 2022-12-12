import pathlib

import panel as pn
import param  # type: ignore

from lumen.filters.base import Filter
from lumen.pipeline import Pipeline
from lumen.state import state as lm_state
from lumen.transforms.base import Transform

from .base import WizardItem
from .fast import FastComponent
from .gallery import Gallery, GalleryItem
from .state import state

ASSETS_DIR = pathlib.Path(__file__).parent / 'assets'


class PipelineEditor(FastComponent):

    filter_type = param.Selector(doc="Select a Filter to add to this Pipeline.")

    transform_type = param.Selector(doc="Select a Transform to add to this Pipeline.")

    filter_items = param.List()

    filters = param.ClassSelector(class_=(list, dict), precedence=-1)

    transform_items = param.List()

    transforms = param.List(precedence=-1)

    preview = param.Parameter()

    sizing_mode = param.String(default='stretch_width')

    source = param.Parameter()

    table = param.Parameter()

    spec = param.Dict(precedence=-1)

    resize = param.Integer(default=0)

    margin = param.Integer(default=10)

    _preview_display = param.String(default='none')

    _scripts = {
        'resize': ['window.dispatchEvent(new Event("resize"))']
    }

    _template = """
    <div style="display: flex; flex-direction: column; ">
      <span style="font-size: 2em"><b>{{ name }}</b></span>
      <span style="font-size: 1em"><b>Pipeline driven by {{ table }} table in the {{ source }} source</b></span>
    </div>
    <div style="display: flex; margin-top: 1em;">
      <div style="display: grid; flex: auto;">
        <label for="filter-select-${id}">
          <b>{{ param.filter_type.label }}</b>
        </label>
        <fast-select id="filter-select" style="min-width: 150px;" value="${filter_type}">
          {% for ftype in param.filter_type.objects %}
          <fast-option id="filter-option-{{ loop.index0 }}" value="{{ ftype }}">{{ ftype.title() }}</fast-option>
          {% endfor %}
        </fast-select>
        <fast-tooltip anchor="filter-select-${id}">{{ param.filter_type.doc }}</fast-tooltip>
      </div>
      <fast-button id="add-filter" appearance="accent" style="margin-top: auto; margin-left: 1em; width: 20px;" onclick="${_add_filter}">
        <b style="font-size: 2em;">+</b>
      </fast-button>
    </div>
    <div id="filters" style="display: flex; flex-wrap: wrap;">
       {% for fitem in filter_items %}
       <div id="filter-item" style="margin: 0.5em; width: 335px; border: var(--accent-fill-rest) 1px solid; border-radius: 0.5em;">
         ${fitem}
         <fast-button id="filt-remove-button-{{ loop.index0 }}" appearance="accent" onclick="${_remove_filter}" style="float: right; z-index: 100;">
           <b style="font-size: 2em;">-</b>
         </fast-button>
       </div>
       {% endfor %}
    </div>
    <div style="display: flex;">
      <div style="display: grid; flex: auto;">
        <label for="transform-select-${id}">
          <b>{{ param.transform_type.label }}</b>
        </label>
        <fast-select id="transform-select" style="min-width: 150px;" value="${transform_type}">
          {% for ftype in param.transform_type.objects %}
          <fast-option id="transform-option-{{ loop.index0 }}" value="{{ ftype }}">{{ ftype.title() }}</fast-option>
          {% endfor %}
        </fast-select>
        <fast-tooltip anchor="transform-select-${id}">{{ param.transform_type.doc }}</fast-tooltip>
      </div>
      <fast-button id="add-transform" appearance="accent" style="margin-top: auto; margin-left: 1em; width: 20px;" onclick="${_add_transform}">
        <b style="font-size: 2em;">+</b>
      </fast-button>
    </div>
    <div id="transforms" style="display: flex; flex-wrap: wrap;">
       {% for titem in transform_items %}
       <div id="transform-item" style="margin: 0.5em; width: 335px; border: var(--accent-fill-rest) 1px solid; border-radius: 0.5em;">
         ${titem}
         <fast-button id="transform-remove-button-{{ loop.index0 }}" appearance="accent" onclick="${_remove_transform}" style="float: right; z-index: 100;">
           <b style="font-size: 2em;">-</b>
         </fast-button>
       </div>
       {% endfor %}
    </div>
    <fast-button id="preview-button" onclick="${_preview}" style="margin-left: auto; margin-top: 1em; z-index: 100;">
      Preview
    </fast-button>
    <fast-divider></faster-divider>
    <div id="preview-area" style="display: ${_preview_display}; margin-top: 2em;">
      <div id="preview" style="flex: auto; margin-left: 1em; overflow-y: auto;">${preview}</div>
    </div>
    """

    thumbnail = ASSETS_DIR / 'source.png'

    _child_config = {
        'filter_items': 'model',
        'transform_items': 'model'
    }

    def __init__(self, **params):
        spec = params.pop('spec', {})
        params.update(**{
            k: v for k, v in spec.items() if k in self.param and k not in params
        })
        filters = params.pop('filters', [])
        transforms = params.pop('transforms', [])
        super().__init__(spec=spec, filters=[], transforms=[], **params)
        theme = 'midnight' #if getattr(pn.config, 'theme', 'default') == 'dark' else 'simple'
        self.preview = pn.widgets.Tabulator(
            sizing_mode='stretch_width', pagination='remote', page_size=12,
            theme=theme, height=400
        )
        self._pipeline = None
        self._populate_selectors()
        for filt in (filters.values() if isinstance(filters, dict) else filters):
            self._add_filter(spec=filt)
        for trans in transforms:
            self._add_transform(spec=trans)

    def _populate_selectors(self):
        filters = param.concrete_descendents(Filter).values()
        self.param.filter_type.objects = ftypes = [f.filter_type for f in filters if f.filter_type]
        self.filter_type = ftypes[0]
        transforms = param.concrete_descendents(Transform).values()
        self.param.transform_type.objects = ttypes = [
            t.transform_type for t in transforms if t.transform_type
        ]
        self.transform_type = ttypes[0]

    @param.depends('_preview_display', watch=True)
    def _update_displayed(self):
        if self._preview_display == 'none' or self._pipeline is not None:
            return
        self._pipeline = Pipeline.from_spec({
            k: v for k, v in self.spec.items() if k not in ('filters', 'transforms')
        })
        filters = self.filters.values() if isinstance(self.filters, dict) else self.filters
        for filt in filters or []:
            if not isinstance(filt, Filter):
                filt = Filter.from_spec(filt, {self._pipeline.table: self._pipeline.schema})
            self._pipeline.add_filter(filt)
        for trans in self.transforms or []:
            if not isinstance(trans, Filter):
                trans = Transform.from_spec(trans)
            self._pipeline.add_transform(trans)
        self._pipeline.param.watch(self._update_preview, ['data'])
        self.preview.value = self._pipeline.data

    def _update_preview(self, event):
        self.preview.value = event.new

    def _update_spec(self, *events):
        self.spec['filters'] = []
        for filt in self.filters:
            filt_spec = filt.to_spec()
            self.spec['filters'].append(filt_spec)
        self.spec['transforms'] = []
        for transform in self.transforms:
            transform_spec = transform.to_spec()
            self.spec['transforms'].append(transform_spec)

    def _preview(self, event):
        self._preview_display = 'flex' if self._preview_display == 'none' else 'none'
        self.resize += 1

    def _remove_filter(self, event):
        index = int(event.node.split('-')[-1])
        self.filters.pop(index)
        self.filter_items.pop(index)
        self.param.trigger('filters')
        self.param.trigger('filter_items')
        self._update_spec()

    def _remove_transform(self, event):
        index = int(event.node.split('-')[-1])
        self.transforms.pop(index)
        self.transform_items.pop(index)
        self.param.trigger('transforms')
        self.param.trigger('transform_items')
        self._update_spec()

    def _add_filter(self, event=None, spec=None):
        self.loading = True
        source = lm_state.sources[self.source]
        schema = source.get_schema(self.table)
        if spec is None:
            spec = {
                'type': self.filter_type,
                'table': self.table
            }
            if schema:
                spec['field'] = list(schema)[0]
        filt = Filter.from_spec(spec, {self.table: schema})
        filt.param.watch(self._update_spec, list(filt.param))
        if self._pipeline:
            self._pipeline.add_filter(filt)

        self.filters.append(filt)
        self.param.trigger('filters')
        params = [p for p in filt.param if p not in ('name', 'schema', 'table', 'widget')]
        self.filter_items.append(pn.Param(filt, parameters=params))
        self.param.trigger('filter_items')
        self._update_spec()
        self.loading = False

    def _add_transform(self, event=None, spec=None):
        self.loading = True
        source = lm_state.sources[self.source]
        schema = source.get_schema(self.table)
        if spec is None:
            spec = {'type': self.transform_type}

        transform = Transform.from_spec(spec)
        transform.param.watch(self._update_spec, list(transform.param))

        for p in transform._field_params:
            transform.param[p].objects = list(schema)

        self.transforms.append(transform)
        if self._pipeline:
            self._pipeline.add_transform(transform)
        self.param.trigger('transforms')
        self.transform_items.append(pn.panel(transform))
        self.param.trigger('transform_items')
        self._update_spec()
        self.loading = False

    def _save(self):
        pass


class PipelinesEditor(WizardItem):
    """
    Declare the data pipelines for your dashboard by giving your pipeline
    a name and adding the source.
    """

    disabled = param.Boolean(default=True)

    pipeline_name = param.String(doc="Enter a name for the pipeline")

    pipelines = param.Dict(default={}, doc="The list of pipelines added to the dashboard.")

    sources = param.List(label='Select a source', doc="""
        Select from list of sources (if None are defined ensure you select at least one Source
        in the source gallery.""")

    source = param.String()

    tables = param.List(label='Select a table', doc="Selects the table driving the pipeline.")

    table = param.String()

    _template = """
    <span style="font-size: 2em">Pipeline Editor</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="display: flex;">
      <form role="form" style="flex: 30%; max-width: 450px; line-height: 2em;">
        <div style="display: grid;">
          <label for="pipeline-name-${id}"><b>{{ param.pipeline_name.label }}</b></label>
          <fast-text-field id="pipeline-name" placeholder="{{ param.pipeline_name.doc }}" value="${pipeline_name}">
          </fast-text-field>
        </div>
        <div style="display: grid;">
          <label for="sources-${id}"><b>{{ param.sources.label }}</b></label>
          <fast-select id="sources" style="max-width: 450px; min-width: 150px;" value="${source}">
          {% for src in sources %}
            <fast-option value="{{ src }}">{{ src.title() }}</fast-option>
          {% endfor %}
          </fast-select>
          <fast-tooltip anchor="sources-${id}">{{ param.sources.doc }}</fast-tooltip>
        </div>
        <div style="display: flex;">
          <div style="display: grid;">
            <label for="tables-${id}"><b>{{ param.tables.label }}</b></label>
            <fast-select id="tables" style="max-width: 450px; min-width: 150px;" value="${table}">
            {% for tbl in tables %}
              <fast-option value="{{ tbl|tojson }}">{{ tbl }}</fast-option>
            {% endfor %}
            </fast-select>
            <fast-tooltip anchor="tables-${id}">{{ param.tables.doc }}</fast-tooltip>
          </div>
          <fast-button id="submit" appearance="accent" style="margin-top: auto; margin-left: 1em; width: 20px;" onclick="${_add_pipeline}" disabled="${disabled}">
            <b style="font-size: 2em;">+</b>
          </fast-button>
        </div>
      </form>
      <div id="pipelines" style="flex: 70%; margin-left: 1em; margin-right: 2em;">
        {% for pipeline in pipelines.values() %}
        <div id="pipeline-container">${pipeline}</div>
        <fast-divider></faster-divider>
        {% endfor %}
      </div>
    </div>
    """

    _dom_events = {'pipeline-name': ['keyup']}

    def __init__(self, **params):
        super().__init__(**params)
        state.sources.param.watch(self._update_sources, 'items')
        self._watchers = {}
        self._update_sources()

    @param.depends('source', watch=True, on_init=True)
    def _update_tables(self):
        if self.source in lm_state.sources:
            self.tables = lm_state.sources[self.source].get_tables()
            if not self.table and self.tables:
                self.table = self.tables[0]

    def _update_sources(self, *events):
        for name, item in state.sources.items.items():
            if name not in self._watchers:
                self._watchers[name] = item.param.watch(self._update_sources, 'selected')
        self.sources = [source for source, item in state.sources.items.items() if item.selected]
        if not self.source and self.sources:
            self.source = self.sources[0]

    @param.depends('pipeline_name', 'table', watch=True)
    def _enable_add(self):
        self.disabled = not bool(self.pipeline_name and self.table)

    def _add_pipeline(self, event):
        self.spec[self.pipeline_name] = spec = {
            'name': self.pipeline_name,
            'source': self.source,
            'table': self.table
        }
        editor = PipelineEditor(
            name=self.pipeline_name, spec=spec, source=self.source,
            sizing_mode='stretch_width', table=self.table
        )
        self.pipelines[self.pipeline_name] = editor
        self.param.trigger('pipelines')
        self.pipeline_name = ''
        self.ready = True


class PipelineGalleryItem(GalleryItem):

    disabled = param.Boolean(default=False)

    editor = param.ClassSelector(class_=PipelineEditor, precedence=-1)

    thumbnail = param.Filename()

    def __init__(self, **params):
        if 'description' not in params:
            spec = params['editor'].spec
            params['description'] = f"Pipeline driven by {spec['table']} on {spec['source']} source."
        super().__init__(**params)
        self.view = pn.pane.PNG(self.thumbnail, height=200, max_width=300, align='center')
        self._modal_content = [self.editor]

    @param.depends('selected', watch=True, on_init=True)
    def _add_spec(self):
        pipelines = state.spec['pipelines']
        if self.selected:
            pipelines[self.name] = self.spec
            lm_state.pipelines[self.name] = Pipeline.from_spec(dict(self.spec, name=self.name))
        elif self.name in pipelines:
            del pipelines[self.name]
            del lm_state.pipelines[self.name]


class PipelineGallery(WizardItem, Gallery):
    """
    Select the pipelines to add to your dashboard application.
    """

    path = param.Foldername()

    pipelines = param.Dict(default={}, doc="The list of pipelines added to the dashboard.", precedence=-1)

    _template = """
    <span style="font-size: 1.5em">Pipelines</span>
    <fast-divider></fast-divider>
    <span style="font-size: 1.2em; font-weight: bold;">{{ __doc__ }}</p>
    <div id="items" style="margin: 1em 0; display: flex; flex-wrap: wrap; gap: 1em;">
      {% for item in items.values() %}
      <fast-card id="pipeline-container" style="width: 350px; height: 400px;">
        ${item}
      </fast-card>
      {% endfor %}
      <fast-card id="pipelines-container-new" style="height: 400px; width: 350px; padding: 1em;">
        <div style="display: grid;">
          <span style="font-size: 1.25em; font-weight: bold;">Add new pipeline</span>
          <i id="add-button" onclick="${_open_modal}" class="fa fa-plus" style="font-size: 14em; margin: 0.2em auto;" aria-hidden="true"></i>
        </div>
      </fast-card>
    </div>
    """

    _gallery_item = PipelineGalleryItem

    _editor_type = PipelineEditor

    def __init__(self, **params):
        super().__init__(**params)
        self._items = self.items
        self._editor = PipelinesEditor(spec={}, margin=10)
        self._save_button = pn.widgets.Button(name='Save pipelines')
        self._save_button.on_click(self._save_pipelines)
        self._modal_content = [self._editor, self._save_button]
        self._watchers = {}
        state.sources.param.watch(self._update_items, 'items')
        self._update_items()
        for name, item in self._items.items():
            self.pipelines[name] = item.editor

    @param.depends('spec', watch=True)
    def _update_params(self):
        for name, pipeline in self.spec.items():
            self.pipelines[name] = editor = PipelineEditor(name=name, spec=pipeline)
            self.items[name] = PipelineGalleryItem(
                name=name, spec=pipeline, selected=True, editor=editor,
                thumbnail=editor.thumbnail
            )
        self.param.trigger('pipelines')
        self.param.trigger('items')

    def _update_items(self, *events):
        for name, item in state.sources.items.items():
            if name not in self._watchers:
                self._watchers[name] = item.param.watch(self._update_items, 'selected')
        for name, item in self.items.items():
            if name not in self._items:
                self._items[name] = item
        self.items = {
            name: item for name, item in self._items.items()
            if (item.spec.get('source') in lm_state.sources or
                item.spec.get('pipeline') in lm_state.pipelines)
        }
        self.param.trigger('items')

    def _add_pipeline(self, event):
        state.modal[:] = self._modal_content
        self._editor._update_tables()
        state.template.open_modal()

    def _save_pipelines(self, event):
        for name, pipeline in self._editor.pipelines.items():
            self.spec[name] = pipeline.spec
            item = PipelineGalleryItem(
                name=name, spec=pipeline.spec, margin=0, selected=True,
                editor=pipeline, thumbnail=pipeline.thumbnail
            )
            self._items[name] = item
            self.items[name] = item
            self.pipelines[name] = pipeline
        self.param.trigger('items')
        self.param.trigger('pipelines')
        self._editor.pipelines = {}
        state.template.close_modal()
