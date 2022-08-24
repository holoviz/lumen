import pathlib

import panel as pn
import param

from lumen.filters import Filter
from lumen.pipeline import Pipeline
from lumen.sources import Source
from lumen.transforms import Transform

from .base import WizardItem
from .fast import FastComponent
from .gallery import Gallery, GalleryItem
from .state import state

ASSETS_DIR = pathlib.Path(__file__).parent / 'assets'


class PipelineEditor(FastComponent):

    filter_type = param.Selector()

    transform_type = param.Selector()

    filter_items = param.List()

    filters = param.List(precedence=-1)

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
    <div id="filters" style="display: flex;">
       {% for fitem in filter_items %}
       <div id="filter-item" style="margin: 0.5em;">
         ${fitem}
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
    <div id="transforms" style="display: flex;">
       {% for titem in transform_items %}
       <div id="transform-item" style="margin: 0.5em;">
         ${titem}
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
        super().__init__(**{k: v for k, v in params.items() if k in self.param})
        theme = 'midnight' #if getattr(pn.config, 'theme', 'default') == 'dark' else 'simple'
        self.preview = pn.widgets.Tabulator(
            sizing_mode='stretch_width', pagination='remote', page_size=12,
            theme=theme, height=400
        )
        self._pipeline = None
        filters = param.concrete_descendents(Filter).values()
        self.param.filter_type.objects = [f.filter_type for f in filters if f.filter_type]
        transforms = param.concrete_descendents(Transform).values()
        self.param.transform_type.objects = [t.transform_type for t in transforms if t.transform_type]

    @param.depends('_preview_display', watch=True)
    def _update_displayed(self):
        if self._preview_display != 'none' and self._pipeline is None:
            self._pipeline = Pipeline.from_spec({
                k: v for k, v in self.spec.items() if k not in ('filters', 'transforms')
            })
            for filt in self.filters:
                self._pipeline.add_filter(filt)
            for trans in self.transforms:
                self._pipeline.add_transform(trans)
            self._pipeline.param.watch(self._update_preview, ['data'])
            self.preview.value = self._pipeline.data

    def _update_preview(self, event):
        print('Update')
        self.preview.value = event.new

    def _update_spec(self, *events):
        self.spec['filters'] = []
        for filt in self.filters:
            filt_spec = dict(type=filt.filter_type, **filt.param.values())
            del filt_spec['name']
            del filt_spec['schema']
            self.spec['filters'].append(filt_spec)
        self.spec['transforms'] = []
        for transform in self.transforms:
            transform_spec = dict(type=transform.transform_type, **transform.param.values())
            del transform_spec['name']
            self.spec['transforms'].append(transform_spec)

    def _preview(self, event):
        self._preview_display = 'flex' if self._preview_display == 'none' else 'none'
        self.resize += 1

    def _add_filter(self, event):
        self.loading = True
        source = Source.from_spec(state.spec['sources'][self.source])
        schema = source.get_schema()
        table = list(schema)[0]
        filt = Filter.from_spec({
            'type': self.filter_type,
            'table': table,
            'field': list(schema[table])[0]
        }, schema)
        filt.param.watch(self._update_spec, list(filt.param))
        if self._pipeline:
            self._pipeline.add_filter(filt)

        self.filters.append(filt)
        self.param.trigger('filters')
        params = [p for p in filt.param if p not in ('name', 'schema', 'table')]
        self.filter_items.append(pn.Param(filt, parameters=params))
        self.param.trigger('filter_items')
        self._update_spec()
        self.loading = False

    def _add_transform(self, event):
        self.loading = True
        source = Source.from_spec(state.spec['sources'][self.source])
        schema = source.get_schema()

        transform = Transform.from_spec({'type': self.transform_type})
        transform.param.watch(self._update_spec, list(transform.param))

        fields = [f for t in schema.values() for f in t]
        for p in transform._field_params:
            transform.param[p].objects = fields

        self.transforms.append(transform)
        if self._pipeline:
            print('Add transform')
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

    sources = param.List(label='Select a source')

    source = param.String()

    tables = param.List(label='Select a table')

    table = param.String()

    _template = """
    <span style="font-size: 2em">Pipeline Editor</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="display: flex;">
      <form role="form" style="flex: 30%; max-width: 300px; line-height: 2em;">
        <div style="display: grid;">
          <label for="pipeline-name-${id}"><b>{{ param.pipeline_name.label }}</b></label>
          <fast-text-field id="pipeline-name" placeholder="{{ param.pipeline_name.doc }}" value="${pipeline_name}">
          </fast-text-field>
        </div>
        <div style="display: grid;">
          <label for="sources-${id}"><b>{{ param.sources.label }}</b></label>
          <fast-select id="sources" style="max-width: 300px; min-width: 150px;" value="${source}">
          {% for src in sources %}
            <fast-option value="{{ src }}">{{ src.title() }}</fast-option>
          {% endfor %}
          </fast-select>
          <fast-tooltip anchor="sources-${id}">{{ param.sources.doc }}</fast-tooltip>
        </div>
        <div style="display: flex;">
          <div style="display: grid;">
            <label for="tables-${id}"><b>{{ param.tables.label }}</b></label>
            <fast-select id="tables" style="max-width: 300px; min-width: 150px;" value="${table}">
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
        self._source = None
        state.sources.param.watch(self._update_sources, 'sources')

    @param.depends('source', watch=True)
    def _update_tables(self):
        source = state.sources.sources[self.source]
        spec = dict(source.spec, cache_dir=None)
        spec.pop('filters', None)
        self._source = Source.from_spec(spec)
        self.tables = self._source.get_tables()

    def _update_sources(self, event):
        self.sources = list(event.new)
        if not self.source and self.sources:
            self.source = self.sources[0]

    @param.depends('pipeline_name', watch=True)
    def _enable_add(self):
        self.disabled = not bool(self.pipeline_name)

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

    _template = """
    <span style="font-size: 1.2em; font-weight: bold;">{{ name }}</p>
    <fast-switch id="selected" checked=${selected} style="float: right;"></fast-switch>
    <div id="details" style="margin: 1em 0; max-width: 320px;">
      ${view}
    </div>
    <p style="height: 4em; max-width: 320px;">{{ description }}</p>
    <fast-button id="edit-button" style="width: 320px; margin: 1em 0;" onclick="${_open_modal}">Edit</fast-button>
    """

    def __init__(self, **params):
        if 'description' not in params:
            spec = params['editor'].spec
            params['description'] = f"Pipeline driven by {spec['table']} on {spec['source']} source."
        super().__init__(**params)
        self.view = pn.pane.PNG(self.thumbnail, height=200, max_width=300, align='center')
        self._modal_content = [self.editor]

    @param.depends('selected', watch=True)
    def _add_spec(self):
        pipelines = state.spec['pipelines']
        if self.selected:
            pipelines[self.name] = self.spec
        elif self.name in pipelines:
            del pipelines[self.name]


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
        for name, item in self.items.items():
            self.pipelines[name] = item.editor
        self._editor = PipelinesEditor(spec={}, margin=10)
        self._save_button = pn.widgets.Button(name='Save pipelines')
        self._save_button.on_click(self._save_pipelines)
        self._modal_content = [self._editor, self._save_button]

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

    def _add_pipeline(self, event):
        state.modal[:] = self._modal_content
        state.template.open_modal()

    def _save_pipelines(self, event):
        for name, pipeline in self._editor.pipelines.items():
            self.spec[name] = pipeline.spec
            item = PipelineGalleryItem(
                name=name, spec=pipeline.spec, margin=0, selected=True,
                editor=pipeline, thumbnail=pipeline.thumbnail
            )
            self.items[name] = item
            self.pipelines[name] = pipeline
        self.param.trigger('items')
        self.param.trigger('pipelines')
        self._editor.pipelines = {}
        state.template.close_modal()
