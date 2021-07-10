import os
import pathlib
import yaml

import panel as pn
import param
import requests

from panel.reactive import ReactiveHTML

from lumen.sources import Source
from .base import WizardItem
from .fast import FastComponent, FastTabs
from .gallery import Gallery, GalleryItem
from .state import state


ASSETS_DIR = pathlib.Path(__file__).parent / 'assets'



class SourceEditor(FastComponent):

    cache_dir = param.String(label="Cache directory (optional)", precedence=1, doc="""
        Enter a relative cache directory.""")

    preview = param.Parameter(default=FastTabs(sizing_mode='stretch_width'))

    source_type = param.String(default="")

    shared = param.Boolean(default=False, precedence=1, doc="""
        Whether the Source can be shared.""")

    sizing_mode = param.String(default='stretch_width')

    source = param.Parameter(precedence=-1)

    spec = param.Dict(precedence=-1)

    resize = param.Integer(default=0)

    margin = param.Integer(default=5)

    _scripts = {
        'resize': ['window.dispatchEvent(new Event("resize"))']
    }

    _template = """
    <span style="font-size: 2em"><b>{{ name }} - {{ source_type }}</b></span>
    <div id="preview">${preview}</div>
    """

    thumbnail = ASSETS_DIR / 'source.png'

    def __new__(cls, **params):
        if cls is not SourceEditor:
            return super().__new__(cls)
        editors = param.concrete_descendents(cls)
        source_type = params.get('spec', {}).get('type')
        for editor in editors.values():
            if editor.source_type == source_type:
                return super().__new__(editor)
        return super().__new__(cls)

    def __init__(self, **params):
        super().__init__(**{k: v for k, v in params.items() if k in self.param})

    def _update_spec(self, *events):
        for event in events:
            self.spec[event.name] = event.new

    def _preview(self, event):
        source = Source.from_spec(self.spec)
        tables = source.get_tables()
        def load_table(event):
            table = tables[event.new]
            tabs[event.new][1].object = source.get(table)
        tabs = []
        for i, table in enumerate(tables):
            tabs.append((table, pn.widgets.Tabulator(
                None if i > 0 else source.get(table), sizing_mode='stretch_width',
                pagination='remote', page_size=8, theme='midnight'
            )))
        self.preview[:] = tabs
        self.preview.param.watch(load_table, 'active')
        self.resize += 1

    def _save(self):
        pass


class SourceGalleryItem(GalleryItem):

    editor = param.ClassSelector(class_=SourceEditor, precedence=-1)

    thumbnail = param.Filename()

    _template = """
    <span style="font-size: 1.2em; font-weight: bold;">{{ name }}</p>
    <fast-switch id="selected" checked=${selected} style="float: right;"></fast-switch>
    <div id="details" style="margin: 1em 0; max-width: 320px;">
      ${view}
    </div>
    <p style="height: 4em; max-width: 320px;">{{ description }}</p>
    <fast-button id="edit-button" style="width: 320px;" onclick="${_open_modal}">Edit</fast-button>
    """

    def __init__(self, **params):
        if 'description' not in params:
            params['description'] = ''
        super().__init__(**params)
        self.view = pn.pane.PNG(self.thumbnail, height=200, max_width=300, align='center')
        self._modal_content = [self.editor]

    @param.depends('selected', watch=True)
    def _add_spec(self):
        sources = state.spec['sources']
        if self.selected:
            sources[self.name] = self.spec
        elif self.name in sources:
            del sources[self.name]


class SourceGallery(WizardItem, Gallery):
    """
    Select the data sources to add to your dashboard specification.
    """

    path = param.Foldername()

    sources = param.Dict(default={}, doc="The list of sources added to the dashboard.", precedence=-1)

    _template = """
    <span style="font-size: 1.5em">Sources</span>
    <fast-divider></fast-divider>
    <span style="font-size: 1.2em; font-weight: bold;">{{ __doc__ }}</p>
    <div id="items" style="margin: 1em 0; display: flex; flex-wrap: wrap; gap: 1em;">
    {% for item in items.values() %}
      <fast-card id="source-container" style="width: 350px; height: 380px;">
        ${item}
      </fast-card>
    {% endfor %}
      <fast-card id="sources-container-new" style="height: 380px; width: 350px; padding: 1em;">
        <div style="display: grid;">
          <span style="font-size: 1.25em; font-weight: bold;">Add new source</span>
          <i id="add-button" onclick="${_open_modal}" class="fa fa-plus" style="font-size: 14em; margin: 0.2em auto;" aria-hidden="true"></i>
        </div>
      </fast-card>
    </div>
    """

    _gallery_item = SourceGalleryItem
    
    _editor_type = SourceEditor

    def __init__(self, **params):
        super().__init__(**params)
        for name, item in self.items.items():
            self.sources[name] = item.editor
        self._editor = SourcesEditor(spec={}, margin=10)
        self._save_button = pn.widgets.Button(name='Save sources')
        self._save_button.on_click(self._save_sources)
        self._modal_content = [self._editor, self._save_button]

    @param.depends('spec', watch=True)
    def _update_params(self):
        for name, source in self.spec.items():
            self.sources[name] = editor = SourceEditor(name=name, spec=source)
            self.items[name] = SourceGalleryItem(
                name=name, spec=source, selected=True, editor=editor,
                thumbnail=editor.thumbnail
            )
        self.param.trigger('sources')
        self.param.trigger('items')

    def _add_source(self, event):
        state.modal[:] = self._modal_content
        state.template.open_modal()

    def _save_sources(self, event):
        for name, source in self._editor.sources.items():
            path = pathlib.Path(self.path) / f'{name}.yaml'
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml.dump(source.spec))
            self.spec[name] = source.spec
            item = SourceGalleryItem(
                name=name, spec=source.spec, margin=0, selected=True,
                editor=source, thumbnail=source.thumbnail
            )
            self.items[name] = item
            self.sources[name] = source
        self.param.trigger('items')
        self.param.trigger('sources')
        self._editor.sources = {}
        state.template.close_modal()



class IntakeSourceEditor(SourceEditor):
    """
    Declare the Intake catalog either by entering the URI of the catalog file, uploading a file or manually entering the catalog file.
    """

    uri = param.String(precedence=1)

    source_type = param.String(default='intake', readonly=True)

    editor = param.Parameter(precedence=-1)

    upload = param.Parameter(precedence=-1)

    display = param.String(default="block")
    
    _template = """
    <span style="font-size: 1.5em">{{ name }} - Intake Source</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <form>
      <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 25%; min-width: 200px; margin-right: 1em;">
          <div style="display: grid;">
            <label for="URI"><b>URI</b></label>
            <fast-text-field id="uri" placeholder="Enter a URI" value="${uri}">
            </fast-text-field>
          </div>
          <div id="upload" style="display: grid; margin-top: 1em;">${upload}</div>
        </div>
        <div style="flex: 70%; min-width: 600px; display: ${display}">
          <label for="catalog"><b>Catalog</b></label>
          <div id="catalog">${editor}</div>
        </div>
      </div>
      <div style="display: flex; margin-top: 1em;">
        <div style="display: grid; margin-right: 1em;">
          <label for="cache_dir"><b>{{ param.cache_dir.label }}</b></label>
          <fast-text-field id="cache_dir" placeholder="{{ param.cache_dir.doc }}" value="${cache_dir}" style="min-width: 300px;">
          </fast-text-field>
        </div>
        <div style="display: grid;">
          <label for="shared"><b>{{ param.shared.label }}</b></label>
          <fast-checkbox value="${shared}"></fast-checkbox>
        </div>
      </div>
    </form>
    <fast-button id="preview-button" onclick="${_preview}" style="position: absolute; right: 5px; margin-top: 1.5em; z-index: 100;">
      Preview
    </fast-button>
    <fast-divider></fast-divider>
    <div id="preview">${preview}</div>
    """

    _dom_events = {'cache_dir': ['keyup'], 'uri': ['keyup']}

    def __init__(self, **params):
        import lumen.sources.intake # noqa
        params.pop('source_type', None)
        self.editor = pn.widgets.Ace(language='yaml', theme='dracula', margin=0, sizing_mode='stretch_width')
        self.upload = pn.widgets.FileInput(sizing_mode='stretch_width', margin=0)
        super().__init__(**params)

    @property
    def thumbnail(self):
        return pathlib.Path(__file__).parent / 'assets' / 'intake.png'

    @param.depends('upload.value', watch=True)
    def _upload_catalog(self):
        self.editor.value = self.upload.value.decode('utf-8')

    @param.depends('editor.value', watch=True)
    def _update_catalog(self):
        self.spec['catalog'] = yaml.safe_load(self.editor.value)

    @param.depends('uri', watch=True)
    def _load_file(self):
        uri = os.path.expanduser(self.uri)
        if os.path.isfile(uri):
            with open(uri) as f:
                self.editor.value = f.read()
        else:
            self.editor.value = requests.get(self.uri).content


class IntakeDremioSourceEditor(SourceEditor):
    """
    Provide a Dremio URI.
    """

    cert = param.String(default=None)

    load_schema = param.Boolean(default=False)

    tls = param.Boolean(doc="Enable TLS")

    uri = param.String(doc="Enter a URI")

    source_type = param.String(default='intake_dremio', readonly=True)

    _template = """
    <span style="font-size: 1.5em">{{ name }} - Intake Dremio Source</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <form>
      <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 25%; min-width: 200px; margin-right: 1em;">
          <div style="display: grid;">
            <label for="URI"><b>URI</b></label>
            <fast-text-field id="uri" placeholder="Enter a URI" value="${uri}">
            </fast-text-field>
          </div>
          <fast-checkbox id="load_schema" checked="${load_schema}">Load schema</fast-checkbox>
          <fast-checkbox id="tls" checked="${tls}">Enable TLS</fast-checkbox>
          <div style="display: grid;">
            <label for="cert"><b>Certificate</b></label>
            <fast-text-field id="cert" disabled=${tls} placeholder="Enter path to a certificate" value="${cert}">
            </fast-text-field>
          </div>
        </div>
      </div>
      <div style="display: flex; margin-top: 1em;">
        <div style="display: grid; margin-right: 1em;">
          <label for="cache_dir"><b>{{ param.cache_dir.label }}</b></label>
          <fast-text-field id="cache_dir" placeholder="{{ param.cache_dir.doc }}" value="${cache_dir}" style="min-width: 300px;">
          </fast-text-field>
        </div>
        <div style="display: grid;">
          <label for="shared"><b>{{ param.shared.label }}</b></label>
          <fast-checkbox value="${shared}"></fast-checkbox>
        </div>
      </div>
    </form>
    <fast-button id="preview-button" onclick="${_preview}" style="position: absolute; right: 5px; margin-top: 1.5em; z-index: 100;">
      Preview
    </fast-button>
    <fast-divider></fast-divider>
    <div id="preview">${preview}</div>
    """

    def __init__(self, **params):
        import lumen.sources.intake # noqa
        super().__init__(**params)

    @param.depends('cert', 'load_schema', 'tls', 'uri', watch=True)
    def _update_spec(self, *events):
        for event in events:
            self.spec[event.name] = event.new

    @property
    def thumbnail(self):
        return pathlib.Path(__file__).parent / 'assets' / 'intake.png'


class FileSourceTable(ReactiveHTML):

    uri = param.String(doc="Enter a URI")

    margin = param.Integer(default=0)

    remove = param.Boolean(default=False, precedence=-1)

    sizing_mode = param.String(default='stretch_width')

    _template = """
    <form style="display: flex; flex-wrap: wrap; margin-top: 0.5em;">
      <div style="flex: 25%; min-width: 150px; display: grid; margin-right: 1em;">
        <label for="name"><b>Table Name</b></label>
        <fast-text-field id="name" placeholder="Enter a name" value="${name}"></fast-text-field>
        </div>
      <div style="flex: 60%; min-width: 300px; display: grid; margin-right: 1em;">
        <label for="uri"><b>URI</b></label>
        <fast-text-field id="uri" placeholder="{{ param.uri.doc }}" value="${uri}"></fast-text-field>
      </div>
      <fast-button style="width: 20px; margin-top: auto;" id="remove-source" onclick="${_remove}" appearance="accent">
        <b>－</b>
      </fast-button>
    </form>
    """

    _dom_events = {'uri': ['keyup'], 'name': ['keyup']}

    def _remove(self, event):
        self.remove = True


class FileSourceEditor(SourceEditor):
    """
    Declare a list of tables by providing a name and a URI to a local or remote file for each.
    """

    source_type = param.String(default='file', readonly=True)

    tables = param.List()

    kwargs = param.Dict(default={})

    _template = """
    <span style="font-size: 1.5em">{{ name }} - File Source</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div id="tables">
      <fast-button id="add-table" onclick="${_add_table}" appearance="outline" style="float: right">
        <b>+</b>
      </fast-button>
      <span style="font-size: 1.2em; margin: 1em 0;"><b>Tables</b></span>
      ${tables}
    </div>
    <div style="display: flex; margin-top: 1em;">
      <div style="display: grid; margin-right: 1em;">
        <label for="cache_dir"><b>{{ param.cache_dir.label }}</b></label>
        <fast-text-field id="cache_dir" placeholder="{{ param.cache_dir.doc }}" value="${cache_dir}" style="min-width: 300px;">
        </fast-text-field>
      </div>
      <div style="display: grid;">
        <label for="shared"><b>{{ param.shared.label }}</b></label>
        <fast-checkbox value="${shared}"></fast-checkbox>
      </div>
    </div>
    <fast-button id="preview-button" onclick="${_preview}" style="position: absolute; right: 5px; margin-top: 1.5em; z-index: 100;">
      Preview
    </fast-button>
    <fast-divider></fast-divider>
    <div id="preview">${preview}</div>
    """

    _dom_events = {'cache_dir': ['keyup']}

    def __init__(self, **params):
        if 'tables' in params:
            tables = []
            for name, table in params['tables'].items():
                tables.append(FileSourceTable(name=name, uri=table))
            params['tables'] = tables
        params.pop('source_type', None)
        super().__init__(**params)

    @property
    def thumbnail(self):
        assets = pathlib.Path(__file__).parent / 'assets'
        exts = {table.uri.split('.')[-1] for table in self.tables}
        if len(exts) == 1:
            filename = assets/ f'{list(exts)[0]}.png'
            if os.path.isfile(filename):
                return filename
    
    def _add_table(self, event=None):
        table = FileSourceTable()
        table.param.watch(self._remove_table, 'remove')
        self.tables += [table]

    def _remove_table(self, event):
        self.tables.remove(event.obj)
        self.param.trigger('tables')



class SourcesEditor(WizardItem):
    """
    Declare the data sources for your dashboard by giving your source
    a name, selecting the source type and adding the source.
    """

    disabled = param.Boolean(default=True)

    sources = param.Dict(default={}, doc="The list of sources added to the dashboard.")

    source_name = param.String(doc="Enter a name for the source")

    source_type = param.Selector(doc="Select the type of source")

    _template = """
    <span style="font-size: 2em">Source Editor</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div style="display: flex;">
      <form role="form" style="flex: 20%; max-width: 250px; line-height: 2em;">
        <div style="display: grid;">
          <label for="source-name-${id}"><b>{{ param.source_name.label }}</b></label>
          <fast-text-field id="source-name" placeholder="{{ param.source_name.doc }}" value="${source_name}">
          </fast-text-field>
        </div>
        <div style="display: flex;">
          <div style="display: grid; flex: auto;">
            <label for="type-${id}">
              <b>{{ param.source_type.label }}</b>
            </label>
            <fast-select id="source-select" style="min-width: 150px;" value="${source_type}">
              {% for stype in param.source_type.objects %}
              <fast-option id="source-option-{{ loop.index0 }}" value="{{ stype }}">{{ stype.title() }}</fast-option>
              {% endfor %}
            </fast-select>
            <fast-tooltip anchor="type-${id}">{{ param.source_type.doc }}</fast-tooltip>
          </div>
          <fast-button id="submit" appearance="accent" style="margin-top: auto; margin-left: 1em; width: 20px;" onclick="${_add_source}" disabled="${disabled}">
            <b style="font-size: 2em;">+</b>
          </fast-button>
        </div>
      </form>
      <div id="sources" style="flex: 75%; margin-left: 1em;">
        {% for source in sources.values() %}
        <div id="source-container">${source}</div>
        <fast-divider></faster-divider>
        {% endfor %}
      </div>
    </div>
    """

    _dom_events = {'source-name': ['keyup']}

    def __init__(self, **params):
        super().__init__(**params)
        sources = param.concrete_descendents(Source)
        self.param.source_type.objects = types = [
            source.source_type for source in sources.values()
        ]+['intake', 'intake_dremio']
        if self.source_type is None and types:
            self.source_type = types[0]

    @param.depends('source_name', watch=True)
    def _enable_add(self):
        self.disabled = not bool(self.source_name)

    def _add_source(self, event):
        self.spec[self.source_name] = spec = {'type': self.source_type}
        editor = SourceEditor(
            type=self.source_type, name=self.source_name, spec=spec,
            sizing_mode='stretch_width'
        )
        self.sources[self.source_name] = editor
        self.param.trigger('sources')
        self.source_name = ''
        self.ready = True
