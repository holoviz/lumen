from __future__ import annotations

import os
import pathlib

from typing import ClassVar, Type

import panel as pn
import param  # type: ignore
import requests
import yaml

from panel.reactive import ReactiveHTML

from lumen.sources.base import Source
from lumen.state import state as lm_state
from lumen.util import catch_and_notify

from .base import WizardItem
from .fast import FastComponent
from .gallery import Editor, Gallery, GalleryItem
from .state import state

ASSETS_DIR = pathlib.Path(__file__).parent / 'assets'


class SourceEditor(FastComponent, Editor):

    cache_dir = param.String(label="Cache directory (optional)", precedence=1, doc="""
        Enter a relative cache directory.""")

    form = param.Parameter()

    preview = param.Parameter()

    source_type = param.String(default="")

    shared = param.Boolean(default=False, precedence=1, doc="""
        Whether the Source can be shared.""")

    sizing_mode = param.String(default='stretch_width')

    source = param.Parameter(precedence=-1)

    spec = param.Dict(precedence=-1)

    resize = param.Integer(default=0)

    margin = param.Integer(default=10)

    _preview_display = param.String(default='none')

    _scripts = {
        'resize': ['window.dispatchEvent(new Event("resize"))']
    }

    _template = """
    <span style="font-size: 2em"><b>{{ name }} - {{ source_type }}</b></span>
    <div id="preview-area" style="display: ${_preview_display}; margin-top: 2em;">
      <form id="form" role="form" style="flex: 30%; max-width: 250px; line-height: 2em; margin-top">${form}</form>
      <div id="preview" style="flex: auto; margin-left: 1em; overflow-y: auto;">${preview}</div>
    </div>
    """

    _default_thumbnail = ASSETS_DIR / 'source.png'

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
        spec = params.pop('spec', {})
        params.update(**{
            k: v for k, v in spec.items() if k in self.param and k not in params
        })
        self._source = None
        self._thumbnail = params.pop('thumbnail', None)
        super().__init__(spec=spec, **params)
        self.form = pn.Column(sizing_mode='stretch_width')
        theme = 'midnight' if getattr(pn.config, 'theme', 'default') == 'dark' else 'simple'
        self.preview = pn.widgets.Tabulator(
            sizing_mode='stretch_width', pagination='remote', page_size=12,
            theme=theme, height=400
        )
        self._select_table = pn.widgets.Select(
            name='Select table', margin=0, sizing_mode='stretch_width'
        )
        self._load_table = pn.widgets.Button(
            name='Load table', sizing_mode='stretch_width', margin=(15, 0)
        )
        self._load_table.on_click(self._load_table_data)
        self.form[:] = [self._select_table, self._load_table]

    @catch_and_notify
    def _load_table_data(self, event):
        if self._source is None:
            self._update_preview()
        self.preview.value = self._source.get(self._select_table.value)

    @property
    def thumbnail(self):
        if self._thumbnail:
            return self._thumbnail
        return self._default_thumbnail

    def _update_spec(self, *events):
        for event in events:
            self.spec[event.name] = event.new

    @param.depends('spec', watch=True)
    def _update_preview(self):
        if self._preview_display == 'none':
            return
        self._source = Source.from_spec(self.spec)
        self._select_table.options = self._source.get_tables()

    def _preview(self, event):
        self._preview_display = 'flex' if self._preview_display == 'none' else 'none'
        self._update_preview()
        self.resize += 1

    def _save(self):
        pass


class SourceGalleryItem(GalleryItem):

    editor = param.ClassSelector(class_=SourceEditor, precedence=-1)

    thumbnail = param.Filename()

    def __init__(self, **params):
        if 'description' not in params:
            params['description'] = ''
        super().__init__(**params)
        self.view = pn.pane.PNG(self.thumbnail, height=200, max_width=300, align='center')
        self._modal_content = [self.editor]

    @param.depends('selected', on_init=True, watch=True)
    def _add_spec(self):
        sources = state.spec['sources']
        if self.selected:
            spec = self.spec.copy()
            spec.pop('metadata', None)
            source = Source.from_spec(dict(spec, name=self.name))
            lm_state.sources[self.name] = source
            sources[self.name] = spec
        elif self.name in sources:
            del sources[self.name]
            del lm_state.sources[self.name]


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
      <fast-card id="source-container" style="width: 350px; height: 400px;">
        ${item}
      </fast-card>
    {% endfor %}
      <fast-card id="sources-container-new" style="height: 400px; width: 350px; padding: 1em;">
        <div style="display: grid;">
          <span style="font-size: 1.25em; font-weight: bold;">Add new source</span>
          <i id="add-button" onclick="${_open_modal}" class="fa fa-plus" style="font-size: 14em; margin: 0.2em auto;" aria-hidden="true"></i>
        </div>
      </fast-card>
    </div>
    """

    _gallery_item = SourceGalleryItem

    _editor_type: ClassVar[Type[SourceEditor]] = SourceEditor

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
            lm_state.sources[name] = Source.from_spec(dict(source.spec, name=name))
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
        <div style="flex: 70%; min-width: 600px; display: block">
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
          <fast-checkbox id="shared" value="${shared}"></fast-checkbox>
        </div>
      </div>
    </form>
    <div style="display: flex; justify-content:flex-end; margin-right: 2em;">
      <fast-button id="preview-button" onclick="${_preview}">Preview</fast-button>
    </div>
    <div id="preview-area" style="display: ${_preview_display}; margin-top: 2em;">
      <form id="form" role="form" style="flex: 30%; max-width: 250px; line-height: 2em;">${form}</form>
      <div id="preview" style="flex: auto; margin-left: 1em; overflow-y: auto;">${preview}</div>
    </div>
    """

    _dom_events = {'cache_dir': ['keyup'], 'uri': ['keyup']}

    _default_thumbnail = pathlib.Path(__file__).parent / 'assets' / 'intake.png'

    def __init__(self, **params):
        import lumen.sources.intake  # noqa
        params.pop('source_type', None)
        self.editor = pn.widgets.Ace(language='yaml', theme='dracula', margin=0, sizing_mode='stretch_width')
        self.upload = pn.widgets.FileInput(sizing_mode='stretch_width', margin=0)
        super().__init__(**params)

    @param.depends('upload.value', watch=True)
    def _upload_catalog(self):
        self.editor.value = self.upload.value.decode('utf-8')

    @param.depends('editor.value', watch=True)
    def _update_catalog(self):
        self.spec['catalog'] = yaml.safe_load(self.editor.value)

    @catch_and_notify
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

    username = param.String(doc="Enter a username")

    password = param.String(doc="Enter a password")

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
          <div style="display: grid;">
            <label for="username"><b>Username</b></label>
            <fast-text-field id="username" placeholder="Enter your username" value="${username}">
            </fast-text-field>
          </div>
          <div style="display: grid;">
            <label for="password"><b>Password</b></label>
            <fast-text-field id="password" type="password" placeholder="Enter password" value="${password}">
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
          <fast-checkbox id="shared" value="${shared}"></fast-checkbox>
        </div>
      </div>
    </form>
    <div style="display: flex; justify-content:flex-end; margin-right: 2em;">
      <fast-button id="preview-button" onclick="${_preview}">Preview</fast-button>
    </div>
    <div id="preview-area" style="display: ${_preview_display}; margin-top: 2em;">
      <form id="form" role="form" style="flex: 30%; max-width: 250px; line-height: 2em;">${form}</form>
      <div id="preview" style="flex: auto; margin-left: 1em; overflow-y: auto;">${preview}</div>
    </div>
    """

    _dom_events = {'uri': ['keyup'], 'username': ['keyup'], 'password': ['keyup']}

    _default_thumbnail = pathlib.Path(__file__).parent / 'assets' / 'intake.png'

    def __init__(self, **params):
        import lumen.sources.intake  # noqa
        super().__init__(**params)

    @param.depends('cert', 'load_schema', 'tls', 'uri', 'password', 'username', watch=True)
    def _update_spec(self):
        for p in ('cert', 'load_schema', 'tls', 'uri', 'password', 'username'):
            self.spec[p] = getattr(self, p)



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

    table_editors = param.List()

    kwargs = param.Dict(default={})

    _template = """
    <span style="font-size: 1.5em">{{ name }} - File Source</span>
    <p>{{ __doc__ }}</p>
    <fast-divider></fast-divider>
    <div id="tables">
      <fast-button id="add-table" onclick="${_add_table}" appearance="outline" style="float: right; margin-right: 2em;">
        <b>+</b>
      </fast-button>
      <span style="font-size: 1.2em; margin: 1em 0;"><b>Tables</b></span>
      ${table_editors}
    </div>
    <div style="display: flex; margin-top: 1em;">
      <div style="display: grid; margin-right: 1em;">
        <label for="cache_dir"><b>{{ param.cache_dir.label }}</b></label>
        <fast-text-field id="cache_dir" placeholder="{{ param.cache_dir.doc }}" value="${cache_dir}" style="min-width: 300px;">
        </fast-text-field>
      </div>
      <div style="display: grid;">
        <label for="shared"><b>{{ param.shared.label }}</b></label>
        <fast-checkbox id="shared" value="${shared}"></fast-checkbox>
      </div>
    </div>
    <div style="display: flex; justify-content:flex-end; margin-right: 2em;">
      <fast-button id="preview-button" onclick="${_preview}">Preview</fast-button>
    </div>
    <div id="preview-area" style="display: ${_preview_display}; margin-top: 2em;">
      <form id="form" role="form" style="flex: 30%; max-width: 300px; line-height: 2em;">${form}</form>
      <div id="preview" style="flex: auto; margin-left: 2em; overflow-y: auto;">${preview}</div>
    </div>
    """

    _dom_events = {'cache_dir': ['keyup']}

    def __init__(self, **params):
        super().__init__(**params)
        self._table_watchers = {}
        for name, table in self.spec.get('tables', {}).items():
            self._add_table(name=name, uri=table)

    @property
    def thumbnail(self):
        if self._thumbnail:
            return self._thumbnail
        assets = pathlib.Path(__file__).parent / 'assets'
        exts = {table.uri.split('.')[-1] for table in self.table_editors}
        if len(exts) == 1:
            filename = assets/ f'{list(exts)[0]}.png'
            if os.path.isfile(filename):
                return filename

    @param.depends('table_editors', watch=True)
    @catch_and_notify
    def _update_spec(self, *events):
        self.spec['tables'] = {t.name: t.uri for t in self.table_editors}
        self.param.trigger('spec')

    @catch_and_notify
    def _add_table(self, event=None, **kwargs):
        table = FileSourceTable(**kwargs)
        remove = table.param.watch(self._remove_table, 'remove')
        update = table.param.watch(self._update_spec, 'uri')
        self._table_watchers[table.name] = (remove, update)
        self.table_editors += [table]
        self.resize += 1

    @catch_and_notify
    def _remove_table(self, event):
        self.table_editors.remove(event.obj)
        watchers = self._table_watchers[event.obj.name]
        for w in watchers:
            event.obj.param.unwatch(w)
        self.param.trigger('table_editors')
        self.resize -= 1


class SourcesEditor(WizardItem):
    """
    Declare the data sources for your dashboard by giving your source
    a name, selecting the source type and adding the source.
    """

    disabled = param.Boolean(default=True)

    sources = param.Dict(default={}, doc="The list of sources added to the dashboard.")

    source_name = param.String(doc="Enter a name for the source")

    source_type = param.Selector(doc="Select the type of source")

    resize = param.Integer(default=0)

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
      <div id="sources" style="flex: 75%; margin-left: 1em; margin-right: 1em;">
        {% for source in sources.values() %}
        <div id="source-container">${source}</div>
        <fast-divider></faster-divider>
        {% endfor %}
      </div>
    </div>
    """

    _dom_events = {'source-name': ['keyup']}

    _scripts = {
        'resize': ['window.dispatchEvent(new Event("resize"))']
    }

    def __init__(self, **params):
        super().__init__(**params)
        sources = param.concrete_descendents(Source)
        self.param.source_type.objects = types = [
            source.source_type for source in sources.values()
            if source.source_type is not None
        ]+['intake', 'intake_dremio']
        if self.source_type is None and types:
            self.source_type = types[0]

    @param.depends('source_name', watch=True)
    def _enable_add(self):
        self.disabled = not bool(self.source_name)

    @catch_and_notify
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
        self.resize += 1
