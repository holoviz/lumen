import os
import pathlib

import param
import yaml

from panel.reactive import ReactiveHTML

from ..util import expand_spec
from .state import state


class GalleryItem(ReactiveHTML):

    description = param.String(default="", doc="""
        A description for the gallery component.""")

    spec = param.Dict(default={}, precedence=-1, doc="""
        The dictionary specification of the gallery item.""")

    selected = param.Boolean(default=False, doc="""
        Whether the item has been selected.""")

    view = param.Parameter(doc="""
        A Panel view of the contents of the gallery item.""")

    margin = param.Integer(default=0)

    thumbnail = param.Filename(precedence=-1)

    sizing_mode= param.String(default='stretch_both')

    __abstract = True

    _template = """
    <span style="font-size: 1.2em; font-weight: bold;">{{ name }}</p>
    <fast-switch id="selected" checked=${selected} style="float: right"></fast-switch>
    <div id="details" style="margin: 1em 0;">
      ${view}
    </div>
    <p style="height: 4em; max-width: 320px;">{{ description }}</p>
    <fast-button id="edit-button" style="width: 320px;" onclick="${_open_modal}">Edit</fast-button>
    """

    def _open_modal(self, event):
        if state.modal.objects == self._modal_content:
            state.template.open_modal()
            return
        state.modal.loading = True
        state.modal[:] = self._modal_content
        state.modal.loading = False
        state.template.open_modal()


class Gallery(ReactiveHTML):

    path = param.Foldername()

    items = param.Dict(default={})

    sizing_mode = param.String(default='stretch_width', readonly=True)

    hidden = param.Boolean(default=True)

    _editor_type = None

    _gallery_item = GalleryItem

    _glob_pattern = '*.y*ml'

    __abstract = True

    def __init__(self, **params):
        path = params.get('path', self.path)
        components = pathlib.Path(path).glob(self._glob_pattern)
        params['items'] = items = {}
        for component in components:
            with open(component, encoding='utf-8') as f:
                yaml_spec = f.read()
                spec = yaml.safe_load(expand_spec(yaml_spec))
            if not spec:
                continue
            metadata = spec.pop('metadata', {})
            if 'name' in metadata:
                name = metadata['name']
            else:
                name = '.'.join(component.name.split('.')[:-1])
            if 'thumbnail' in metadata:
                thumbnail = metadata['thumbnail']
                if not pathlib.Path(thumbnail).is_absolute():
                    thumbnail = path / pathlib.Path(thumbnail)
            else:
                thumbnail = '.'.join(str(component).split('.')[:-1]) + '.png'
            kwargs = {'name': name, 'spec': spec}
            if os.path.isfile(thumbnail):
                kwargs['thumbnail'] = thumbnail
            if self._editor_type:
                kwargs['editor'] = self._editor_type(**kwargs)
            if 'description' in metadata:
                kwargs['description'] = metadata['description']
            kwargs = self._preprocess_kwargs(kwargs)
            items[name] = item = self._gallery_item(**kwargs)
            item.param.watch(self._selected, ['selected'])
        super().__init__(**params)

    def _preprocess_kwargs(self, kwargs):
        return kwargs

    def _selected(self, event):
        """
        Called when a GalleryItem is selected
        """

    def _open_modal(self, event):
        if state.modal.objects == self._modal_content:
            state.template.open_modal()
            return
        state.modal.loading = True
        state.template.open_modal()
        state.modal[:] = self._modal_content
        state.modal.loading = False
