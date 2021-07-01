import os
import pathlib
import yaml

import param

from panel.reactive import ReactiveHTML

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

    thumbnail = param.Filename()

    __abstract = True

    def __init__(self, **params):
        spec = params.get('spec', {})
        if 'metadata' in spec:
            params.update(spec['metadata'])
        super().__init__(**params)

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

    _gallery_item = GalleryItem
    _editor_type = None 

    _glob_pattern = '*.y*ml'

    __abstract = True
    
    def __init__(self, **params):
        path = params.get('path', self.path)
        components = pathlib.Path(path).glob(self._glob_pattern)
        params['items'] = items = {}
        for source in components:
            with open(source, encoding='utf-8') as f:
                spec = yaml.safe_load(f.read())
            if not spec:
                continue
            name = '.'.join(source.name.split('.')[:-1])
            thumbnail = '.'.join(str(source).split('.')[:-1]) + '.png'
            kwargs = {'name': name, 'spec': spec}
            if os.path.isfile(thumbnail):
                kwargs['thumbnail'] = thumbnail
            if self._editor_type:
                kwargs['editor'] = self._editor_type(**dict(spec, **kwargs))
            items[name] = item = self._gallery_item(**kwargs)
            item.param.watch(self._selected, ['selected'])
        super().__init__(**params)

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
