"""
The Download classes makes it possible to download data.
"""
from __future__ import annotations

from io import BytesIO, StringIO
from typing import (
    IO, Any, ClassVar, Dict, List, Tuple,
)

import param  # type: ignore

from panel.viewable import Viewer

from .base import MultiTypeComponent
from .panel import DownloadButton
from .util import slugify

DOWNLOAD_FORMATS = ['csv', 'xlsx', 'json', 'parquet']


class Download(MultiTypeComponent, Viewer):
    """
    `Download` is a plugin component for `View` components that adds a download button.
    """

    color = param.Color(default='grey', allow_None=True, doc="""
      The color of the download button.""")

    filename = param.String(default=None, doc="""
      The filename of the downloaded table.
      File extension is added automatic based on the format.
      If filename is not defined, it will be the name of the orignal table of the view.""")

    format = param.ObjectSelector(default=None, objects=DOWNLOAD_FORMATS, doc="""
      The format to download the data in.""")

    hide = param.Boolean(default=False, doc="""
      Whether the download button hides when not in focus.""")

    index = param.Boolean(default=True, doc="""
      Whether the downloaded table has an index.""")

    kwargs = param.Dict(default={}, doc="""
      Keyword arguments passed to the serialization function, e.g.
      data.to_csv(file_obj, **kwargs).""")

    size = param.Integer(default=18, doc="""
      The size of the download button.""")

    view = param.Parameter(doc="Holds the current view.")

    # Specification configuration
    _internal_params: ClassVar[List[str]] = ['view', 'name']
    _required_keys: ClassVar[List[str | Tuple[str, ...]]] = ['format']
    _validate_params: ClassVar[bool] = True

    download_type: ClassVar[str | None] = 'default'

    @classmethod
    def validate(cls, spec: Dict[str, Any] | str, context: Dict[str, Any] | None = None):
        if isinstance(spec, str):
            spec = {'format': spec}
        if 'type' not in spec:
            spec['type'] = 'default'

        return super().validate(spec, context)

    def __bool__(self) -> bool:
        return self.format is not None

    def _table_data(self) -> IO:
        io: IO[Any]
        if self.format in ('json', 'csv'):
            io = StringIO()
        else:
            io = BytesIO()
        data = self.view.get_data()
        if self.format == 'csv':
            data.to_csv(io, index=self.index, **self.kwargs)
        elif self.format == 'json':
            data.to_json(io, index=self.index, **self.kwargs)
        elif self.format == 'xlsx':
            data.to_excel(io, index=self.index, **self.kwargs)
        elif self.format == 'parquet':
            data.to_parquet(io, index=self.index, **self.kwargs)
        io.seek(0)
        return io

    def __panel__(self) -> DownloadButton:
        filename = self.filename or slugify(self.view.pipeline.table)
        filename = f'{filename}.{self.format}'
        return DownloadButton(
            callback=self._table_data, filename=filename, color=self.color,
            size=18, hide=self.hide
        )

    @classmethod
    def from_spec(cls, spec: Dict[str, Any] | str) -> MultiTypeComponent | None:
        spec = dict(spec)
        cls_type = spec.pop("type", "default")
        return cls._get_type(cls_type)(**spec)

    def to_spec(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        spec = super().to_spec(context)
        if len(spec) == 1 and "type" in spec:
            return {}
        return spec
