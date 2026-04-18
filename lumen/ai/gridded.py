from __future__ import annotations

from typing import Any

from ..pipeline import Pipeline
from ..sources.xarray_sql import XArraySQLSource
from ..util import check_xarray_available


def detect_gridded(pipeline: Pipeline) -> dict[str, Any] | None:
    """Return gridded metadata if pipeline.source is an XArraySQLSource, else None.

    Returns a dict with keys ``dims``, ``coords``, ``data_vars``, ``source_type``,
    or ``None`` for tabular pipelines.
    """
    source = pipeline.source
    if not isinstance(source, XArraySQLSource):
        return None
    if not check_xarray_available():
        return None
    ds = source.dataset
    return {
        'source_type': 'xarray',
        'dims': list(ds.dims),
        'coords': {k: list(ds.coords[k].shape) for k in ds.coords},
        'data_vars': list(ds.data_vars),
    }
