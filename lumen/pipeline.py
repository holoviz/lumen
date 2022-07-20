from __future__ import annotations

from typing import (
    Any, Dict, List, Optional, Type, Union,
)

import panel as pn
import param

from .filters import Filter, ParamFilter
from .sources import Source
from .state import state
from .transforms import Filter as FilterTransform, SQLTransform, Transform


class Pipeline(param.Parameterized):
    """
    A Pipeline represents a data Source along with any number of
    Filters and general purpose Transforms. A pipeline can be used
    to drive one or more visual outputs or leveraged as a standalone
    component to encapsulate multiple data processing steps.
    """

    data = param.DataFrame(doc="The current data on this source.")

    schema = param.Dict(doc="The schema of the input data.")

    source = param.ClassSelector(
        class_=Source,
        doc="The Source this pipeline is fed by."
    )

    pipeline = param.ClassSelector(
        class_=None,
        doc="Optionally a pipeline may be chained to another pipeline."
    )

    filters = param.List(
        item_type=Filter,
        doc="A list of Filters to apply to the source data."
    )

    sql_transforms = param.List(
        item_type=SQLTransform,
        doc="A list of SQLTransforms to apply to the source data."
    )

    transforms = param.List(
        item_type=Transform,
        doc="A list of Transforms to apply to the source data."
    )

    table = param.String(
        doc="The name of the table driving this pipeline."
    )

    def __init__(self, *, source, table, **params):
        if 'schema' not in params:
            params['schema'] = source.get_schema(table)
        super().__init__(source=source, table=table, **params)
        self._init_callbacks()

    def _init_callbacks(self):
        for filt in self.filters:
            filt.param.watch(self._update_data, ['value'])
        refs = set()
        for transform in self.transforms+self.sql_transforms:
            if transform.controls:
                transform.param.watch(self._update_data, transform.controls)
            refs |= {
                var.split('.')[1] for var in transform.refs
                if var.startswith('$variables.')
            }
            for fp in transform._field_params:
                if isinstance(transform.param[fp], param.Selector):
                    transform.param[fp].objects = list(self.schema)
        if refs:
            state.variables.param.watch(self._update_data, list(refs))
        if self.pipeline is not None:
            self.pipeline.param.watch(self._update_data, 'data')

    @property
    def refs(self):
        refs = self.source.refs.copy()
        for filt in self.filters:
            for ref in filt.refs:
                if ref not in refs:
                    refs.append(ref)
        for transform in self.sql_transforms+self.transforms:
            for ref in transform.refs:
                if ref not in refs:
                    refs.append(ref)
        return refs

    def _update_data(self, *events: param.Event):
        query = {}

        # Compute Filter query
        for filt in self.filters:
            filt_query = filt.query
            if (filt_query is not None and not getattr(filt, 'disabled', None) and
                (filt.table is None or filt.table == self.table)):
                query[filt.field] = filt_query

        if self.pipeline is None:
            # Compute SQL transform expression
            if self.sql_transforms:
                if not self.source._supports_sql:
                    raise ValueError(
                        'Can only use sql transforms source that support them. '
                        f'Found source typed {self.source.source_type!r} instead.'
                    )
                query['sql_transforms'] = self.sql_transforms

            data = self.source.get(self.table, **query)
        else:
            data = FilterTransform.apply_to(
                self.pipeline.data, conditions=list(query.items())
            )

        # Apply ParamFilter
        for filt in self.filters:
            if not isinstance(filt, ParamFilter):
                continue
            from holoviews import Dataset
            if filt.value is not None:
                ds = Dataset(data)
                data = ds.select(filt.value).data

        # Apply transforms
        for transform in self.transforms:
            data = transform.apply(data)

        self.data = data

    @classmethod
    def from_spec(
        cls, spec: Dict[str, Any], source: Optional[Source] = None,
        source_filters: Optional[List[Filter]] = None
    ):
        params = dict(spec)

        # Resolve source
        if 'source' in spec:
            source = spec['source']
            if isinstance(source, dict):
                source = Source.from_spec(source)
            elif isinstance(source, str):
                if source in state.sources:
                    source = state.sources[source]
                else:
                    source = state.load_source(source, state.spec['sources'][source])
        params['source'] = source

        # Resolve filters
        params['filters'] = filters = []
        filter_specs = spec.pop('filters', {})
        if filter_specs:
            table = spec.get('table')
            schema = source.get_schema(table)
        for fname, filt_spec in filter_specs.items():
            filt_spec = dict(filt_spec, table=table, name=fname)
            filt = Filter.from_spec(filt_spec, {table: schema}, source_filters)
            filters.append(filt)

        # Resolve transforms
        transform_specs = spec.pop('transforms', [])
        params['transforms'] = [Transform.from_spec(tspec) for tspec in transform_specs]
        sql_transform_specs = spec.pop('sql_transforms', [])
        params['sql_transforms'] = [Transform.from_spec(tspec) for tspec in sql_transform_specs]
        return cls(**params)

    def add_filter(self, filt: Union[Filter, Type[Filter]], field: Optional[str] = None):
        """
        Add a filter to the pipeline.

        Arguments
        ---------
        filt: Filter | Type[Filter]
           The filter instance or filter type to add.
        field: str | None
           The field to filter on (required to instantiate Filter type).
        """
        if not isinstance(filt, Filter):
            tspec = f'{filt.__module__}.{filt.__name__}'
            filt = Filter.from_spec(
                {'type': tspec, 'field': field, 'table': self.table},
                {self.table: self._schema}
            )
        self.filters.append(filt)
        filt.param.watch(self._update_data, ['value'])
        self._update_data()

    def add_transform(self, transform: Transform):
        """
        Add a (SQL)Transform to the pipeline.

        Arguments
        ---------
        filt: Transform
           The Transform instance to add.
        """
        self.transforms.append(transform)
        fields = list(self._schema)
        for fparam in transform._field_params:
            transform.param[fparam].objects = fields
            transform.param.update(**{fparam: fields})
        transform.param.watch(self._update_data, transform.controls)
        self._update_data()

    def chain(self, filters=None, transforms=None):
        params = {
            'filters': filters or [],
            'transforms': transforms or [],
            'sql_transforms': [],
            'pipeline': self.pipeline
        }
        return self.clone(**params)

    def clone(self, **params) -> Pipeline:
        """
        Create a new instance of the pipeline with optionally overridden parameter values.
        """
        return type(self)(**dict({p: v for p, v in self.param.values().items()
                                  if p != 'name'}, **params))

    @property
    def control_panel(self) -> pn.Column:
        col = pn.Column(sizing_mode='stretch_width')
        if self.filters:
            col.append('<div style="font-size: 1.5em; font-weight: bold;">Filters</div>')
        for f in self.filters:
            w = f.panel
            if w is not None:
                col.append(w)
        transforms = (self.transforms+self.sql_transforms)
        if transforms:
            col.append('<div style="font-size: 1.5em; font-weight: bold;">Transforms</div>')
        for t in transforms:
            w = t.control_panel
            if w is not None:
                col.append(w)
        return col


Pipeline.param.pipeline.class_ = Pipeline
