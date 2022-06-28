from __future__ import annotations

from typing import Type, Union

import panel as pn
import param

from .filters import Filter, ParamFilter
from .sources import Source
from .state import state
from .transforms import Filter as FilterTransform, SQLTransform, Transform


class Pipeline(param.Parameterized):

    data = param.DataFrame(doc="The current data on this source.")

    schema = param.Dict(doc="The schema of the input data.")

    source = param.ClassSelector(
        class_=Source, doc="The Source this pipeline is fed by."
    )

    filters = param.List(
        item_type=Filter, doc="A list of Filters to apply to the source data."
    )

    sql_transforms = param.List(
        item_type=SQLTransform, doc="A list of SQLTransforms to apply to the source data."
    )

    transforms = param.List(
        item_type=Transform, doc="A list of Transforms to apply to the source data."
    )

    table = param.String(doc="The name of the table driving this pipeline.")

    def __init__(self, *, source, table, **params):
        params['schema'] = source.get_schema(table)
        super().__init__(source=source, table=table, **params)
        for transform in self.transforms+self.sql_transforms:
            for fp in transform._field_params:
                if isinstance(transform.param[fp], param.Selector):
                    transform.param[fp].objects = list(self.schema)

    @classmethod
    def from_spec(cls, spec, source=None, filters=None):
        params = dict(spec)

        # Resolve sources
        if 'source' in spec:
            source = spec['source']
            if isinstance(source, dict):
                source = Source.from_spec(source)
            elif isinstance(source, str):
                source = state.sources[source]
        else:
            params['source'] = source

        # Resolve schema
        params['filters'] = filters = list(filters) if filters else []
        filter_specs = spec.pop('filters', [])
        if filter_specs:
            table = spec.get('table')
            schema = source.get_schema(table)
        for filt_spec in filter_specs:
            filt_spec = dict(filt_spec, table=table)
            filt = Filter.from_spec(filt_spec, {table: schema}, None)
            filters.append(filt)

        # Resolve transforms
        transform_specs = spec.pop('transforms', [])
        params['transforms'] = [Transform.from_spec(tspec) for tspec in transform_specs]
        sql_transform_specs = spec.pop('sql_transforms', [])
        params['sql_transforms'] = [Transform.from_spec(tspec) for tspec in sql_transform_specs]

        return cls(**params)

    def _update_data(self, *events):
        query = {}
        if self.sql_transforms:
            if not self.source._supports_sql:
                raise ValueError(
                    'Can only use sql transforms source that support them. '
                    f'Found source typed {self.source.source_type!r} instead.'
                )
            query['sql_transforms'] = self.sql_transforms
        for filt in self.filters:
            filt_query = filt.query
            if (filt_query is not None and not getattr(filt, 'disabled', None) and
                (filt.table is None or filt.table == self.table)):
                query[filt.field] = filt_query
        data = self.source.get(self.table, **query)
        for transform in self.transforms:
            data = transform.apply(data)
        if len(data):
            data = FilterTransform.apply_to(data, conditions=list(query.items()))
        for filt in self.filters:
            if not isinstance(filt, ParamFilter):
                continue
            from holoviews import Dataset
            if filt.value is not None:
                ds = Dataset(data)
                data = ds.select(filt.value).data
        self.data = data

    def add_filter(self, filt: Union[Filter, Type[Filter]], field=None):
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
        self.transforms.append(transform)
        fields = list(self._schema)
        for fparam in transform._field_params:
            transform.param[fparam].objects = fields
            transform.param.update(**{fparam: fields})
        transform.param.watch(self._update_data, transform.controls)
        self._update_data()

    @property
    def control_panel(self):
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
