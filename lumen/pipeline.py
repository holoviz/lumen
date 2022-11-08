from __future__ import annotations

import sys

from itertools import product
from typing import (
    Any, Dict, List, Optional, Type, Union,
)

import panel as pn
import param
import tqdm

from panel.widgets import Widget

from .base import Component
from .filters import Filter, ParamFilter, WidgetFilter
from .sources import Source
from .state import state
from .transforms import Filter as FilterTransform, SQLTransform, Transform
from .util import catch_and_notify, get_dataframe_schema
from .validation import ValidationError, match_suggestion_message


class DataFrame(param.DataFrame):
    """
    DataFrame parameter that resolves data on access.
    """

    def __get__(self, obj, objtype):
        if (obj is not None and obj.__dict__.get(self._internal_name) is None) or (obj._stale and obj.auto_update):
            obj._update_data(force=True)
        return super().__get__(obj, objtype)


def expand_queries(values, groups=('filters', 'variables')):
    spec_groups = []
    for group in groups:
        if group in values:
            var_names, var_values = zip(*values[group].items())
            variable_space = (dict(zip(var_names, vs)) for vs in product(*var_values))
        else:
            variable_space = [{}]
        spec_groups.append(list(variable_space))
    return [dict(zip(groups, group)) for group in product(*spec_groups)]


class Pipeline(Component):
    """
    `Pipeline` encapsulates filters and transformations applied to a `Source` table.

    A `Pipeline` ingests data from a `Source` table or another
    `Pipeline` applying the declared `Filter` and `(SQL)Transform`
    definitions. It can be used to drive one or more visual outputs or
    leveraged as a standalone component to encapsulate multiple data
    processing steps.
    """

    data = DataFrame(doc="The current data on this source.")

    schema = param.Dict(doc="The schema of the input data.")

    auto_update = param.Boolean(default=True, constant=True, doc="""
        Whether changes in filters, transforms and references automatically
        trigger updates in the data or whether an update has to be triggered
        manually using the update event or the update button in the UI."""
    )

    source = param.ClassSelector(class_=Source, constant=True, doc="""
        The Source this pipeline is fed by."""
    )

    pipeline = param.ClassSelector(class_=None, doc="""
        Optionally a pipeline may be chained to another pipeline."""
    )

    filters = param.List(item_type=Filter, doc="""
        A list of Filters to apply to the source data."""
    )

    sql_transforms = param.List(item_type=SQLTransform, doc="""
        A list of SQLTransforms to apply to the source data."""
    )

    transforms = param.List(item_type=Transform, doc="""
        A list of Transforms to apply to the source data."""
    )

    table = param.String(doc="""
        The name of the table driving this pipeline."""
    )

    update = param.Event(label='Apply update', doc="""
        Update event trigger (if manual update is set)."""
    )

    _stale = param.Boolean(default=False, precedence=-1, doc="""
        Whether the pipeline is stale."""
    )

    _internal_params = ['data', 'name', 'schema']
    _required_fields = [('source', 'pipeline')]

    def __init__(self, *, source, table, **params):
        if 'schema' not in params:
            params['schema'] = source.get_schema(table)
        super().__init__(source=source, table=table, **params)
        self._update_widget = pn.Param(self.param['update'], widgets={'update': {'button_type': 'success'}})[0]
        self._init_callbacks()

    @param.depends('_stale', watch=True)
    def _handle_stale(self):
        self._update_widget.button_type = 'warning' if self._stale else 'success'

    def _init_callbacks(self):
        self.param.watch(self._update_data, ['filters', 'sql_transforms', 'transforms', 'table', 'update'])
        self.source.param.watch(self._update_data, self.source._reload_params)
        for filt in self.filters:
            filt.param.watch(self._update_data, ['value'])
        for transform in self.transforms+self.sql_transforms:
            transform.param.watch(self._update_data, list(transform.param))
            for fp in transform._field_params:
                if isinstance(transform.param[fp], param.Selector):
                    transform.param[fp].objects = list(self.schema)
        if self.pipeline is not None:
            self.pipeline.param.watch(self._update_data, 'data')

    def _update_refs(self, *events):
        self._update_data()

    def to_spec(self, context=None):
        """
        Exports the full specification to reconstruct this component.

        Parameters
        ----------
        context: Dict[str, Any]
          Context contains the specification of all previously
          serialized components, e.g. to allow resolving of
          references.

        Returns
        -------
        Declarative specification of this component.
        """
        spec = super().to_spec(context=context)
        if 'pipeline' in spec and 'source' in spec:
            del spec['source']
        if context is None:
            return spec
        if 'pipelines' not in context:
            context['pipelines'] = {}
        for type_name in ('pipeline', 'source'):
            if type_name not in spec:
                continue
            obj = spec.pop(type_name)
            plural = f'{type_name}s'
            obj_name = getattr(self, type_name).name
            if plural not in context:
                context[plural] = {}
            context[plural][obj_name] = obj
            spec[type_name] = obj_name
            break
        context['pipelines'][self.name] = spec
        return spec

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

    def _compute_data(self):
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
                    raise ValidationError(
                        'Can only use sql transforms source that support them. '
                        f'Found source typed {self.source.source_type!r} instead.'
                    )
                query['sql_transforms'] = self.sql_transforms
            data = self.source.get(self.table, **query)
        else:
            pipelines = []
            current = self
            while current.pipeline is not None:
                pipelines.append(current.pipeline)
                current = current.pipeline
            if self.pipeline.data is None or any(p._stale for p in pipelines):
                self.pipeline._update_data(force=True)
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
        return data

    @catch_and_notify
    def _update_data(self, *events: param.Event, force: bool = False):
        if self._update_widget.loading:
            return
        if not force and not self.auto_update and not any(e.name == 'update' for e in events):
            self._stale = True
            return

        self._update_widget.loading = True
        try:
            self.data = self._compute_data()
        except Exception as e:
            raise e
        else:
            self._stale = False
        finally:
            self._update_widget.loading = False

    @classmethod
    def _validate_source(cls, source_spec, spec, context):
        if isinstance(source_spec, str):
            if source_spec not in context['sources']:
                msg = f'Pipeline specified non-existent source {source_spec!r}.'
                msg = match_suggestion_message(source_spec, list(context['sources']), msg)
                raise ValidationError(msg, spec, source_spec)
            return source_spec
        return Source.validate(source_spec, context)

    @classmethod
    def _validate_pipeline(cls, pipeline_spec, spec, context):
        if isinstance(pipeline_spec, str):
            if pipeline_spec not in context['pipelines']:
                msg = f'Pipeline specified non-existent pipeline {pipeline_spec!r}.'
                msg = match_suggestion_message(pipeline_spec, list(context['pipelines']), msg)
                raise ValidationError(msg, spec, pipeline_spec)
            return pipeline_spec
        return Pipeline.validate(pipeline_spec, context)

    @classmethod
    def _validate_filters(cls, filter_specs, spec, context):
        for filter_spec in (filter_specs if isinstance(filter_specs, list) else filter_specs.values()):
            if not isinstance(filter_spec, str):
                continue
            elif not isinstance(spec['source'], str):
                raise ValidationError(
                    'Pipeline may only reference filters by name if the Source has been '
                    'defined by reference. Please simply move the filter definition from '
                    'the Source to the Pipeline.', spec, filter_spec
                )
            source = context['sources'][spec['source']]
            if 'filters' not in source:
                raise ValidationError(
                    f'Pipeline could not resolve {filter_spec!r} filter on {spec["source"]} source, '
                    'the source does not define any filters. ', spec, filter_spec
                )
            elif filter_spec not in source['filters']:
                msg = f'Pipeline could not resolve {filter_spec!r} filter on {spec["source"]} source.'
                msg = match_suggestion_message(filter_spec, list(source['filters']), msg)
                raise ValidationError(msg, spec, filter_spec)
        return cls._validate_dict_or_list_subtypes('filters', Filter, filter_specs, spec, context)

    @classmethod
    def validate(cls, spec, context=None):
        if isinstance(spec, str):
            if spec not in context['pipelines']:
                msg = f'Referenced non-existent pipeline {spec!r}.'
                msg = match_suggestion_message(spec, list(context['pipelines']), msg)
                raise ValidationError(msg, spec, spec)
            return spec
        return super().validate(spec, context)

    @classmethod
    def from_spec(
        cls, spec: Dict[str, Any] | str, source: Optional[Source] = None,
        source_filters: Optional[List[Filter]] = None
    ):
        if isinstance(spec, str):
            return state.pipelines[spec]

        spec = spec.copy()
        if source is not None:
            spec['source'] = source
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
        elif 'pipeline' in spec:
            pipeline = spec['pipeline']
            if isinstance(pipeline, dict):
                pipeline = Pipeline.from_spec(pipeline)
            elif isinstance(pipeline, str) and pipeline in state.pipelines:
                pipeline = state.pipelines[pipeline]
            else:
                raise ValidationError('Pipeline {pipeline!r} could not be resolved.', spec)
            params['pipeline'] = pipeline
            params['source'] = source = pipeline.source
            params['table'] = pipeline.table

        # Validate table
        table = params.get('table')
        tables = source.get_tables()
        if table is None:
            if len(tables) > 1:
                raise ValidationError(
                    "The Pipeline specification does not contain a table and the "
                    "supplied Source has multiple tables. Please specify one of the "
                    f"following tables: {tables} in the pipeline specification.",
                    spec
                )
            params['table'] = table = tables[0]
        elif table not in tables:
            if hasattr(source, '_get_source'):
                # Certain sources perform fuzzy matching so we use the
                # internal API to see if the table has a match
                source._get_source(table)
            else:
                raise ValidationError(
                    "The Pipeline specification references a table that is not "
                    "available on the specified source. ", spec
                )

        # Resolve filters
        params['filters'] = filters = []
        filter_specs = spec.pop('filters', {})
        if filter_specs:
            params['schema'] = schema = source.get_schema(table)
        for filt_spec in (filter_specs.items() if isinstance(filter_specs, dict) else filter_specs):
            if isinstance(filt_spec, tuple):
                filt_spec = dict(filt_spec[1], table=table, name=filt_spec[0])
            else:
                filt_spec = dict(filt_spec, table=table)
            filt = Filter.from_spec(filt_spec, {table: schema}, source_filters)
            filters.append(filt)

        # Resolve transforms
        transform_specs = spec.pop('transforms', [])
        params['transforms'] = [Transform.from_spec(tspec) for tspec in transform_specs]
        sql_transform_specs = spec.pop('sql_transforms', [])
        params['sql_transforms'] = [Transform.from_spec(tspec) for tspec in sql_transform_specs]

        # Instantiate and populate pre-cache
        precache = params.pop('precache', None)
        obj = cls(**params)
        if precache:
            obj.precache(precache)
        return obj

    def add_filter(self, filt: Union[Filter, Type[Filter], Widget], field: Optional[str] = None, **kwargs):
        """
        Add a filter to the pipeline.

        Arguments
        ---------
        filt: Filter | Type[Filter]
           The filter instance or filter type to add.
        field: str | None
           The field to filter on (required to instantiate Filter type).
        """
        if isinstance(filt, str):
            filt = Filter._get_type(filt)
        elif isinstance(filt, Widget):
            filt = WidgetFilter(
                widget=filt, field=field, table=self.table,
                schema={self.table: self.schema}
            )
        if not isinstance(filt, Filter):
            tspec = f'{filt.__module__}.{filt.__name__}'
            filt = Filter.from_spec(
                dict({'type': tspec, 'field': field, 'table': self.table}, **kwargs),
                {self.table: self.schema}
            )
        self.filters.append(filt)
        filt.param.watch(self._update_data, ['value'])
        self._stale = True

    def add_transform(self, transform: Transform, **kwargs):
        """
        Add a (SQL)Transform to the pipeline.

        Arguments
        ---------
        filt: Transform
           The Transform instance to add.
        """
        if isinstance(transform, str):
            transform = Transform._get_type(transform)(**kwargs)
        if isinstance(transform, SQLTransform):
            self.sql_transforms.append(transform)
        else:
            self.transforms.append(transform)
        fields = list(self.schema)
        for fparam in transform._field_params:
            transform.param[fparam].objects = fields
            transform.param.update(**{fparam: kwargs.get(fparam, fields)})
        transform.param.watch(self._update_data, list(transform.param))
        self._stale = True

    def chain(
        self,
        filters: Optional[List[Filter]]=None,
        transforms: Optional[List[Transform]] = None,
        sql_transforms: Optional[List[Transform]] = None,
        **kwargs
    ):
        """
        Chains additional filtering, transform and sql_transform operations
        on an existing pipeline. Note that if one or more sql_transforms
        are provided the pipeline is cloned rather than applying the
        operations on top of the existing pipeline.

        Arguments
        ---------
        filters: List[Filter] | None
          Additional filters to apply on top of existing pipeline.
        transforms: List[Transform] | None
          Additional transforms to apply on top of existing pipeline.
        sql_transforms: List[SQLTransform] | None
          Additional filters to apply on top of existing pipeline.

        Returns
        -------
        Pipeline
        """
        if sql_transforms:
            params = {
                'filters': self.filters + (filters or []),
                'transforms': self.transforms + (transforms or []),
                'sql_transforms': self.sql_transforms + (sql_transforms or []),
                'data': None
            }
        else:
            params = {
                'filters': filters or [],
                'transforms': transforms or [],
                'sql_transforms': [],
                'pipeline': self,
                'schema': get_dataframe_schema(self.data)['items']['properties'],
                'data': None
            }
        chain_update = kwargs.pop('_chain_update', False)
        params.update(kwargs)
        new = self.clone(**params)
        if chain_update and not new.auto_update:
            self.param.watch(lambda e: new.param.trigger('update'), 'update')
        return new

    def clone(self, **params) -> Pipeline:
        """
        Create a new instance of the pipeline with optionally overridden parameter values.
        """
        return type(self)(**dict({p: v for p, v in self.param.values().items()
                                  if p != 'name'}, **params))

    def precache(self, queries: Dict[str, Dict[str, []]] | List[Dict[str, Dict[str, Any]]]) -> None:
        """
        Populates the cache of the Source with the provided queries.

        Queries can be provided in two formats:

          - A dictionary containing 'filters' and 'variables'
            dictionaries each containing lists of values to compute
            a cross-product for, e.g.

              {
                'filters': {
                  <filter>': ['a', 'b', 'c', ...],
                  ...
                },
                'variables': {
                  <variable>: [0, 2, 4, ...],
                  ...
                }
              }
          - A list containing dictionaries of explicit values
            for each filter and variables.

              [{
                 'filters': {<filter>: 'a'},
                 'variables': {<variable>: 0}
               },
               {
                 'filters': {<filter>: 'a'},
                 'variables': {<variable>: 1}
               },
               ...
              ]
        """
        if not self.source.cache_dir or not self.precache:
            return

        if not isinstance(queries, list):
            queries = expand_queries(queries)

        old_auto = self.auto_update
        self.auto_update = False
        restore = None
        for qspec in tqdm.tqdm(queries, leave=True, file=sys.stdout):
            try:
                previous = self._set_spec(qspec)
                if restore is None:
                    restore = previous
                self._update_data(force=True)
            except Exception as e:
                raise e
        if restore:
            self._set_spec(restore)
            self._update_data(force=True)
        self.auto_update = old_auto

    def _set_spec(self, spec):
        previous = {'variables': {}, 'filters': {}}
        filters = self.traverse('filters')
        for var_name, var_val in spec.get('variables', {}).items():
            variable = state.variables._vars[var_name]
            previous['variables'][var_name] = variable.value
            variable.value = var_val
        filt_spec = spec.get('filters', {})
        for filt in filters:
            if filt.field in filt_spec:
                previous['filters'][filt.field] = filt.value
                filt.value = filt_spec[filt.field]
        return previous

    def traverse(self, type) -> List[Transform] | List[Filter]:
        """
        Returns all Filter or Transform objects in a potentially chained
        pipeline.
        """
        if type not in ('filters', 'transforms', 'sql_transforms'):
            raise TypeError(f'May only traverse Pipeline filters, transforms or sql_transforms, not {type}')
        objects = []
        pipeline = self
        while pipeline is not None:
            objects.extend(getattr(self, type))
            pipeline = pipeline.pipeline
        return objects

    @property
    def control_panel(self) -> pn.Column:
        col = pn.Column()
        filters = [filt.panel for filt in self.traverse('filters')]
        if any(filters):
            col.append('<div style="font-size: 1.5em; font-weight: bold;">Filters</div>')
        col.extend([filt for filt in filters if filt is not None])
        transforms = self.traverse('transforms')+self.traverse('sql_transforms')
        controls = [t.control_panel for t in transforms if t.controls]
        if controls:
            col.append('<div style="font-size: 1.5em; font-weight: bold;">Transforms</div>')
        col.extend(controls)
        variables, variable_controls = [], []
        for transform in transforms:
            for ref in transform.refs:
                if ref.startswith('$variable'):
                    variable = state.variables._vars[ref.split('$variables.')[1]]
                    if variable not in variables:
                        variables.append(variable)
        for variable in variables:
            vpanel = variable.panel
            if vpanel is not None:
                variable_controls.append(vpanel)
        if variable_controls:
            col.append('<div style="font-size: 1.5em; font-weight: bold;">Variables</div>')
        col.extend(variable_controls)
        if not self.auto_update:
            col.append(self._update_widget)
        return col


Pipeline.param.pipeline.class_ = Pipeline
