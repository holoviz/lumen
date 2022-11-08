import pathlib

import pandas as pd

from panel.widgets import Select

from lumen.filters import ConstantFilter
from lumen.pipeline import Pipeline
from lumen.sources.intake_sql import IntakeSQLSource
from lumen.state import state
from lumen.transforms import Columns
from lumen.transforms.sql import SQLColumns, SQLGroupBy


def test_pipeline_source_only(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    pipeline = Pipeline(source=source, table='test')

    pd.testing.assert_frame_equal(pipeline.data, mixed_df)

    source.tables = {'test': 'test2.csv'}
    pd.testing.assert_frame_equal(pipeline.data, mixed_df[::-1].reset_index(drop=True))

def test_pipeline_change_table(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    source.tables = {'test': 'test.csv', 'test2': 'test2.csv'}
    pipeline = Pipeline(source=source, table='test')

    pd.testing.assert_frame_equal(pipeline.data, mixed_df)

    pipeline.table = 'test2'
    pd.testing.assert_frame_equal(pipeline.data, mixed_df[::-1].reset_index(drop=True))

def test_pipeline_with_filter(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    cfilter = ConstantFilter(field='A', value=(1, 2))
    pipeline = Pipeline(source=source, filters=[cfilter], table='test')

    expected = mixed_df.iloc[1:3]
    pd.testing.assert_frame_equal(pipeline.data, expected)

    # Update
    cfilter.value = (0, 2)
    expected = mixed_df.iloc[0:3]
    pd.testing.assert_frame_equal(pipeline.data, expected)

def test_pipeline_manual_with_filter(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    cfilter = ConstantFilter(field='A', value=(1, 2))
    pipeline = Pipeline(source=source, filters=[cfilter], table='test', auto_update=False)

    expected = mixed_df.iloc[1:3]
    pd.testing.assert_frame_equal(pipeline.data, expected)

    # Update filter
    cfilter.value = (0, 2)
    pd.testing.assert_frame_equal(pipeline.data, expected)

    # Trigger pipeline update
    pipeline.param.trigger('update')
    expected = mixed_df.iloc[0:3]
    pd.testing.assert_frame_equal(pipeline.data, expected)

def test_pipeline_with_transform(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))

    transform = Columns(columns=['A', 'B'])
    pipeline = Pipeline(source=source, transforms=[transform], table='test')

    expected = mixed_df[['A', 'B']]
    pd.testing.assert_frame_equal(pipeline.data, expected)

    # Update
    transform.columns = ['B', 'C']
    expected = mixed_df[['B', 'C']]
    pd.testing.assert_frame_equal(pipeline.data, expected)

def test_pipeline_manual_with_transform(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))

    transform = Columns(columns=['A', 'B'])
    pipeline = Pipeline(source=source, transforms=[transform], table='test', auto_update=False)

    expected = mixed_df[['A', 'B']]
    pd.testing.assert_frame_equal(pipeline.data, expected)

    # Update transform
    transform.columns = ['B', 'C']
    pd.testing.assert_frame_equal(pipeline.data, expected)

    # Trigger update
    pipeline.param.trigger('update')
    expected = mixed_df[['B', 'C']]
    pd.testing.assert_frame_equal(pipeline.data, expected)

def test_pipeline_with_sql_transform(mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = IntakeSQLSource(
        uri=str(root / 'catalog.yml'), root=str(root)
    )

    transform = SQLColumns(columns=['A', 'B'])
    pipeline = Pipeline(source=source, table='test_sql', sql_transforms=[transform])

    df = mixed_df[['A', 'B']]
    pd.testing.assert_frame_equal(pipeline.data, df)

    # Update
    transform.columns = ['B', 'C']
    df = mixed_df[['B', 'C']]
    pd.testing.assert_frame_equal(pipeline.data, df)

def test_pipeline_chained_with_filter(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))

    cfilter1 = ConstantFilter(field='A', value=(1, 3))
    cfilter2 = ConstantFilter(field='B', value=1.0)
    pipeline1 = Pipeline(source=source, filters=[cfilter1], table='test')
    pipeline2 = pipeline1.chain(filters=[cfilter2])

    expected = mixed_df.iloc[[1, 3]]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    # Update
    cfilter1.value = (2, 3)
    expected = mixed_df.iloc[[3]]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    cfilter2.value = 0.0
    expected = mixed_df.iloc[[2]]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

def test_pipeline_manual_chained_with_filter(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))

    cfilter1 = ConstantFilter(field='A', value=(1, 3))
    cfilter2 = ConstantFilter(field='B', value=1.0)
    pipeline1 = Pipeline(source=source, filters=[cfilter1], table='test', auto_update=False)
    pipeline2 = pipeline1.chain(filters=[cfilter2], _chain_update=True)

    expected = mixed_df.iloc[[1, 3]]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    # Update filter
    cfilter1.value = (2, 3)
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    # Trigger update
    pipeline1.param.trigger('update')
    expected = mixed_df.iloc[[3]]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    # Update chained filter
    cfilter2.value = 0.0
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    # Trigger update
    pipeline2.param.trigger('update')
    expected = mixed_df.iloc[[2]]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

def test_pipeline_chained_with_transform(make_filesource, mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))

    cfilter = ConstantFilter(field='A', value=(1, 2))
    pipeline1 = Pipeline(source=source, filters=[cfilter], table='test')
    transform = Columns(columns=['A', 'B'])
    pipeline2 = pipeline1.chain(transforms=[transform])

    expected = mixed_df.iloc[1:3][['A', 'B']]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    # Update
    cfilter.value = (2, 3)
    expected = mixed_df.iloc[2:4][['A', 'B']]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    transform.columns = ['B', 'C']
    expected = mixed_df.iloc[2:4][['B', 'C']]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

def test_pipeline_collapsed_with_sql_transform(mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = IntakeSQLSource(
        uri=str(root / 'catalog.yml'), root=str(root)
    )

    cfilter = ConstantFilter(field='A', value=(1, 2))
    pipeline1 = Pipeline(source=source, filters=[cfilter], table='test_sql')
    transform = SQLColumns(columns=['A', 'B'])
    pipeline2 = pipeline1.chain(sql_transforms=[transform])

    expected = mixed_df.iloc[1:3][['A', 'B']].reset_index(drop=True)
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    # Update
    cfilter.value = (2, 3)
    expected = mixed_df.iloc[2:4][['A', 'B']].reset_index(drop=True)
    pd.testing.assert_frame_equal(pipeline2.data, expected)

    transform.columns = ['B', 'C']
    expected = mixed_df.iloc[2:4][['B', 'C']].reset_index(drop=True)
    pd.testing.assert_frame_equal(pipeline2.data, expected)

def test_pipeline_chained_with_sql_transform(mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = IntakeSQLSource(
        uri=str(root / 'catalog.yml'), root=str(root)
    )

    transform1 = SQLColumns(columns=['A', 'B'])
    pipeline1 = Pipeline(source=source, sql_transforms=[transform1], table='test_sql')
    cfilter1 = ConstantFilter(field='A', value=(1, 2))
    pipeline2 = pipeline1.chain(filters=[cfilter1])
    cfilter2 = ConstantFilter(field='B', value=(0, 2))
    pipeline3 = pipeline2.chain(filters=[cfilter2])

    expected = mixed_df.iloc[1:3][['A', 'B']]
    pd.testing.assert_frame_equal(pipeline3.data, expected)

    # Update
    cfilter1.value = (1, 3)
    expected = mixed_df.iloc[1:4][['A', 'B']]
    pd.testing.assert_frame_equal(pipeline3.data, expected)

    transform1.columns = ['B', 'C']
    expected = mixed_df[['B', 'C']]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

def test_pipeline_chained_manual_update_with_sql_transform(mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = IntakeSQLSource(
        uri=str(root / 'catalog.yml'), root=str(root)
    )

    transform1 = SQLColumns(columns=['A', 'B'])
    pipeline1 = Pipeline(
        source=source, sql_transforms=[transform1], table='test_sql',
        auto_update=False, name='Pipeline 1'
    )
    cfilter1 = ConstantFilter(field='A', value=(1, 2))
    pipeline2 = pipeline1.chain(filters=[cfilter1], auto_update=False, name='Pipeline 2')
    cfilter2 = ConstantFilter(field='B', value=(0, 2))
    pipeline3 = pipeline2.chain(filters=[cfilter2], auto_update=False, name='Pipeline 3')

    expected = mixed_df.iloc[1:3][['A', 'B']]
    pd.testing.assert_frame_equal(pipeline3.data, expected)

    # Update
    cfilter1.value = (1, 3)
    pd.testing.assert_frame_equal(pipeline3.data, expected)
    pipeline3.param.trigger('update')
    expected = mixed_df.iloc[1:4][['A', 'B']]
    pd.testing.assert_frame_equal(pipeline3.data, expected)

    transform1.columns = ['B', 'C']
    pd.testing.assert_frame_equal(pipeline3.data, expected)
    pipeline3.param.trigger('update')
    expected = mixed_df[['B', 'C']]
    pd.testing.assert_frame_equal(pipeline3.data, expected)

    cfilter2.value = (0, 0)
    pd.testing.assert_frame_equal(pipeline3.data, expected)
    pipeline3.param.trigger('update')
    expected = mixed_df.iloc[::2][['B', 'C']]
    pd.testing.assert_frame_equal(pipeline3.data, expected)

def test_not_removing_type(penguins_file):
    spec = {
        "source": {"type": "file", "tables": {"penguins": penguins_file}},
    }
    spec_org = spec.copy()
    Pipeline.from_spec(spec)
    Pipeline.from_spec(spec)
    assert spec == spec_org

def test_load_chained_pipeline(penguins_file):
    spec = {
        "pipelines": {
            "penguins": {
                "source": {
                    "type": "file",
                    "tables": {"penguins": penguins_file}
                }
            },
            "penguins_chained": {
                "pipeline": "penguins"
            }
        }
    }
    state.spec = spec
    pipelines = state.load_pipelines()
    assert 'penguins' in pipelines
    assert 'penguins_chained' in pipelines
    assert pipelines['penguins_chained'].pipeline is pipelines['penguins']

def test_pipeline_with_sql_transform_nested_widget_vars(mixed_df):
    root = pathlib.Path(__file__).parent / 'sources'
    source = IntakeSQLSource(
        uri=str(root / 'catalog.yml'), root=str(root)
    )

    sel = Select(options=['A', 'B'], value='A')
    transform = SQLGroupBy(aggregates={'AVG': sel}, by=['C'])
    pipeline = Pipeline(source=source, table='test_sql', sql_transforms=[transform])

    df = mixed_df[['C', 'A']]
    pd.testing.assert_frame_equal(pipeline.data, df)

    # Update
    sel.value = 'B'
    df = mixed_df[['C', 'B']]
    pd.testing.assert_frame_equal(pipeline.data, df)
