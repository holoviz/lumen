import pathlib

import pandas as pd

from lumen.filters import ConstantFilter
from lumen.pipeline import Pipeline
from lumen.sources.intake_sql import IntakeSQLSource
from lumen.transforms import Columns
from lumen.transforms.sql import SQLColumns


def test_pipeline_source_only(make_filesource):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    pipeline = Pipeline(source=source, table='test')
    pipeline._update_data()
    expected = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(pipeline.data, expected)

def test_pipeline_with_filter(make_filesource):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    pipeline = Pipeline(source=source, filters=[ConstantFilter(field='A', value=(1, 2))], table='test')
    pipeline._update_data()
    expected = pd._testing.makeMixedDataFrame().iloc[1:3]
    pd.testing.assert_frame_equal(pipeline.data, expected)

def test_pipeline_with_transform(make_filesource):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    pipeline = Pipeline(source=source, transforms=[Columns(columns=['A', 'B'])], table='test')
    pipeline._update_data()
    expected = pd._testing.makeMixedDataFrame()[['A', 'B']]
    pd.testing.assert_frame_equal(pipeline.data, expected)

def test_pipeline_with_sql_transform():
    root = pathlib.Path(__file__).parent / 'sources'
    source = IntakeSQLSource(
        uri=str(root / 'catalog.yml'), root=str(root)
    )
    pipeline = Pipeline(source=source, table='test_sql', sql_transforms=[SQLColumns(columns=['A', 'B'])])
    pipeline._update_data()
    df = pd._testing.makeMixedDataFrame()[['A', 'B']]
    pd.testing.assert_frame_equal(pipeline.data, df)

def test_pipeline_chained_with_filter(make_filesource):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    pipeline1 = Pipeline(source=source, filters=[ConstantFilter(field='A', value=(1, 3))], table='test')
    pipeline2 = pipeline1.chain(filters=[ConstantFilter(field='B', value=1.0)])
    pipeline2._update_data()
    expected = pd._testing.makeMixedDataFrame().iloc[[1, 3]]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

def test_pipeline_chained_with_transform(make_filesource):
    root = pathlib.Path(__file__).parent / 'sources'
    source = make_filesource(str(root))
    pipeline1 = Pipeline(source=source, filters=[ConstantFilter(field='A', value=(1, 2))], table='test')
    pipeline2 = pipeline1.chain(transforms=[Columns(columns=['A', 'B'])])
    pipeline2._update_data()
    expected = pd._testing.makeMixedDataFrame().iloc[1:3][['A', 'B']]
    pd.testing.assert_frame_equal(pipeline2.data, expected)

def test_pipeline_chained_with_sql_transform():
    root = pathlib.Path(__file__).parent / 'sources'
    source = IntakeSQLSource(
        uri=str(root / 'catalog.yml'), root=str(root)
    )
    pipeline1 = Pipeline(source=source, filters=[ConstantFilter(field='A', value=(1, 2))], table='test_sql')
    pipeline2 = pipeline1.chain(sql_transforms=[SQLColumns(columns=['A', 'B'])])
    pipeline2._update_data()
    df = pd._testing.makeMixedDataFrame().iloc[1:3][['A', 'B']].reset_index(drop=True)
    pd.testing.assert_frame_equal(pipeline2.data, df)
