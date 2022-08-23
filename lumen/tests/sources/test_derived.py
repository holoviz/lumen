import pytest

import os

import pandas as pd

from lumen.sources.base import DerivedSource


@pytest.fixture
def original(make_filesource):
    root = os.path.dirname(__file__)
    return make_filesource(root)


@pytest.fixture
def expected_table():
    df = pd._testing.makeMixedDataFrame()
    return df.iloc[:3]


@pytest.fixture
def expected_schema():
    schema = {
        'A': {'inclusiveMaximum': 2.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'B': {'inclusiveMaximum': 1.0, 'inclusiveMinimum': 0.0, 'type': 'number'},
        'C': {'enum': ['foo1', 'foo2', 'foo3'], 'type': 'string'},
        'D': {
            'format': 'datetime',
            'type': 'string',
            'inclusiveMaximum': '2009-01-05T00:00:00',
            'inclusiveMinimum': '2009-01-01T00:00:00'
        }
    }
    return schema


@pytest.fixture
def mirror_transforms():
    transforms = [{'type': 'iloc', 'end': 3}]
    return transforms


@pytest.fixture
def mirror_filters():
    filters = [{'type': 'constant', 'field': 'A', 'value': (0, 2)}]
    return filters


@pytest.fixture
def tables_transforms(mirror_transforms):
    spec = {
        'type': 'derived',
        'tables': {
            'derived': {
                'source': 'original',
                'table': 'test',
                'transforms': mirror_transforms
            }
        }
    }
    return spec


@pytest.fixture
def tables_filters(mirror_filters):
    spec = {
        'type': 'derived',
        'tables': {
            'derived': {
                'source': 'original',
                'table': 'test',
                'filters': mirror_filters
            }
        }
    }
    return spec


def test_derived_source_resolve_module_type():
    assert DerivedSource._get_type('lumen.sources.base.DerivedSource') is DerivedSource


def test_derived_mirror_source(original):
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
    })
    assert derived.get_tables() == original.get_tables()
    assert derived.get_schema() == original.get_schema()
    for table in original.get_tables():
        pd.testing.assert_frame_equal(derived.get(table), original.get(table))


def test_derived_mirror_source_transforms(original, mirror_transforms, expected_table, expected_schema):
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
        'transforms': mirror_transforms,
    })
    assert derived.get_tables() == original.get_tables()
    assert derived.get_schema('test') == expected_schema
    pd.testing.assert_frame_equal(derived.get('test'), expected_table)


def test_derived_mirror_source_filters(original, mirror_filters, expected_table, expected_schema):
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
        'filters': mirror_filters
    })
    assert derived.get_tables() == original.get_tables()
    assert derived.get_schema('test') == expected_schema
    pd.testing.assert_frame_equal(derived.get('test'), expected_table)


def test_derived_tables_source(original):
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'tables': {'derived': {'source': 'original', 'table': 'test'}}
    })
    assert derived.get_tables() == ['derived']
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(derived.get('derived'), df)
    assert original.get_schema('test') == derived.get_schema('derived')


def test_derived_tables_source_transforms(original, tables_transforms, expected_table, expected_schema):
    derived = DerivedSource.from_spec(tables_transforms)
    assert derived.get_tables() == ['derived']
    pd.testing.assert_frame_equal(derived.get('derived'), expected_table)
    assert derived.get_schema('derived') == expected_schema


def test_derived_tables_source_filters(original, tables_filters, expected_table, expected_schema):
    derived = DerivedSource.from_spec(tables_filters)
    assert derived.get_tables() == ['derived']
    pd.testing.assert_frame_equal(derived.get('derived'), expected_table)
    assert derived.get_schema('derived') == expected_schema
