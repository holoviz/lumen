import os

import pandas as pd

from lumen.sources.base import FileSource, DerivedSource


def test_derived_mirror_source():
    root = os.path.dirname(__file__)
    original = FileSource(tables={'test': 'test.csv'}, root=root, kwargs={'parse_dates': ['D']})
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
    }, sources={'original': original})

    assert derived.get_tables() == ['test']
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(derived.get('test'), df)
    assert original.get_schema() == derived.get_schema()


def test_derived_mirror_source_transforms():
    root = os.path.dirname(__file__)
    original = FileSource(tables={'test': 'test.csv'}, root=root, kwargs={'parse_dates': ['D']})
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
        'transforms': [{'type': 'iloc', 'end': 3}]
    }, sources={'original': original})

    assert derived.get_tables() == ['test']
    df = pd._testing.makeMixedDataFrame().iloc[:3]
    pd.testing.assert_frame_equal(derived.get('test'), df)
    assert derived.get_schema('test') == {
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


def test_derived_mirror_source_filters():
    root = os.path.dirname(__file__)
    original = FileSource(tables={'test': 'test.csv'}, root=root, kwargs={'parse_dates': ['D']})
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
        'filters': [{'type': 'constant', 'field': 'A', 'value': (0, 2)}]
    }, sources={'original': original})

    assert derived.get_tables() == ['test']
    df = pd._testing.makeMixedDataFrame().iloc[:3]
    pd.testing.assert_frame_equal(derived.get('test'), df)
    assert derived.get_schema('test') == {
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


def test_derived_tables_source():
    root = os.path.dirname(__file__)
    original = FileSource(tables={'test': 'test.csv'}, root=root, kwargs={'parse_dates': ['D']})
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'tables': {'derived': {'source': 'original', 'table': 'test'}}
    }, sources={'original': original})

    assert derived.get_tables() == ['derived']
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(derived.get('derived'), df)
    assert original.get_schema('test') == derived.get_schema('derived')


def test_derived_tables_source_transforms():
    root = os.path.dirname(__file__)
    original = FileSource(tables={'test': 'test.csv'}, root=root, kwargs={'parse_dates': ['D']})
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'tables': {
            'derived': {
                'source': 'original',
                'table': 'test',
                'transforms': [{'type': 'iloc', 'end': 3}]
            }
        }
    }, sources={'original': original})

    assert derived.get_tables() == ['derived']
    df = pd._testing.makeMixedDataFrame().iloc[:3]
    pd.testing.assert_frame_equal(derived.get('derived'), df)
    assert derived.get_schema('derived') == {
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


def test_derived_tables_source_filters():
    root = os.path.dirname(__file__)
    original = FileSource(tables={'test': 'test.csv'}, root=root, kwargs={'parse_dates': ['D']})
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'tables': {
            'derived': {
                'source': 'original',
                'table': 'test',
                'filters': [{'type': 'constant', 'field': 'A', 'value': (0, 2)}]
            }
        }
    }, sources={'original': original})

    assert derived.get_tables() == ['derived']
    df = pd._testing.makeMixedDataFrame().iloc[:3]
    pd.testing.assert_frame_equal(derived.get('derived'), df)
    assert derived.get_schema('derived') == {
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
