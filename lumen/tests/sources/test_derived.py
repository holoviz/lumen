import os

import pandas as pd

from lumen.sources.base import DerivedSource


def test_derived_mirror_source(make_filesource):
    root = os.path.dirname(__file__)
    original = make_filesource(root)
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
    })

    assert derived.get_tables() == ['test']
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(derived.get('test'), df)
    assert original.get_schema() == derived.get_schema()


def test_derived_mirror_source_transforms(make_filesource):
    root = os.path.dirname(__file__)
    make_filesource(root)
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
        'transforms': [{'type': 'iloc', 'end': 3}]
    })

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


def test_derived_mirror_source_filters(make_filesource):
    root = os.path.dirname(__file__)
    make_filesource(root)
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'source': 'original',
        'filters': [{'type': 'constant', 'field': 'A', 'value': (0, 2)}]
    })

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


def test_derived_tables_source(make_filesource):
    root = os.path.dirname(__file__)
    original = make_filesource(root)
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'tables': {'derived': {'source': 'original', 'table': 'test'}}
    })

    assert derived.get_tables() == ['derived']
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(derived.get('derived'), df)
    assert original.get_schema('test') == derived.get_schema('derived')


def test_derived_tables_source_transforms(make_filesource):
    root = os.path.dirname(__file__)
    make_filesource(root)
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'tables': {
            'derived': {
                'source': 'original',
                'table': 'test',
                'transforms': [{'type': 'iloc', 'end': 3}]
            }
        }
    })

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


def test_derived_tables_source_filters(make_filesource):
    root = os.path.dirname(__file__)
    make_filesource(root)
    derived = DerivedSource.from_spec({
        'type': 'derived',
        'tables': {
            'derived': {
                'source': 'original',
                'table': 'test',
                'filters': [{'type': 'constant', 'field': 'A', 'value': (0, 2)}]
            }
        }
    })

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
