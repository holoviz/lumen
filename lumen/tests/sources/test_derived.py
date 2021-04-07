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
