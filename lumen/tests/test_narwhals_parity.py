"""Differential matrix: every ported transform must agree with pandas.

Each case below is run once per backend and compared against the pandas result
for the same spec, rather than against a hand-written expectation. Hand-written
expectations are what let a divergence through: they encode what the author
believed pandas does, and the author is exactly who was wrong.

Adding a case here is cheaper than discovering the divergence in a review, so
new specs belong in CASES rather than in a bespoke test.
"""
import datetime as dt

import pandas as pd
import pytest

from lumen.transforms.base import (
    Aggregate, Columns, DropNA, Filter, Iloc, Melt, Rename, Sample, Sort,
)
from lumen.util import as_pandas

# Frames the transforms are exercised against. Each one carries a shape that
# has produced a real divergence at some point: nulls, NaN, mixed dtypes.
FRAMES = {
    'plain': {'g': ['b', 'a', 'b', 'a'], 'v': [1.0, 2.0, 3.0, 4.0], 'i': [4, 3, 2, 1]},
    'nulls': {'g': ['b', None, 'b', 'a'], 'v': [1.0, 2.0, None, 4.0], 'i': [4, 3, 2, 1]},
    'nan': {'g': ['b', 'a', 'b', 'a'], 'v': [1.0, float('nan'), 3.0, 4.0], 'i': [4, 3, 2, 1]},
    'single': {'g': ['a'], 'v': [1.0], 'i': [1]},
    'inf': {'g': ['b', 'a', 'b', 'a'], 'v': [1.0, float('inf'), 3.0, 4.0],
            'i': [4, 3, 2, 1]},
    'nankey': {'g': [1.0, float('nan'), 1.0, 2.0], 'v': [1.0, 2.0, 3.0, 4.0],
               'i': [4, 3, 2, 1]},
    'infkey': {'g': [1.0, float('inf'), 1.0, 2.0], 'v': [1.0, 2.0, 3.0, 4.0],
               'i': [4, 3, 2, 1]},
}

# (transform, kwargs, frame name). Anything a spec can legally express.
CASES = [
    (Sort, {'by': ['v']}, 'plain'),
    (Sort, {'by': ['v']}, 'nan'),
    (Sort, {'by': ['v'], 'ascending': False}, 'nan'),
    (Sort, {'by': ['g', 'v']}, 'nan'),
    (Sort, {'by': ['v']}, 'nulls'),
    (Sort, {'by': ['g', 'v']}, 'plain'),
    (Sort, {'by': ['v'], 'ascending': False}, 'nulls'),
    (Sort, {'by': ['g', 'v'], 'ascending': [True, False]}, 'plain'),
    (Sort, {'by': []}, 'plain'),
    (Columns, {'columns': ['g', 'v']}, 'plain'),
    (Columns, {'columns': ['v']}, 'nulls'),
    (Iloc, {'start': 1, 'end': 3}, 'plain'),
    (Iloc, {'end': 2}, 'plain'),
    (Iloc, {'start': 2}, 'plain'),
    (Iloc, {'start': 0, 'end': 99}, 'single'),
    (Aggregate, {'by': ['g'], 'with_index': False}, 'plain'),
    (Aggregate, {'by': ['g'], 'with_index': False}, 'nulls'),
    (Aggregate, {'by': ['g'], 'with_index': False}, 'nan'),
    (Aggregate, {'by': ['g'], 'with_index': False}, 'inf'),
    (Aggregate, {'by': ['g'], 'with_index': False}, 'nankey'),
    (Aggregate, {'by': ['g'], 'with_index': False}, 'infkey'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'median'}, 'plain'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'first'}, 'nan'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'std'}, 'plain'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'var'}, 'plain'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'count'}, 'nulls'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'sum'}, 'plain'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'min'}, 'nulls'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'nunique'}, 'plain'),
    (Aggregate, {'by': ['g'], 'with_index': False, 'method': 'prod'}, 'plain'),
    (Aggregate, {'by': ['g'], 'columns': ['v'], 'with_index': False}, 'plain'),
    (DropNA, {}, 'nulls'),
    (DropNA, {}, 'nan'),
    (DropNA, {'subset': ['v']}, 'nulls'),
    (DropNA, {'how': 'all'}, 'nulls'),
    (Rename, {'columns': {'g': 'group'}}, 'plain'),
    (Rename, {'columns': {'absent': 'z'}}, 'plain'),
    (Rename, {'mapper': {'g': 'group'}, 'axis': 1}, 'plain'),
    (Rename, {'mapper': {'g': 'group'}, 'axis': 'columns'}, 'plain'),
    (Melt, {'id_vars': ['g']}, 'plain'),
    (Melt, {'id_vars': ['g'], 'value_vars': ['v']}, 'plain'),
    (Melt, {'id_vars': ['g'], 'value_vars': []}, 'plain'),
    (Melt, {'id_vars': ['g'], 'value_vars': ['g', 'v']}, 'plain'),
    (Melt, {'id_vars': [], 'value_vars': ['v']}, 'nulls'),
    (Melt, {'id_vars': ['g'], 'var_name': 'field'}, 'plain'),
    (Filter, {'conditions': [('g', 'a')]}, 'plain'),
    (Filter, {'conditions': [('v', [None])]}, 'nulls'),
    (Filter, {'conditions': [('g', [None])]}, 'nulls'),
    (Filter, {'conditions': [('v', [1.0, None])]}, 'nulls'),
    (Filter, {'conditions': [('g', ['a', 'b'])]}, 'nulls'),
    (Filter, {'conditions': [('v', (1.0, 3.0))]}, 'nulls'),
    (Filter, {'conditions': [('v', [(1.0, 1.0), (4.0, 4.0)])]}, 'nulls'),
    (Filter, {'conditions': [('i', 3), ('g', ['a'])]}, 'plain'),
]


def assert_native_path(caplog, backend):
    """Fail if a non-pandas backend quietly took the pandas path.

    Without this the matrix compares pandas against pandas and passes no matter
    what the port does, which would let the whole thing rot green.
    """
    if backend == 'pandas':
        return
    assert 'converted to pandas' not in caplog.text, (
        'the narwhals path was not taken; the comparison is vacuous'
    )


def _missing_to_none(row):
    """Collapse every spelling of missing to one, so rows compare.

    pandas uses NaN where polars and pyarrow use null, including for missing
    strings. The distinction is a pandas artifact rather than a difference in
    the data, and both spellings behave the same through Filter.
    """
    return [None if isinstance(v, float) and v != v else v for v in row]


def normalize(frame):
    """Comparable form: pandas, index dropped, rows and columns sorted.

    Row order is compared separately by the ordering cases; here the point is
    that the same rows and values come out of every backend.
    """
    result = as_pandas(frame).reset_index(drop=True)
    result = result[sorted(result.columns)]
    if len(result):
        result = result.sort_values(list(result.columns), na_position='last')
    return result.reset_index(drop=True)


@pytest.mark.parametrize("transform, kwargs, frame_name", [
    pytest.param(t, k, f, id=f"{t.__name__}-{f}-{sorted(k.items())}")
    for t, k, f in CASES
])
def test_transform_matches_pandas(constructor, transform, kwargs, frame_name):
    data = FRAMES[frame_name]
    reference = normalize(transform.apply_to(pd.DataFrame(data), **kwargs))
    result = normalize(transform.apply_to(constructor(data), **kwargs))
    pd.testing.assert_frame_equal(result, reference, check_dtype=False)


# Specs that deliberately fall back: pandas-only options, or aggregations whose
# narwhals meaning differs from the pandas one. Each is still checked for the
# same answer above; this list only records that the fallback is intentional.
FALLS_BACK = {
    ('Aggregate', 'nunique'), ('Aggregate', 'prod'), ('Aggregate', 'median'),
    ('Aggregate', 'first'), ('Aggregate', 'count'),
    ('DropNA', 'all'), ('Melt', '[]'), ('Sort', 'noby'),
}


@pytest.mark.parametrize("transform, kwargs, frame_name", [
    pytest.param(t, k, f, id=f"{t.__name__}-{f}-{sorted(k.items())}")
    for t, k, f in CASES
])
def test_native_path_is_actually_taken(constructor, caplog, transform, kwargs, frame_name):
    marker = (
        transform.__name__,
        kwargs.get('method') or ('[]' if kwargs.get('value_vars') == [] else None)
        or ('all' if kwargs.get('how') == 'all' else None)
        or ('noby' if kwargs.get('by') == [] else None)
    )
    if marker in FALLS_BACK:
        pytest.skip('documented fallback')
    with caplog.at_level('WARNING'):
        transform.apply_to(constructor(FRAMES[frame_name]), **kwargs)
    assert_native_path(caplog, constructor.__module__.split('.')[0])


ORDERED = [c for c in CASES if c[0] in (Sort, Iloc, Aggregate, Filter)]


@pytest.mark.parametrize("transform, kwargs, frame_name", [
    pytest.param(t, k, f, id=f"{t.__name__}-{f}-{sorted(k.items())}")
    for t, k, f in ORDERED
])
def test_row_order_matches_pandas(constructor, transform, kwargs, frame_name):
    """Row order is a contract for anything that sorts, slices or groups."""
    data = FRAMES[frame_name]
    reference = as_pandas(transform.apply_to(pd.DataFrame(data), **kwargs))
    result = as_pandas(transform.apply_to(constructor(data), **kwargs))
    assert [_missing_to_none(r) for r in result.values.tolist()] == \
        [_missing_to_none(r) for r in reference.values.tolist()]


def test_schema_matches_pandas_over_dtypes(constructor):
    """Every dtype a backend can express must describe the same as pandas."""
    from lumen.util import get_dataframe_schema
    data = {
        'i': pd.Series([1, 2, 3], dtype='int64'),
        'u': pd.Series([1, 2, 3], dtype='uint8'),
        'f32': pd.Series([1.5, 2.5, 3.5], dtype='float32'),
        'f': [1.5, None, 3.5],
        's': ['a', None, 'b'],
        'b': [True, False, True],
        'd': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
    }
    reference = get_dataframe_schema(pd.DataFrame(data))['items']['properties']
    schema = get_dataframe_schema(constructor(data))['items']['properties']
    for properties in (reference, schema):
        for spec in properties.values():
            if 'enum' in spec:
                spec['enum'] = _missing_to_none(spec['enum'])
    assert schema == reference


def test_schema_matches_pandas_tz_aware(constructor):
    from lumen.util import get_dataframe_schema
    data = {'t': pd.date_range('2020-01-01', periods=3, tz='UTC')}
    reference = get_dataframe_schema(pd.DataFrame(data))['items']['properties']
    assert get_dataframe_schema(constructor(data))['items']['properties'] == reference


@pytest.mark.parametrize("value", [
    'a string', 3, dt.date(2020, 1, 1), ['a', 'nope'], (0, 99), None,
])
def test_filter_never_crashes_on_odd_values(constructor, value):
    """Filter values arrive from URLs and widgets, so nothing may raise."""
    data = {'g': ['b', 'a'], 'v': [1.0, 2.0]}
    for column in ('g', 'v'):
        try:
            expected = len(Filter.apply_to(pd.DataFrame(data), conditions=[(column, value)]))
        except Exception:
            continue  # pandas rejects it too, so no contract to match
        assert len(as_pandas(
            Filter.apply_to(constructor(data), conditions=[(column, value)])
        )) == expected
