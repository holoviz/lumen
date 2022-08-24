import pandas as pd


def source_get_tables(source, source_tables):
    assert source.get_tables() == list(source_tables.keys())
    return True


def source_table_cache_key(source, table, kwargs1, kwargs2):
    key1 = source._get_key(table, **kwargs1)
    key2 = source._get_key(table, **kwargs2)
    assert key1 == key2
    return True


def source_filter(source, table_column_value_type, dask, expected_df):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    if dask is not None:
        kwargs['__dask'] = dask
    filtered = source.get(table, **kwargs)
    pd.testing.assert_frame_equal(filtered, expected_df)
    return True


def source_get_query_cache(source, table_column_value_type, dask, expected_df):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    if dask:
        source.get(table, __dask=dask, **kwargs)
    else:
        source.get(table, **kwargs)
    cache_key = source._get_key(table, **kwargs)
    assert cache_key in source._cache
    cached_df = source._cache[cache_key]
    if dask:
        cached_df = cached_df.compute()
    pd.testing.assert_frame_equal(cached_df, expected_df)
    cache_key = source._get_key(table, **kwargs)
    assert cache_key in source._cache
    return True


def source_clear_cache(source, table_column_value_type, dask):
    table, column, value, _ = table_column_value_type
    kwargs = {column: value}
    cache_key = source._get_key(table, **kwargs)
    if dask is not None:
        kwargs['__dask'] = dask
    source.get(table, **kwargs)
    assert cache_key in source._cache
    source.clear_cache()
    assert len(source._cache) == 0
    return True
