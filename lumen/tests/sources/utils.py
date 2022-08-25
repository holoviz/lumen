import pandas as pd


def source_get_tables(source, source_tables):
    assert source.get_tables() == list(source_tables.keys())
    for table in source_tables:
        pd.testing.assert_frame_equal(source.get(table), source_tables[table])
    return True


def source_table_cache_key(source, table, kwargs1={}, kwargs2={}):
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


def source_get_cache_query(source, table_column_value_type, dask, expected_df):
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


def source_get_cache_no_query(source, table, expected_df, dask=False):
    source.get(table, __dask=dask)
    cache_key = source._get_key(table)
    assert cache_key in source._cache
    assert len(source._cache) == 1
    cached_df = source._cache[cache_key]
    if dask:
        cached_df = cached_df.compute()
    pd.testing.assert_frame_equal(cached_df, expected_df)
    cache_key = source._get_key(table)
    assert cache_key in source._cache
    assert len(source._cache) == 1
    return True


def source_get_schema_cache(source, table):
    source.get_schema(table)
    assert len(source._schema_cache) == 1
    assert table in source._schema_cache
    source.get_schema(table)
    assert len(source._schema_cache) == 1
    assert table in source._schema_cache
    return True


def source_get_schema_update_cache(source, table):
    # for some source type,
    # source.get_schema() updates both source._schema_cache and source._cache
    source.get_schema(table)
    assert len(source._schema_cache) == 1
    assert len(source._cache) == 1
    # schema for this table is now cached, call source.get_schema() again does not update cache
    source.get_schema(table)
    assert len(source._schema_cache) == 1
    assert len(source._cache) == 1
    return True


def source_clear_cache_get_query(source, table, kwargs={}):
    source.get(table, **kwargs)
    assert len(source._cache) == 1
    source.clear_cache()
    assert len(source._cache) == 0
    return True


def source_clear_cache_get_schema(source, table):
    source.get_schema(table)
    assert len(source._schema_cache) == 1
    source.clear_cache()
    assert len(source._schema_cache) == 0
    return True
