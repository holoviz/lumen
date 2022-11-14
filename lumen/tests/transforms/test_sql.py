import datetime as dt

from lumen.transforms.sql import (
    SQLColumns, SQLDistinct, SQLFilter, SQLGroupBy, SQLLimit, SQLMinMax,
)


def test_sql_group_by_single_column():
    assert (
        SQLGroupBy.apply_to('SELECT * FROM TABLE', by=['A'], aggregates={'AVG': 'B'}) ==
        """SELECT\n    A, AVG(B) AS B\nFROM ( SELECT * FROM TABLE )\nGROUP BY A"""
    )

def test_sql_group_by_multi_columns():
    assert (
        SQLGroupBy.apply_to('SELECT * FROM TABLE', by=['A'], aggregates={'AVG': ['B', 'C']}) ==
        """SELECT\n    A, AVG(B) AS B, AVG(C) AS C\nFROM ( SELECT * FROM TABLE )\nGROUP BY A"""
    )

def test_sql_limit():
    assert (
        SQLLimit.apply_to('SELECT * FROM TABLE', limit=10) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nLIMIT 10"""
    )

def test_sql_columns():
    assert (
        SQLColumns.apply_to('SELECT * FROM TABLE', columns=['A', 'B']) ==
        """SELECT\n    A, B\nFROM ( SELECT * FROM TABLE )"""
    )

def test_sql_distinct():
    assert (
        SQLDistinct.apply_to('SELECT * FROM TABLE', columns=['A', 'B']) ==
        """SELECT DISTINCT\n    A, B\nFROM ( SELECT * FROM TABLE )"""
    )

def test_sql_min_max():
    assert (
        SQLMinMax.apply_to('SELECT * FROM TABLE', columns=['A', 'B']) ==
        """SELECT\n    MIN(A) as A_min, MAX(A) as A_max, MIN(B) as B_min, MAX(B) as B_max\nFROM ( SELECT * FROM TABLE )"""
    )

def test_sql_filter_none():
    assert (
        SQLFilter.apply_to('SELECT * FROM TABLE', conditions=[('A', None)]) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nWHERE ( A IS NULL )"""
    )

def test_sql_filter_scalar():
    assert (
        SQLFilter.apply_to('SELECT * FROM TABLE', conditions=[('A', 1)]) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nWHERE ( A = 1 )"""
    )


def test_sql_filter_isin():
    assert (
        SQLFilter.apply_to('SELECT * FROM TABLE', conditions=[('A', ['A', 'B', 'C'])]) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nWHERE ( A IN ('A', 'B', 'C') )"""
    )

def test_sql_filter_datetime():
    assert (
        SQLFilter.apply_to('SELECT * FROM TABLE', conditions=[('A', dt.datetime(2017, 4, 14))]) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nWHERE ( A = '2017-04-14 00:00:00' )"""
    )

def test_sql_filter_date():
    assert (
        SQLFilter.apply_to('SELECT * FROM TABLE', conditions=[('A', dt.date(2017, 4, 14))]) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nWHERE ( A BETWEEN '2017-04-14 00:00:00' AND '2017-04-14 23:59:59' )"""
    )

def test_sql_filter_date_range():
    assert (
        SQLFilter.apply_to('SELECT * FROM TABLE', conditions=[('A', (dt.date(2017, 2, 22), dt.date(2017, 4, 14)))]) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nWHERE ( A BETWEEN '2017-02-22 00:00:00' AND '2017-04-14 23:59:59' )"""
    )

def test_sql_filter_datetime_range():
    assert (
        SQLFilter.apply_to('SELECT * FROM TABLE', conditions=[('A', (dt.datetime(2017, 2, 22), dt.datetime(2017, 4, 14)))]) ==
        """SELECT\n    *\nFROM ( SELECT * FROM TABLE )\nWHERE ( A BETWEEN '2017-02-22 00:00:00' AND '2017-04-14 00:00:00' )"""
    )
