import dataclasses
import sqlite3

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
import sqlglot

pytest.importorskip("duckdb")

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from lumen.agent import AgentSession, facts
from lumen.agent.authoring import add_sql
from lumen.agent.data import (
    describe_table, list_tables, metaset, run_sql,
)
from lumen.agent.facts import (
    Lineage, answer_notes, closest, dates_note, filter_notes, grain_note,
    query_notes, table_grain, table_notes, text_filters,
)
from lumen.agent.scenarios.fixtures import fixture_source
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.data.sharing import sharing
from lumen.data.stats import SAMPLED, get_stats_store
from lumen.sources.duckdb import DuckDBSource
from lumen.sources.sqlalchemy import SQLAlchemySource

TODAY = date(2026, 11, 25)
TEXAS_SHARE = "SELECT SUM(CASE WHEN t_state = '{}' THEN t_cap ELSE 0 END) / SUM(t_cap) AS texas_share FROM turbines"
LAST_MONTH = (
    "SELECT COUNT(*) AS quakes FROM quakes WHERE \"time\" >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' "
    "AND \"time\" < DATE_TRUNC('month', CURRENT_DATE)"
)


@pytest.fixture
def today(monkeypatch):
    monkeypatch.setattr(facts, "today", lambda: TODAY)
    return TODAY


@pytest.fixture
def context():
    sessions = []

    def make(source):
        session = AgentSession(source, FunctionModel(lambda m, i: ModelResponse([TextPart("")])))
        sessions.append(session)
        return SimpleNamespace(deps=session.deps)

    yield make
    for session in sessions:
        session.close()


async def stats_of(source, table):
    return (await metaset(source)).get_stats(f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}")


async def test_grain_names_the_key_and_the_totals_repeated_on_each_row_of_a_project():
    source = fixture_source("turbines")
    grain = await table_grain(source, "turbines", await stats_of(source, "turbines"))
    assert grain.key == ["case_id"] and grain.unique == []
    [group] = grain.groups
    assert (group.column, group.values, group.low, group.high) == ("p_name", 90, 3, 17)
    assert {"t_state", "p_year", "p_tnum", "p_cap"} <= set(group.fixed)
    assert not {"case_id", "xlong", "ylat"} & set(group.fixed)
    assert group.totals == ["p_tnum is its row count", "p_cap is its SUM of t_cap / 1000"]
    note = grain_note(grain)
    assert note.startswith("grain: one row per case_id. p_name repeats, 90 values on 3 to 17 rows each; ")
    assert note.endswith(
        "Per p_name, p_tnum is its row count and p_cap is its SUM of t_cap / 1000: totals that a SUM over rows "
        "multiplies by the p_name's row count."
    )


async def test_grain_finds_a_key_of_two_columns_and_says_nothing_without_one():
    frame = pd.DataFrame({"state": ["TX", "CA", "IA"] * 4, "year": [2020] * 3 + [2021] * 3 + [2022] * 3 + [2023] * 3,
                          "mw": [float(i) for i in range(12)]})
    source = DuckDBSource.from_df({"capacity": frame})
    grain = await table_grain(source, "capacity", await stats_of(source, "capacity"))
    assert grain_note(grain) == "grain: one row per (state, year)."

    penguins = fixture_source("penguins")
    assert grain_note(await table_grain(penguins, "penguins", await stats_of(penguins, "penguins"))) is None


async def test_table_notes_take_one_query_per_profile_and_only_exact_statistics(monkeypatch):
    queries = []
    run = facts._run

    async def counting(adapter, sql, params):
        queries.append(sql)
        return await run(adapter, sql, params)

    monkeypatch.setattr(facts, "_run", counting)
    source = fixture_source("turbines")
    stats = await stats_of(source, "turbines")
    notes = await table_notes(source, "turbines", stats)
    assert [note.split(":")[0] for note in notes] == ["grain", "dates"]
    assert await table_notes(source, "turbines", stats) == notes
    assert len(queries) == 1
    assert await table_notes(source, "turbines", dataclasses.replace(stats, method=SAMPLED)) == []
    monkeypatch.setattr(sharing, "share", "none")
    assert await table_notes(source, "turbines", stats) == []
    assert len(queries) == 1


async def test_dates_read_relative_periods_from_the_latest_date_when_the_data_ends_before_today():
    quakes = fixture_source("quakes")
    stats = await stats_of(quakes, "quakes")
    span = "dates: time 2024-01-01 01:13:00 to 2025-12-31 22:15:00."
    assert dates_note(stats, TODAY) == (
        f"{span} The data ends before today: unless the user asked about calendar dates, read last month, "
        "this year and the like from its latest date, not from today, and say which dates you used."
    )
    assert dates_note(stats, date(2025, 12, 31)) == span
    turbines = fixture_source("turbines")
    years = await stats_of(turbines, "turbines")
    assert dates_note(years, TODAY).startswith("dates: p_year 1999 to 2022. The data ends before today")
    assert dates_note(years, date(2022, 6, 1)) == "dates: p_year 1999 to 2022."


async def test_list_tables_and_describe_table_say_what_a_row_is_and_where_the_dates_end(context, today):
    ctx = context(fixture_source("turbines_states"))
    listing = await list_tables(ctx)
    assert "turbines (978 rows)\n  grain: one row per case_id. p_name repeats" in listing
    assert "states (24 rows)\n  grain: one row per state; name is unique too.\n  state VARCHAR" in listing
    lines = (await describe_table(ctx, "turbines", columns=["t_cap"])).splitlines()
    assert lines[0] == "turbines (978 rows)"
    assert lines[1].startswith("  grain: one row per case_id.")
    assert lines[2].startswith("  dates: p_year 1999 to 2022. The data ends before today")
    assert lines[3:] == ["  t_cap BIGINT 1000..3000"]


@pytest.mark.parametrize(("frame", "expected"), [
    (pd.DataFrame({"n": pd.Series([], dtype=int)}), "The query returned no rows, so this is not an answer yet"),
    (pd.DataFrame({"n": [0]}), "The result's only numbers are zero or NULL (n = 0)"),
    (pd.DataFrame({"share": [None], "n": [0]}, dtype=float), "The result's only numbers are zero or NULL (share = NULL, n = 0)"),
    (pd.DataFrame({"year": [2000, 2001], "mw": [0.0, None]}), "In this result, mw is zero or NULL in all 2 rows"),
    (pd.DataFrame({"state": ["TX", "CA"], "share": [1.0, 1.0]}), "In this result, all 2 rows hold the same numbers (share = 1)"),
    (pd.DataFrame({"n": [242]}), None),
    (pd.DataFrame({"n": [242], "missing": [0]}), None),
    (pd.DataFrame({"state": ["TX", "CA"], "mw": [3.5, 1.0]}), None),
    (pd.DataFrame({"state": ["TX", "TX"]}), None),
], ids=["empty", "zero", "null", "zero-column", "constant", "number", "number-and-zero", "series", "text"])
def test_answer_notes_say_when_a_result_answers_nothing(frame, expected):
    notes = answer_notes(frame)
    if expected is None:
        assert notes == []
    else:
        [note] = notes
        assert note.startswith(expected)


@pytest.mark.parametrize(("text", "values", "expected"), [
    ("Texas", ["CA", "TX", "IA"], ["TX"]),
    ("New York", ["NY", "NE", "CA"], ["NY"]),
    ("tx", ["TX", "CA"], ["TX"]),
    ("california", ["Nevada", "California"], ["California"]),
    ("Californa", ["Nevada", "California"], ["California"]),
    ("Northern", ["offshore Northern California", "Nevada"], ["offshore Northern California"]),
    ("Mars", ["CA", "TX"], []),
])
def test_closest_values_cover_case_abbreviations_parts_and_spelling(text, values, expected):
    assert closest(text, values) == expected


async def test_a_text_filter_no_row_matches_names_the_closest_stored_values(context):
    ctx = context(fixture_source("turbines"))
    await list_tables(ctx)
    text = await add_sql(ctx, "data", "texas_share", TEXAS_SHARE.format("Texas"))
    assert "Note: The result's only numbers are zero or NULL (texas_share = 0)" in text
    assert "Note: No row of turbines has t_state = 'Texas'. The closest stored values: 'TX' (103 rows)." in text
    text = await add_sql(ctx, "data", "texas_share", TEXAS_SHARE.format("TX"))
    assert "closest" not in text and "zero or NULL" not in text
    text = await run_sql(ctx, "SELECT * FROM turbines WHERE LOWER(t_state) IN ('texas', 'iowa')")
    assert "Note: No row of turbines has t_state = 'texas', ignoring case. The closest stored values: 'TX' (103 rows)." in text
    assert "Note: No row of turbines has t_state = 'iowa', ignoring case. The closest stored values: 'IA' (61 rows)." in text
    assert "ignoring case" not in await run_sql(ctx, "SELECT * FROM turbines WHERE LOWER(t_state) = 'tx'")


@pytest.mark.parametrize("share", ["schema", "none"])
async def test_filter_notes_are_withheld_when_the_sharing_level_hides_values(context, monkeypatch, share):
    ctx = context(fixture_source("turbines"))
    await list_tables(ctx)
    monkeypatch.setattr(sharing, "share", share)
    text = await run_sql(ctx, TEXAS_SHARE.format("Texas"))
    assert "zero or NULL" in text and "No row" not in text and "'TX'" not in text


async def test_filter_notes_are_withheld_for_personal_data(context, monkeypatch):
    people = pd.DataFrame({"email": ["al@yahoo.com", "bo@yahoo.com", "cy@hotmail.com"] * 3, "team": ["a", "b", "c"] * 3})
    ctx = context(DuckDBSource.from_df({"people": people}))
    await list_tables(ctx)
    monkeypatch.setattr(sharing, "mask_pii", True)
    for sql in ("SELECT COUNT(*) AS n FROM people WHERE email = 'bob@example.org'",
                "SELECT COUNT(*) AS n FROM people WHERE email LIKE '%gmail%'"):
        assert "No row" not in await run_sql(ctx, sql)


async def test_a_pattern_says_which_rows_it_misses_that_hold_an_abbreviation_of_its_text(context):
    ctx = context(fixture_source("quakes"))
    await list_tables(ctx)
    text = await run_sql(ctx, "SELECT COUNT(*) AS quakes FROM quakes WHERE place LIKE '%California%'")
    assert ("Note: place LIKE '%California%' matches 21 rows of quakes, not the 221 rows with 'CA' as a word, "
            "e.g. ") in text
    text = await run_sql(ctx, "SELECT COUNT(*) AS quakes FROM quakes WHERE place LIKE '%, CA' OR place ILIKE '%california%'")
    assert "Note:" not in text


async def test_a_query_reading_today_learns_where_the_data_ends(context, today):
    ctx = context(fixture_source("quakes"))
    await list_tables(ctx)
    text = await add_sql(ctx, "data", "quakes_last_month", LAST_MONTH)
    assert "Note: The result's only numbers are zero or NULL (quakes = 0)" in text
    assert ("Note: The query reads today's date, but quakes.time ends at 2025-12-31 22:15:00, before today. Unless "
            "the user asked about calendar dates, read last month, this year and the like from that latest date, not "
            "from today, and say which dates you used.") in text
    text = await run_sql(ctx, "SELECT COUNT(*) AS quakes FROM quakes WHERE \"time\" >= '2026-10-01'")
    assert "Note: The data it filters on: quakes.time runs from 2024-01-01 01:13:00 to 2025-12-31 22:15:00." in text
    text = await run_sql(ctx, "SELECT COUNT(*) AS quakes FROM quakes WHERE \"time\" >= TIMESTAMP '2025-12-01'")
    assert "Note:" not in text


async def test_a_query_reading_today_says_nothing_when_the_data_reaches_today(context, monkeypatch):
    monkeypatch.setattr(facts, "today", lambda: date(2025, 12, 31))
    ctx = context(fixture_source("quakes"))
    await list_tables(ctx)
    assert "today" not in await run_sql(ctx, LAST_MONTH)


@pytest.mark.parametrize("sql", [
    "WITH s AS (SELECT CASE WHEN t_state = 'TX' THEN 'Texas' ELSE 'Other' END AS t_state, t_cap FROM turbines) "
    "SELECT SUM(t_cap) AS cap FROM s WHERE t_state = 'Texas'",
    "SELECT COUNT(*) AS n FROM (SELECT UPPER(p_name) AS p_name FROM turbines) AS x WHERE p_name = 'NORTH RIDGE'",
    "SELECT CASE WHEN t_state IN ('TX', 'OK') THEN 'South' ELSE 'Other' END AS region, SUM(t_cap) AS cap "
    "FROM turbines GROUP BY 1 HAVING region = 'South'",
    "SELECT * FROM (SELECT p_name, CASE WHEN p_cap > 30 THEN 'big' ELSE 'small' END AS t_manu FROM turbines) "
    "WHERE t_manu = 'big'",
], ids=["cte-case", "subquery-upper", "having-alias", "subquery-case"])
async def test_a_derived_column_named_like_a_table_column_is_not_checked_against_its_values(context, sql):
    ctx = context(fixture_source("turbines"))
    await list_tables(ctx)
    assert "No row" not in await run_sql(ctx, sql)


async def test_a_column_passed_through_unchanged_is_checked_through_ctes_and_earlier_results(context):
    ctx = context(fixture_source("turbines"))
    await list_tables(ctx)
    text = await run_sql(ctx, "WITH s AS (SELECT t_state AS state, t_cap FROM turbines) "
                              "SELECT SUM(t_cap) AS cap FROM s WHERE state = 'Texas'")
    assert "Note: No row of turbines has t_state = 'Texas'. The closest stored values: 'TX' (103 rows)." in text
    await add_sql(ctx, "data", "by_region", "SELECT CASE WHEN t_state IN ('TX', 'OK') THEN 'South' ELSE 'Other' END "
                                             "AS t_state, SUM(t_cap) AS cap FROM turbines GROUP BY 1")
    assert "No row" not in await add_sql(ctx, "by_region", "south", "SELECT * FROM by_region WHERE t_state = 'South'")
    await add_sql(ctx, "data", "by_state", "SELECT t_state, SUM(t_cap) AS cap FROM turbines GROUP BY 1")
    text = await add_sql(ctx, "by_state", "texas", "SELECT * FROM by_state WHERE t_state = 'Texas'")
    assert "Note: No row of turbines has t_state = 'Texas'." in text


async def test_a_pattern_on_a_column_with_more_values_than_read_states_only_a_minimum(context):
    places = [f"{i} km N of Town{i}, Nevada" for i in range(1500) for _ in range(5)]
    places += [f"{i} km S of City{i}, California" for i in range(600)]
    places += [f"{i} km E of Spot{i}, CA" for i in range(50) for _ in range(6)]
    ctx = context(DuckDBSource.from_df({"quakes": pd.DataFrame({"place": places})}))
    await list_tables(ctx)
    text = await run_sql(ctx, "SELECT COUNT(*) AS n FROM quakes WHERE place LIKE '%California%'")
    assert "| 600 |" in text
    assert "Note: In quakes, place LIKE '%California%' misses at least 300 rows with 'CA' as a word, e.g. " in text
    assert "matches" not in text


async def test_a_value_on_one_row_of_a_group_and_null_on_the_others_is_not_repeated():
    sizes, caps = {"a": 2, "b": 3, "c": 4, "d": 5}, {"a": 2.0, "b": 3.5, "c": 4.0, "d": 6.25}
    frame = pd.DataFrame([
        {"project": p, "turbine": f"{p}{i}", "t_cap": caps[p] * 1000 / n, "p_cap": caps[p] if i == 0 else None}
        for p, n in sizes.items() for i in range(n)
    ])
    source = DuckDBSource.from_df({"t": frame})
    grain = await table_grain(source, "t", await stats_of(source, "t"))
    assert "p_cap" not in grain_note(grain)
    assert all("p_cap" not in group.fixed and not group.totals for group in grain.groups)


async def test_a_text_key_comes_before_a_whole_number_that_happens_to_be_unique():
    source = DuckDBSource.from_df({"t": pd.DataFrame({"population": [100, 250, 31, 47, 99],
                                                      "state": ["TX", "CA", "IA", "NY", "WA"]})})
    assert grain_note(await table_grain(source, "t", await stats_of(source, "t"))) == "grain: one row per state."


async def test_two_groups_are_too_few_to_call_a_column_their_row_count():
    source = DuckDBSource.from_df({"t": pd.DataFrame({"team": ["a", "a", "b", "b", "b"], "division": [2, 2, 3, 3, 3],
                                                      "player": ["p1", "p2", "p3", "p4", "p5"]})})
    grain = await table_grain(source, "t", await stats_of(source, "t"))
    assert grain.groups[0].fixed == ["division"] and grain.groups[0].totals == []


async def test_value_queries_are_cached_per_profile(context, monkeypatch):
    queries = []
    run = facts._run

    async def counting(adapter, sql, params):
        queries.append(sql)
        return await run(adapter, sql, params)

    ctx = context(fixture_source("quakes"))
    await list_tables(ctx)
    monkeypatch.setattr(facts, "_run", counting)
    for _ in range(2):
        await run_sql(ctx, "SELECT COUNT(*) AS n FROM quakes WHERE place LIKE '%California%'")
    assert len(queries) == 1


async def test_fact_queries_keep_the_profilers_cap_on_billed_bytes(monkeypatch):
    calls = []
    monkeypatch.setattr(facts, "get_adapter", lambda source, max_bytes_billed=None: calls.append(max_bytes_billed))
    facts._adapter(object())
    assert calls == [get_stats_store().profiler.max_bytes_billed]


@pytest.mark.parametrize(("dialect", "sql", "expected", "note"), [
    ("duckdb", "SELECT * FROM turbines WHERE t_state = 'tx'", ("=", False, False), "No row of turbines has t_state = 'tx'."),
    ("tsql", "SELECT * FROM turbines WHERE t_state = 'tx'", ("=", True, True), None),
    ("mysql", "SELECT * FROM turbines WHERE t_state = 'TX '", ("=", True, True), None),
    ("sqlite", "SELECT * FROM turbines WHERE t_state LIKE '%tx%'", ("LIKE", True, False), None),
    ("postgres", "SELECT * FROM turbines WHERE t_state LIKE '%tx%'", ("LIKE", False, False),
     "t_state LIKE '%tx%' matches 0 rows of turbines, not the 103 rows that match it ignoring case"),
])
async def test_text_filters_compare_as_the_dialect_does(dialect, sql, expected, note):
    source = fixture_source("turbines")
    stats = {"turbines": await stats_of(source, "turbines")}
    tree = sqlglot.parse_one(sql, read=dialect)
    [found] = text_filters(tree, Lineage(tree, source, stats, dialect), dialect)
    assert (found.op, found.caseless, found.padded) == expected
    notes = filter_notes(found, [("TX", 103), ("CA", 103)], True)
    assert notes == [] if note is None else notes[0].startswith(note)


async def test_facts_on_a_sqlite_source(tmp_path, today):
    path = tmp_path / "wind.sqlite"
    connection = sqlite3.connect(path)
    fixture_source("turbines").execute("SELECT * FROM turbines").to_sql("turbines", connection, index=False)
    connection.close()
    source = SQLAlchemySource(url=f"sqlite:///{path}")
    stats = await stats_of(source, "turbines")
    duckdb = fixture_source("turbines")
    assert await table_notes(source, "turbines", stats) == await table_notes(
        duckdb, "turbines", await stats_of(duckdb, "turbines"))
    store = get_stats_store()
    sql = "SELECT COUNT(*) AS n FROM turbines WHERE t_state = 'Texas' OR t_manu LIKE '%vestas%'"
    notes = await query_notes(source, sql, source.execute(sql), lambda table: store.get(source, table))
    assert "No row of turbines has t_state = 'Texas'. The closest stored values: 'TX' (103 rows)." in notes
    assert not any("vestas" in note for note in notes)
