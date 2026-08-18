"""
Deterministic data-quality profiling for query results.

Reports the problems a result carries (missing values, duplicate rows,
placeholder numbers, untrimmed text, numbers stored as text, constant columns
and outliers) as sentences an LLM can act on. No LLM call is involved: the
whole point is to reach a verdict cheaply enough that a clean result costs
nothing, so an expensive follow-up only happens when there is a real problem.

Checks work block-wise rather than column-by-column. On a 5000x200 frame the
per-column form spent 123ms computing quantiles that one ``DataFrame.quantile``
call does in 38ms, and column loops were what made DuckDB look competitive.
"""
from __future__ import annotations

import pandas as pd

from .utils import PROFILE_SAMPLE_ROWS, format_float, log_debug

# A column is only worth reporting once a non-trivial share of it is null;
# a single missing value in a million rows is not a finding worth an LLM call.
LINT_NULL_THRESHOLD = 0.01
# Placeholder numbers that instruments and legacy exports write in place of a
# missing value. They pass as data through SUM/AVG and silently wreck results.
LINT_SENTINELS = (-9999, -999, 9999)
# Tukey's "far out" fence, not the usual 1.5 "outside" fence. At 1.5 a normal
# distribution puts ~0.7% of its values beyond the fence, so every clean numeric
# column reports outliers and the cleaning pass never short-circuits; measured on
# 5000-row gaussian samples, 1.5 fired on 200/200 columns and 3.0 on 3/200.
LINT_IQR_MULTIPLIER = 3.0
# Quartiles computed from fewer points than this describe the sample rather than
# a distribution, so no outlier verdict is drawn from them.
LINT_MIN_QUANTILE_ROWS = 4
# Every column is always checked. This caps only how many are named in a
# finding's text, so a wide result with a systemic problem cannot produce a
# sentence longer than the query it describes.
LINT_MAX_NAMED_COLUMNS = 8
# Values inspected before deciding a text column might be numeric. Nearly every
# genuine text column fails on its first value, so probing a short head avoids
# parsing whole columns to learn that (13x faster on a 150-column frame).
LINT_NUMERIC_PROBE_ROWS = 20


def _format_column_hits(hits: dict[str, str]) -> str:
    """Render ``{column: detail}`` pairs for a finding, truncated to a readable length."""
    listed = list(hits.items())[:LINT_MAX_NAMED_COLUMNS]
    rendered = [f'"{col}" ({detail})' for col, detail in listed]
    if len(hits) > len(listed):
        rendered.append(f"and {len(hits) - len(listed)} more columns")
    return ", ".join(rendered)


def _lint_nulls(df: pd.DataFrame) -> list[str]:
    fractions = df.isnull().mean()
    dirty = fractions[fractions > LINT_NULL_THRESHOLD]
    if dirty.empty:
        return []
    hits = {col: f"{frac:.1%} null" for col, frac in dirty.items()}
    return [f"Missing values in {len(dirty)} column(s): {_format_column_hits(hits)}."]


def _lint_duplicates(df: pd.DataFrame) -> list[str]:
    duplicates = int(df.duplicated().sum())
    if not duplicates:
        return []
    return [f"{duplicates} of {len(df)} rows are exact duplicates of an earlier row."]


def _lint_sentinels(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return []
    counts = numeric.isin(LINT_SENTINELS).sum()
    hits = {col: f"{int(count)} rows" for col, count in counts.items() if count}
    if not hits:
        return []
    return [
        f"Placeholder numbers {list(LINT_SENTINELS)} appear as data in "
        f"{_format_column_hits(hits)}; they almost certainly encode missing values."
    ]


def _lint_whitespace(df: pd.DataFrame) -> list[str]:
    hits = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].dropna()
        # The .str accessor yields NaN for non-string entries, so an object
        # column holding mixed types contributes only its actual strings.
        stripped = values.str.strip()
        padded = int((stripped.notna() & (stripped != values)).sum())
        blank = int((stripped == "").sum())
        if padded or blank:
            hits[col] = f"{padded} padded, {blank} empty"
    if not hits:
        return []
    return [f"Untrimmed or empty text in {_format_column_hits(hits)}."]


def _lint_numeric_text(df: pd.DataFrame) -> list[str]:
    hits = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].dropna()
        if values.empty:
            continue
        if pd.to_numeric(values.head(LINT_NUMERIC_PROBE_ROWS), errors="coerce").isna().any():
            continue
        if pd.to_numeric(values, errors="coerce").notna().all():
            hits[col] = "every value parses as a number"
    if not hits:
        return []
    return [f"Numbers stored as text in {_format_column_hits(hits)}; aggregating these needs a cast."]


def _lint_constant(df: pd.DataFrame) -> list[str]:
    # Below two rows every column is trivially constant, which says nothing
    # about the data.
    if len(df) < 2:
        return []
    hits = {col: "one distinct value" for col, count in df.nunique(dropna=False).items() if count <= 1}
    if not hits:
        return []
    return [f"Constant column(s) carrying no information: {_format_column_hits(hits)}."]


def _lint_outliers(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return []
    numeric = numeric.loc[:, numeric.count() >= LINT_MIN_QUANTILE_ROWS]
    if numeric.empty:
        return []

    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    # A zero IQR means the middle half of a column is a single value, so Tukey
    # fences collapse onto it and flag every other value. That is the normal
    # shape of a coded or flag column, where "outlier" is meaningless, so drop
    # those columns rather than guess which degenerate case each one is.
    fence = (LINT_IQR_MULTIPLIER * (q3 - q1)).dropna()
    fence = fence[fence > 0]
    if fence.empty:
        return []

    spread = numeric[fence.index]
    low, high = q1[fence.index] - fence, q3[fence.index] + fence
    # NaN compares False on both sides, so missing values never count as outliers.
    counts = ((spread < low) | (spread > high)).sum()
    hits = {
        col: f"{int(count)} outside [{format_float(low[col])}, {format_float(high[col])}]"
        for col, count in counts.items() if count
    }
    if not hits:
        return []
    return [f"IQR outliers in {_format_column_hits(hits)}."]


def lint_data(df: pd.DataFrame, actionable_only: bool = False) -> list[str]:
    """
    Report data-quality problems in a query result, phrased so an LLM can fix them in SQL.

    An empty list means the result looked clean and no follow-up work is warranted.

    Parameters
    ----------
    df : pd.DataFrame
        The query result to profile.
    actionable_only : bool
        Report only problems a SQL rewrite should act on, omitting constant
        columns and outliers. Use this to decide whether a rewrite is worth
        requesting, and to choose what the rewriting prompt is shown.

    Returns
    -------
    list[str]
        One sentence per problem found, or an empty list.
    """
    if df.empty:
        return []

    checks = (_lint_nulls, _lint_duplicates, _lint_sentinels, _lint_whitespace, _lint_numeric_text)
    if not actionable_only:
        # Reported for the reader but never worth rewriting a query for: a
        # constant column is the normal shape of a result filtered to one value,
        # so dropping it deletes something the user asked for, and an outlier is
        # data, so filtering it changes the answer rather than cleaning it.
        checks += (_lint_constant, _lint_outliers)

    sampled = len(df) > PROFILE_SAMPLE_ROWS
    if sampled:
        # Fixed seed so the same result always yields the same findings; a
        # profiler that reported different problems on each run would be noise.
        df = df.sample(PROFILE_SAMPLE_ROWS, random_state=0)

    findings = []
    for check in checks:
        try:
            findings.extend(check(df))
        except Exception as e:
            # Profiling must never break a query that already succeeded. Object
            # columns holding lists, dicts or geometries make several of the
            # pandas operations above raise rather than return.
            log_debug(f"lint_data check {check.__name__} skipped: {e}")

    if findings and sampled:
        findings.append(
            f"Counts above come from a random {PROFILE_SAMPLE_ROWS}-row sample of the result, not the full table."
        )
    return findings
