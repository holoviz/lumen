"""
Model-free helpers shared by :mod:`lumen.data` and :mod:`lumen.ai`.

They render values, errors, schemas and SQL for a language model to read,
and size text in tokens, without importing any LLM library.
"""
from __future__ import annotations

import asyncio
import difflib
import math
import re

from shutil import get_terminal_size
from typing import Any, Literal

import pandas as pd
import sqlglot

from ..config import SOURCE_TABLE_SEPARATOR
from ..pipeline import Pipeline
from ..sources.base import Source
from ..transforms import SQLRemoveSourceSeparator
from ..util import log
from .sharing import shares_values, sharing

# Rows beyond which a frame is sampled before being profiled. Shared by
# describe_data_sync and lint_data so the summary an LLM reads and the findings
# it is asked to act on are drawn from the same amount of data.
PROFILE_SAMPLE_ROWS = 5000

# Terminal colour codes, which sqlglot puts in its error messages
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# Matches a column name ending in a stem + integer index, with an optional
# separator so all common series conventions are caught: "X_pca_12", "PC1",
# "dim-1", "emb.2". Used to collapse machine-generated numbered column series.
INDEXED_COLUMN_RE = re.compile(r"^(?P<stem>.+?)[_.\-]?(?P<idx>\d+)$")

# Tokenizer used to size prompt payloads. o200k_base is what semchunk resolves
# vector_store's "gpt-4o-mini" default to, so chunking and truncation agree.
TOKEN_ENCODING = "o200k_base"

# Used when the tokenizer cannot be loaded. Deliberately below the ~4 chars/token
# of English prose: dense YAML and aligned numeric tables run nearer 2.5, and
# under-estimating the ratio over-estimates tokens, so we truncate early rather
# than blowing a budget.
FALLBACK_CHARS_PER_TOKEN = 3.0

# Resolved lazily by _get_token_encoder. Key presence (not its value)
# distinguishes "not yet resolved" from "resolved, and unavailable".
_TOKEN_ENCODER_CACHE: dict[str, Any] = {}


def format_float(num):
    """
    Process a float value, returning numeric types instead of strings.
    For very large/small numbers, returns a float that will display in scientific notation.
    """
    if not isinstance(num, (int, float)):
        return num
    if pd.isna(num) or math.isinf(num):
        return num

    # For integers, return as int
    if num == int(num):
        # But if it's a very large integer, keep as float for potential sci notation
        if abs(num) >= 1e5:
            return f"{num:.1e}"  # Format as scientific notation
        return int(num)

    # For normal range numbers, round but keep as float
    elif 0.01 <= abs(num) < 1e6:
        return round(num, 1)

    # For very small or very large numbers, return as-is
    # These will naturally display in scientific notation when converted to string
    else:
        return f"{num:.1e}"  # Format as scientific notation


def format_error(error: BaseException) -> str:
    """Render an error as ``Type: message`` for the model, without terminal colour codes."""
    return ANSI_ESCAPE.sub("", f"{type(error).__name__}: {error}")


# Credentials that error messages commonly quote, e.g. an HTTP error naming
# the request URL with ``?key=...`` or ``https://user:token@host``.
URL_CREDENTIALS = re.compile(r"(?<=://)[^/\s@]+@")
URL_QUERY = re.compile(r"(?P<url>\b[a-z][\w+.-]*://[^\s?#'\"<>]*)\?[^\s#'\"<>)]+", re.IGNORECASE)


def format_tool_error(error: BaseException | str, max_length: int = 1000) -> str:
    """
    Render an error for a tool result that is sent to the provider.

    Tool results are re-sent every round, so the text is truncated, and URL
    credentials and query strings are redacted so secrets do not leak.
    """
    text = error if isinstance(error, str) else format_error(error)
    text = URL_CREDENTIALS.sub("<redacted>@", URL_QUERY.sub(r"\g<url>?<redacted>", text))
    return truncate_string(text, max_length)


def process_enums(spec, num_cols, limit=None, include_enum=True, reduce_enums=True, include_empty_fields=False):
    """
    Process enum values in a schema field specification.

    Parameters
    ----------
    spec : dict
        The specification dictionary for this field
    num_cols : int
        The total number of columns in the schema (used for enum reduction calculation)
    limit : int, optional
        The limit used in schema generation
    include_enum : bool, default True
        Whether to include enum values
    reduce_enums : bool, default True
        Whether to reduce the number of enum values for readability
    include_empty_fields : bool, default False
        Whether to include fields that are empty

    Returns
    -------
    tuple[dict, bool]
        The updated spec dictionary and a boolean indicating if this field is empty
    """
    is_empty = False

    if "enum" not in spec:
        return spec, is_empty

    # Scale max_enums based on average enum length (inverse relationship)
    # Short enums (categories) = show more, long enums (text) = show fewer
    if reduce_enums:
        enum_values = spec["enum"]
        # Calculate average length, excluding None values
        non_none_values = [v for v in enum_values if v is not None]
        if non_none_values:
            avg_length = sum(len(str(v)) for v in non_none_values) / len(non_none_values)
        else:
            avg_length = 0

        # Exponential decay: short enums get more slots, long enums get fewer
        # Formula: base_max = max(2, 10 * e^(-0.025 * avg_length))
        # 20: Maximum base enums (for very short values)
        # 0.025: Decay rate (higher = faster drop-off)
        # max(2, ...): Minimum floor (always show at least 2)
        base_max_enums = max(2, int(10 * math.exp(-0.04 * avg_length)))

        # Still consider number of columns but with less weight
        max_enums = max(2, int(base_max_enums * math.exp(-0.05 * max(0, num_cols - 10))))
    else:
        max_enums = 2
    truncate_limit = min(limit or 2, max_enums)

    if not include_enum or len(spec["enum"]) == 0:
        spec.pop("enum")
    elif len(spec["enum"]) > max_enums:
        # if there are only 10 enums, fine; let's keep them all
        # else, that assumes this column is like glasbey 100+ categories vs category10
        # so truncate to 5 and add "..."
        spec["enum"] = spec["enum"][:truncate_limit] + ["..."]
    elif limit and len(spec["enum"]) == 1 and spec["enum"][0] is None:
        spec["enum"] = [f"(unknown; truncated to {limit} rows)"]
    elif not include_empty_fields and len(spec["enum"]) == 1 and (
        spec["enum"][0] is None or (isinstance(spec["enum"][0], str) and spec["enum"][0].strip() == "")
    ):
        is_empty = True

    # truncate each enum to 100 characters if they exist
    if "enum" in spec:
        spec["enum"] = [
            enum if enum is None or not isinstance(enum, str) or len(enum) < 100 else f"{enum[:100]} ..."
            for enum in spec["enum"]
        ]

    return spec, is_empty


async def get_schema(
    source: Source | Pipeline,
    table: str | None = None,
    include_type: bool = True,
    include_min_max: bool = True,
    include_enum: bool = True,
    include_count: bool = False,
    include_empty_fields: bool = False,
    include_nans: bool = False,
    reduce_enums: bool = True,
    shuffle: bool = True,
    **get_kwargs
):
    if not shares_values():
        include_min_max = include_enum = False
        include_type = include_type and sharing.share != "none"
    if isinstance(source, Pipeline):
        schema = await asyncio.to_thread(source.get_schema)
    else:
        if "limit" not in get_kwargs:
            get_kwargs["limit"] = 100
        schema = await asyncio.to_thread(source.get_schema, table, shuffle=shuffle, **get_kwargs)
    schema = dict(schema)

    # first pop regardless to prevent
    # argument of type 'numpy.int64' is not iterable
    count = schema.pop("__len__", None)

    if not include_type:
        for spec in schema.values():
            if "type" in spec:
                spec.pop("type")

    empty_fields = []
    if include_min_max:
        for field, spec in schema.items():
            min_val = 0
            max_val = 0
            if "inclusiveMinimum" in spec:
                min_val = spec.pop("inclusiveMinimum")
                if isinstance(min_val, (int, float)):
                    min_val = format_float(min_val)
                if include_nans or not pd.isna(min_val):
                    spec["min"] = min_val
            if "inclusiveMaximum" in spec:
                max_val = spec.pop("inclusiveMaximum")
                if isinstance(max_val, (int, float)):
                    max_val = format_float(max_val)
                if include_nans or not pd.isna(max_val):
                    spec["max"] = max_val
            if not include_empty_fields and pd.isna(min_val) and pd.isna(max_val):
                empty_fields.append(field)
                continue
    else:
        for field, spec in schema.items():
            min_val = 0
            max_val = 0
            if "inclusiveMinimum" in spec:
                min_val = spec.pop("inclusiveMinimum")
            if "inclusiveMaximum" in spec:
                max_val = spec.pop("inclusiveMaximum")
            if "min" in spec:
                min_val = spec.pop("min")
            if "max" in spec:
                max_val = spec.pop("max")
            if not include_empty_fields and pd.isna(min_val) and pd.isna(max_val):
                empty_fields.append(field)
                continue

    num_cols = len(schema)
    for field, spec in schema.items():
        if "type" in spec:
            if not include_empty_fields and spec["type"] is None:
                spec.pop("type")
                continue
            if spec["type"] == "string":
                spec["type"] = "str"
            elif spec["type"] == "integer":
                spec["type"] = "int"
            elif spec["type"] == "number":
                spec["type"] = "num"
            elif spec["type"] == "boolean":
                spec["type"] = "bool"

            # If it's obvious that this is a numeric range, remove type
            # Yes `{'min': 2017, 'max': 2020}`  # implicitly infer
            # Not `{'type': 'num', 'min': '2.0e+06', 'max': '6.2e+06'}`  # explicitly show type
            if isinstance(spec.get("min"), (int, float)) and isinstance(spec.get("max"), (int, float)):
                spec.pop("type")

        # Process enums using the extracted function
        if "enum" in spec:
            spec, is_empty = process_enums(
                spec, num_cols,
                limit=get_kwargs.get("limit"),
                include_enum=include_enum,
                reduce_enums=reduce_enums,
                include_empty_fields=include_empty_fields
            )
            if is_empty:
                empty_fields.append(field)
                continue

    # Format any other numeric values in the schema
    for spec in schema.values():
        for key, value in spec.items():
            if isinstance(value, (int, float)) and key not in ['type']:
                spec[key] = format_float(value)

    # remove any fields that are empty
    for field in empty_fields:
        schema[field] = "<null>"

    for field, spec in schema.items():
        if isinstance(spec, dict) and not shares_values(field, spec.get("enum") or ()):
            # A personal-data column keeps its type but not its values
            for key in ("enum", "min", "max"):
                spec.pop(key, None)

    if count is not None and include_count:
        schema["__len__"] = count
    return schema


# The types numpy cannot hold, and what to read them as instead.
_OBJECT_DTYPE_CASTS = {'decimal': 'float64', 'date': 'datetime64[us]'}


def normalize_object_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return df with columns numpy is able to describe.

    numpy has no decimal and no date, so both sit in an object column, which
    anything reading dtypes takes for text: the summary reads it as
    categorical, so a money column loses its mean and range and comes out as
    an enum of decimal.Decimal that yaml.safe_load cannot even read back, and
    the exploration preview reports it as str.

    Inferred from the values rather than the source dtype, because pandas
    produces such a column for a DECIMAL result of its own, so this is not
    only about the frame having arrived from another library.
    """
    casts = {
        col: _OBJECT_DTYPE_CASTS[kind]
        for col in df.columns
        if (kind := pd.api.types.infer_dtype(df[col])) in _OBJECT_DTYPE_CASTS
    }
    return df.astype(casts) if casts else df


def clean_sql(sql_expr: str, dialect: str | None = None, prettify: bool = False) -> str:
    """
    Cleans up a SQL expression generated by an LLM by removing
    backticks, fencing and extraneous space and semi-colons.
    """
    sql_expr = SQLRemoveSourceSeparator.apply_to(sql_expr)
    cleaned_sql = sql_expr.replace("```sql", "").replace("```", "")
    if dialect != 'bigquery':
        cleaned_sql = cleaned_sql.replace('`', '"')
    clean_sql = cleaned_sql.strip().rstrip(";")
    if prettify:
        # 'any' is a legacy BaseSQLSource dialect value that sqlglot no
        # longer accepts (28.x raises 'Unknown dialect any'); normalise it
        # to sqlglot's dialect-agnostic mode so SQLAgent's chat path keeps
        # working for sources that have not declared a concrete dialect.
        sql_dialect = None if dialect == "any" else dialect
        clean_sql = sqlglot.transpile(
            sql_expr,
            read=sql_dialect,
            pretty=True
        )[0]
    return clean_sql


def format_msg_content(content: Any) -> str:
    """
    Format a message content field for logging.

    Handles plain strings, multimodal lists (text + images), and
    arbitrary objects.  Non-string items are represented as compact
    ``Type[id]`` placeholders so that base64 blobs never flood logs.

    Parameters
    ----------
    content : Any
        The message content — a string, a list of mixed items, or any object.

    Returns
    -------
    str
        A human-readable, log-safe representation of the content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(f"<{type(item).__name__}[{id(item)}]>")
        return " + ".join(parts)
    return f"<{type(content).__name__}[{id(content)}]>"


def log_debug(msg: Any, offset: int = 24, prefix: str = "", suffix: str = "", show_sep: Literal["above", "below"] | None = None, show_length: bool = False):
    """
    Log a debug message with a separator line above and below.
    """
    terminal_cols, _ = get_terminal_size()
    terminal_cols -= offset  # Account for the timestamp and log level
    delimiter = "_" * terminal_cols
    if show_sep == "above":
        log.debug(f"\033[90m{delimiter}\033[0m")
    if prefix:
        log.debug(prefix)

    if not isinstance(msg, (str, list)):
        log_debug(format_msg_content(msg))
    elif isinstance(msg, list):
        for m in msg:
            log.debug(m if isinstance(m, str) else format_msg_content(m))
    else:
        log.debug(msg)
    if show_length:
        num_tokens = count_tokens(msg) if isinstance(msg, str) else sum(count_tokens(m) for m in msg)
        log.debug(f"Characters: \033[94m{len(msg)}\033[0m Tokens: \033[94m{num_tokens}\033[0m")
    if suffix:
        log.debug(suffix)
    if show_sep == "below":
        log.debug(f"\033[90m{delimiter}\033[0m")


def slug_to_table_name(slug: str) -> str:
    """Extract just the table name from a source-qualified slug."""
    return slug.split(SOURCE_TABLE_SEPARATOR, 1)[-1]


def find_slug_by_table_name(table_name: str, candidates: dict | list) -> str | None:
    """Find a slug in *candidates* whose table-name suffix matches *table_name*."""
    keys = candidates if isinstance(candidates, (list, set, frozenset)) else candidates.keys()
    return next(
        (k for k in keys if slug_to_table_name(k) == table_name),
        None,
    )


def parse_table_slug(table: str, sources: list[Source], normalize: bool = True) -> tuple[Source | None, str]:
    if SOURCE_TABLE_SEPARATOR in table:
        a_source_name, a_table = table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
        a_source_obj = next((s for s in sources if s.name == a_source_name), None)
    else:
        a_source_obj = sources[0] if sources else None
        a_table = table
    if normalize:
        a_table = a_source_obj.normalize_table(a_table)
    return a_source_obj, a_table


def truncate_string(s, max_length=30, ellipsis="..."):
    if len(s) <= max_length:
        return s
    part_length = (max_length - len(ellipsis)) // 2
    return f"{s[:part_length]}{ellipsis}{s[-part_length:]}"


def closest_names(name: str, options, n: int = 3) -> list[str]:
    """Return up to ``n`` entries of ``options`` that most resemble ``name``, ignoring case."""
    by_lower: dict[str, str] = {}
    for option in options:
        by_lower.setdefault(str(option).lower(), str(option))
    matches = difflib.get_close_matches(str(name).lower(), list(by_lower), n=n, cutoff=0.5)
    return [by_lower[match] for match in matches]


def format_unknown_name(kind: str, name: Any, options, max_listed: int = 30) -> str:
    """
    Describe an unknown ``kind`` (tool, table, column, ...) so a model can correct itself.

    Lists the closest matches first, then the valid names, capped at
    ``max_listed`` so a large catalog does not flood the transcript.
    """
    options = sorted({str(option) for option in options})
    message = f"Unknown {kind} {name!r}."
    close = closest_names(str(name), options)
    if close:
        message += f" Closest matches: {', '.join(close)}."
    if options:
        listed = ", ".join(options[:max_listed])
        more = f" (and {len(options) - max_listed} more)" if len(options) > max_listed else ""
        message += f" Valid {kind}s: {listed}{more}."
    return message


def _get_token_encoder():
    """
    Return a cached tiktoken encoder, or ``None`` if one cannot be loaded.

    Resolution is deferred to first use: ``tiktoken.get_encoding`` downloads the
    BPE vocabulary over HTTPS unless it is already in ``$TIKTOKEN_CACHE_DIR``,
    so importing this module must not trigger it. Any failure (no tiktoken,
    offline, upstream 5xx) is logged once and yields ``None``, which callers
    treat as "estimate from character length" — sizing a prompt payload is never
    worth raising into an in-flight conversation.
    """
    if "encoder" in _TOKEN_ENCODER_CACHE:
        return _TOKEN_ENCODER_CACHE["encoder"]
    try:
        # Deferred so a missing tiktoken degrades to the character estimate
        # rather than breaking the import.
        import tiktoken

        encoder = tiktoken.get_encoding(TOKEN_ENCODING)
    except Exception as e:
        log.warning(
            f"Could not load the {TOKEN_ENCODING!r} tokenizer ({type(e).__name__}: {e}); "
            f"token counts will be estimated from character length at "
            f"~{FALLBACK_CHARS_PER_TOKEN} chars/token."
        )
        encoder = None
    _TOKEN_ENCODER_CACHE["encoder"] = encoder
    return encoder


def count_tokens(text: str) -> int:
    """
    Estimate how many tokens *text* occupies in an LLM prompt.

    Counts with the :data:`TOKEN_ENCODING` tokenizer. Lumen also targets
    Anthropic, Google, Mistral, Bedrock and LiteLLM models, whose tokenizers
    differ, so this is an estimate for every provider — use it to size payloads,
    not to predict billing. Falls back to a character-length estimate when the
    tokenizer is unavailable.
    """
    if not text:
        return 0
    encoder = _get_token_encoder()
    if encoder is None:
        return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)
    return len(encoder.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int, marker: str = "truncated") -> str:
    """
    Trim *text* to roughly *max_tokens*, cutting on a line boundary.

    Prompt payloads are YAML, whitespace-aligned tables and markdown, where a
    mid-line cut leaves a fragment the model may misread as data — so the text
    is cut back to the last complete line that fits. The appended note reports
    what was dropped, letting the model distinguish "this is everything" from
    "there is more", rather than inferring it from a bare ellipsis.

    Parameters
    ----------
    text : str
        Text to trim.
    max_tokens : int
        Approximate token budget, per :func:`count_tokens`.
    marker : str
        Word used in the appended note.

    Returns
    -------
    str
        *text* unchanged when it already fits, otherwise the leading lines that
        fit followed by ``... (truncated, showing N of M tokens)``.
    """
    total = count_tokens(text)
    if total <= max_tokens:
        return text

    # Cut proportionally on characters first, then walk back a line at a time
    # until the result plus its note fits. The first guess is usually right;
    # the loop only corrects for uneven token density within the text.
    note_budget = count_tokens(f"\n... ({marker}, showing {max_tokens} of {total} tokens)")
    budget = max(max_tokens - note_budget, 1)
    lines = text.split("\n")
    keep = max(1, int(len(lines) * budget / total))
    while keep > 1 and count_tokens("\n".join(lines[:keep])) > budget:
        keep -= 1
    kept = "\n".join(lines[:keep])
    if count_tokens(kept) > budget:
        # A single line exceeds the budget on its own; fall back to a character
        # cut so we still respect the cap.
        kept = kept[: max(1, int(len(kept) * budget / count_tokens(kept)))]
    shown = count_tokens(kept)
    return f"{kept}\n... ({marker}, showing {shown} of {total} tokens)"


def collapse_indexed_columns(
    names: list[str], min_series: int = 8, max_gaps: int = 5
) -> list[str]:
    """
    Collapse machine-generated numbered column series into a single entry.

    A *series* is a group of columns that share a stem and differ only by a
    trailing integer index (e.g. ``X_pca_0 … X_pca_99`` or ``PCs_0 … PCs_99``),
    such as embedding/PCA matrices or one-hot expansions. Individually the names
    carry no meaning, so listing every one wastes prompt budget without adding
    signal. A run of at least *min_series* members collapses; a near-complete
    run is still collapsed and its missing indices are named. Genuinely distinct
    names, short runs, and sparse/heavily-gapped runs (which includes any
    step > 1 series) are returned unchanged and in their original position.

    Parameters
    ----------
    names : list[str]
        Column names in their original order.
    min_series : int
        Minimum present members before a stem is collapsed. Smaller runs stay
        expanded since the token savings don't justify hiding the columns.
    max_gaps : int
        Most missing indices tolerated within a run's span. A run with more
        holes than this stays expanded, since the gaps are likely meaningful
        (and too numerous to name compactly). This also excludes step > 1
        series, whose "gaps" exceed the budget.

    Returns
    -------
    list[str]
        Column names with each qualifying series replaced by a single
        ``"{first}..{last} ({n} cols)"`` entry — with ``", missing <indices>"``
        appended when the run has holes. The real first/last column names are
        used, so the original separator and any zero-padding are preserved. The
        entry sits where the series first appeared.

    Examples
    --------
    >>> collapse_indexed_columns(["obs_id", "PCs_0", "PCs_1", "PCs_2"], min_series=2)
    ['obs_id', 'PCs_0..PCs_2 (3 cols)']
    >>> collapse_indexed_columns(["x_0", "x_1", "x_3"], min_series=2)
    ['x_0..x_3 (3 cols, missing 2)']
    >>> collapse_indexed_columns(["gender", "age", "smoking_status"])
    ['gender', 'age', 'smoking_status']
    """
    stem_members: dict[str, list[tuple[int, str]]] = {}
    name_stem: dict[str, str] = {}
    for name in names:
        match = INDEXED_COLUMN_RE.match(name)
        if match:
            stem = match.group("stem")
            stem_members.setdefault(stem, []).append((int(match.group("idx")), name))
            name_stem[name] = stem

    collapsible: dict[str, str] = {}
    for stem, members in stem_members.items():
        indices = [idx for idx, _ in members]
        # Skip small runs, and any with duplicate indices (not a clean series).
        if len(indices) < min_series or len(set(indices)) != len(indices):
            continue
        lo, hi = min(indices), max(indices)
        # Collapse a contiguous run, or a near-complete one whose few holes we
        # can name. A sparse or heavily-gapped run (including any step > 1
        # series) exceeds max_gaps and stays expanded — its holes may matter.
        missing = sorted(set(range(lo, hi + 1)) - set(indices))
        if len(missing) > max_gaps:
            continue
        # Label from the real endpoint names so the source's separator and any
        # zero-padding (``_``, ``-``, ``.`` or none) are preserved.
        lo_name = next(name for idx, name in members if idx == lo)
        hi_name = next(name for idx, name in members if idx == hi)
        label = f"{lo_name}..{hi_name} ({len(indices)} cols"
        if missing:
            label += f", missing {', '.join(map(str, missing))}"
        collapsible[stem] = label + ")"

    result: list[str] = []
    emitted: set[str] = set()
    for name in names:
        stem = name_stem.get(name)
        if stem in collapsible:
            if stem not in emitted:
                result.append(collapsible[stem])
                emitted.add(stem)
        else:
            result.append(name)
    return result
