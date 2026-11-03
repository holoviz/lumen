"""
How much of the data Lumen AI shows the LLM.

``share`` sets the level for every prompt: ``"sample"`` sends row values,
value counts, ranges and chart images, as Lumen always has; ``"schema"``
sends table and column names, types, row counts and null fractions, but no
values; ``"none"`` sends table and column names only. Queries still run on
the full data locally, so answers, tables and charts are unaffected; only
what the model reads to write SQL and explanations changes.

``mask_pii`` withholds the values of columns that look like personal data
(emails, phone numbers, names, addresses, ID numbers) even at ``"sample"``.

Both start from the ``LUMEN_AI_SHARE`` and ``LUMEN_AI_MASK_PII`` environment
variables and can be set with ``lumen.ai.configure``.
"""
from __future__ import annotations

import os
import re

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import pandas as pd
import param

SHARE_LEVELS = ("sample", "schema", "none")

MASK = "<masked>"

# Column names that hold personal data, matched against the name's words,
# e.g. "customer_email" or "BillingAddress"
_PII_NAME_WORDS = {
    "address", "birthdate", "birthday", "dob", "email", "firstname", "fullname",
    "iban", "lastname", "mobile", "passport", "phone", "postcode", "ssn",
    "street", "surname", "telephone", "zipcode",
}
_PII_NAME_PAIRS = {
    ("birth", "date"), ("credit", "card"), ("card", "number"), ("date", "birth"), ("of", "birth"), ("first", "name"),
    ("full", "name"), ("ip", "address"), ("last", "name"), ("social", "security"),
    ("zip", "code"), ("postal", "code"),
}
_PII_VALUE = re.compile(
    r"^\s*(?:"
    r"[\w.+-]+@[\w-]+\.[\w.-]+"            # email
    r"|\+?\d[\d\s().-]{7,}\d"              # phone number
    r"|\d{3}-\d{2}-\d{4}"                  # US social security number
    r"|(?:\d{1,3}\.){3}\d{1,3}"            # IPv4 address
    r")\s*$"
)


class DataSharing(param.Parameterized):

    share = param.Selector(default=os.environ.get("LUMEN_AI_SHARE", "sample"), objects=list(SHARE_LEVELS), doc="""
        What of the data prompts may contain: "sample" (values), "schema"
        (structure without values) or "none" (names only).""")

    mask_pii = param.Boolean(default=os.environ.get("LUMEN_AI_MASK_PII", "").lower() in ("1", "true"), doc="""
        Withhold the values of columns that look like personal data.""")


sharing = DataSharing()


def _name_words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return [word for word in re.split(r"[^a-z0-9]+", spaced.lower()) if word]


def looks_like_pii(column: str, values: Iterable[Any] = ()) -> bool:
    """Whether a column's name, or most of the given values, look like personal data."""
    words = _name_words(column)
    if _PII_NAME_WORDS & set(words) or _PII_NAME_WORDS & {"".join(words)}:
        return True
    if any(pair in _PII_NAME_PAIRS for pair in pairwise(words)):
        return True
    strings = [value for value in values if isinstance(value, str)][:50]
    return bool(strings) and sum(bool(_PII_VALUE.match(value)) for value in strings) > len(strings) / 2


def shares_values(column: str | None = None, values: Iterable[Any] = ()) -> bool:
    """Whether row values, of one column if given, may appear in a prompt."""
    if sharing.share != "sample":
        return False
    return column is None or not (sharing.mask_pii and looks_like_pii(column, values))


def mask_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of df with the values of personal-data columns masked, if masking is on."""
    if not sharing.mask_pii:
        return df
    sample = df.head(50)
    masked = [
        col for col in df.columns
        if looks_like_pii(col, sample[col].dropna().tolist() if sample[col].dtype == object else ())
    ]
    if not masked:
        return df
    df = df.copy()
    df[masked] = MASK
    return df
