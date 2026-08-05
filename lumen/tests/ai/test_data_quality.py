import numpy as np
import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.data_quality import lint_data
from lumen.ai.utils import PROFILE_SAMPLE_ROWS


class TestLintData:
    """Deterministic data-quality findings used to drive SQLAgent's cleaning pass."""

    def test_clean_frame_reports_nothing(self):
        df = pd.DataFrame({
            "id": range(20),
            "category": ["a", "b"] * 10,
            "value": [float(i) for i in range(20)],
        })
        assert lint_data(df) == []

    def test_empty_frame_reports_nothing(self):
        assert lint_data(pd.DataFrame({"a": [], "b": []})) == []

    def test_nulls_reported_with_percentage(self):
        df = pd.DataFrame({"value": [1.0] * 90 + [None] * 10, "other": range(100)})
        findings = lint_data(df)
        assert any('Missing values' in f and '"value" (10.0% null)' in f for f in findings)

    def test_nulls_below_threshold_ignored(self):
        """One missing value in a thousand is not worth an extra LLM round trip."""
        df = pd.DataFrame({"value": [1.0] * 999 + [None]})
        assert not any("Missing values" in f for f in lint_data(df))

    def test_duplicate_rows_reported(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        findings = lint_data(df)
        assert any("1 of 4 rows are exact duplicates" in f for f in findings)

    def test_sentinel_numbers_reported(self):
        df = pd.DataFrame({"temp": [12.5, 13.0, -9999.0, 14.0, -9999.0, 15.0]})
        findings = lint_data(df)
        assert any("Placeholder numbers" in f and '"temp" (2 rows)' in f for f in findings)

    def test_untrimmed_and_empty_text_reported(self):
        df = pd.DataFrame({"name": ["alice", " bob ", "", "dave"]})
        findings = lint_data(df)
        assert any("Untrimmed or empty text" in f and "1 padded, 1 empty" in f for f in findings)

    def test_numbers_stored_as_text_reported(self):
        df = pd.DataFrame({"amount": ["1.5", "2.5", "3.5", "4.5"]})
        findings = lint_data(df)
        assert any("Numbers stored as text" in f and '"amount"' in f for f in findings)

    def test_genuine_text_not_flagged_as_numeric(self):
        df = pd.DataFrame({"name": ["alice", "bob", "carol", "dave"]})
        assert not any("Numbers stored as text" in f for f in lint_data(df))

    def test_constant_column_reported(self):
        df = pd.DataFrame({"region": ["north"] * 10, "value": range(10)})
        findings = lint_data(df)
        assert any("Constant column" in f and '"region"' in f for f in findings)

    def test_single_row_frame_reports_no_constant_columns(self):
        """Every column of a one-row result is trivially constant."""
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        assert not any("Constant column" in f for f in lint_data(df))

    def test_iqr_outliers_reported(self):
        df = pd.DataFrame({"revenue": [float(i) for i in range(50)] + [100000.0]})
        findings = lint_data(df)
        assert any("IQR outliers" in f and '"revenue" (1 outside' in f for f in findings)

    def test_degenerate_iqr_column_reports_no_outliers(self):
        """A flag-shaped column has a zero IQR, where Tukey fences flag every
        minority value; skipping it is deliberate, not an oversight."""
        df = pd.DataFrame({"is_active": [0.0] * 50 + [1.0]})
        assert not any("IQR outliers" in f for f in lint_data(df))

    def test_uniform_numeric_column_has_no_outliers(self):
        df = pd.DataFrame({"value": [float(i) for i in range(100)]})
        assert not any("IQR outliers" in f for f in lint_data(df))

    def test_ordinary_random_data_reports_nothing(self):
        """The property the cleaning pass depends on: clean data costs no LLM call.
        At the usual 1.5 IQR fence a gaussian column always reports outliers."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "measure": rng.normal(size=5000),
            "count": rng.integers(0, 100, size=5000),
            "label": rng.choice(["alpha", "beta", "gamma"], size=5000),
        })
        assert lint_data(df) == []

    def test_large_frame_notes_that_counts_are_sampled(self):
        df = pd.DataFrame({
            "value": [1.0] * (PROFILE_SAMPLE_ROWS * 2),
            "flag": [None] * (PROFILE_SAMPLE_ROWS * 2),
        })
        findings = lint_data(df)
        assert findings
        assert findings[-1].startswith(f"Counts above come from a random {PROFILE_SAMPLE_ROWS}-row sample")

    def test_constant_and_outlier_findings_are_not_actionable(self):
        """Filtering to one region then grouping is an ordinary query, not a defect:
        it must be reported without provoking a rewrite that drops the column."""
        df = pd.DataFrame({
            "region": ["north"] * 40,
            "month": list(range(40)),
            "revenue": [100.0 + i for i in range(40)],
        })
        assert any("Constant column" in f for f in lint_data(df))
        assert lint_data(df, actionable_only=True) == []

    def test_actionable_findings_still_reported(self):
        df = pd.DataFrame({"name": ["alice", " bob ", "", "dave"]})
        assert any("Untrimmed" in f for f in lint_data(df, actionable_only=True))

    def test_unhashable_values_do_not_raise(self):
        """A geometry or list column must not break a query that already succeeded."""
        df = pd.DataFrame({"geom": [[1, 2], [3, 4], [1, 2]], "value": [1.0, 2.0, 3.0]})
        assert isinstance(lint_data(df), list)

    def test_findings_cap_the_columns_they_name(self):
        df = pd.DataFrame({f"c{i}": ["  padded  "] * 5 for i in range(20)})
        finding = next(f for f in lint_data(df) if "Untrimmed" in f)
        assert "and 12 more columns" in finding

    def test_every_column_is_checked_even_when_not_all_are_named(self):
        """The cap is on the finding's wording, not on how much is inspected:
        all 20 columns are counted, only the first 8 are named."""
        df = pd.DataFrame({f"c{i}": [1.0] * 90 + [None] * 10 for i in range(20)})
        finding = next(f for f in lint_data(df) if "Missing values" in f)
        assert finding.startswith("Missing values in 20 column(s):")
        assert "and 12 more columns" in finding
