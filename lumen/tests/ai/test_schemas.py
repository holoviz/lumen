"""Tests for lumen.ai.schemas — Metaset rendering."""
import pytest
import yaml

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.schemas import (
    Column, DocumentChunk, Metaset, TableCatalogEntry,
)
from lumen.config import SOURCE_TABLE_SEPARATOR

SEP = SOURCE_TABLE_SEPARATOR


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _entry(slug, *, derived_from=None, created_order=0, similarity=1.0,
           columns=None, description=None, sql_expr=None, metadata=None):
    return TableCatalogEntry(
        table_slug=slug,
        similarity=similarity,
        columns=columns or [],
        derived_from=derived_from or [],
        created_order=created_order,
        description=description,
        sql_expr=sql_expr,
        metadata=metadata or {},
    )


def _metaset(entries, schemas=None, docs=None, schema_tables=None):
    catalog = {e.table_slug: e for e in entries}
    return Metaset(
        query=None, catalog=catalog, schemas=schemas,
        docs=docs, schema_tables=schema_tables,
    )


def _col(name, description=None):
    return Column(name=name, description=description)


# ---------------------------------------------------------------
# Properties: has_schemas, has_docs, get_docs
# ---------------------------------------------------------------

class TestMetasetProperties:

    def test_has_schemas_false_when_none(self):
        ms = _metaset([_entry(f"S{SEP}t")])
        assert ms.has_schemas is False

    def test_has_schemas_false_when_empty(self):
        ms = _metaset([_entry(f"S{SEP}t")], schemas={})
        assert ms.has_schemas is False

    def test_has_schemas_true(self):
        ms = _metaset([_entry(f"S{SEP}t")], schemas={f"S{SEP}t": {"col": {"type": "int"}}})
        assert ms.has_schemas is True

    def test_has_docs_false_when_none(self):
        ms = _metaset([_entry(f"S{SEP}t")])
        assert ms.has_docs is False

    def test_has_docs_true(self):
        doc = DocumentChunk(filename="f.md", text="hello", similarity=0.9)
        ms = _metaset([_entry(f"S{SEP}t")], docs=[doc])
        assert ms.has_docs is True

    def test_get_docs_returns_empty_when_none(self):
        ms = _metaset([_entry(f"S{SEP}t")])
        assert ms.get_docs() == []

    def test_get_docs_returns_docs(self):
        doc = DocumentChunk(filename="f.md", text="hello", similarity=0.9)
        ms = _metaset([_entry(f"S{SEP}t")], docs=[doc])
        assert ms.get_docs() == [doc]


# ---------------------------------------------------------------
# get_top_tables
# ---------------------------------------------------------------

class TestGetTopTables:

    def test_returns_all_sorted_by_similarity(self):
        a = _entry(f"S{SEP}low", similarity=0.2)
        b = _entry(f"S{SEP}high", similarity=0.9)
        c = _entry(f"S{SEP}mid", similarity=0.5)
        ms = _metaset([a, b, c])
        assert ms.get_top_tables() == [f"S{SEP}high", f"S{SEP}mid", f"S{SEP}low"]

    def test_n_limits_results(self):
        entries = [_entry(f"S{SEP}t{i}", similarity=i / 10) for i in range(10)]
        ms = _metaset(entries)
        assert len(ms.get_top_tables(n=3)) == 3

    def test_offset_skips(self):
        a = _entry(f"S{SEP}first", similarity=1.0)
        b = _entry(f"S{SEP}second", similarity=0.5)
        ms = _metaset([a, b])
        result = ms.get_top_tables(offset=1)
        assert result == [f"S{SEP}second"]

    def test_n_and_offset(self):
        entries = [_entry(f"S{SEP}t{i}", similarity=1.0 - i * 0.1) for i in range(5)]
        ms = _metaset(entries)
        result = ms.get_top_tables(n=2, offset=1)
        assert len(result) == 2
        assert f"S{SEP}t0" not in result  # skipped by offset


# ---------------------------------------------------------------
# schema_tables controls which tables get full detail
# ---------------------------------------------------------------

class TestSchemaTables:

    def test_schema_tables_limits_primary_output(self):
        a = _entry(f"S{SEP}shown", columns=[_col("x")])
        b = _entry(f"S{SEP}hidden", columns=[_col("y")])
        ms = _metaset([a, b], schema_tables=[f"S{SEP}shown"])
        output = ms.table_context(n_others=10)
        parsed_lines = output.strip().split("\n")
        # "shown" should be a primary table with column detail
        assert any("x" in line for line in parsed_lines)
        # "hidden" should only appear in "Others available"
        assert "Others available:" in output
        assert "hidden" in output

    def test_schema_tables_none_uses_similarity(self):
        high = _entry(f"S{SEP}high", similarity=1.0, columns=[_col("a")])
        low = _entry(f"S{SEP}low", similarity=0.1, columns=[_col("b")])
        ms = _metaset([high, low])
        output = ms.table_context(n=1, n_others=10)
        # high-similarity table should be primary
        assert "a" in output.split("Others available")[0]

    def test_schema_tables_empty_list_shows_no_primaries(self):
        a = _entry(f"S{SEP}tbl", columns=[_col("x")])
        ms = _metaset([a], schema_tables=[])
        output = ms.table_list(n_others=10)
        assert "Others available:" in output
        assert "tbl" in output


# ---------------------------------------------------------------
# Column rendering
# ---------------------------------------------------------------

class TestColumnRendering:

    def test_table_context_includes_column_names(self):
        e = _entry(f"S{SEP}t", columns=[_col("alpha"), _col("beta")])
        ms = _metaset([e])
        output = ms.table_context()
        assert "alpha" in output
        assert "beta" in output

    def test_table_list_omits_columns(self):
        e = _entry(f"S{SEP}t", columns=[_col("alpha"), _col("beta")])
        ms = _metaset([e])
        output = ms.table_list()
        # table_list has include_columns=False
        assert "alpha" not in output

    def test_compact_context_shows_schema_detail(self):
        slug = f"S{SEP}t"
        e = _entry(slug, columns=[_col("price")])
        schemas = {slug: {"price": {"type": "num", "min": 1, "max": 100}, "__len__": 50}}
        ms = _metaset([e], schemas=schemas)
        output = ms.compact_context()
        assert "price" in output
        assert "n_rows" in output


# ---------------------------------------------------------------
# Schema row count (n_rows)
# ---------------------------------------------------------------

class TestRowCount:

    def test_n_rows_shown_when_schema_has_len(self):
        slug = f"S{SEP}t"
        e = _entry(slug)
        # n_rows is computed as len(schema_dict) which counts keys
        # (column entries + __len__), not the __len__ value itself.
        schemas = {slug: {"col": {"type": "int"}, "__len__": 42}}
        ms = _metaset([e], schemas=schemas)
        output = ms.compact_context()
        parsed = yaml.safe_load(output)
        assert "n_rows" in parsed["t"]
        assert parsed["t"]["n_rows"] == len(schemas[slug])

    def test_n_rows_absent_without_schema(self):
        e = _entry(f"S{SEP}t")
        ms = _metaset([e])
        output = ms.table_list(include_lineage=True)
        assert "n_rows" not in output


# ---------------------------------------------------------------
# Description rendering
# ---------------------------------------------------------------

class TestDescriptionRendering:

    def test_description_shown_as_info(self):
        e = _entry(f"S{SEP}t", description="Sales data by region")
        ms = _metaset([e])
        output = ms.table_context()
        assert "Sales data by region" in output

    def test_no_description_no_info_key(self):
        e = _entry(f"S{SEP}t")
        ms = _metaset([e])
        output = ms.table_context()
        assert "info" not in output


# ---------------------------------------------------------------
# Documentation rendering
# ---------------------------------------------------------------

class TestDocRendering:

    def test_docs_appended_to_output(self):
        e = _entry(f"S{SEP}t")
        doc = DocumentChunk(filename="guide.md", text="Use this table for analysis", similarity=0.8)
        ms = _metaset([e], docs=[doc])
        output = ms.table_list(include_docs=True)
        assert "<documentation>" in output
        assert "guide.md" in output
        assert "Use this table for analysis" in output

    def test_docs_omitted_when_flag_false(self):
        e = _entry(f"S{SEP}t")
        doc = DocumentChunk(filename="guide.md", text="text", similarity=0.8)
        ms = _metaset([e], docs=[doc])
        output = ms.table_list(include_docs=False)
        assert "<documentation>" not in output


# ---------------------------------------------------------------
# Source display: single vs multiple sources
# ---------------------------------------------------------------

class TestSourceDisplay:

    def test_single_source_hides_prefix_by_default(self):
        e = _entry(f"MySource{SEP}tbl")
        ms = _metaset([e])
        output = ms.table_list()
        assert "MySource" not in output
        assert "tbl" in output

    def test_show_source_true_includes_prefix(self):
        e = _entry(f"MySource{SEP}tbl")
        ms = _metaset([e])
        output = ms.table_list(show_source=True)
        assert "MySource" in output

    def test_multiple_sources_show_prefix_by_default(self):
        a = _entry(f"SrcA{SEP}tbl")
        b = _entry(f"SrcB{SEP}tbl2")
        ms = _metaset([a, b])
        output = ms.table_list()
        assert "SrcA" in output
        assert "SrcB" in output


# ---------------------------------------------------------------
# n_others parameter
# ---------------------------------------------------------------

class TestNOthers:

    def test_n_others_limits_others_section(self):
        entries = [_entry(f"S{SEP}t{i}") for i in range(10)]
        ms = _metaset(entries, schema_tables=[f"S{SEP}t0"])
        output = ms.table_list(n_others=3)
        others_section = output.split("Others available:")[-1] if "Others available:" in output else ""
        # Count lines starting with "- "
        others_count = sum(1 for line in others_section.strip().split("\n") if line.startswith("- "))
        assert others_count == 3

    def test_n_others_zero_hides_section(self):
        a = _entry(f"S{SEP}shown")
        b = _entry(f"S{SEP}hidden")
        ms = _metaset([a, b], schema_tables=[f"S{SEP}shown"])
        output = ms.table_list(n_others=0)
        assert "Others available:" not in output


# ---------------------------------------------------------------
# Empty catalog
# ---------------------------------------------------------------

class TestEmptyCatalog:

    def test_table_list_empty(self):
        ms = _metaset([])
        assert "No data sources" in ms.table_list()

    def test_table_context_empty(self):
        ms = _metaset([])
        assert "No data sources" in ms.table_context()

    def test_compact_context_empty(self):
        ms = _metaset([])
        assert "No data sources" in ms.compact_context()

    def test_full_context_empty(self):
        ms = _metaset([])
        assert "No data sources" in ms.full_context()

    def test_str_empty(self):
        ms = _metaset([])
        assert "No data sources" in str(ms)


# ---------------------------------------------------------------
# Lineage in rendered output
# ---------------------------------------------------------------

class TestLineageRendering:
    """Verify that lineage annotations appear (or don't) in rendered context."""

    @pytest.fixture
    def metaset_with_lineage(self):
        raw = _entry(f"Src{SEP}data_csv")
        derived = _entry(
            f"Src{SEP}avg_by_season",
            derived_from=[f"Src{SEP}data_csv"],
            created_order=1,
        )
        return _metaset([raw, derived])

    def test_table_list_includes_lineage_when_requested(self, metaset_with_lineage):
        output = metaset_with_lineage.table_list(include_lineage=True)
        assert "derived_from" in output
        assert "data_csv" in output

    def test_table_list_omits_lineage_by_default(self, metaset_with_lineage):
        output = metaset_with_lineage.table_list()
        assert "derived_from" not in output

    def test_compact_context_includes_lineage_by_default(self, metaset_with_lineage):
        ms = metaset_with_lineage
        ms.schemas = {
            f"Src{SEP}data_csv": {"__len__": 100},
            f"Src{SEP}avg_by_season": {"__len__": 12},
        }
        output = ms.compact_context()
        assert "derived_from" in output

    def test_latest_marker_on_highest_created_order(self, metaset_with_lineage):
        output = metaset_with_lineage.table_list(include_lineage=True)
        parsed = yaml.safe_load(output)
        assert parsed["avg_by_season"]["latest"] is True
        assert "latest" not in parsed.get("data_csv", {})

    def test_step_included_in_primary_table_yaml(self, metaset_with_lineage):
        output = metaset_with_lineage.table_list(include_lineage=True)
        parsed = yaml.safe_load(output)
        assert parsed["avg_by_season"]["step"] == 1

    def test_others_available_shows_lineage_hint(self):
        raw = _entry(f"Src{SEP}data_csv")
        d1 = _entry(f"Src{SEP}derived_a", derived_from=[f"Src{SEP}data_csv"], created_order=1)
        d2 = _entry(f"Src{SEP}derived_b", derived_from=[f"Src{SEP}derived_a"], created_order=2)
        ms = _metaset([raw, d1, d2])
        ms.schema_tables = [f"Src{SEP}data_csv"]
        output = ms.table_list(include_lineage=True, n_others=10)
        assert "(from data_csv)" in output
        assert "(from derived_a)" in output
        assert "★" in output

    def test_others_available_no_lineage_hint_when_disabled(self):
        raw = _entry(f"Src{SEP}data_csv")
        d1 = _entry(f"Src{SEP}derived_a", derived_from=[f"Src{SEP}data_csv"], created_order=1)
        ms = _metaset([raw, d1])
        ms.schema_tables = [f"Src{SEP}data_csv"]
        output = ms.table_list(include_lineage=False, n_others=10)
        assert "(from" not in output
        assert "★" not in output

    def test_original_table_has_no_lineage_keys_even_when_enabled(self):
        raw = _entry(f"Src{SEP}sales")
        ms = _metaset([raw])
        output = ms.table_list(include_lineage=True)
        assert "derived_from" not in output
        assert "step" not in output
        assert "latest" not in output

    def test_lineage_strips_source_prefix_from_derived_from(self):
        raw = _entry(f"LongSourceName{SEP}data")
        derived = _entry(
            f"LongSourceName{SEP}agg",
            derived_from=[f"LongSourceName{SEP}data"],
            created_order=1,
        )
        ms = _metaset([raw, derived])
        output = ms.table_list(include_lineage=True)
        parsed = yaml.safe_load(output)
        # derived_from should show bare table name, not the full slug
        assert parsed["agg"]["derived_from"] == ["data"]


# ---------------------------------------------------------------
# Deduplication — stale source entries filtered from output
# ---------------------------------------------------------------

class TestDeduplication:

    def test_duplicate_table_under_two_sources_shows_only_newest(self):
        old = _entry(f"SrcOld{SEP}my_table", created_order=1)
        new = _entry(f"SrcNew{SEP}my_table", created_order=2)
        ms = _metaset([old, new])
        output = ms.table_list(include_lineage=True, show_source=True)
        assert "SrcNew" in output
        assert "SrcOld" not in output

    def test_dedup_keeps_higher_created_order(self):
        a = _entry(f"S1{SEP}t", created_order=0)
        b = _entry(f"S2{SEP}t", created_order=3)
        c = _entry(f"S3{SEP}t", created_order=1)
        ms = _metaset([a, b, c])
        output = ms.table_list(show_source=True)
        assert "S2" in output
        assert "S1" not in output
        assert "S3" not in output

    def test_no_false_dedup_when_names_differ(self):
        a = _entry(f"Src{SEP}alpha")
        b = _entry(f"Src{SEP}beta")
        ms = _metaset([a, b])
        output = ms.table_list()
        assert "alpha" in output
        assert "beta" in output

    def test_others_available_also_deduplicated(self):
        raw = _entry(f"SrcNew{SEP}raw_data")
        old_d = _entry(f"SrcOld{SEP}derived", derived_from=[f"SrcOld{SEP}raw_data"], created_order=1)
        new_d = _entry(f"SrcNew{SEP}derived", derived_from=[f"SrcNew{SEP}raw_data"], created_order=2)
        ms = _metaset([raw, old_d, new_d])
        ms.schema_tables = [f"SrcNew{SEP}raw_data"]
        output = ms.table_list(include_lineage=True, n_others=10, show_source=True)
        assert output.count("derived") == 1

    def test_schema_tables_stale_slug_filtered(self):
        """schema_tables referencing a stale slug should not appear."""
        old = _entry(f"SrcOld{SEP}tbl", created_order=0)
        new = _entry(f"SrcNew{SEP}tbl", created_order=1)
        ms = _metaset([old, new], schema_tables=[f"SrcOld{SEP}tbl"])
        output = ms.table_list(show_source=True)
        # The stale slug is not an active slug, so it should be filtered
        assert "SrcOld" not in output


# ---------------------------------------------------------------
# _resolve_table_slug
# ---------------------------------------------------------------

class TestResolveTableSlug:

    def test_resolves_bare_name(self):
        ms = _metaset([_entry(f"Src{SEP}my_table")])
        assert ms._resolve_table_slug("my_table") == f"Src{SEP}my_table"

    def test_returns_exact_slug(self):
        slug = f"Src{SEP}my_table"
        ms = _metaset([_entry(slug)])
        assert ms._resolve_table_slug(slug) == slug

    def test_returns_none_for_unknown(self):
        ms = _metaset([_entry(f"Src{SEP}other")])
        assert ms._resolve_table_slug("missing") is None

    def test_returns_none_for_none_input(self):
        ms = _metaset([_entry(f"Src{SEP}tbl")])
        assert ms._resolve_table_slug(None) is None


# ---------------------------------------------------------------
# Realistic rendered output
# ---------------------------------------------------------------

class TestRealisticOutput:
    """Verify rendered YAML against a realistic multi-column derived table."""

    @pytest.fixture
    def oni_metaset(self):
        raw_slug = f"Src{SEP}data_oni_csv"
        derived_slug = f"Src{SEP}aggregated_metrics_by_season"
        raw = _entry(
            raw_slug,
            columns=[
                _col("season"), _col("year"), _col("sst_c"),
                _col("anom_c"), _col("cumulative"), _col("ntotal"), _col("oni"),
            ],
        )
        derived = _entry(
            derived_slug,
            derived_from=[raw_slug],
            created_order=1,
            columns=[
                _col("season"), _col("count_records"), _col("avg_sst_c"),
                _col("avg_anom_c"), _col("sum_cumulative"), _col("sum_ntotal"),
            ],
        )
        schemas = {
            raw_slug: {
                "season": {"type": "str", "enum": ["AMJ", "OND", "ASO", "SON", "DJF", "NDJ", "FMA", "JFM", "..."]},
                "year": {"min": 1950, "max": 2024},
                "sst_c": {"min": 24.6, "max": 29.2},
                "anom_c": {"min": -1.6, "max": 2.6},
                "cumulative": {"min": 1, "max": 41},
                "ntotal": {"min": 2, "max": 50},
                "oni": {"type": "str", "enum": ["neutral", "el_nino", "la_nina"]},
                "__len__": 900,
            },
            derived_slug: {
                "season": {"type": "str", "enum": ["AMJ", "OND", "ASO", "SON", "DJF", "NDJ", "FMA", "JFM", "..."]},
                "count_records": {"min": 74, "max": 75},
                "avg_sst_c": {"min": 26.5, "max": 27.6},
                "avg_anom_c": {"type": "num", "min": "-9.5e-03", "max": 0},
                "sum_cumulative": {"min": 530, "max": 704},
                "sum_ntotal": {"min": 1061, "max": 1161},
                "__len__": 7,
            },
        }
        return _metaset([raw, derived], schemas=schemas)

    def test_compact_context_derived_table_structure(self, oni_metaset):
        output = oni_metaset.compact_context()
        parsed = yaml.safe_load(output)
        derived = parsed["aggregated_metrics_by_season"]

        assert derived["derived_from"] == ["data_oni_csv"]
        assert derived["step"] == 1
        assert derived["latest"] is True
        assert "season" in derived["columns"]
        assert "avg_sst_c" in derived["columns"]
        assert "count_records" in derived["columns"]

    def test_compact_context_original_table_has_no_lineage(self, oni_metaset):
        output = oni_metaset.compact_context()
        parsed = yaml.safe_load(output)
        raw = parsed["data_oni_csv"]

        assert "derived_from" not in raw
        assert "step" not in raw
        assert "latest" not in raw
        assert "season" in raw["columns"]
        assert "year" in raw["columns"]

    def test_compact_context_column_schemas_present(self, oni_metaset):
        output = oni_metaset.compact_context()
        parsed = yaml.safe_load(output)
        derived_cols = parsed["aggregated_metrics_by_season"]["columns"]

        # Numeric range columns should have min/max
        assert derived_cols["avg_sst_c"]["min"] == 26.5
        assert derived_cols["avg_sst_c"]["max"] == 27.6
        assert derived_cols["count_records"]["min"] == 74
        assert derived_cols["count_records"]["max"] == 75

        # Enum column should have enum list
        assert "enum" in derived_cols["season"]

    def test_table_list_with_lineage_shows_derivation(self, oni_metaset):
        output = oni_metaset.table_list(include_lineage=True)
        parsed = yaml.safe_load(output)

        assert "aggregated_metrics_by_season" in parsed
        assert parsed["aggregated_metrics_by_season"]["derived_from"] == ["data_oni_csv"]
        assert parsed["aggregated_metrics_by_season"]["latest"] is True

        # Original table should have no lineage keys
        raw_data = parsed.get("data_oni_csv", {})
        assert "derived_from" not in raw_data

    def test_table_list_without_lineage_is_clean(self, oni_metaset):
        output = oni_metaset.table_list(include_lineage=False)
        assert "derived_from" not in output
        assert "step" not in output
        assert "latest" not in output

    def test_others_available_with_derived_hint(self, oni_metaset):
        oni_metaset.schema_tables = [f"Src{SEP}data_oni_csv"]
        output = oni_metaset.table_list(include_lineage=True, n_others=10)
        assert "(from data_oni_csv)" in output
        assert "\u2605" in output  # star on the derived table

    def test_full_context_includes_all_details(self, oni_metaset):
        output = oni_metaset.full_context()
        parsed = yaml.safe_load(output)

        # Both tables present with columns and lineage
        assert "data_oni_csv" in parsed
        assert "aggregated_metrics_by_season" in parsed
        assert "columns" in parsed["data_oni_csv"]
        assert "columns" in parsed["aggregated_metrics_by_season"]
        assert parsed["aggregated_metrics_by_season"]["derived_from"] == ["data_oni_csv"]
