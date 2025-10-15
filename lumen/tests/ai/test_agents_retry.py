import pytest
import yaml

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.agents import LumenBaseAgent


class TestRetryOutputbyLine:
    """Test cases for the extracted static methods from _retry_output_by_line."""

    def test_prepare_lines_for_retry_without_target_keys(self):
        """Test _prepare_lines_for_retry when no target keys are specified."""
        original_output = "line1\nline2\nline3"
        
        lines, targeted, original_spec = LumenBaseAgent._prepare_lines_for_retry(
            original_output, None
        )
        
        assert lines == ["line1", "line2", "line3"]
        assert targeted is False
        assert original_spec is None

    def test_prepare_lines_for_retry_with_target_keys(self):
        """Test _prepare_lines_for_retry when target keys are specified."""
        original_yaml = """
        spec:
          chart:
            type: bar
            data: [1, 2, 3]
        other: value
        """
        retry_target_keys = ["spec", "chart"]
        
        lines, targeted, original_spec = LumenBaseAgent._prepare_lines_for_retry(
            original_yaml, retry_target_keys
        )
        
        assert targeted is True
        assert original_spec is not None
        assert "spec" in original_spec
        # The lines should now contain only the targeted section as YAML
        reconstructed = yaml.safe_load("\n".join(lines))
        assert reconstructed["type"] == "bar"
        assert reconstructed["data"] == [1, 2, 3]

    def test_prepare_lines_for_retry_with_invalid_yaml(self):
        """Test _prepare_lines_for_retry with invalid YAML when targeting."""
        invalid_yaml = "invalid: yaml: content:"
        retry_target_keys = ["spec"]
        
        with pytest.raises(yaml.YAMLError):
            LumenBaseAgent._prepare_lines_for_retry(invalid_yaml, retry_target_keys)
    
    def test_sql_spec_modification_real_example(self):
        """Test the complete retry workflow with a real SQL specification example."""
        original_yaml = 'source:\n  tables:\n    average_sst_during_el_nino_oni_csv: \'SELECT AVG("sst_c") AS "average_sst_el_nino"\n\n      FROM "oni_csv"\n\n      WHERE "oni" = \'\'el_nino\'\'\'\n    oni_csv: oni.csv\n  type: duckdb\n  uri: \':memory:\'\nsql_transforms:\n- limit: 1000000\n  pretty: true\n  read: duckdb\n  type: sql_limit\n  write: duckdb\ntable: average_sst_during_el_nino_oni_csv\n'
        retry_target_keys = ["source", "tables"]

        # Test prepare_lines_for_retry
        lines, targeted, original_spec = LumenBaseAgent._prepare_lines_for_retry(
            original_yaml, retry_target_keys
        )
        
        assert targeted is True
        assert original_spec is not None
        assert "source" in original_spec
        assert "tables" in original_spec["source"]
        
        # The lines should contain the extracted tables section
        reconstructed_tables = yaml.safe_load("\n".join(lines))
        assert "average_sst_during_el_nino_oni_csv" in reconstructed_tables
        assert "oni_csv" in reconstructed_tables
        assert reconstructed_tables["oni_csv"] == "oni.csv"
        
        # Test apply_line_changes_to_output with real LineChange
        from lumen.ai.models import LineChange
        from lumen.ai.utils import set_nested

        # Simulate the line change that modifies oni_csv
        line_changes = [LineChange(line_no=5, replacement="oni_csv: SELECT * FROM oni.csv")]
        
        # Mock apply_changes to return what it would actually return
        expected_modified_tables = reconstructed_tables.copy()
        expected_modified_tables["oni_csv"] = "SELECT * FROM oni.csv"
        result = LumenBaseAgent._apply_line_changes_to_output(
            lines, line_changes, targeted=True,
            original_spec=original_spec, retry_target_keys=retry_target_keys
        )
        
        # Parse the result and verify the modification was applied correctly
        final_spec = yaml.safe_load(result)
        assert final_spec["source"]["tables"]["oni_csv"] == "SELECT * FROM oni.csv"
        # Verify other parts of the spec remain unchanged
        assert final_spec["sql_transforms"][0]["limit"] == 1000000
        assert final_spec["table"] == "average_sst_during_el_nino_oni_csv"
        # Verify the SQL query for the other table is preserved
        assert "SELECT AVG" in final_spec["source"]["tables"]["average_sst_during_el_nino_oni_csv"]
