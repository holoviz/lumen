"""
Tests for CodeSourceControls, URLSourceControls, and ParametricSourceControls.
"""
import io
import json

from unittest.mock import AsyncMock, patch

import pandas as pd
import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.controls import (
    CodeSourceControls, SourceResult, URLSourceControls,
)
from lumen.ai.controls.ingest.download import DownloadSourceControls
from lumen.ai.controls.ingest.parametric import ParametricSourceControls
from lumen.ai.controls.ingest.utils import (
    read_html_tables, read_json_to_dataframe,
)
from lumen.sources.duckdb import DuckDBSource

from .conftest import (
    async_fetch_data, raises_error, returns_list_of_dicts, returns_none,
    sync_fetch_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# CodeSourceControls Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeSourceControlsRegistration:
    """Tests for action registration in CodeSourceControls."""

    def test_single_function_registration(self, code_controls_single_func):
        """Single callable registered with auto-generated name from __name__."""
        controls = code_controls_single_func
        assert len(controls._actions) == 1
        assert "Sync Fetch Data" in controls._actions
        assert controls._action_selector.visible is False

    def test_dict_functions_registration(self, code_controls_dict_funcs):
        """Dict of {name: callable} registered as actions."""
        controls = code_controls_dict_funcs
        assert len(controls._actions) == 2
        assert "Fetch Data" in controls._actions
        assert "List Items" in controls._actions
        assert controls._action_selector.visible is True
        assert set(controls._action_selector.options) == {"Fetch Data", "List Items"}

    def test_instance_methods_registration(self, code_controls_instance_methods):
        """Object instance + method names registered as bound methods."""
        controls = code_controls_instance_methods
        assert len(controls._actions) == 2
        assert "Get Prices" in controls._actions
        assert "Get Details" in controls._actions
        assert callable(controls._actions["Get Prices"])
        assert callable(controls._actions["Get Details"])


@pytest.mark.asyncio
class TestCodeSourceControlsFetchData:
    """Tests for _fetch_data in CodeSourceControls."""

    async def test_sync_function_runs_in_thread(self, code_controls_single_func):
        """Sync function runs via asyncio.to_thread and returns SourceResult."""
        result = await code_controls_single_func._fetch_data(
            "Sync Fetch Data", ticker="AAPL", limit=5
        )

        assert isinstance(result, SourceResult)
        assert len(result.sources) == 1
        assert result.table == "data"

        df = result.sources[0].get("data")
        assert len(df) == 5
        assert list(df["ticker"]) == ["AAPL"] * 5

    async def test_async_function_awaited_directly(self, code_controls_async):
        """Async function awaited directly without thread pool."""
        result = await code_controls_async._fetch_data(
            "Async Fetch", ticker="MSFT", limit=3
        )

        assert isinstance(result, SourceResult)
        df = result.sources[0].get("data")
        assert len(df) == 3
        assert list(df["ticker"]) == ["MSFT"] * 3

    @pytest.mark.parametrize(
        "func,action_name,kwargs,expected_len,expected_col,expected_values",
        [
            (returns_list_of_dicts, "List Items", {"n": 4}, 4, "id", [0, 1, 2, 3]),
        ],
        ids=["list_of_dicts"],
    )
    async def test_return_type_conversion(
        self, context, source_catalog, func, action_name, kwargs, expected_len, expected_col, expected_values
    ):
        """Various return types converted to DataFrame correctly."""
        controls = CodeSourceControls(
            functions={action_name: func},
            context=context,
            source_catalog=source_catalog,
        )
        result = await controls._fetch_data(action_name, **kwargs)

        assert isinstance(result, SourceResult)
        df = result.sources[0].get("data")
        assert len(df) == expected_len
        assert list(df[expected_col]) == expected_values

    async def test_none_result_returns_empty(self, context, source_catalog):
        """None result returns SourceResult.empty with message."""
        controls = CodeSourceControls(
            functions={"Returns None": returns_none},
            context=context,
            source_catalog=source_catalog,
        )
        result = await controls._fetch_data("Returns None")

        assert isinstance(result, SourceResult)
        assert len(result.sources) == 0
        assert "no data" in result.message.lower()

    async def test_exception_returns_error_result(self, context, source_catalog):
        """Exception caught and returned as error SourceResult."""
        controls = CodeSourceControls(
            functions={"Raises Error": raises_error},
            context=context,
            source_catalog=source_catalog,
        )
        result = await controls._fetch_data("Raises Error")

        assert isinstance(result, SourceResult)
        assert len(result.sources) == 0
        assert "error" in result.message.lower()
        assert "Test error" in result.message


class TestCodeSourceControlsParamOverrides:
    """Tests for parameter overrides in CodeSourceControls."""

    def test_full_replacement_with_param_instance(self, context, source_catalog):
        """param.Parameter instance fully replaces auto-detected param."""
        controls = CodeSourceControls(
            functions={"Fetch Data": sync_fetch_data},
            param_overrides={
                "Fetch Data": {
                    "ticker": param.Selector(
                        default="AAPL",
                        objects=["AAPL", "MSFT", "GOOGL"],
                    ),
                },
            },
            context=context,
            source_catalog=source_catalog,
        )

        ticker_param = controls._action_models["Fetch Data"].param["ticker"]
        assert isinstance(ticker_param, param.Selector)
        assert ticker_param.default == "AAPL"
        assert ticker_param.objects == ["AAPL", "MSFT", "GOOGL"]

    def test_dict_merge_updates_existing_param(self, context, source_catalog):
        """Dict override merges into existing param, preserving type."""
        controls = CodeSourceControls(
            functions={"Fetch Data": sync_fetch_data},
            param_overrides={
                "Fetch Data": {
                    "limit": {"default": 20, "bounds": (1, 100)},
                },
            },
            context=context,
            source_catalog=source_catalog,
        )

        limit_param = controls._action_models["Fetch Data"].param["limit"]
        assert limit_param.default == 20
        assert limit_param.bounds == (1, 100)

    def test_skip_params_excludes_from_introspection(self, context, source_catalog):
        """Parameters in skip_params excluded from action model."""

        def func_with_internal(ticker: str, raw: bool = False, params: dict = None):
            return pd.DataFrame({"ticker": [ticker]})

        controls = CodeSourceControls(
            functions={"Fetch": func_with_internal},
            skip_params=frozenset({"self", "cls", "return", "raw", "params"}),
            context=context,
            source_catalog=source_catalog,
        )

        param_names = [n for n in controls._action_models["Fetch"].param if n != "name"]
        assert "ticker" in param_names
        assert "raw" not in param_names
        assert "params" not in param_names


class TestCodeSourceControlsAgentIntegration:
    """Tests for agent integration in CodeSourceControls."""

    def test_as_tools_returns_all_actions(self, code_controls_dict_funcs):
        """as_tools() returns all registered action callables."""
        tools = code_controls_dict_funcs.as_tools()

        assert len(tools) == 2
        names = {name for name, _ in tools}
        assert names == {"Fetch Data", "List Items"}
        assert all(callable(func) for _, func in tools)

    @pytest.mark.asyncio
    async def test_load_action_invokes_fetch_data(self, code_controls_single_func):
        """load_action() programmatic entry point works correctly."""
        result = await code_controls_single_func.load_action(
            "Sync Fetch Data", ticker="TSLA", limit=3
        )

        assert isinstance(result, SourceResult)
        df = result.sources[0].get("data")
        assert len(df) == 3
        assert list(df["ticker"]) == ["TSLA"] * 3


# ─────────────────────────────────────────────────────────────────────────────
# URLSourceControls Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestURLSourceControlsSubclassPattern:
    """Tests for the subclass pattern in URLSourceControls."""

    def test_class_level_params_detected(self, url_controls):
        """Class-level params detected via _get_query_param_names."""
        param_names = url_controls._get_query_param_names()

        assert "region" in param_names
        assert "year" in param_names
        assert "url_template" not in param_names  # Base class param excluded
        assert "vector_store" not in param_names  # Base class param excluded

    def test_uses_actions_false(self, url_controls):
        """Subclass pattern has _uses_actions=False."""
        assert url_controls._uses_actions is False
        assert len(url_controls._actions) == 0

    def test_as_tools_synthesizes_callable_from_params(self, url_controls):
        """as_tools() creates callable with signature matching class params."""
        import inspect

        tools = url_controls.as_tools()

        assert len(tools) == 1
        name, func = tools[0]
        assert name == "TestURL"  # "Controls" suffix removed
        assert callable(func)

        sig = inspect.signature(func)
        assert "region" in sig.parameters
        assert "year" in sig.parameters


@pytest.mark.asyncio
class TestURLSourceControlsFetchData:
    """Tests for _fetch_data in URLSourceControls."""

    @pytest.mark.parametrize(
        "filename,content,expected_table,expected_len,expected_col,expected_values",
        [
            ("data.csv", b"name,value\na,1\nb,2\n", "data", 2, "name", ["a", "b"]),
            (
                "data.json",
                json.dumps([{"id": 1}, {"id": 2}]).encode(),
                "data",
                2,
                "id",
                [1, 2],
            ),
            (
                "api.json",
                json.dumps({"data": [{"x": 10}, {"x": 20}]}).encode(),
                "api",
                2,
                "x",
                [10, 20],
            ),
            (
                "page.html",
                b"<html><body><table><tr><th>col1</th><th>col2</th></tr><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></table></body></html>",
                "page",
                2,
                "col1",
                ["a", "b"],
            ),
        ],
        ids=["csv", "json_array", "json_nested_data_key", "html_table"],
    )
    async def test_file_format_parsing(
        self, url_controls, filename, content, expected_table, expected_len, expected_col, expected_values
    ):
        """Various file formats parsed correctly."""
        with patch(
            "lumen.ai.controls.ingest.url.download_file",
            new_callable=AsyncMock,
            return_value=(filename, content, None),
        ):
            result = await url_controls._fetch_data(
                "TestURLControls", region="us", year=2024
            )

        assert len(result.sources) == 1
        df = result.sources[0].get(expected_table)
        assert len(df) == expected_len
        assert list(df[expected_col]) == expected_values

    async def test_url_template_interpolation(self, url_controls):
        """Parameters correctly interpolated into URL template."""
        csv_content = b"id,value\n1,100\n"

        with patch(
            "lumen.ai.controls.ingest.url.download_file",
            new_callable=AsyncMock,
            return_value=("data.csv", csv_content, None),
        ):
            result = await url_controls._fetch_data(
                "TestURLControls", region="eu", year=2023
            )

        assert isinstance(result, SourceResult)
        assert len(result.sources) == 1

    async def test_missing_param_returns_error(self, url_controls):
        """Missing URL template param returns error SourceResult."""
        result = await url_controls._fetch_data(
            "TestURLControls", region="us"  # missing 'year'
        )

        assert len(result.sources) == 0
        assert "missing param" in result.message.lower()

    @pytest.mark.parametrize(
        "filename,content,error_msg,expected_message_contains",
        [
            ("error.txt", b"ERROR: Invalid request", None, "api error"),
            ("empty.csv", b"col1,col2\n", None, "no data"),
            (None, None, "Connection timeout", "download failed"),
        ],
        ids=["api_error_response", "empty_response", "download_error"],
    )
    async def test_error_handling(
        self, url_controls, filename, content, error_msg, expected_message_contains
    ):
        """Various error conditions handled correctly."""
        with patch(
            "lumen.ai.controls.ingest.url.download_file",
            new_callable=AsyncMock,
            return_value=(filename, content, error_msg),
        ):
            result = await url_controls._fetch_data(
                "TestURLControls", region="us", year=2024
            )

        assert len(result.sources) == 0
        assert expected_message_contains in result.message.lower()


class TestJSONParsing:
    """Tests for read_json_to_dataframe utility."""

    @pytest.mark.parametrize(
        "json_data,expected_len,expected_col,expected_values",
        [
            ([{"a": 1}, {"a": 2}], 2, "a", [1, 2]),
            ({"data": [{"b": 10}, {"b": 20}]}, 2, "b", [10, 20]),
            ({"results": [{"c": 100}]}, 1, "c", [100]),
            ({"items": [{"d": 1}, {"d": 2}]}, 2, "d", [1, 2]),
            ({"key": "value", "num": 42}, 1, "key", ["value"]),
        ],
        ids=["list", "dict_with_data", "dict_with_results", "dict_with_items", "single_dict"],
    )
    def test_json_structures(self, json_data, expected_len, expected_col, expected_values):
        """Various JSON structures parsed to DataFrame correctly."""
        content = json.dumps(json_data).encode()
        df = read_json_to_dataframe(content)

        assert len(df) == expected_len
        assert list(df[expected_col]) == expected_values


class TestHTMLParsing:
    """Tests for read_html_tables utility."""

    @pytest.mark.parametrize(
        "html_content,expected_len,expected_col,expected_values",
        [
            (
                "<table><tr><th>name</th></tr><tr><td>alice</td></tr><tr><td>bob</td></tr></table>",
                2,
                "name",
                ["alice", "bob"],
            ),
            (
                "<html><body><table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table></body></html>",
                1,
                "x",
                [1],
            ),
        ],
        ids=["simple_table", "full_html_doc"],
    )
    def test_html_table_structures(self, html_content, expected_len, expected_col, expected_values):
        """Various HTML table structures parsed to DataFrame correctly."""
        result = read_html_tables(html_content, "data")

        assert len(result) == 1
        df = result["data"]
        assert len(df) == expected_len
        assert list(df[expected_col]) == expected_values

    def test_html_multiple_tables_returns_all(self):
        """When multiple tables exist, returns all as separate tables."""
        html = """
        <html>
        <table><tr><th>small</th></tr><tr><td>1</td></tr></table>
        <table><tr><th>large</th></tr><tr><td>a</td></tr><tr><td>b</td></tr><tr><td>c</td></tr></table>
        </html>
        """
        result = read_html_tables(html, "page")

        assert len(result) == 2
        assert "page_table0" in result
        assert "page_table1" in result
        assert len(result["page_table0"]) == 1
        assert len(result["page_table1"]) == 3

    def test_html_no_tables_raises(self):
        """HTML with no tables raises an error."""
        html = "<html><body><p>No tables here</p></body></html>"

        with pytest.raises((ValueError, ImportError)):
            read_html_tables(html, "data")


class TestSourceResult:
    """Tests for SourceResult factory methods and document handling."""

    def test_from_document_sets_document_only_flag(self):
        """from_document() creates result with document_only=True."""
        from lumen.ai.controls.ingest.result import SourceResult

        result = SourceResult.from_document("Indexed content as 'page'")

        assert result.document_only is True
        assert result.sources == []
        assert "Indexed" in result.message

    def test_from_source_has_document_only_false(self):
        """from_source() creates result with document_only=False."""
        from lumen.ai.controls.ingest.result import SourceResult
        from lumen.sources.duckdb import DuckDBSource

        source = DuckDBSource(uri=":memory:", ephemeral=True, tables={})
        result = SourceResult.from_source(source, table="test", message="Loaded data")

        assert result.document_only is False
        assert len(result.sources) == 1

    def test_empty_result_has_document_only_false(self):
        """empty() creates result with document_only=False."""
        from lumen.ai.controls.ingest.result import SourceResult

        result = SourceResult.empty("No data found")

        assert result.document_only is False
        assert result.sources == []

    def test_str_returns_message(self):
        """__str__ returns the message for LLM responses."""
        from lumen.ai.controls.ingest.result import SourceResult

        result = SourceResult.from_document("Document indexed successfully")

        assert str(result) == "Document indexed successfully"


class TestDownloadSourceControlsDocumentExtraction:
    """Tests for HTML document text extraction in DownloadSourceControls."""

    def test_extract_html_text_removes_scripts_and_styles(self):
        """_extract_html_text removes script and style elements."""
        from lumen.ai.controls.ingest.download import DownloadSourceControls

        controls = DownloadSourceControls()
        html = b"""
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
        <script>alert('hello');</script>
        <p>Important content here</p>
        <nav>Navigation links</nav>
        <footer>Footer text</footer>
        </body>
        </html>
        """
        text = controls._extract_html_text(html)

        assert "Important content" in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_extract_html_text_handles_bytes(self):
        """_extract_html_text handles bytes input."""
        from lumen.ai.controls.ingest.download import DownloadSourceControls

        controls = DownloadSourceControls()
        html = b"<html><body><p>Test content</p></body></html>"

        text = controls._extract_html_text(html)

        assert "Test content" in text


# ─────────────────────────────────────────────────────────────────────────────
# ParametricSourceControls Base Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParametricSourceControlsBase:
    """Tests for base functionality in ParametricSourceControls."""

    @pytest.mark.parametrize(
        "fixture_name,expected_visible",
        [
            ("code_controls_single_func", False),
            ("code_controls_dict_funcs", True),
        ],
        ids=["single_action_hidden", "multiple_actions_visible"],
    )
    def test_action_selector_visibility(self, fixture_name, expected_visible, request):
        """Action selector visibility based on number of actions."""
        controls = request.getfixturevalue(fixture_name)
        assert controls._action_selector.visible is expected_visible

    @pytest.mark.parametrize(
        "fixture_name,expected_uses_actions",
        [
            ("code_controls_single_func", True),
            ("url_controls", False),
        ],
        ids=["action_pattern", "subclass_pattern"],
    )
    def test_uses_actions_property(self, fixture_name, expected_uses_actions, request):
        """_uses_actions correctly identifies control pattern."""
        controls = request.getfixturevalue(fixture_name)
        assert controls._uses_actions is expected_uses_actions

    def test_action_models_created_for_each_action(self, code_controls_dict_funcs):
        """Action models created as Parameterized instances."""
        assert "Fetch Data" in code_controls_dict_funcs._action_models
        assert "List Items" in code_controls_dict_funcs._action_models

        for model in code_controls_dict_funcs._action_models.values():
            assert isinstance(model, param.Parameterized)


@pytest.mark.asyncio
class TestParametricSourceControlsLoad:
    """Tests for _load dispatch in ParametricSourceControls."""

    async def test_action_pattern_uses_model_values(self, code_controls_single_func):
        """_load() uses parameter values from action model."""
        model = code_controls_single_func._action_models["Sync Fetch Data"]
        model.ticker = "NVDA"
        model.limit = 7

        result = await code_controls_single_func._load()

        df = result.sources[0].get("data")
        assert len(df) == 7
        assert list(df["ticker"]) == ["NVDA"] * 7

    async def test_subclass_pattern_uses_instance_values(self, url_controls):
        """_load() uses class-level parameter values."""
        url_controls.region = "eu"
        url_controls.year = 2023

        with patch(
            "lumen.ai.controls.ingest.url.download_file",
            new_callable=AsyncMock,
            return_value=("data.csv", b"x,y\n1,2\n", None),
        ):
            result = await url_controls._load()

        assert isinstance(result, SourceResult)
        assert len(result.sources) == 1


class TestApplyOverrides:
    """Tests for _apply_overrides static method."""

    @pytest.mark.parametrize(
        "initial_param,override,expected_type,expected_default",
        [
            (param.String(default="old"), param.Integer(default=42), param.Integer, 42),
            (param.Integer(default=10), {"default": 20}, param.Integer, 20),
        ],
        ids=["full_replacement", "dict_merge"],
    )
    def test_override_application(self, initial_param, override, expected_type, expected_default):
        """Overrides applied correctly based on type."""
        params_dict = {"x": initial_param}
        ParametricSourceControls._apply_overrides(params_dict, {"x": override})

        assert isinstance(params_dict["x"], expected_type)
        assert params_dict["x"].default == expected_default

    def test_dict_merge_adds_bounds(self):
        """Dict merge can add new attributes like bounds."""
        params_dict = {"x": param.Integer(default=10)}
        ParametricSourceControls._apply_overrides(params_dict, {"x": {"bounds": (0, 100)}})

        assert params_dict["x"].bounds == (0, 100)

    def test_nonexistent_param_ignored(self):
        """Override for nonexistent param silently ignored."""
        params_dict = {"x": param.String(default="value")}
        ParametricSourceControls._apply_overrides(params_dict, {"y": {"default": "other"}})

        assert "y" not in params_dict
        assert params_dict["x"].default == "value"
