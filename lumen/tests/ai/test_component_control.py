import datetime as dt

import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

import panel_material_ui as pmui

from panel.viewable import Viewer

from lumen.ai.agents import ComponentControlAgent
from lumen.ai.tools import ComponentController


class Config(param.Parameterized):
    """Configuration of the model."""

    n_estimators = param.Integer(default=100, bounds=(10, 500), doc="Number of trees.")

    criterion = param.Selector(default="gini", objects=["gini", "entropy"], doc="Split criterion.")

    normalize = param.Boolean(default=True, doc="Whether to normalize inputs.")

    weights = param.List(default=[1.0], item_type=float, doc="Class weights.")

    _private = param.String(default="hidden", precedence=-1)


@pytest.fixture
def widgets():
    return {
        "slider": pmui.FloatSlider(
            label="Temperature", start=0, end=40, step=0.5, value=21.5,
            description="Target temperature of the simulation",
        ),
        "select": pmui.Select(label="Colormap", options={"Viridis": "viridis", "Plasma": "plasma"}),
        "multi": pmui.MultiChoice(label="Regions", options=["EU", "US", "APAC"], value=["EU"]),
        "range": pmui.RangeSlider(label="Year range", start=2000, end=2024, value=(2010, 2020)),
        "date": pmui.DatePicker(label="Day", value=dt.date(2020, 5, 1)),
        "toggle": pmui.Switch(label="Show outliers", value=False),
        "button": pmui.Button(label="Reset"),
    }


@pytest.fixture
def page(widgets):
    return pmui.Page(
        main=[pmui.Column(*[w for k, w in widgets.items() if k != "button"])],
        sidebar=[widgets["button"]],
    )


def tools_by_name(controller):
    return {tool.name: tool for tool in controller.tools}


def schema(tool):
    return tool._model.model_json_schema()


class TestDiscovery:

    def test_walks_page(self, page):
        controller = ComponentController(components=page)
        assert [spec.key for spec in controller.specs] == [
            "temperature", "colormap", "regions", "year_range", "day", "show_outliers", "reset"
        ]

    def test_summary_lists_labels_values_and_tools(self, page):
        summary = ComponentController(components=page, purpose="Turbine dashboard.").summary()
        assert "Turbine dashboard." in summary
        assert '`temperature` — FloatSlider labelled "Temperature"' in summary
        assert "Target temperature of the simulation" in summary
        assert "value: 21.5 (between 0 and 40; step 0.5)" in summary
        assert "value: 'Viridis' (one of: Viridis, Plasma)" in summary
        assert "Control with: set_temperature" in summary
        assert "Control with: click_reset" in summary

    def test_summary_without_components(self):
        assert "No controllable components" in ComponentController().summary()

    def test_discovery_tool_names_use_namespace(self, page):
        tools = tools_by_name(ComponentController(components=page, namespace="dashboard"))
        assert "list_dashboard_components" in tools
        assert "describe_dashboard_component" in tools

    async def test_describe_tool_reports_parameters_and_docs(self):
        controller = ComponentController(components={"config": Config()})
        tools = tools_by_name(controller)
        described = await tools["describe_ui_component"].function(component="config")
        assert "Configuration of the model." in described
        assert "`n_estimators` (Integer) = 100" in described
        assert "Accepts: between 10 and 500" in described
        assert "Doc: Number of trees." in described
        assert "Control with: set_config" in described

    async def test_describe_tool_rejects_unknown_component(self, page):
        tools = tools_by_name(ComponentController(components=page))
        described = await tools["describe_ui_component"].function(component="nope")
        assert "Unknown component 'nope'" in described

    async def test_list_tool_reflects_current_values(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        widgets["slider"].value = 33.0
        assert "value: 33.0" in await tools["list_ui_components"].function()

    def test_describe_tool_enumerates_components(self, page):
        tools = tools_by_name(ComponentController(components=page))
        assert schema(tools["describe_ui_component"])["properties"]["component"]["enum"][0] == "temperature"


class TestSchemas:

    def test_one_tool_per_component(self, page):
        tools = tools_by_name(ComponentController(components=page))
        assert set(tools) == {
            "list_ui_components", "describe_ui_component", "set_temperature", "set_colormap",
            "set_regions", "set_year_range", "set_day", "set_show_outliers", "click_reset",
        }

    def test_slider_bounds_are_declared(self, page):
        tools = tools_by_name(ComponentController(components=page))
        value = schema(tools["set_temperature"])["properties"]["value"]
        assert {"type": "number", "minimum": 0, "maximum": 40} in value["anyOf"]
        assert "Currently 21.5" in value["description"]

    def test_options_become_literals(self, page):
        tools = tools_by_name(ComponentController(components=page))
        value = schema(tools["set_colormap"])["properties"]["value"]
        assert {"type": "string", "enum": ["Viridis", "Plasma"]} in value["anyOf"]

    def test_multi_select_options_become_list_of_literals(self, page):
        tools = tools_by_name(ComponentController(components=page))
        value = schema(tools["set_regions"])["properties"]["value"]
        assert {"type": "array", "items": {"type": "string", "enum": ["EU", "US", "APAC"]}} in value["anyOf"]

    def test_button_tool_takes_no_arguments(self, page):
        tools = tools_by_name(ComponentController(components=page))
        assert schema(tools["click_reset"])["properties"] == {}

    def test_parameterized_exposes_own_parameters(self):
        tools = tools_by_name(ComponentController(components={"config": Config()}))
        properties = schema(tools["set_config"])["properties"]
        assert set(properties) == {"n_estimators", "criterion", "normalize", "weights"}
        assert properties["n_estimators"]["description"].startswith("Number of trees.")

    def test_explicit_parameters_override(self):
        controller = ComponentController(
            components={"config": Config()}, parameters={"config": ["criterion"]}
        )
        assert set(schema(tools_by_name(controller)["set_config"])["properties"]) == {"criterion"}

    def test_descriptions_are_added_to_the_purpose(self):
        controller = ComponentController(
            components={"config": Config()}, descriptions={"config": "Hyper-parameters."}
        )
        assert "Hyper-parameters." in tools_by_name(controller)["set_config"].purpose


class TestApply:

    async def test_sets_value(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        result = await tools["set_temperature"].function(value=30)
        assert widgets["slider"].value == 30
        assert result == "Updated Temperature: `value` 21.5 → 30.0"

    async def test_coerces_string_number(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        await tools["set_temperature"].function(value="12.25")
        assert widgets["slider"].value == 12.25

    async def test_rejects_out_of_bounds_value(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        result = await tools["set_temperature"].function(value=99)
        assert widgets["slider"].value == 21.5
        assert "above the upper bound 40" in result

    async def test_rejects_out_of_bounds_range(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        result = await tools["set_year_range"].function(value=[1990, 2015])
        assert widgets["range"].value == (2010, 2020)
        assert "below the lower bound 2000" in result

    async def test_resolves_option_label_to_value(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        await tools["set_colormap"].function(value="Plasma")
        assert widgets["select"].value == "plasma"

    async def test_resolves_raw_option_value(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        await tools["set_colormap"].function(value="viridis")
        assert widgets["select"].value == "viridis"

    async def test_rejects_unknown_option(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        result = await tools["set_colormap"].function(value="magma")
        assert widgets["select"].value == "viridis"
        assert "not one of the allowed values: Viridis, Plasma" in result

    async def test_coerces_single_value_to_list(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        await tools["set_regions"].function(value="US")
        assert widgets["multi"].value == ["US"]

    async def test_coerces_boolean_string(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        await tools["set_show_outliers"].function(value="yes")
        assert widgets["toggle"].value is True

    async def test_coerces_date_string(self, page, widgets):
        tools = tools_by_name(ComponentController(components=page))
        await tools["set_day"].function(value="2021-07-04")
        assert widgets["date"].value == dt.date(2021, 7, 4)

    async def test_coerces_datetime_range(self):
        slider = pmui.DatetimeRangeSlider(
            label="Window", start=dt.datetime(2020, 1, 1), end=dt.datetime(2020, 12, 31),
            value=(dt.datetime(2020, 2, 1), dt.datetime(2020, 3, 1)),
        )
        tools = tools_by_name(ComponentController(components=[slider]))
        await tools["set_window"].function(value=["2020-04-01", "2020-06-15T12:00:00"])
        assert slider.value == (dt.datetime(2020, 4, 1), dt.datetime(2020, 6, 15, 12))

    async def test_coerces_list_items_to_item_type(self):
        config = Config()
        tools = tools_by_name(ComponentController(components={"config": config}))
        await tools["set_config"].function(weights=[1, 2.5])
        assert config.weights == [1.0, 2.5]

    async def test_omitted_arguments_are_left_alone(self):
        config = Config()
        tools = tools_by_name(ComponentController(components={"config": config}))
        result = await tools["set_config"].function(n_estimators=250)
        assert (config.n_estimators, config.criterion, config.normalize) == (250, "gini", True)
        assert result == "Updated config: `n_estimators` 100 → 250"

    async def test_reports_parameter_rejected_by_param(self):
        config = Config()
        tools = tools_by_name(ComponentController(components={"config": config}))
        result = await tools["set_config"].function(n_estimators=1000)
        assert config.n_estimators == 100
        assert "must be at most 500" in result

    async def test_click_triggers_button(self, page, widgets):
        clicks = []
        widgets["button"].on_click(clicks.append)
        tools = tools_by_name(ComponentController(components=page))
        result = await tools["click_reset"].function()
        assert len(clicks) == 1
        assert result == "Clicked Reset."


class TestLiveSync:

    def test_added_components_are_picked_up(self, page):
        controller = ComponentController(components=page)
        assert "set_bins" not in tools_by_name(controller)
        page.main[0].append(pmui.IntSlider(label="Bins", start=1, end=100, value=10))
        assert "set_bins" in tools_by_name(controller)

    def test_removed_components_disappear(self, page, widgets):
        controller = ComponentController(components=page)
        page.main[0].remove(widgets["select"])
        assert "set_colormap" not in tools_by_name(controller)

    def test_walks_nested_viewer(self):
        class Dashboard(Viewer):
            def __init__(self, **params):
                super().__init__(**params)
                self.search = pmui.TextInput(label="Search")
                self._layout = pmui.Column(
                    self.search, pmui.Tabs(("A", pmui.Column(pmui.IntInput(label="Limit", value=10))))
                )

            def __panel__(self):
                return self._layout

        assert [spec.key for spec in ComponentController(components=Dashboard()).specs] == ["search", "limit"]

    def test_excluded_components_are_skipped(self, page, widgets):
        controller = ComponentController(components=page, exclude=[widgets["slider"], pmui.Switch])
        keys = [spec.key for spec in controller.specs]
        assert "temperature" not in keys
        assert "show_outliers" not in keys

    def test_duplicate_labels_are_disambiguated(self):
        column = pmui.Column(pmui.IntSlider(label="Count"), pmui.IntSlider(label="Count"))
        assert [spec.key for spec in ComponentController(components=column).specs] == ["count", "count_2"]

    def test_explicitly_named_container_is_not_walked(self, widgets):
        column = pmui.Column(widgets["slider"])
        controller = ComponentController(components={"panel": column})
        assert [spec.key for spec in controller.specs] == ["panel"]


class TestAgent:

    def test_components_build_a_controller(self, page):
        agent = ComponentControlAgent(components=page)
        assert agent.controller.components is page
        assert agent.controller in agent.llm_tools

    def test_controller_is_not_shared_between_instances(self):
        assert ComponentControlAgent().controller is not ComponentControlAgent().controller

    def test_setting_components_updates_the_controller(self, page, widgets):
        agent = ComponentControlAgent()
        agent.components = page
        assert next(spec.key for spec in agent.controller.specs) == "temperature"

    async def test_prompt_context_includes_live_state(self, page, widgets):
        agent = ComponentControlAgent(components=page)
        widgets["slider"].value = 12.0
        prompt_context = await agent._gather_prompt_context("main", [], {})
        assert "value: 12.0" in prompt_context["ui_state"]
