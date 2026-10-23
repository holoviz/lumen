import datetime as dt

import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

import panel_material_ui as pmui

from panel.viewable import Viewer

from lumen.ai.agents import ChatAgent, ComponentControlAgent
from lumen.ai.coordinator import Planner
from lumen.ai.tools import ComponentController


class Config(param.Parameterized):
    """Configuration of the model."""

    n_estimators = param.Integer(default=100, bounds=(10, 500), doc="Number of trees.")

    criterion = param.Selector(default="gini", objects=["gini", "entropy"], doc="Split criterion.")

    normalize = param.Boolean(default=True, doc="Whether to normalize inputs.")

    weights = param.List(default=[1.0], item_type=float, doc="Class weights.")

    threshold = param.Number(default=1.0, allow_None=True, doc="Decision threshold.")

    accuracy = param.Number(default=0.0, constant=True, doc="Accuracy of the last fit.")

    _private = param.String(default="hidden", precedence=-1)

    @param.depends("n_estimators", watch=True)
    def _fit(self):
        with param.parameterized.edit_constant(self):
            self.accuracy = self.n_estimators / 1000


class Labelled(param.Parameterized):

    label = param.String(default="some data")


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


@pytest.fixture
def controller(page, widgets):
    return ComponentController(components=page, actions=[widgets["button"]])


def tools_by_name(controller):
    return {tool.name: tool for tool in controller.as_llm_tools()}


def schema(tool):
    return tool._model.model_json_schema()


async def set_components(controller, **values):
    return await tools_by_name(controller)["set_ui_components"].function(**values)


class TestDiscovery:

    def test_walks_page(self, controller):
        assert [spec.key for spec in controller.specs] == [
            "temperature", "colormap", "regions", "year_range", "day", "show_outliers", "reset"
        ]

    def test_walked_buttons_require_opt_in(self, page):
        assert "reset" not in [spec.key for spec in ComponentController(components=page).specs]

    def test_explicit_buttons_are_actions(self, widgets):
        controller = ComponentController(components=[widgets["button"]])
        assert "click_ui_component" in tools_by_name(controller)

    def test_password_inputs_are_not_walked(self):
        column = pmui.Column(pmui.TextInput(label="Search"), pmui.PasswordInput(label="API key", value="secret"))
        controller = ComponentController(components=column)
        assert [spec.key for spec in controller.specs] == ["search"]

    async def test_explicit_password_input_value_is_hidden(self):
        controller = ComponentController(components={"key": pmui.PasswordInput(value="secret")})
        described = await tools_by_name(controller)["describe_ui_component"].function(component="key")
        assert "secret" not in controller.summary() and "secret" not in described
        assert "<hidden>" in controller.summary()

    def test_keyed_layout_exposes_nothing_by_default(self, widgets):
        controller = ComponentController(components={"panel": pmui.Column(widgets["slider"]), "page": pmui.Page()})
        assert controller.specs == []

    def test_keyed_layout_exposes_listed_parameters(self, widgets):
        controller = ComponentController(components={"panel": pmui.Column(widgets["slider"])}, parameters={"panel": ["visible"]})
        assert [info.name for info in controller.specs[0].settable] == ["visible"]

    def test_invisible_components_are_skipped(self, page, widgets):
        widgets["slider"].visible = False
        assert "temperature" not in [spec.key for spec in ComponentController(components=page).specs]

    def test_disabled_widget_is_listed_but_not_settable(self, page, widgets):
        widgets["slider"].disabled = True
        controller = ComponentController(components=page)
        assert "value: 21.5 (disabled" in controller.summary()
        assert "temperature" not in schema(tools_by_name(controller)["set_ui_components"])["properties"]

    def test_summary_lists_labels_values_and_tools(self, page, widgets):
        summary = ComponentController(components=page, actions=[widgets["button"]], purpose="Turbine dashboard.").summary()
        assert "Turbine dashboard." in summary
        assert '`temperature` — FloatSlider labelled "Temperature"' in summary
        assert "Target temperature of the simulation" in summary
        assert "value: 21.5 (between 0 and 40; step 0.5)" in summary
        assert "value: 'Viridis' (one of: Viridis, Plasma)" in summary
        assert "value: ['EU'] (any of: EU, US, APAC)" in summary
        assert "set with set_ui_components" in summary
        assert "click with click_ui_component" in summary

    def test_summary_without_components(self):
        assert "No controllable components" in ComponentController().summary()

    def test_tool_names_use_namespace(self, page):
        tools = tools_by_name(ComponentController(components=page, namespace="dashboard"))
        assert set(tools) == {"list_dashboard_components", "describe_dashboard_component", "set_dashboard_components"}

    async def test_describe_tool_reports_parameters_and_docs(self):
        tools = tools_by_name(ComponentController(components={"config": Config()}))
        described = await tools["describe_ui_component"].function(component="config")
        assert "Configuration of the model." in described
        assert "`n_estimators` (Integer) = 100" in described
        assert "Accepts: between 10 and 500" in described
        assert "Doc: Number of trees." in described
        assert "`accuracy` (Number) = 0.0" in described
        assert "(read-only)" in described

    async def test_describe_tool_rejects_unknown_component(self, controller):
        described = await tools_by_name(controller)["describe_ui_component"].function(component="nope")
        assert "Unknown component 'nope'" in described

    async def test_describe_tool_resolves_live(self, page, controller):
        describe = tools_by_name(controller)["describe_ui_component"]
        page.main[0].append(pmui.IntSlider(label="Bins", start=1, end=100, value=10))
        assert "IntSlider" in await describe.function(component="bins")

    async def test_list_tool_reflects_current_values(self, controller, widgets):
        tools = tools_by_name(controller)
        widgets["slider"].value = 33.0
        assert "value: 33.0" in await tools["list_ui_components"].function()

    def test_label_parameter_of_parameterized_is_not_a_display_name(self):
        assert ComponentController(components=[Labelled()]).specs[0].key == "labelled"

    def test_constant_parameters_are_reported(self):
        summary = ComponentController(components={"config": Config()}).summary()
        assert "accuracy: 0.0 (read-only)" in summary


class TestKeys:

    def test_duplicate_labels_are_disambiguated(self):
        column = pmui.Column(pmui.IntSlider(label="Count"), pmui.IntSlider(label="Count"))
        assert [spec.key for spec in ComponentController(components=column).specs] == ["count", "count_2"]

    def test_explicit_keys_win_over_labels(self):
        explicit = pmui.FloatSlider(label="Other")
        button = pmui.Button(label="Temperature")
        controller = ComponentController(components={"temperature": explicit}, actions=[button])
        assert {spec.key: spec.component for spec in controller.specs} == {"temperature": explicit, "temperature_2": button}

    def test_colliding_explicit_keys_raise(self):
        controller = ComponentController(components={"My Key": Config(), "my_key": Config()})
        with pytest.raises(ValueError, match="collides"):
            controller.as_llm_tools()

    def test_explicitly_named_container_is_not_walked(self, widgets):
        controller = ComponentController(components={"panel": pmui.Column(widgets["slider"])}, parameters={"panel": ["visible"]})
        assert [spec.key for spec in controller.specs] == ["panel"]

    def test_long_namespace_keeps_tool_names_within_limits(self, page):
        controller = ComponentController(components=page, namespace="a" * 60)
        assert all(len(tool.name) <= 64 for tool in controller.as_llm_tools())

    def test_keyword_labels_become_valid_identifiers(self):
        controller = ComponentController(components=pmui.Column(pmui.TextInput(label="class")))
        assert controller.specs[0].key == "class_"


class TestSchemas:

    def test_fixed_tool_set(self, controller):
        assert set(tools_by_name(controller)) == {
            "list_ui_components", "describe_ui_component", "set_ui_components", "click_ui_component",
        }

    def test_one_optional_argument_per_component(self, controller):
        properties = schema(tools_by_name(controller)["set_ui_components"])["properties"]
        assert list(properties) == ["temperature", "colormap", "regions", "year_range", "day", "show_outliers"]
        assert "required" not in schema(tools_by_name(controller)["set_ui_components"])

    def test_schema_has_no_refs_or_current_values(self, controller):
        rendered = str(schema(tools_by_name(controller)["set_ui_components"]))
        assert "$defs" not in rendered and "$ref" not in rendered
        assert "Currently" not in rendered

    def test_schema_avoids_constructs_gemini_rejects(self, controller):
        def walk(node):
            if isinstance(node, dict):
                assert "prefixItems" not in node
                assert node != {}
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        properties = schema(tools_by_name(controller)["set_ui_components"])["properties"]
        assert properties["year_range"]["type"] == "array"
        walk(properties)

    def test_slider_bounds_are_declared(self, controller):
        value = schema(tools_by_name(controller)["set_ui_components"])["properties"]["temperature"]
        assert (value["type"], value["minimum"], value["maximum"]) == ("number", 0, 40)
        assert "Target temperature of the simulation" in value["description"]

    def test_options_become_enums(self, controller):
        value = schema(tools_by_name(controller)["set_ui_components"])["properties"]["colormap"]
        assert value["enum"] == ["Viridis", "Plasma"]

    def test_multi_select_options_become_list_of_enums(self, controller):
        value = schema(tools_by_name(controller)["set_ui_components"])["properties"]["regions"]
        assert value["items"] == {"type": "string", "enum": ["EU", "US", "APAC"]}

    def test_many_options_are_not_enumerated(self):
        select = pmui.Select(label="Item", options=[f"o{i}" for i in range(80)])
        value = schema(tools_by_name(ComponentController(components=[select]))["set_ui_components"])["properties"]["item"]
        assert value["type"] == "string" and "enum" not in value

    def test_click_tool_enumerates_actions(self, controller, widgets):
        controller.actions = [widgets["button"], pmui.Button(label="Save")]
        controller.components = pmui.Column(widgets["button"])
        component = schema(tools_by_name(controller)["click_ui_component"])["properties"]["component"]
        assert component["enum"] == ["reset", "save"]

    def test_parameterized_takes_object_of_own_parameters(self):
        properties = schema(tools_by_name(ComponentController(components={"config": Config()}))["set_ui_components"])["properties"]
        config = properties["config"]
        assert config["type"] == "object"
        assert set(config["properties"]) == {"n_estimators", "criterion", "normalize", "weights", "threshold"}
        assert config["properties"]["n_estimators"]["description"].startswith("Number of trees.")
        assert {"type": "null"} in config["properties"]["threshold"]["anyOf"]

    def test_explicit_parameters_override(self):
        controller = ComponentController(components={"config": Config()}, parameters={"config": ["criterion"]})
        config = schema(tools_by_name(controller)["set_ui_components"])["properties"]["config"]
        assert set(config["properties"]) == {"criterion"}

    def test_descriptions_are_added_to_the_argument(self):
        controller = ComponentController(components={"config": Config()}, descriptions={"config": "Hyper-parameters."})
        config = schema(tools_by_name(controller)["set_ui_components"])["properties"]["config"]
        assert "Hyper-parameters" in config["description"]


class TestApply:

    async def test_sets_value_and_reports_read_back(self, controller, widgets):
        result = await set_components(controller, temperature=30)
        assert widgets["slider"].value == 30
        assert result == "Updated:\n- `temperature.value` 21.5 → 30.0"

    async def test_sets_several_components_in_one_call(self, controller, widgets):
        await set_components(controller, temperature=30, colormap="Plasma", show_outliers=True)
        assert (widgets["slider"].value, widgets["select"].value, widgets["toggle"].value) == (30, "plasma", True)

    async def test_validates_through_json_type(self, controller, widgets):
        await set_components(controller, temperature="12.25", show_outliers="yes", day="2021-07-04")
        assert widgets["slider"].value == 12.25
        assert widgets["toggle"].value is True
        assert widgets["date"].value == dt.date(2021, 7, 4)

    async def test_rejects_out_of_bounds_value(self, controller, widgets):
        result = await set_components(controller, temperature=99)
        assert widgets["slider"].value == 21.5
        assert "Rejected:\n- `temperature.value`: Input should be less than or equal to 40" in result

    async def test_rejects_out_of_bounds_range(self, controller, widgets):
        result = await set_components(controller, year_range=[1990, 2015])
        assert widgets["range"].value == (2010, 2020)
        assert "below the lower bound 2000" in result

    async def test_resolves_option_label_or_value(self, controller, widgets):
        await set_components(controller, colormap="plasma ")
        assert widgets["select"].value == "plasma"
        await set_components(controller, colormap="viridis")
        assert widgets["select"].value == "viridis"

    async def test_rejects_unknown_option(self, controller, widgets):
        result = await set_components(controller, colormap="magma")
        assert widgets["select"].value == "viridis"
        assert "not one of the allowed values: Viridis, Plasma" in result

    async def test_coerces_single_value_to_list(self, controller, widgets):
        await set_components(controller, regions="US")
        assert widgets["multi"].value == ["US"]

    async def test_coerces_datetime_range(self):
        slider = pmui.DatetimeRangeSlider(
            label="Window", start=dt.datetime(2020, 1, 1), end=dt.datetime(2020, 12, 31),
            value=(dt.datetime(2020, 2, 1), dt.datetime(2020, 3, 1)),
        )
        await set_components(ComponentController(components=[slider]), window=["2020-04-01", "2020-06-15T12:00:00"])
        assert slider.value == (dt.datetime(2020, 4, 1), dt.datetime(2020, 6, 15, 12))

    async def test_coerces_list_items_to_item_type(self):
        config = Config()
        await set_components(ComponentController(components={"config": config}), config={"weights": [1, 2.5]})
        assert config.weights == [1.0, 2.5]

    async def test_omitted_parameters_are_left_alone(self):
        config = Config()
        await set_components(ComponentController(components={"config": config}), config={"n_estimators": 250})
        assert (config.n_estimators, config.criterion, config.normalize) == (250, "gini", True)

    async def test_null_clears_nullable_parameter(self):
        config = Config()
        await set_components(ComponentController(components={"config": config}), config={"threshold": None})
        assert config.threshold is None

    async def test_null_is_ignored_for_non_nullable_parameter(self, controller, widgets):
        result = await set_components(controller, temperature=None, colormap="Plasma")
        assert widgets["slider"].value == 21.5
        assert "Rejected" not in result

    async def test_reports_consequential_changes(self):
        config = Config()
        result = await set_components(ComponentController(components={"config": config}), config={"n_estimators": 250})
        assert "Also changed as a result:\n- `config.accuracy` 0.0 → 0.25" in result

    async def test_reports_values_changed_by_the_application(self, controller, widgets):
        widgets["slider"].param.watch(lambda event: setattr(widgets["slider"], "value", 20.0), "value")
        result = await set_components(controller, temperature=30)
        assert "`temperature.value` 21.5 → 20.0 (requested 30.0, the application changed it)" in result

    async def test_awaits_settle_before_reading_back(self, widgets):
        settled = []

        async def settle():
            settled.append(widgets["slider"].value)

        controller = ComponentController(components=[widgets["slider"]], settle=settle)
        await set_components(controller, temperature=30)
        assert settled == [30]

    async def test_reports_parameter_rejected_by_param(self):
        config = Config()
        result = await set_components(ComponentController(components={"config": config}), config={"n_estimators": 1000})
        assert config.n_estimators == 100
        assert "less than or equal to 500" in result

    async def test_rejects_unknown_component_and_parameter(self):
        result = await set_components(ComponentController(components={"config": Config()}), nope=1, config={"bogus": 1})
        assert "`nope` is not a component that can be set" in result
        assert "`config.bogus` cannot be set" in result

    async def test_validates_against_effects_of_earlier_updates(self):
        first = pmui.Select(label="A", options=["x", "y"], value="x")
        second = pmui.Select(label="B", options=[1, 2], value=1)
        first.param.watch(lambda event: second.param.update(options=[1, 2, 3] if event.new == "y" else [1, 2]), "value")
        result = await set_components(ComponentController(components=[first, second]), a="y", b="3")
        assert second.value == 3, result

    async def test_reports_errors_raised_by_callbacks_with_read_back(self, widgets):
        def fail(event):
            raise RuntimeError("boom")

        widgets["slider"].param.watch(fail, "value")
        result = await set_components(ComponentController(components=[widgets["slider"]]), temperature=30)
        assert "`temperature.value` 21.5 → 30.0" in result
        assert "the application raised an error while applying the change: boom" in result
        assert "Also changed" not in result

    async def test_settle_errors_are_reported(self, widgets):
        def settle():
            raise RuntimeError("timeout")

        result = await set_components(ComponentController(components=[widgets["slider"]], settle=settle), temperature=30)
        assert "settle failed: timeout" in result

    async def test_unchanged_values_are_not_reported_as_updates(self):
        config = Config(threshold=None)
        result = await set_components(ComponentController(components={"config": config}), config={"threshold": None})
        assert result == "Updated:\n- `config.threshold` already None"

    async def test_click_errors_are_returned_to_the_llm(self, controller, widgets):
        def fail(event):
            raise RuntimeError("boom")

        widgets["button"].on_click(fail)
        result = await tools_by_name(controller)["click_ui_component"].function(component="reset")
        assert "The application raised an error in response: boom" in result

    async def test_click_triggers_button_and_reports_changes(self, controller, widgets):
        clicks = []
        widgets["button"].on_click(lambda event: (clicks.append(event), setattr(widgets["slider"], "value", 0.0)))
        result = await tools_by_name(controller)["click_ui_component"].function(component="reset")
        assert len(clicks) == 1
        assert result == "Clicked Reset.\nChanged as a result:\n- `temperature.value` 21.5 → 0.0"


class TestLiveSync:

    def test_added_components_are_picked_up(self, page, controller):
        page.main[0].append(pmui.IntSlider(label="Bins", start=1, end=100, value=10))
        assert "bins" in schema(tools_by_name(controller)["set_ui_components"])["properties"]

    def test_removed_components_disappear(self, page, controller, widgets):
        page.main[0].remove(widgets["select"])
        assert "colormap" not in schema(tools_by_name(controller)["set_ui_components"])["properties"]

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


class TestAgent:

    def test_components_build_a_controller(self, page):
        agent = ComponentControlAgent(components=page)
        assert agent.controller.components is page
        assert agent.controller.as_llm_tools in agent.llm_tools

    def test_controller_is_not_shared_between_instances(self):
        assert ComponentControlAgent().controller is not ComponentControlAgent().controller

    def test_setting_components_updates_the_controller(self, page):
        agent = ComponentControlAgent()
        agent.components = page
        assert next(spec.key for spec in agent.controller.specs) == "temperature"

    async def test_applies_only_with_components(self, page):
        assert not await ComponentControlAgent().applies({})
        assert await ComponentControlAgent(components=page).applies({})

    def test_routing_purpose_lists_controls_without_mutating_purpose(self, page):
        agent = ComponentControlAgent(components=[Config(), page])
        purpose = agent.purpose
        routing = agent.routing_purpose({})
        assert routing.endswith(
            "The application currently exposes: config (n_estimators, criterion, normalize, weights, threshold); "
            "temperature; colormap; regions; year_range; day; show_outliers"
        )
        assert agent.purpose == purpose

    async def test_planner_prompt_shows_routing_purpose(self, llm, page):
        agent = ComponentControlAgent(components=page)
        planner = Planner(llm=llm, agents=[ChatAgent(), agent])
        system = await planner._render_prompt("main", [], {}, agents=planner.agents, tools=[], unmet_dependencies=set(), candidates=[], previous_actors=[], previous_plans=[], follow_up_type="new")
        assert "The application currently exposes: temperature; colormap" in system

    def test_controller_tools_stay_off_other_agents(self, llm, page):
        agent = ComponentControlAgent(components=page)
        planner = Planner(llm=llm, agents=[ChatAgent(), agent])
        chat = next(a for a in planner.agents if isinstance(a, ChatAgent))
        assert agent.controller.as_llm_tools not in chat.llm_tools
        assert agent.controller.as_llm_tools not in planner.llm_tools

    async def test_prompt_context_includes_live_state(self, page, widgets):
        agent = ComponentControlAgent(components=page)
        widgets["slider"].value = 12.0
        prompt_context = await agent._gather_prompt_context("main", [], {})
        assert "value: 12.0" in prompt_context["ui_state"]
        assert prompt_context["set_tool"] == "set_ui_components"
