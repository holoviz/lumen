"""
Outbreak simulator whose scenario is steered by a conversation.

Unlike `penguin_copilot.py`, which walks a whole page for widgets, this demo
hands the `ComponentController` two `param.Parameterized` objects directly: the
model, whose parameters and docstrings are all the assistant needs to reason
about the simulation, and a read-only `Outcome` whose values it can read back
after every change. Because it can both act and observe, the assistant in the
drawer on the right can iterate towards a goal instead of just flipping
switches:

- "What can you control here?"
- "Model a measles-like outbreak in a city of three million"
- "What if we start vaccinating on day 40 at one percent of the population a day?"
- "Save this as the baseline, then find the weakest lockdown that keeps hospital load under 100%"
- "Halve the peak without locking down harder than 50%"

Run it with::

    panel serve examples/ai/epidemic_copilot.py --show
"""

import holoviews as hv
import hvplot.pandas  # noqa
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param

from bokeh.models import NumeralTickFormatter

from lumen.ai import Planner
from lumen.ai.agents import ChatAgent, ComponentControlAgent
from lumen.ai.llm import OpenAI
from lumen.ai.tools import ComponentController

pn.extension(notifications=True, respect_explicit_sizing=True)

HOSPITALISATION_RATE = 0.02

COMPARTMENTS = {
    "Susceptible": "susceptible",
    "Exposed": "exposed",
    "Infected": "infected",
    "Recovered": "recovered",
    "Vaccinated": "vaccinated",
    "Deaths": "deaths",
}

GOOD, BAD, DEMAND = "#2e7d32", "#c62828", "#1976d2"


class Epidemic(param.Parameterized):
    """Compartmental SEIR model of an outbreak in a single, closed population."""

    population = param.Integer(default=1_000_000, bounds=(10_000, 20_000_000), doc="""
        Number of people in the population.""")

    initial_infections = param.Integer(default=50, bounds=(1, 10_000), doc="""
        Number of people already infectious on day zero.""")

    r0 = param.Number(default=2.5, bounds=(0.5, 20.0), doc="""
        Basic reproduction number, i.e. how many people the average case infects
        while everybody around them is still susceptible.""")

    incubation_days = param.Number(default=5.0, bounds=(0.5, 21.0), doc="""
        Average number of days between catching the disease and becoming
        infectious.""")

    infectious_days = param.Number(default=7.0, bounds=(1.0, 30.0), doc="""
        Average number of days a case stays infectious.""")

    fatality_rate = param.Number(default=0.6, bounds=(0.0, 60.0), doc="""
        Percentage of cases that end in death.""")

    intervention_day = param.Integer(default=0, bounds=(0, 365), doc="""
        Day on which an intervention such as a lockdown starts. Zero means no
        intervention at all.""")

    intervention_strength = param.Number(default=0.4, bounds=(0.0, 0.95), doc="""
        Fraction by which the intervention reduces transmission, e.g. 0.5 halves
        the rate at which the disease spreads.""")

    vaccination_day = param.Integer(default=0, bounds=(0, 365), doc="""
        Day on which the vaccination campaign starts. Zero means no campaign.""")

    vaccination_rate = param.Number(default=0.5, bounds=(0.0, 5.0), doc="""
        Percentage of the population vaccinated per day once the campaign has
        started.""")

    vaccine_efficacy = param.Number(default=90.0, bounds=(0.0, 100.0), doc="""
        Percentage of the vaccinated who actually become immune.""")

    hospital_beds_per_100k = param.Number(default=30.0, bounds=(1.0, 500.0), doc="""
        Hospital beds available per 100,000 people; two percent of the people
        infected at any one time are assumed to need one.""")

    days = param.Integer(default=240, bounds=(30, 730), doc="""
        Number of days to simulate.""")

    @property
    def hospital_beds(self) -> float:
        return self.hospital_beds_per_100k * self.population / 100_000

    def simulate(self) -> pd.DataFrame:
        beta = self.r0 / self.infectious_days
        onset_rate = 1 / self.incubation_days
        removal_rate = 1 / self.infectious_days
        fatality = self.fatality_rate / 100
        doses = self.population * self.vaccination_rate / 100 * self.vaccine_efficacy / 100
        susceptible = float(self.population - self.initial_infections)
        infected = float(self.initial_infections)
        exposed = recovered = vaccinated = deaths = 0.0
        rows = []
        for day in range(self.days + 1):
            rows.append((day, susceptible, exposed, infected, recovered, vaccinated, deaths))
            locked_down = self.intervention_day and day >= self.intervention_day
            transmission = beta * (1 - self.intervention_strength if locked_down else 1)
            infections = transmission * susceptible * infected / self.population
            onsets = onset_rate * exposed
            removals = removal_rate * infected
            immunised = doses if self.vaccination_day and day >= self.vaccination_day else 0.0
            immunised = min(immunised, max(susceptible - infections, 0.0))
            susceptible -= infections + immunised
            exposed += infections - onsets
            infected += onsets - removals
            recovered += removals * (1 - fatality)
            deaths += removals * fatality
            vaccinated += immunised
        data = pd.DataFrame(rows, columns=["day", *COMPARTMENTS.values()])
        return data.assign(hospitalised=data.infected * HOSPITALISATION_RATE)


class Outcome(param.Parameterized):
    """Results of the most recent simulation."""

    peak_infections = param.Integer(default=0, constant=True, doc="""
        Largest number of people infected at the same time.""")

    peak_day = param.Integer(default=0, constant=True, doc="""
        Day on which the number of infections peaks.""")

    peak_hospital_load = param.Number(default=0.0, constant=True, doc="""
        Beds needed at the peak as a percentage of the beds available; anything
        above 100 means the health system is overwhelmed.""")

    overflow_days = param.Integer(default=0, constant=True, doc="""
        Number of days on which the demand for hospital beds exceeds capacity.""")

    deaths = param.Integer(default=0, constant=True, doc="""
        Total number of deaths over the course of the outbreak.""")

    infected_share = param.Number(default=0.0, constant=True, doc="""
        Percentage of the population infected over the course of the outbreak.""")

    peak_vs_baseline = param.Number(default=0.0, constant=True, doc="""
        Change in peak infections relative to the saved baseline scenario, in
        percent; negative means this scenario is milder than the baseline.""")


class Reference(param.Parameterized):
    """The saved scenario the current one is compared against."""

    label = param.String(default="baseline")

    data = param.DataFrame(default=None)


PRESETS = {
    "Custom": {},
    "Seasonal flu": dict(r0=1.4, incubation_days=1.5, infectious_days=5.0, fatality_rate=0.1),
    "COVID-like": dict(r0=2.5, incubation_days=5.0, infectious_days=7.0, fatality_rate=0.6),
    "Measles-like": dict(r0=14.0, incubation_days=10.0, infectious_days=8.0, fatality_rate=0.15),
    "Ebola-like": dict(r0=1.8, incubation_days=9.0, infectious_days=10.0, fatality_rate=45.0),
}

DISEASE_PARAMETERS = list(PRESETS["COVID-like"])


def scenario_label(model: Epidemic) -> str:
    label = f"R0 {model.r0:g}"
    if model.intervention_day:
        label += f", day {model.intervention_day} -{model.intervention_strength:.0%}"
    if model.vaccination_day:
        label += f", vaccines day {model.vaccination_day}"
    return label


model = Epidemic()
outcome = Outcome()
reference = Reference(data=model.simulate(), label=scenario_label(model))

preset = pmui.Select(
    label="Scenario preset", options=list(PRESETS), value="COVID-like",
    description="Disease profile to load; picking one overwrites the disease parameters of the model",
)

curves = pmui.MultiChoice(
    label="Curves", options=list(COMPARTMENTS), value=["Infected", "Deaths"],
    description="Compartments to draw in the epidemic curves plot",
)

scale = pmui.RadioButtonGroup(
    label="Y axis", options=["linear", "log"], value="linear",
    description="Scale of the y axis",
)

capacity_line = pmui.Switch(
    label="Show capacity", value=True,
    description="Whether to mark the available hospital beds and shade the demand exceeding them",
)

compare = pmui.Switch(
    label="Compare to baseline", value=True,
    description="Whether to overlay the infections of the saved baseline scenario",
)

save = pmui.Button(label="Save as baseline", icon="bookmark_add", variant="outlined")

reset = pmui.Button(label="Reset scenario", icon="restart_alt", variant="outlined")

def apply_preset(event):
    values = PRESETS[event.new]
    if values:
        model.param.update(**values)


def sync_preset(*events):
    """
    Keep the preset honest about the disease the model describes.

    Applying a preset lands back here, but the name it resolves to is the one
    that was just selected, so param drops the update and nothing loops.
    """
    preset.value = next(
        (name for name, values in PRESETS.items()
         if values and all(getattr(model, key) == value for key, value in values.items())),
        "Custom",
    )


def reset_scenario(event):
    model.param.update(**{
        name: parameter.default for name, parameter in model.param.objects().items()
        if name != "name"
    })


def save_baseline(event):
    reference.param.update(data=model.simulate(), label=scenario_label(model))
    if pn.state.notifications:
        pn.state.notifications.success(f"Saved '{reference.label}' as the baseline.")


def format_value(value) -> str:
    return f"{value:,}" if isinstance(value, int) else f"{value:,g}"


def announce(*events):
    if not pn.state.notifications:
        return
    changes = ", ".join(f"{event.name.replace('_', ' ')} {format_value(event.new)}" for event in events)
    pn.state.notifications.info(f"Scenario updated: {changes}")


preset.param.watch(apply_preset, "value")
model.param.watch(sync_preset, DISEASE_PARAMETERS)
model.param.watch(announce, [name for name in model.param if name != "name"])
reset.on_click(reset_scenario)
save.on_click(save_baseline)


def simulate(baseline, **parameters):
    """
    Run the simulation and record its outcome.

    The model parameters are passed in only so that the bound views re-render
    whenever a slider (or the assistant) changes one of them; ``baseline`` does
    the same for the comparison recorded on the outcome.
    """
    data = model.simulate()
    beds = model.hospital_beds
    peak = float(data.infected.max())
    baseline_peak = None if baseline is None else float(baseline.infected.max())
    with param.parameterized.edit_constant(outcome):
        outcome.param.update(
            peak_infections=int(peak),
            peak_day=int(data.day[data.infected.idxmax()]),
            peak_hospital_load=round(100 * float(data.hospitalised.max()) / beds, 1),
            overflow_days=int((data.hospitalised > beds).sum()),
            deaths=int(data.deaths.iloc[-1]),
            infected_share=round(100 * float(data.recovered.iloc[-1] + data.deaths.iloc[-1]) / model.population, 1),
            peak_vs_baseline=0.0 if not baseline_peak else round(100 * (peak / baseline_peak - 1), 1),
        )
    return data


def metric(title, value, caption, colour=None):
    return pmui.Paper(
        pmui.Typography(title, variant="overline", sx={"opacity": 0.6, "lineHeight": 1.5}),
        pmui.Typography(value, variant="h5", sx={"color": colour} if colour else {}),
        pmui.Typography(caption, variant="caption", sx={"opacity": 0.6}),
        elevation=2, styles={"padding": "6px 16px", "flex": "1 1 170px", "max-width": "320px"},
    )


def metrics(data, baseline, label):
    load, change = outcome.peak_hospital_load, outcome.peak_vs_baseline
    return pn.FlexBox(
        metric("Peak infections", f"{outcome.peak_infections:,}", f"on day {outcome.peak_day}"),
        metric("Total infected", f"{outcome.infected_share:g}%", "of the population"),
        metric("Deaths", f"{outcome.deaths:,}", f"{100 * outcome.deaths / model.population:.2f}% of the population"),
        metric(
            "Peak hospital load", f"{load:g}%",
            f"over capacity on {outcome.overflow_days} days" if outcome.overflow_days else "within capacity",
            BAD if load > 100 else GOOD,
        ),
        metric(
            "Peak vs baseline", f"{change:+g}%", f"baseline: {label}",
            BAD if change > 0 else GOOD if change < 0 else None,
        ),
        sizing_mode="stretch_width", gap="10px", margin=(10, 10, 0, 10),
    )


def headline(data):
    beds = model.hospital_beds
    summary = (
        f"Infections peak at {outcome.peak_infections:,} on day {outcome.peak_day}, "
        f"{outcome.infected_share:g}% of the population is infected in total and "
        f"{outcome.deaths:,} people die."
    )
    if outcome.overflow_days:
        return pmui.Alert(
            f"{summary} Demand peaks at {outcome.peak_hospital_load:g}% of the {beds:,.0f} available "
            f"beds and stays above capacity for {outcome.overflow_days} days.",
            severity="error", sizing_mode="stretch_width",
        )
    return pmui.Alert(
        f"{summary} Hospital demand peaks at {outcome.peak_hospital_load:g}% of the "
        f"{beds:,.0f} available beds, so the health system copes.",
        severity="success", sizing_mode="stretch_width",
    )


def outbreak_plot(data, curves, scale, compare, baseline, label):
    if not curves:
        return pmui.Alert("Select at least one compartment to plot.", severity="info", sizing_mode="stretch_width")
    columns = [COMPARTMENTS[curve] for curve in curves]
    if scale == "log":
        # a log axis cannot render the zeros the compartments start out at
        data = data.assign(**{column: data[column].mask(data[column] < 1) for column in columns})
    elements = [data.hvplot.line(
        x="day", y=columns, responsive=True, height=340, logy=scale == "log",
        ylabel="people", yformatter=NumeralTickFormatter(format="0,0"), grid=True, line_width=2,
    )]
    if model.intervention_day:
        elements.append(hv.VLine(model.intervention_day).opts(color="#616161", line_dash="dotted"))
    if model.vaccination_day:
        elements.append(hv.VLine(model.vaccination_day).opts(color="#0288d1", line_dash="dotted"))
    if compare and baseline is not None:
        elements.append(hv.Curve(baseline, "day", "infected", label=f"infected ({label})").opts(
            color="black", line_dash="dashed", line_width=1.5, alpha=0.7
        ))
    return hv.Overlay(elements).opts(toolbar="above", legend_position="top_right", active_tools=[])


def hospital_plot(data, capacity_line):
    # expressed as a share of capacity, so the 100% line is what matters
    load = data.assign(load=100 * data.hospitalised / model.hospital_beds)
    elements = [load.hvplot.area(
        x="day", y="load", responsive=True, height=220, alpha=0.35,
        color=DEMAND, ylabel="% of capacity", grid=True, label="demand",
    )]
    if capacity_line:
        # a band between the capacity and the demand shades the unmet demand
        band = load.assign(capacity=100.0, over=load.load.clip(lower=100))
        elements.append(hv.Area(band, "day", ["capacity", "over"], label="over capacity").opts(
            color=BAD, alpha=0.5, line_alpha=0
        ))
        elements.append(hv.HLine(100).opts(color="#212121", line_dash="dashed", line_width=1.5))
    return hv.Overlay(elements).opts(toolbar=None, legend_position="top_right")


data = pn.bind(
    simulate, reference.param.data,
    **{name: model.param[name] for name in model.param if name != "name"},
)

main = pn.Row(
    pn.Column(
        pn.panel(pn.bind(metrics, data, reference.param.data, reference.param.label), sizing_mode="stretch_width"),
        pn.panel(pn.bind(headline, data), sizing_mode="stretch_width", margin=10),
        pmui.Card(
            pn.panel(
                pn.bind(outbreak_plot, data, curves, scale, compare, reference.param.data, reference.param.label),
                sizing_mode="stretch_width",
            ),
            title="Epidemic curves", sizing_mode="stretch_width", margin=10,
        ),
        pmui.Card(
            pn.panel(pn.bind(hospital_plot, data, capacity_line), sizing_mode="stretch_width"),
            title="Hospital demand", sizing_mode="stretch_width", margin=10,
        ),
    ),
    sizing_mode="stretch_both"
)

page = pmui.Page(
    title="Outbreak Simulator",
    sidebar=[
        pmui.Column(
            preset,
            pmui.Accordion(
                ("Disease", pmui.Column(
                    pmui.FloatSlider.from_param(model.param.r0, step=0.1),
                    pmui.FloatSlider.from_param(model.param.incubation_days, step=0.5),
                    pmui.FloatSlider.from_param(model.param.infectious_days, step=0.5),
                    pmui.FloatSlider.from_param(model.param.fatality_rate, step=0.1),
                )),
                ("Response", pmui.Column(
                    pmui.IntSlider.from_param(model.param.intervention_day, step=5),
                    pmui.FloatSlider.from_param(model.param.intervention_strength, step=0.05),
                    pmui.IntSlider.from_param(model.param.vaccination_day, step=5),
                    pmui.FloatSlider.from_param(model.param.vaccination_rate, step=0.1),
                    pmui.FloatSlider.from_param(model.param.vaccine_efficacy, step=5),
                )),
                ("Population", pmui.Column(
                    pmui.IntSlider.from_param(model.param.population, step=50_000),
                    pmui.IntSlider.from_param(model.param.initial_infections, step=10),
                    pmui.FloatSlider.from_param(model.param.hospital_beds_per_100k, step=5),
                    pmui.IntSlider.from_param(model.param.days, step=10),
                )),
                active=[0, 1], sizing_mode="stretch_width",
            ),
            pmui.Row(reset, save),
            pmui.Divider(),
            pmui.Typography("Plot", variant="overline"),
            curves, scale, capacity_line, compare,
        )
    ],
    main=[main],
    sidebar_width=340,
)

controller = ComponentController(
    components={
        "model": model,
        "outcome": outcome,
        "scenario_preset": preset,
        "curves": curves,
        "y_axis": scale,
        "capacity_line": capacity_line,
        "compare_to_baseline": compare,
        "save_baseline": save,
        "reset_scenario": reset,
    },
    # Outcome parameters are constant, so they are reported to the LLM but no
    # tool is generated for them; listing them makes the result of a change
    # readable and lets the assistant iterate towards a target.
    parameters={"outcome": [
        "peak_infections", "peak_day", "peak_hospital_load", "overflow_days",
        "deaths", "infected_share", "peak_vs_baseline",
    ]},
    descriptions={
        "model": "The epidemiological scenario that is re-simulated whenever one of its parameters changes",
        "outcome": "Read-only results of the latest simulation; read these back after changing the model",
        "save_baseline": "Stores the current scenario as the baseline that later scenarios are compared against",
        "reset_scenario": "Restores every model parameter to its default",
    },
    purpose="""
        Model and display options of an outbreak simulator. The model is
        re-simulated on every change and the outcome reflects the result, so a
        scenario can be tuned towards a target such as keeping the peak
        hospital load below 100%.""",
)

assistant = Planner(
    agents=[ChatAgent, ComponentControlAgent(controller=controller)],
    # Searching for a setting that meets a target takes many write-then-read
    # rounds; the smaller models tend to stop after the first write.
    llm=OpenAI(model_kwargs={"default": {"model": "gpt-5.4"}, "ui": {"model": "gpt-5.4-mini"}}),
)

assistant.interface.send(
    "I can drive this simulation. Try *'model a measles-like outbreak in a city of three million'* or "
    "*'find the weakest lockdown that keeps hospital load under 100%'*.",
    user="Assistant", respond=False,
)

main.append(
    pmui.Drawer(
        assistant, anchor="right", variant="docked",
        dock_icon="chat", dock_position="end",
        inline=True, size=400, width_policy="min"
    )
)

page.servable()
