"""
Palmer Penguins explorer with a natural language copilot.

Every widget on the page is discovered by the `ComponentController`, which
hands one tool per widget to the assistant docked on the right, so the
conversation drives the dashboard:

- "Which controls do you have and what are they set to?"
- "Show only Gentoo penguins from Biscoe"
- "Plot flipper length against body mass, colour by sex and make the points bigger"
- "Narrow it down to the heaviest third of the penguins"
- "Reset the filters"

Run it with::

    panel serve examples/ai/penguin_copilot.py --show
"""

import hvplot.pandas  # noqa
import pandas as pd
import panel as pn
import panel_material_ui as pmui

from lumen.ai import Planner
from lumen.ai.agents import ChatAgent, ComponentControlAgent
from lumen.ai.llm import OpenAI
from lumen.ai.tools import ComponentController

pn.extension(notifications=True)

DATA_URL = "https://datasets.holoviz.org/penguins/v1/penguins.csv"
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

df = pd.read_csv(DATA_URL).dropna()

species = pmui.MultiChoice(
    label="Species", options=sorted(df.species.unique()), value=sorted(df.species.unique()),
    description="Species to include; an empty selection includes all of them",
)

islands = pmui.MultiChoice(
    label="Islands", options=sorted(df.island.unique()), value=sorted(df.island.unique()),
    description="Islands the penguins were observed on",
)

sex = pmui.RadioButtonGroup(
    label="Sex", options=["all", "male", "female"], value="all",
    description="Restrict the selection to one sex",
)

bill_length = pmui.RangeSlider(
    label="Bill length", start=30, end=60, step=0.5, value=(30, 60),
    description="Range of bill lengths in millimetres",
)

body_mass = pmui.RangeSlider(
    label="Body mass", start=2500, end=6500, step=50, value=(2500, 6500),
    description="Range of body masses in grams",
)

x = pmui.Select(label="X axis", options=MEASURES, value="bill_length_mm")

y = pmui.Select(label="Y axis", options=MEASURES, value="body_mass_g")

color_by = pmui.Select(
    label="Colour by", options=["species", "island", "sex"], value="species",
    description="Column the points and histograms are grouped and coloured by",
)

point_size = pmui.IntSlider(
    label="Point size", start=20, end=200, step=10, value=60,
    description="Size of the scatter markers",
)

show_distribution = pmui.Switch(
    label="Show distribution", value=True,
    description="Whether to show the histogram below the scatter plot",
)

reset = pmui.Button(label="Reset filters", icon="restart_alt", variant="outlined")

FILTERS = (species, islands, sex, bill_length, body_mass)


def reset_filters(event):
    for widget in FILTERS:
        widget.value = widget.param.value.default


reset.on_click(reset_filters)


def filter_data(species, islands, sex, bill_length, body_mass):
    data = df
    if species:
        data = data[data.species.isin(species)]
    if islands:
        data = data[data.island.isin(islands)]
    if sex != "all":
        data = data[data.sex == sex]
    return data[data.bill_length_mm.between(*bill_length) & data.body_mass_g.between(*body_mass)]


def scatter(data, x, y, color_by, point_size):
    if data.empty:
        return pmui.Alert("No penguins match the current filters.", severity="warning", sizing_mode="stretch_width")
    return data.hvplot.scatter(
        x=x, y=y, by=color_by, size=point_size, alpha=0.7, responsive=True,
        height=380, legend="top_left", hover_cols=["species", "island", "sex"],
    ).opts(toolbar="above")


def distribution(data, x, color_by, show_distribution):
    if not show_distribution or data.empty:
        return None
    return data.hvplot.hist(
        y=x, by=color_by, bins=25, alpha=0.6, responsive=True, height=220, legend=False,
    ).opts(toolbar=None)


def stats(data, y):
    if data.empty:
        return pmui.Row()
    cards = [("Penguins", f"{len(data):,}"), ("Species", f"{data.species.nunique()}")]
    cards.append((f"Mean {y.replace('_', ' ')}", f"{data[y].mean():.1f}"))
    return pmui.Row(*[
        pmui.Paper(
            pmui.Typography(title, variant="overline"),
            pmui.Typography(value, variant="h5"),
            elevation=1, sizing_mode="stretch_width", styles={"padding": "8px 16px"},
        )
        for title, value in cards
    ], sizing_mode="stretch_width")


data = pn.bind(filter_data, species, islands, sex, bill_length, body_mass)

page = pmui.Page(
    title="Penguin Explorer",
    sidebar=[
        pmui.Column(
            pmui.Typography("Filters", variant="overline"),
            species, islands, sex, bill_length, body_mass, reset,
            pmui.Divider(),
            pmui.Typography("Plot", variant="overline"),
            x, y, color_by, point_size, show_distribution,
        )
    ],
    main=[
        pn.panel(pn.bind(stats, data, y), sizing_mode="stretch_width"),
        pmui.Card(
            pn.panel(pn.bind(scatter, data, x, y, color_by, point_size), sizing_mode="stretch_width"),
            pn.panel(pn.bind(distribution, data, x, color_by, show_distribution), sizing_mode="stretch_width"),
            title="Measurements", sizing_mode="stretch_width",
        ),
    ],
    sidebar_width=320,
)

controller = ComponentController(
    components=page,
    purpose="""
        Filters and plot options of a dashboard exploring the Palmer Penguins
        dataset, which measures the bills, flippers and body mass of penguins
        observed on three islands.""",
    descriptions={"reset_filters": "Restores every filter to its default, i.e. selects all the penguins again"},
)

assistant = Planner(
    agents=[ChatAgent, ComponentControlAgent(controller=controller)],
    llm=OpenAI(),
)

assistant.interface.send(
    "I can drive this dashboard. Try *'show only Gentoo penguins from Biscoe'* or "
    "*'colour by sex and plot flipper length against body mass'*.",
    user="Assistant", respond=False,
)

page.main.append(
    pmui.Drawer(
        assistant, anchor="right", variant="docked", size=460,
        dock_icon="chat", dock_position="middle",
    )
)

page.servable()
