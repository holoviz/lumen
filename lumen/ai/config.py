from __future__ import annotations

import asyncio

from pathlib import Path, PosixPath

import panel as pn
import yaml

from instructor.utils import disable_pydantic_error_url

from ..config import SOURCE_TABLE_SEPARATOR  # NOQA: F401


class LlmSetupError(Exception):
    """
    Raised when an error occurs during the setup of the LLM.
    """


class RetriesExceededError(Exception):
    """
    Raised when the maximum number of retries is exceeded.
    """


THIS_DIR = Path(__file__).parent
PROMPTS_DIR = THIS_DIR / "prompts"

GETTING_STARTED_SUGGESTIONS = [
    "What datasets are available?",
    "What's interesting to analyze?",
    "Can you visualize the data?",
]

DEMO_MESSAGES = [
    "What datasets are available?",
    "Show me the the first dataset and its columns.",
    "What are some interesting analyses I can do?",
    "Perform an analysis using the first suggestion.",
    "Show me a plot of these results.",
]

DEFAULT_EMBEDDINGS_PATH = Path("embeddings")

UNRECOVERABLE_ERRORS = (
    ImportError,
    LlmSetupError,
    RecursionError,
    asyncio.CancelledError,
)

PROVIDED_SOURCE_NAME = "ProvidedSource00000"

# Vega-lite zoomable map configuration
VEGA_ZOOMABLE_MAP_ITEMS = {
    "params": [
        {"name": "tx", "update": "width / 2"},
        {"name": "ty", "update": "height / 2"},
        {
            "name": "scale",
            "value": 300,
            "on": [{"events": {"type": "wheel", "consume": True}, "update": "clamp(scale * pow(1.003, -event.deltaY * pow(16, event.deltaMode)), 150, 50000)"}],
        },
        {"name": "angles", "value": [0, 0], "on": [{"events": "pointerdown", "update": "[rotateX, centerY]"}]},
        {"name": "cloned", "value": None, "on": [{"events": "pointerdown", "update": "copy('projection')"}]},
        {"name": "start", "value": None, "on": [{"events": "pointerdown", "update": "invert(cloned, xy())"}]},
        {"name": "drag", "value": None, "on": [{"events": "[pointerdown, window:pointerup] > window:pointermove", "update": "invert(cloned, xy())"}]},
        {"name": "delta", "value": None, "on": [{"events": {"signal": "drag"}, "update": "[drag[0] - start[0], start[1] - drag[1]]"}]},
        {"name": "rotateX", "value": -240, "on": [{"events": {"signal": "delta"}, "update": "angles[0] + delta[0]"}]},
        {"name": "centerY", "value": 40, "on": [{"events": {"signal": "delta"}, "update": "clamp(angles[1] + delta[1], -60, 60)"}]},
    ],
    "projection": {
        "scale": {"signal": "scale"},
        "rotate": [{"signal": "rotateX"}, 0, 0],
        "center": [0, {"signal": "centerY"}],
        "translate": [{"signal": "tx"}, {"signal": "ty"}]
    }
}

# Map outline layer configurations
VEGA_MAP_LAYER = {
    "world": {
        "data": {
            "url": "https://vega.github.io/vega-datasets/data/world-110m.json",
            "format": {"type": "topojson", "feature": "countries"}
        },
        "mark": {"type": "geoshape", "fill": None, "stroke": "black"}
    },
    "usa": {
        "data": {
            "url": "https://vega.github.io/vega-datasets/data/us-10m.json",
            "format": {"type": "topojson", "feature": "states"}
        },
        "mark": {"type": "geoshape", "fill": None, "stroke": "black"}
    }
}

def str_presenter(dumper, data):
    if "\n" in data:  # Only use literal block for strings containing newlines
        return dumper.represent_scalar("tag:yaml.org,2002:str", data.strip(), style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def tuple_presenter(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)


def path_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.resolve()))

pn.chat.ChatStep.min_width = 375
pn.chat.ChatStep.collapsed_on_success = False

disable_pydantic_error_url()
yaml.add_representer(str, str_presenter)
yaml.add_representer(tuple, tuple_presenter)
yaml.add_representer(PosixPath, path_representer)
yaml.add_representer(Path, path_representer)
