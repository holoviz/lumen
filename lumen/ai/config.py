from __future__ import annotations

import asyncio

from pathlib import Path

import panel as pn
import yaml

from instructor.utils import disable_pydantic_error_url


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

SOURCE_TABLE_SEPARATOR = "__@__"
PROVIDED_SOURCE_NAME = "ProvidedSource00000"


def str_presenter(dumper, data):
    if "\n" in data:  # Only use literal block for strings containing newlines
        return dumper.represent_scalar("tag:yaml.org,2002:str", data.strip(), style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

def tuple_presenter(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)

pn.chat.ChatStep.min_width = 375
pn.chat.ChatStep.collapsed_on_success = False

disable_pydantic_error_url()
yaml.add_representer(str, str_presenter)
yaml.add_representer(tuple, tuple_presenter)
