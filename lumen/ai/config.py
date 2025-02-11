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
    "What datasets do you have?",
    "Tell me about the dataset.",
    "Create a plot of the dataset.",
    "Find the min and max of the values.",
]

DEMO_MESSAGES = [
    "What data is available?",
    "Can I see the first one?",
    "Tell me about the dataset.",
    "What could be interesting to analyze?",
    "Perform a SQL query on one of these.",
    "Show it to me as a scatter plot.",
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


pn.chat.ChatStep.min_width = 375
pn.chat.ChatStep.collapsed_on_success = False

disable_pydantic_error_url()
yaml.add_representer(str, str_presenter)
