from pathlib import Path

import panel as pn


class LlmSetupError(Exception):
    """
    Raised when an error occurs during the setup of the LLM.
    """


THIS_DIR = Path(__file__).parent

FUZZY_TABLE_LENGTH = 10

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
)

pn.chat.ChatStep.min_width = 350
