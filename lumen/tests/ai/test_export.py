import datetime as dt
import json
import os

import pytest

from panel.chat import ChatMessage

from lumen.pipeline import Pipeline
from lumen.sources.intake import IntakeSource
from lumen.views import Table

try:
    from lumen.ai.coordinator import Coordinator
    from lumen.ai.export import (
        LumenOutput, export_notebook, format_markdown, format_output,
        make_preamble,
    )
except ImportError:
    pytest.skip("Skipping tests that require lumen.ai", allow_module_level=True)


@pytest.fixture
def source():
    root = os.path.dirname(__file__)
    parent = os.path.dirname(root) + "/sources"
    return IntakeSource(uri=os.path.join(parent, "catalog.yml"), root=root)


@pytest.fixture
def source_tables(mixed_df):
    df_test = mixed_df.copy()
    df_test_sql = mixed_df.copy()
    df_test_sql_none = mixed_df.copy()
    df_test_sql_none["C"] = ["foo1", None, "foo3", None, "foo5"]
    tables = {
        "test": df_test,
        "test_sql": df_test_sql,
        "test_sql_with_none": df_test_sql_none,
    }
    return tables


@pytest.fixture
def fixed_datetime(monkeypatch):
    fixed_time = dt.datetime(2024, 4, 27, 12, 0, 0)

    class FixedDateTime:
        @classmethod
        def now(cls):
            return fixed_time

    monkeypatch.setattr(dt, "datetime", FixedDateTime)
    return fixed_time


def test_make_preamble(fixed_datetime):
    preamble = "# Initial preamble content"
    cells = make_preamble(preamble)

    assert len(cells) == 2

    # Check the header cell
    header_cell = cells[0]
    assert header_cell.cell_type == "markdown"
    expected_header = f"# Lumen.ai - Chat Logs {fixed_datetime}"
    assert header_cell.source == expected_header

    # Check the imports cell
    imports_cell = cells[1]
    assert imports_cell.cell_type == "code"
    expected_source = (
        "# Initial preamble content\nimport lumen as lm\n"
        "import panel as pn\n\npn.extension('tabulator')"
    )
    assert imports_cell.source == expected_source


def test_format_markdown_with_https_avatar():
    msg = ChatMessage(
        object="This is a test message.",
        user="User",
        avatar="https://example.com/avatar.png",
    )
    cell = format_markdown(msg)[0]

    assert cell.cell_type == "markdown"
    expected_avatar = (
        '<img src="https://example.com/avatar.png" width=45 height=45></img>'
    )
    expected_header = (
        f'<div style="display: flex; flex-direction: row; font-weight: bold; '
        f'font-size: 2em;">{expected_avatar}<span style="margin-left: 0.5em">User</span></div>\n'
    )
    expected_content = "\nThis is a test message."
    expected_source = f"{expected_header}{expected_content}"
    assert cell.source == expected_source


def test_format_markdown_with_text_avatar():
    msg = ChatMessage(
        object="Bot response here.",
        user="Bot",
        avatar="B",
    )
    cell = format_markdown(msg)[0]

    assert cell.cell_type == "markdown"
    expected_avatar = "<span>B</span>"
    expected_header = (
        f'<div style="display: flex; flex-direction: row; font-weight: bold; '
        f'font-size: 2em;">{expected_avatar}<span style="margin-left: 0.5em">Bot</span></div>\n'
    )
    expected_content = "\n> Bot response here."
    expected_source = f"{expected_header}{expected_content}"
    assert cell.source == expected_source


# requires async because internally uses async_executor
async def test_format_output_pipeline(source):
    pipeline = Pipeline(source=source, table="test")
    msg = ChatMessage(object=LumenOutput(component=pipeline), user="User")

    cells = format_output(msg)
    cells[0].pop("id")
    uri = source.uri
    if os.name == "nt":
        uri = uri.replace("\\", "\\\\")
    expected_source = (
        f"pipeline = lm.Pipeline.from_spec({{\n"
        f'  "source": {{\n'
        f'    "uri": "{uri}",\n'
        f'    "type": "intake"\n'
        f"  }},\n"
        f'  "table": "test"\n'
        f"}})\n"
        f"pipeline"
    )

    assert cells == [
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": expected_source,
            "outputs": [],
        }
    ]


# requires async because internally uses async_executor
async def test_format_output_view(source):
    pipeline = Pipeline(source=source, table="test")
    table = Table(pipeline=pipeline)
    msg = ChatMessage(object=LumenOutput(component=table), user="User")

    cells = format_output(msg)
    cells[0].pop("id")
    uri = source.uri
    if os.name == "nt":
        uri = uri.replace("\\", "\\\\")
    assert cells == [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "view = lm.View.from_spec({\n"
            '  "pipeline": {\n'
            '    "source": {\n'
            f'      "uri": "{uri}",\n'
            '      "type": "intake"\n'
            "    },\n"
            '    "table": "test"\n'
            "  },\n"
            '  "type": "table"\n'
            "})\n"
            "view",
        },
    ]


# requires async because internally uses async_executor
async def test_export_notebook(source):
    coordinator = Coordinator()
    coordinator.interface.objecs = [
        ChatMessage(
            object=LumenOutput(component=Pipeline(source=source, table="test")),
            user="User",
        ),
        ChatMessage(
            object=LumenOutput(
                component=Table(pipeline=Pipeline(source=source, table="test"))
            ),
            user="Bot",
        ),
    ]

    cells = json.loads(export_notebook(coordinator.interface))
    cells["cells"][0].pop("source")
    cells["cells"][0].pop("id")
    cells["cells"][1].pop("id")
    assert cells == {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import lumen as lm\n",
                    "import panel as pn\n",
                    "\n",
                    "pn.extension('tabulator')",
                ],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
