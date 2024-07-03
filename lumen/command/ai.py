from __future__ import annotations

import argparse
import ast
import os
import sys

import bokeh.command.util  # type: ignore

from bokeh.application.handlers.code import CodeHandler  # type: ignore
from bokeh.command.util import (  # type: ignore
    build_single_handler_application as _build_application, die,
)
from bokeh.document import Document
from panel.command import Serve, main as _pn_main, transform_cmds
from panel.io.server import Application
from panel.io.state import set_curdoc

from .. import __version__
from ..config import config

SOURCE_CODE = """
import pathlib

import lumen as lm
import lumen.ai as lmai
import panel as pn

from lumen.sources.duckdb import DuckDBSource

pn.extension("tabulator", "codeeditor", inline=False, template="fast")

llm = lmai.llm.OpenAI()

lmai.memory["current_source"] = DuckDBSource(
    tables=["{table}"],
    uri=":memory:",
    initializers=["INSTALL httpfs;", "LOAD httpfs;"],
)

assistant = lmai.Assistant(
    llm=llm,
    agents=[
        lmai.agents.SourceAgent,
        lmai.agents.TableAgent,
        lmai.agents.TableListAgent,
        lmai.agents.SQLAgent,
        lmai.agents.PipelineAgent,
        lmai.agents.hvPlotAgent,
        lmai.agents.ChatAgent,
    ],
)
assistant.servable("Lumen.ai")
assistant.controls().servable(area="sidebar")
"""


class AIHandler(CodeHandler):
    ''' Modify Bokeh documents by using Lumen AI on a dataset.

    '''

    def __init__(self, *args, **kwargs):
        '''

        Keywords:
            filename (str) : a path to a dataset

        '''
        if 'filename' not in kwargs:
            raise ValueError('Must pass a filename to Lumen AI')
        table = os.path.abspath(kwargs['filename'])
        if table.endswith(".parq"):
            table = f"read_parquet('{table}')"
        elif table.endswith(".csv"):
            table = f"read_csv('{table}')"
        elif table.endswith(".json"):
            table = f"read_json_auto('{table}')"
        else:
            raise ValueError('Unsupported file format. Please provide a .parq, .csv, or .json file.')

        kwargs['source'] = SOURCE_CODE.format(table=table)
        super().__init__(*args, **kwargs)


    def modify_document(self, doc: Document) -> None:
        # Temporary fix for issues with --warm flag
        with set_curdoc(doc):
            super().modify_document(doc)


def build_single_handler_application(path, argv):
    if not os.path.isfile(path):
        return _build_application(path, argv)

    handler = AIHandler(filename=path)
    if handler.failed:
        raise RuntimeError("Error loading %s:\n\n%s\n%s " % (path, handler.error, handler.error_detail))

    application = Application(handler)

    return application


bokeh.command.util.build_single_handler_application = build_single_handler_application


def main(args=None):
    """Merges commands offered by pyct and bokeh and provides help for both"""
    start, template_vars = None, None
    for i, arg in enumerate(sys.argv):
        if '--template-vars' in arg:
            start = i
            if '=' in arg:
                end = i
                template_vars = arg.split('=')[1]
            else:
                end = i+1
                template_vars = sys.argv[end]
            break

    if start is not None:
        sys.argv = sys.argv[:start] + sys.argv[end+1:]
        config.template_vars = ast.literal_eval(template_vars)

    _pn_main()

    parser = argparse.ArgumentParser(
        prog="lumen-ai", epilog="See '<command> --help' to read about a specific subcommand."
    )

    parser.add_argument(
        '-v', '--version', action='version', version=__version__
    )

    subs = parser.add_subparsers(help="Sub-commands")

    serve_parser = subs.add_parser(Serve.name, help=Serve.help)
    Serve(parser=serve_parser)

    sys.argv = transform_cmds(sys.argv)
    args = parser.parse_args(sys.argv[1:])

    try:
        ret = args.invoke(args)
    except Exception as e:
        die("ERROR: " + str(e))

    if ret is False:
        sys.exit(1)
    elif ret is not True and isinstance(ret, int) and ret != 0:
        sys.exit(ret)

if __name__ == "__main__":
    main()
