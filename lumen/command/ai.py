from __future__ import annotations

import argparse
import ast
import os
import sys

import bokeh.command.util  # type: ignore

from bokeh.application.handlers.code import CodeHandler  # type: ignore
from bokeh.command.util import die  # type: ignore
from panel.command import Serve, transform_cmds
from panel.io.server import Application

from ..config import config

SOURCE_CODE = """
import pathlib

import lumen as lm
import lumen.ai as lmai
import panel as pn

from lumen.sources.duckdb import DuckDBSource

pn.extension("tabulator", "codeeditor", inline=False, template="fast")

llm = lmai.llm.OpenAI()

{table_initializer}

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
            no_data (bool) : if True, do not load data

        '''
        table_initializer = ""
        if 'filename' in kwargs:
            table = os.path.abspath(kwargs['filename'])
            if table.endswith(".parq") or table.endswith(".parquet"):
                table = f"read_parquet('{table}')"
            elif table.endswith(".csv"):
                table = f"read_csv('{table}')"
            elif table.endswith(".json"):
                table = f"read_json_auto('{table}')"
            else:
                raise ValueError('Unsupported file format. Please provide a .parq, .parquet, .csv, or .json file.')

            table_initializer = f"""
lmai.memory["current_source"] = DuckDBSource(
    tables=["{table}"],
    uri=":memory:",
)
            """

        if 'no_data' in kwargs:
            kwargs.pop('no_data')
            kwargs['filename'] = 'no_data'

        kwargs['source'] = SOURCE_CODE.format(table_initializer=table_initializer)
        super().__init__(*args, **kwargs)


def build_single_handler_application(path: str | None, argv):

    if path is None or not os.path.isfile(path):
        handler = AIHandler(no_data=True)
    else:
        handler = AIHandler(filename=path)

    if handler.failed:
        raise RuntimeError("Error loading %s:\n\n%s\n%s " % (path, handler.error, handler.error_detail))

    application = Application(handler)
    return application

def build_single_handler_applications(paths: list[str], argvs: dict[str, list[str]] | None = None) -> dict[str, Application]:
    ''' Custom to allow for standalone `lumen-ai` command to launch without data'''
    applications: dict[str, Application] = {}
    argvs = argvs or {}

    if 'no_data' in sys.argv:
        application = build_single_handler_application(None, [])
        applications['/'] = application
    else:
        for path in paths:
            application = build_single_handler_application(path, argvs.get(path, []))

            route = application.handlers[0].url_path()

            if not route:
                if '/' in applications:
                    raise RuntimeError(f"Don't know the URL path to use for {path}")
                route = '/'
            applications[route] = application

    return applications

bokeh.command.util.build_single_handler_application = build_single_handler_application
bokeh.command.subcommands.serve.build_single_handler_applications = build_single_handler_applications

def main(args=None):
    start, template_vars = None, None
    for i, arg in enumerate(sys.argv):
        if '--template-vars' in arg:
            start = i
            if '=' in arg:
                end = i
                template_vars = arg.split('=')[1]
            else:
                end = i + 1
                template_vars = sys.argv[end]
            break

    if start is not None:
        sys.argv = sys.argv[:start] + sys.argv[end + 1:]
        config.template_vars = ast.literal_eval(template_vars)

    parser = argparse.ArgumentParser(
        prog="lumen-ai",
        description="""
        Lumen AI - Launch Lumen AI applications easily.\n\n To start the application without any
          data, simply run 'lumen-ai' with no additional arguments. You can upload data through
          the chat interface afterwards.
          """,
        epilog="See '<command> --help' to read about a specific subcommand."
    )

    parser.add_argument(
        '-v', '--version', action='version', version='Lumen AI 1.0.0'
    )

    subs = parser.add_subparsers(help="Sub-commands")

    serve_parser = subs.add_parser(Serve.name, help=
                                   """
                                   Run a bokeh server to serve the Lumen AI application.
                                   This command should be followed by dataset paths or directories
                                   to add to the chat memory, which can be a .parq, .parquet, .csv,
                                   or .json file. run `lumen-ai serve --help` for more options)
                                   """)
    serve_command = Serve(parser=serve_parser)
    serve_parser.set_defaults(invoke=serve_command.invoke)

    if len(sys.argv) > 1 and sys.argv[1] in ('--help', '-h'):
        args = parser.parse_args(sys.argv[1:])
        args.invoke(args)
        sys.exit()

    if len(sys.argv) == 1:
    # If no command is specified, start the server with an empty application
        sys.argv.extend(['serve', 'no_data'])

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
