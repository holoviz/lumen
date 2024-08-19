from __future__ import annotations

import argparse
import ast
import glob
import os
import sys

from textwrap import dedent

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

pn.extension("tabulator", "codeeditor", "filedropper", inline=False, template="fast")

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
        lmai.agents.AnalysisAgent(analyses=[lmai.analysis.Join])
    ],
)
assistant.servable("Lumen.ai")
assistant.controls().servable(area="sidebar")
"""

VALID_EXTENSIONS = ['.parq', '.parquet', '.csv', '.json']

class AIHandler(CodeHandler):
    ''' Modify Bokeh documents by using Lumen AI on a dataset.

    '''

    def __init__(self, no_data: bool = False, filename: str | None = None, **kwargs) -> None:
        '''

        Keywords:
            filename (str) : the path to the dataset or a stringified list of paths to multiple datasets
                used as tables under the same source. Wildcards are supported.
            no_data (bool) : if True, launch app without data

        '''
        table_initializer = ""
        expanded_files = []
        if no_data:
            filename = 'no_data'
        elif filename:
            input_tables = ast.literal_eval(filename)
            if isinstance(input_tables, str):
                input_tables = [input_tables]

            for pattern in input_tables:
                pattern = pattern.strip()
                expanded_files.extend([f for f in glob.glob(pattern) if any(f.endswith(ext) for ext in VALID_EXTENSIONS)])

            if not expanded_files:
                raise ValueError(f"No valid files found matching the pattern(s) provided: {input_tables}")

            expanded_files = list(set(expanded_files))
            tables = []
            for table in expanded_files:
                if table.endswith(('.parq', '.parquet')):
                    table = f"read_parquet('{table}')"
                elif table.endswith(".csv"):
                    table = f"read_csv('{table}')"
                elif table.endswith(".json"):
                    table = f"read_json_auto('{table}')"
                tables.append(table)

            table_initializer = dedent(
                f"""
                lmai.memory["current_source"] = DuckDBSource(
                    tables={tables},
                    uri=":memory:",
                )
                """
            )

        source = SOURCE_CODE.format(table_initializer=table_initializer)
        super().__init__(filename=filename, source=source, **kwargs)


def build_single_handler_application(tables: str | None, argv):
    if tables is None or (not os.path.isfile(tables) and "[" not in tables):
        handler = AIHandler(no_data=True)
    else:
        handler = AIHandler(filename=tables)

    if handler.failed:
        raise RuntimeError(f"Error loading {tables}:\n\n{handler.error}\n{handler.error_detail} ")

    application = Application(handler)
    return application

def build_single_handler_applications(paths: list[str], argvs: dict[str, list[str]] | None = None) -> dict[str, Application]:
    ''' Custom to allow for standalone `lumen-ai` command to launch without data'''
    applications: dict[str, Application] = {}
    argvs = argvs or {}

    if 'no_data' in sys.argv:
        application = build_single_handler_application(None, [])
    else:
        tables = str(paths)
        application = build_single_handler_application(tables, argvs.get(tables, []))
    applications['/lumen_ai'] = application
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
        import traceback
        traceback.print_exc()
        die("ERROR: " + str(e))

    if ret is False:
        sys.exit(1)
    elif ret is not True and isinstance(ret, int) and ret != 0:
        sys.exit(ret)


if __name__ == "__main__":
    main()
