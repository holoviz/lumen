from __future__ import annotations

import argparse
import sys

import bokeh.command.util  # type: ignore

from bokeh.application.handlers.code import CodeHandler  # type: ignore
from bokeh.command.util import die  # type: ignore
from panel.command import Serve, transform_cmds
from panel.io.application import Application

SOURCE_CODE = """
import lumen.ai as lmai
lmai.ExplorerUI([{tables}]).servable()"""

VALID_EXTENSIONS = ['.parq', '.parquet', '.csv', '.json']


class AIHandler(CodeHandler):
    ''' Modify Bokeh documents by using Lumen AI on a dataset.

    '''

    def __init__(self, tables: list[str], **kwargs) -> None:
        tables = list({table for table in tables if any(table.endswith(ext) for ext in VALID_EXTENSIONS)})
        source = SOURCE_CODE.format(tables=','.join([repr(t) for t in tables]))
        super().__init__(filename='lumen_ai.py', source=source, **kwargs)


def build_single_handler_applications(paths: list[str], argvs: dict[str, list[str]] | None = None) -> dict[str, Application]:
    ''' Custom to allow for standalone `lumen-ai` command to launch without data'''
    handler = AIHandler(paths)
    if handler.failed:
        raise RuntimeError(f"Error loading {paths}:\n\n{handler.error}\n{handler.error_detail} ")
    return {'/lumen_ai': Application(handler)}

bokeh.command.subcommands.serve.build_single_handler_applications = build_single_handler_applications


def main(args=None):
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
    serve_parser = subs.add_parser(
        Serve.name, help="""
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
