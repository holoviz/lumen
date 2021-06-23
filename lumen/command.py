import argparse
import ast
import os
import sys

import bokeh.command.util

from bokeh.application.handlers.code import CodeHandler
from bokeh.command.util import build_single_handler_application as _build_application, die
from panel.command import main as _pn_main
from panel.io.server import Application

from . import __version__
from .config import config
from .dashboard import Defaults, load_yaml
from .state import state


class YamlHandler(CodeHandler):
    ''' Modify Bokeh documents by creating Dashboard from Lumen yaml spec.

    '''

    def __init__(self, *args, **kwargs):
        '''

        Keywords:
            filename (str) : a path to a Yaml (".yaml"/".yml") file

        '''
        if 'filename' not in kwargs:
            raise ValueError('Must pass a filename to YamlHandler')
        filename = os.path.abspath(kwargs['filename'])
        kwargs['source'] = f"from lumen import Dashboard; Dashboard('{filename}', load_global=False).servable();"
        super().__init__(*args, **kwargs)

        if filename not in config.yamls:
            config.yamls.append(filename)

        # Initialize cached and shared sources
        with open(filename) as f:
            yaml_spec = f.read()
        state.spec = spec = load_yaml(yaml_spec)
        config._root = os.path.abspath(os.path.dirname(filename))
        warm = any(flag in sys.argv for flag in ('--dev', '--autoreload', '--warm'))
        Defaults.from_spec(spec.get('defaults', {})).apply()
        if warm:
            config.load_local_modules()
            state.load_global_sources(clear_cache=not config.dev)


def build_single_handler_application(path, argv):
    if not os.path.isfile(path) or not (path.endswith(".yml") or path.endswith(".yaml")):
        return _build_application(path, argv)

    handler = YamlHandler(filename=path, arg=argv)
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

    if len(sys.argv) == 1 or sys.argv[1] not in ('-v', '--version'):
        _pn_main()
        return

    parser = argparse.ArgumentParser(
        prog="lumen", epilog="See '<command> --help' to read about a specific subcommand."
    )

    parser.add_argument(
        '-v', '--version', action='version', version=__version__
    )

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
