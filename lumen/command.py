import argparse
import ast
import os
import sys
import yaml

import bokeh.command.util

from bokeh.application import Application
from bokeh.application.handlers.code import CodeHandler
from bokeh.command.util import build_single_handler_application as _build_application, die
from panel.command import main as _pn_main

from . import __version__
from .sources import Source
from .util import expand_spec


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
        filename = kwargs['filename']
        kwargs['source'] = f"from lumen import Dashboard; Dashboard('{filename}').servable();"
        super().__init__(*args, **kwargs)

        # Initialize cached and shared sources
        from . import config
        root = os.path.abspath(os.path.dirname(filename))
        with open(filename) as f:
            yaml_spec = f.read()
        expanded = expand_spec(yaml_spec, config.template_vars)
        spec = yaml.load(expanded, Loader=yaml.Loader)
        for name, source_spec in spec.get('sources', {}).items():
            if source_spec.get('shared'):
                config.sources[name] = source = Source.from_spec(
                    source_spec, config.sources, root=root)
                if source.cache_dir:
                    source.clear_cache()


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
        from . import config
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
