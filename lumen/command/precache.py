import argparse

from pathlib import Path

from bokeh.command.subcommand import Argument, Subcommand

from lumen.config import config
from lumen.state import state
from lumen.variables import Variables


class Precache(Subcommand):
    ''' Subcommand to run precaching pass on all sources.

    '''

    name = "precache"

    help = "Precaches Lumen YAML"

    args = (
        ('files', Argument(
            metavar = 'DIRECTORY-OR-SCRIPT',
            nargs   = '*',
            help    = "The app directories or scripts to serve (serve empty document if not specified)",
            default = None,
        )),
    )

    def invoke(self, args: argparse.Namespace):
        from ..dashboard import load_yaml
        for yaml_file in args.files:
            config.root = str(Path(yaml_file).parent)
            state.spec = load_yaml(Path(yaml_file).read_text())
            variables = Variables.from_spec(state.spec.get('variables', {}))
            for var_name, var in variables._vars.items():
                state.variables[var_name] = var
            for name, source_spec in state.spec['sources'].items():
                state.load_source(name, source_spec)
            state.load_pipelines(auto_update=False)
            state.reset()
