import argparse

from pathlib import Path

from bokeh.command.subcommand import Argument, Subcommand  # type: ignore
from panel.command import Serve

from ..ui.launcher import Launcher
from ..ui.state import state as ui_state
from ..util import resolve_module_reference


class Builder(Subcommand):

    name = 'builder'

    help = "Launch the Lumen Builder UI"

    args = (
        tuple(
            (name, arg) for name, arg in Serve.args
            if name not in ('files', '--args', '--glob')
        ) + (
        ('--launcher', Argument(
            metavar = 'LAUNCHER',
            type    = str,
            help    = "The Launcher plugin to use",
            default = 'lumen.ui.launcher.LocalLauncher'
        )),
        ('--components', Argument(
            metavar = 'COMPONENTS',
            type    = str,
            help    = "Directory containing components"
        ))
    ))

    def __init__(self, parser: argparse.ArgumentParser, serve: Serve) -> None:
        super().__init__(parser)
        self.serve = serve

    def invoke(self, args: argparse.Namespace) -> None:
        # Set panel serve arguments
        args.glob = False
        args.args = []
        args.files = [str(Path(__file__).parent.parent / 'ui')]

        # Set up UI state
        ui_state.launcher = resolve_module_reference(args.launcher, Launcher)
        if args.components:
            ui_state.components = args.components

        self.serve.invoke(args)
