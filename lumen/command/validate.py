import argparse

from pathlib import Path

from bokeh.command.subcommand import Argument, Subcommand


class Validate(Subcommand):
    ''' Subcommand to validate a Lumen YAML.

    '''

    name = "validate"

    help = "Validates a Lumen YAML"

    args = (
        ('file', Argument(
            metavar = 'YAML-FILE',
            help    = "The application to validate",
            default = None,
        )),
    )

    def invoke(self, args: argparse.Namespace):
        from ..dashboard import Dashboard, load_yaml
        spec = load_yaml(Path(args.file).read_text())
        Dashboard.validate(spec)
