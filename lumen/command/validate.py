import argparse

from pathlib import Path

import yaml

from bokeh.command.subcommand import Argument, Subcommand

from ..validation import ValidationError


class Validate(Subcommand):
    ''' Subcommand to validate a Lumen YAML.

    '''

    name = "validate"

    help = "Validates a Lumen YAML"

    args = (
        ('files', Argument(
            metavar = 'DIRECTORY-OR-SCRIPT',
            nargs   = '*',
            help    = "The app directories or scripts to serve (serve empty document if not specified)",
            default = None,
        )),
    )

    def invoke(self, args: argparse.Namespace):
        from ..dashboard import Dashboard, load_yaml
        for yaml_file in args.files:
            spec = load_yaml(Path(yaml_file).read_text())
            try:
                output = Dashboard.validate(spec)
            except ValidationError as e:
                print(e)
                continue
            if output == spec:
                print(f'Validation found no issues with {yaml_file}.')
            else:
                print(f'Validation made a number of changes to {yaml_file}.\n\n', yaml.dump(output))
