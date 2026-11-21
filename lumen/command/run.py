import argparse
import json
import sys

from pathlib import Path

from bokeh.command.subcommand import Argument, Subcommand


def _pairs(values: list[str] | None, flag: str) -> dict[str, str]:
    pairs = {}
    for value in values or []:
        name, sep, rest = value.partition("=")
        if not sep or not name:
            raise ValueError(f"{flag} takes NAME=VALUE, got {value!r}.")
        pairs[name] = rest
    return pairs


class Run(Subcommand):
    ''' Subcommand to run a FlowDash dashboard of Lumen nodes headlessly.

    '''

    name = "run"

    help = "Runs a FlowDash dashboard without a page or a model and reports each node"

    args = (
        ('dashboard', Argument(
            metavar = 'DASHBOARD',
            help    = "The dashboard file, in the FlowDash YAML format",
        )),
        ('--param', Argument(
            action  = 'append',
            metavar = 'NAME=VALUE',
            help    = "Bind a parameter of the dashboard. A source parameter takes a file, directory, "
                      "URL or connection string, or table=path entries separated by commas.",
        )),
        ('--source', Argument(
            action  = 'append',
            metavar = 'REF=DATA',
            help    = "Give the source nodes naming REF the data DATA, read like a source parameter.",
        )),
        ('--json', Argument(
            action  = 'store_true',
            help    = "Write the report as JSON.",
        )),
        ('--rows', Argument(
            type    = int,
            default = 5,
            help    = "How many rows of each output the JSON report shows (default 5).",
        )),
    )

    def invoke(self, args: argparse.Namespace) -> int:
        from ..flowdash.run import (
            USAGE, RunReport, format_report, run_dashboard,
        )

        try:
            params = _pairs(args.param, "--param")
            sources = _pairs(args.source, "--source")
            text = Path(args.dashboard).read_bytes()
        except (ValueError, OSError) as exc:
            report = RunReport("usage", USAGE, message=str(exc))
        else:
            report = run_dashboard(text, params, sources, rows=args.rows)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))  # noqa: T201
        else:
            print(format_report(report), file=sys.stdout if report.exit_code == 0 else sys.stderr)
        return report.exit_code
