"""
The %%lumen and %lumen magics: ask a source a question from a cell.

    %load_ext lumen.ai.notebook

    %%lumen penguins --as heaviest
    Which species was heaviest in {year}?

The cell is the question, with `{name}` filled in from the notebook's
variables (`{{` and `}}` for literal braces). The answer is shown and kept
as `heaviest`, or `_lumen` without --as. `%lumen penguins How many rows?`
asks in one line, where IPython fills in `{name}` itself, and --confirm
previews the plan as ask(confirm=True) does.
"""
from __future__ import annotations

import argparse
import shlex

from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_cell_magic, magics_class

from .answer import ask

_parser = argparse.ArgumentParser(prog="%lumen", add_help=False, exit_on_error=False)
_parser.add_argument("source")
_parser.add_argument("--as", dest="name", default="_lumen")
_parser.add_argument("--confirm", action="store_true")


@magics_class
class LumenMagics(Magics):

    @line_cell_magic
    def lumen(self, line: str, cell: str | None = None):
        try:
            args, words = _parser.parse_known_args(shlex.split(line))
        except (argparse.ArgumentError, ValueError) as e:
            raise UsageError(f"%lumen source [--as name] [--confirm] question: {e}") from e
        namespace = self.shell.user_ns
        if args.source not in namespace:
            raise UsageError(f"No variable named {args.source!r} to ask; read your data with lumen.read first.")
        if cell is None:
            # IPython has already filled in {name} on a magic's line
            question = " ".join(words).strip()
        else:
            try:
                question = cell.strip().format_map(namespace)
            except KeyError as e:
                raise UsageError(f"The question names {{{e.args[0]}}}, which is not a variable here.") from e
        if not question:
            raise UsageError("Write the question after the source, or in the cell below %%lumen.")
        answer = ask(namespace[args.source], question, confirm=args.confirm)
        namespace[args.name] = answer
        return answer


def load_ipython_extension(ipython) -> None:
    ipython.register_magics(LumenMagics)
