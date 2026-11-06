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

import re

from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_cell_magic, magics_class

from .answer import ask

_USAGE = "%lumen source [--as name] [--confirm] question"
_FLAG = re.compile(r"\s*--(?:(confirm)|as[ =]\s*(\S+))(?=\s|$)")


def _parse(line: str) -> tuple[str, str, bool, str]:
    """
    The source, the name to keep the answer as, whether to confirm, and the
    rest of the line verbatim. Only the source and the flags before the
    question are parsed, so quotes and braces in the question are left alone.
    """
    source, _, rest = line.strip().partition(" ")
    if not source or source.startswith("--"):
        raise UsageError(f"Start with the variable to ask: {_USAGE}")
    name, confirm = "_lumen", False
    while match := _FLAG.match(rest):
        if match.group(1):
            confirm = True
        else:
            name = match.group(2)
        rest = rest[match.end():]
    return source, name, confirm, rest.strip()


@magics_class
class LumenMagics(Magics):

    @line_cell_magic
    def lumen(self, line: str, cell: str | None = None):
        source, name, confirm, words = _parse(line)
        namespace = self.shell.user_ns
        if source not in namespace:
            raise UsageError(f"No variable named {source!r} to ask; read your data with lumen.read first.")
        if cell is None:
            # IPython has already filled in {name} on a magic's line
            question = words
        else:
            try:
                question = cell.strip().format_map(namespace)
            except KeyError as e:
                raise UsageError(f"The question names {{{e.args[0]}}}, which is not a variable here.") from e
            except (ValueError, IndexError, AttributeError) as e:
                raise UsageError(
                    f"The question has a brace that is not a variable ({e}); write {{{{ and }}}} for a literal brace."
                ) from e
        if not question:
            raise UsageError("Write the question after the source, or in the cell below %%lumen.")
        answer = ask(namespace[source], question, confirm=confirm)
        namespace[name] = answer
        return answer


def load_ipython_extension(ipython) -> None:
    ipython.register_magics(LumenMagics)
