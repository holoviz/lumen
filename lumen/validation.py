from __future__ import annotations

import re
import textwrap

from collections.abc import Callable, Sequence
from difflib import get_close_matches
from inspect import signature
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from inspect import Signature

BOLD = '\033[1m'
END = '\033[0m'


def get_param_info(sig: Signature) -> tuple[list[str], list[Any]]:
    ''' Find parameters with defaults and return them.

    Arguments:
        sig (Signature) : a function signature

    Returns:
        tuple(list, list) : parameters with defaults

    '''
    defaults = []
    for param in sig.parameters.values():
        if param.default is not param.empty:
            defaults.append(param.default)
    return [name for name in sig.parameters], defaults

def validate_callback(
    callback: Callable[..., Any], fargs: Sequence[str],
    what: str ="Callback functions"
) -> None:
    '''Bokeh-internal function to check callback signature'''
    sig = signature(callback)
    formatted_args = str(sig)
    error_msg = what + " must have signature func(%s), got func%s"

    all_names, default_values = get_param_info(sig)

    nargs = len(all_names) - len(default_values)
    if nargs != len(fargs):
        raise ValueError(error_msg % (", ".join(fargs), formatted_args))

def validate_parameters(params: list[str], expected: list[str], name: str) -> None:
    for p in params:
        if p not in expected:
            first_parth_msg = f"'{p}' not a parameter for '{name}'"
            msg = match_suggestion_message(p, expected, first_parth_msg)
            raise ValueError(msg)

def match_suggestion_message(
    word: str,
    possibilities: list[str],
    msg: str = '',
    n: int = 3
) -> str:
    match = get_close_matches(word, possibilities, n)
    if match:
        if len(match) > 1:
            match_str = "', '".join(match[:-1]) + f" or '{match[-1]}"
        else:
            match_str = match[0]
        if msg:
            msg = msg[:-1] if msg.endswith(".") else msg
            msg = msg + f". Did you mean '{match_str}'?"
        else:
            msg = f"Did you mean '{match_str}'?"
    return msg

def reverse_match_suggestion(
    word: str,
    possibilities: list[str],
    msg: str
) -> tuple[str, str | None]:
    match = get_close_matches(word, list(possibilities), n=1)
    if match:
        msg = f'{msg} Did you mean {word!r}?'
        return msg, match[0]
    return msg, None


class ValidationError(ValueError):
    """
    A ValidationError is raised when the specification of a component has missing
    required keys, an incorrect value or is otherwise malformed.
    """

    def __init__(self, msg, spec=None, attr=None):
        if spec:
            snippet = yaml.dump(spec, sort_keys=False)
            if attr:
                snippet = re.sub(rf"\b{attr}\b", f'{BOLD}{attr}{END}', snippet)
            snippet = textwrap.indent(snippet, '    ')
            if '!!python' not in snippet:
                msg = f'{msg}\n\n{snippet}'
        super().__init__(msg)
