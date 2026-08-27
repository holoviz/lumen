"""Read hvPlot's option descriptions out of its converter docstring.

hvPlot's explorer controls carry the type, default and bounds of every option
they expose, but roughly half of them document nothing, and a param's ``doc`` is
what the plot agent's model turns into the description the LLM reads. The prose
for those options exists, in ``HoloViewsConverter``'s class docstring, so it is
read from there rather than restated here.
"""
from __future__ import annotations

import inspect
import logging
import re

from functools import cache

from griffe import Docstring, DocstringSectionKind  # type: ignore
from hvplot.converter import HoloViewsConverter  # type: ignore

# hvPlot groups its options under headings of its own ("Data Options",
# "Geographic Options", ...). griffe's numpy parser only treats a section as
# parameters when it is titled "Parameters", and reads everything else as an
# admonition, so the headings are renamed before it sees them.
_SECTION_UNDERLINE = re.compile(r'^-+$')

# Options sharing a description are documented on one line, e.g.
# "logx/logy : bool". griffe keeps the first name and discards the rest.
_GROUPED_OPTION = re.compile(r'^(\w+(?:/\w+)+)\s*:', re.MULTILINE)

# reStructuredText that is meaningful in rendered docs and noise in a prompt.
_ROLE = re.compile(r':\w+:`[^`]*?([^`:/]+)`')
_LITERAL = re.compile(r'``([^`]+)``')
_DIRECTIVE = re.compile(r'\.\.\s+\w+::')

# A sentence break, minus the abbreviations hvPlot's prose actually uses.
_SENTENCE = re.compile(r'(?<!\be\.g)(?<!\bi\.e)(?<!\betc)\.(?:\s|$)')

MAX_DESCRIPTION = 200


def _as_parameters(doc: str) -> str:
    """Retitle hvPlot's option sections so griffe parses them as parameters."""
    lines = doc.split('\n')
    retitled: list[str] = []
    index = 0
    while index < len(lines) - 1:
        title, underline = lines[index], lines[index + 1]
        if title.strip() and not title.startswith(' ') and _SECTION_UNDERLINE.match(underline.strip()):
            retitled += ['Parameters', '-' * len('Parameters')]
            index += 2
            continue
        retitled.append(lines[index])
        index += 1
    return '\n'.join(retitled + lines[-1:])


def _summarize(description: str) -> str:
    """Reduce a docstring entry to one plain sentence."""
    description = _DIRECTIVE.split(description)[0]
    description = _ROLE.sub(r'\1', description)
    description = _LITERAL.sub(r'\1', description)
    description = ' '.join(description.split())
    sentence = _SENTENCE.split(description)[0].strip(' ,;')
    if len(sentence) >= MAX_DESCRIPTION:
        sentence = sentence[:MAX_DESCRIPTION - 1].rsplit(' ', 1)[0]
    return f'{sentence}.' if sentence else ''


def _grouped_names(doc: str) -> dict[str, list[str]]:
    """Map the surviving name of each grouped entry to the names griffe dropped."""
    groups = {}
    for match in _GROUPED_OPTION.finditer(doc):
        first, *rest = match.group(1).split('/')
        groups[first] = rest
    return groups


@cache
def hvplot_param_docs() -> dict[str, str]:
    """Map an hvPlot option name to a one-sentence description of it."""
    doc = inspect.cleandoc(HoloViewsConverter.__doc__ or '')
    # griffe reports every continuation line hvPlot indents by three spaces, and
    # every grouped entry it could not type. None of it is actionable here.
    griffe_log = logging.getLogger('griffe')
    level = griffe_log.level
    griffe_log.setLevel(logging.ERROR)
    try:
        sections = Docstring(_as_parameters(doc), lineno=1, parser='numpy').parse()
    finally:
        griffe_log.setLevel(level)
    docs = {
        parameter.name: summary
        for section in sections if section.kind is DocstringSectionKind.parameters
        for parameter in section.value
        if (summary := _summarize(parameter.description or ''))
    }
    if not docs:
        raise RuntimeError(
            'Could not read any option descriptions from hvPlot. Its converter '
            'docstring is laid out differently than expected.'
        )
    for first, rest in _grouped_names(doc).items():
        for name in rest:
            docs.setdefault(name, docs.get(first, ''))
    return docs
