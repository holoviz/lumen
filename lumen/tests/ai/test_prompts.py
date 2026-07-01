"""Conventions and syntax checks for the bundled prompt templates.

These guard the conventions documented in ``lumen/ai/prompts/GUIDANCE.md``:
every template must parse, top-level sections use ``##`` (no stray single-``#``
markdown headers), and headings carry no trailing colon. Markdown-looking lines
inside fenced code blocks (e.g. YAML ``#`` comments) are ignored.
"""
import re

import jinja2
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.config import PROMPTS_DIR

PROMPT_FILES = sorted(PROMPTS_DIR.glob("**/*.jinja2"))

# Single leading '#' followed by a space and content (i.e. a Markdown H1).
_H1 = re.compile(r"^#(?!#)\s+\S")
# Any Markdown heading ending in a colon.
_TRAILING_COLON = re.compile(r"^#{1,6}\s+.*:\s*$")


def _markdown_heading_lines(text):
    """Yield (lineno, line) for heading-looking lines outside ``` fences."""
    in_fence = False
    for i, line in enumerate(text.split("\n"), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("#"):
            yield i, line


def _rel(path):
    return path.relative_to(PROMPTS_DIR)


def test_prompt_files_discovered():
    assert PROMPT_FILES, f"No prompt templates found under {PROMPTS_DIR}"


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_template_parses(path):
    source = path.read_text(encoding="utf-8")
    try:
        jinja2.Environment().parse(source)
    except jinja2.TemplateSyntaxError as exc:
        pytest.fail(f"{_rel(path)} failed to parse: {exc}")


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_no_stray_h1_headers(path):
    source = path.read_text(encoding="utf-8")
    offenders = [
        f"  line {ln}: {line}"
        for ln, line in _markdown_heading_lines(source)
        if _H1.match(line)
    ]
    assert not offenders, (
        f"{_rel(path)} uses single-'#' headers; sections should use '##'/'###':\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_no_trailing_colon_headings(path):
    source = path.read_text(encoding="utf-8")
    offenders = [
        f"  line {ln}: {line}"
        for ln, line in _markdown_heading_lines(source)
        if _TRAILING_COLON.match(line)
    ]
    assert not offenders, (
        f"{_rel(path)} has headings ending in ':' (drop the colon):\n"
        + "\n".join(offenders)
    )
