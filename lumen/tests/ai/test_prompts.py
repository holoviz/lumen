"""Conventions and syntax checks for the bundled prompt templates.

These guard the conventions documented in ``lumen/ai/prompts/GUIDANCE.md``:
every template must parse, top-level sections use ``##`` (no stray single-``#``
markdown headers), and headings carry no trailing colon. Markdown-looking lines
inside fenced code blocks (e.g. YAML ``#`` comments) are ignored.
"""
import re
import unicodedata

import jinja2
import pytest

try:
    import lumen.ai
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from pathlib import Path

from lumen.ai.config import PROMPTS_DIR

PROMPT_FILES = sorted(PROMPTS_DIR.glob("**/*.jinja2"))
AI_DIR = Path(lumen.ai.__file__).parent

# Single leading '#' followed by a space and content (i.e. a Markdown H1).
_H1 = re.compile(r"^#(?!#)\s+\S")
# Any Markdown heading ending in a colon.
_TRAILING_COLON = re.compile(r"^#{1,6}\s+.*:\s*$")
# A Jinja tag or expression that strips the whitespace on *both* sides. Placed
# directly under a heading, the leading '-' eats the newline that separates the
# heading from its body, so the two render glued together.
_BOTH_SIDES_STRIPPED = re.compile(r"^\{[%{]-.*-[%}]\}$")
# Guard on a key's presence rather than its value, e.g. ``{% if 'sql' in memory %}``.
_EXISTENCE_GUARD = re.compile(r"\{%-?\s*if\s+(['\"][^'\"]+['\"])\s+in\s+memory\s*-?%\}")
# Words the house style reserves for genuinely load-bearing rules.
_EMPHASIS = re.compile(r"\b(CRITICAL|MUST|NEVER|ALWAYS|IMPORTANT)\b")
_MAX_EMPHASIS = 2
# ``PROMPTS_DIR / "Actor" / "main.jinja2"`` as written in the Python sources.
_TEMPLATE_REGISTRATION = re.compile(
    r"""PROMPTS_DIR\s*((?:/\s*["'][^"']+["']\s*)+)"""
)
# '★' marks derived tables in schemas.py output, so templates must be able to
# name it; the degree sign appears in example unit strings.
_ALLOWED_SYMBOLS = {"★", "°"}


def _markdown_heading_lines(text):
    """Yield (lineno, line) for heading-looking lines outside ``` fences."""
    in_fence = False
    for i, line in enumerate(text.split("\n"), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("#"):
            yield i, line


def _heading_text(line):
    return re.sub(r"^#+\s*", "", line).strip()


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


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_headings_not_glued_to_body(path):
    """
    A heading followed by a ``{%- ... -%}`` tag renders glued to its body.

    The tag's leading ``-`` strips the newline after the heading, so
    ``## Instructions`` + ``{%- if x -%}`` renders as ``## InstructionsBody...``.
    This is invisible in the source and only shows up in the rendered prompt,
    which is why it needs a mechanical check.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    headings = {ln for ln, _ in _markdown_heading_lines("\n".join(lines))}
    offenders = [
        f"  line {ln}: {lines[ln - 1]}\n    followed by: {lines[ln].strip()}"
        for ln in sorted(headings)
        if ln < len(lines) and _BOTH_SIDES_STRIPPED.match(lines[ln].strip())
    ]
    assert not offenders, (
        f"{_rel(path)} has heading(s) whose following Jinja tag strips the newline "
        f"after them, gluing the heading to its body. Drop the tag's leading '-':\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_guards_test_emptiness_not_existence(path):
    """
    ``{% if 'k' in memory %}`` renders a dangling header when ``memory['k']`` is
    empty or ``None``, since the key is present either way. Guard on the value.
    """
    offenders = [
        f"  line {ln}: {line.strip()}"
        for ln, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1)
        if _EXISTENCE_GUARD.search(line)
    ]
    assert not offenders, (
        f"{_rel(path)} guards on key existence, which renders a dangling section "
        f"when the value is empty. Use `memory.get('k')` (or "
        f"`memory.get('k') is not none` when an empty value is still meaningful):\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_no_emoji(path):
    """
    One voice per prompt: prose, not the terse caps-and-emoji dialect. Emoji also
    cost tokens without adding information the words don't already carry.
    """
    offenders = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        found = {
            char
            for char in line
            if ord(char) > 127
            and char not in _ALLOWED_SYMBOLS
            and (unicodedata.category(char) in ("So", "Sk") or ord(char) == 0xFE0F)
        }
        if found:
            offenders.append(f"  line {ln}: {''.join(sorted(found))} in {line.strip()}")
    assert not offenders, (
        f"{_rel(path)} uses emoji/pictographs; state the meaning in words instead:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_emphasis_budget(path):
    """
    A rule stated once clearly beats the same rule restated with rising emphasis.
    Cap the shouty words so they keep their signal where they are genuinely needed.
    """
    matches = _EMPHASIS.findall(path.read_text(encoding="utf-8"))
    assert len(matches) <= _MAX_EMPHASIS, (
        f"{_rel(path)} uses {len(matches)} emphasis markers "
        f"({', '.join(sorted(set(matches)))}), over the budget of {_MAX_EMPHASIS}. "
        f"Reserve CRITICAL/MUST/NEVER/ALWAYS/IMPORTANT for load-bearing rules."
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_headings_are_not_all_caps(path):
    """Headings use Title Case; ``## EXAMPLES`` and ``## Examples`` are one section."""
    offenders = []
    for ln, line in _markdown_heading_lines(path.read_text(encoding="utf-8")):
        text = _heading_text(line)
        letters = [char for char in text if char.isalpha()]
        if letters and all(char.isupper() for char in letters):
            offenders.append(f"  line {ln}: {line}")
    assert not offenders, (
        f"{_rel(path)} has ALL-CAPS headings; use Title Case:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(_rel(p)))
def test_examples_blocks_are_headed_and_marked_illustrative(path):
    """
    Example content renders in the same prompt, with the same formatting, as the
    user's live data. A heading separates the two and the illustrative note tells
    the model which is which, so it does not cite placeholder figures as findings.
    """
    source = path.read_text(encoding="utf-8")
    if not re.search(r"\{%-?\s*block\s+examples\s*-?%\}", source):
        return
    body = re.split(r"\{%-?\s*block\s+examples\s*-?%\}", source, maxsplit=1)[1]
    body = re.split(r"\{%-?\s*endblock", body, maxsplit=1)[0]
    if not body.strip():
        # Base templates declare an empty block for children to override.
        return

    headings = [_heading_text(line) for _, line in _markdown_heading_lines(body)]
    examples = [h for h in headings if h.lower().startswith("example")]
    assert examples, (
        f"{_rel(path)} has a non-empty examples block with no '## Examples' heading; "
        f"add one so example content is delimited from the user's live data."
    )
    assert any("illustrative" in h.lower() for h in examples), (
        f"{_rel(path)} has example headings {examples} that are not marked "
        f"illustrative. Use e.g. '## Examples (illustrative — names below are "
        f"placeholders, not real data)' so the model does not cite them as data."
    )


def _registered_templates():
    """Yield (source_file, lineno, path) for every ``PROMPTS_DIR / ...`` in the code."""
    for source in sorted(AI_DIR.glob("**/*.py")):
        for ln, line in enumerate(source.read_text(encoding="utf-8").split("\n"), start=1):
            for match in _TEMPLATE_REGISTRATION.finditer(line):
                parts = re.findall(r"""["']([^"']+)["']""", match.group(1))
                yield source, ln, PROMPTS_DIR.joinpath(*parts)


def test_registered_templates_exist():
    """
    Every ``PROMPTS_DIR / ...`` path named in the code resolves to a file on disk.

    A registration pointing at a missing template raises only when that prompt is
    first rendered, which for a base class whose subclasses all override it may be
    never — so the typo survives indefinitely.
    """
    registrations = list(_registered_templates())
    assert registrations, "Found no template registrations to check"
    missing = [
        f"  {source.relative_to(AI_DIR)}:{ln} -> {template.relative_to(PROMPTS_DIR)}"
        for source, ln, template in registrations
        if not template.exists()
    ]
    assert not missing, (
        "These registered prompt templates do not exist on disk:\n" + "\n".join(missing)
    )
