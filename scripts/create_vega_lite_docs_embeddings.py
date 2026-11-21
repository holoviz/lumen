import argparse
import asyncio
import hashlib
import json
import re
import statistics

from collections import Counter

import panel as pn
import requests

from lumen.ai.config import (
    LUMEN_CACHE_DIR, VEGA_LITE_DOCS_NUMPY_DB_FILE,
    VEGA_LITE_DOCS_OPENAI_DB_FILE,
)
from lumen.ai.embeddings import Embeddings, NumpyEmbeddings, OpenAIEmbeddings
from lumen.ai.llm import OpenAI as OpenAILLM
from lumen.ai.vector_store import DuckDBVectorStore
from lumen.config import dump_yaml

# Pinned to the last v5 release, because lumen's normalize_vegalite_spec stamps
# every generated spec with `.../vega-lite/v5.json`. Tracking `next` would drift
# onto v6 docs and feed v6-only properties into a v5 pipeline, producing
# validation failures the model has no way to diagnose. Bump this in step with
# whatever normalize_vegalite_spec targets.
VL_REF = "v5.23.0"
REPO_RAW = f"https://raw.githubusercontent.com/vega/vega-lite/{VL_REF}"
RAW_BASE = f"{REPO_RAW}/site/docs"
EXAMPLES_BASE = f"{REPO_RAW}/examples/specs"
SITE_BASE = "https://vega.github.io/vega-lite"
DOCS_BASE = f"{SITE_BASE}/docs"

MAX_SECTION_CHARS = 4000
MIN_SECTION_CHARS = 120
# Docs pages and example specs are immutable at a pinned ref, so cache the HTTP
# responses on disk: otherwise every rebuild refetches ~420 URLs before any
# embedding work starts. VL_REF is in the path because pn.cache keys on the
# function argument (a bare docs path or example name), which does not change
# when the ref does -- so bumping VL_REF alone would serve stale v5 content.
HTTP_CACHE_PATH = str(LUMEN_CACHE_DIR / f"vega_lite_docs_http_{VL_REF}")
# Matches the placeholder VegaLiteAgent/main.jinja2 uses in its own examples.
TABLE_PLACEHOLDER = "<TABLE_NAME>"
# A merge should attach a runt to nearby context, not accumulate a mega-chunk.
# Without this cap, selection.md collapsed 9 sections into 4 and diluted the
# specific ones (Point Selection Properties) below retrieval range.
MERGE_TARGET_MAX = 1200

PAGES = [
    ("Overview / Structural", [
        ("Spec", "spec.md"),
        ("Title", "view/title.md"),
        ("Size", "view/size.md"),
        ("Data", "data.md"),
    ]),
    ("Transforms", [
        # Overview pages live at <dir>/<dir>.md (same as composition/composition.md).
        ("Transform", "transform/transform.md"),
        ("Aggregate", "transform/aggregate.md"),
        ("Bin", "transform/bin.md"),
        ("Calculate", "transform/calculate.md"),
        ("Density", "transform/density.md"),
        ("Filter", "transform/filter.md"),
        ("Flatten", "transform/flatten.md"),
        ("Fold", "transform/fold.md"),
        ("Impute", "transform/impute.md"),
        ("Join Aggregate", "transform/joinaggregate.md"),
        ("Loess", "transform/loess.md"),
        ("Lookup", "transform/lookup.md"),
        ("Pivot", "transform/pivot.md"),
        ("Quantile", "transform/quantile.md"),
        ("Regression", "transform/regression.md"),
        ("Sample", "transform/sample.md"),
        ("Stack", "transform/stack.md"),
        ("Time Unit", "transform/timeunit.md"),
        ("Window", "transform/window.md"),
    ]),
    ("Marks", [
        ("Mark", "mark/mark.md"),
        ("Arc", "mark/arc.md"),
        ("Area", "mark/area.md"),
        ("Bar", "mark/bar.md"),
        ("Box Plot", "mark/boxplot.md"),
        ("Circle", "mark/circle.md"),
        ("Error Band", "mark/errorband.md"),
        ("Error Bar", "mark/errorbar.md"),
        ("Geoshape", "mark/geoshape.md"),
        ("Image", "mark/image.md"),
        ("Line", "mark/line.md"),
        ("Point", "mark/point.md"),
        ("Rect", "mark/rect.md"),
        ("Rule", "mark/rule.md"),
        ("Square", "mark/square.md"),
        ("Text", "mark/text.md"),
        ("Tick", "mark/tick.md"),
        ("Trail", "mark/trail.md"),
    ]),
    ("Encoding", [
        # Channel index (x, y, color, size, shape, tooltip, order, row, column,
        # text, href, detail) -- highest-value page for request -> spec mapping.
        ("Encoding", "encoding.md"),
        ("Axis", "encoding/axis.md"),
        # encoding/band.md exists on `next` (v6) but not at v5.23.0.
        ("Condition", "encoding/condition.md"),
        ("Datum", "encoding/datum.md"),
        ("Field", "encoding/field.md"),
        ("Format", "encoding/format.md"),
        ("Header", "encoding/header.md"),
        ("Legend", "encoding/legend.md"),
        ("Scale", "encoding/scale.md"),
        ("Sort", "encoding/sort.md"),
        ("Type", "encoding/type.md"),
        ("Value", "encoding/value.md"),
    ]),
    ("Projection", [
        ("Projection", "projection.md"),
    ]),
    ("View Composition", [
        ("View Composition", "composition/composition.md"),
        ("Facet", "composition/facet.md"),
        ("Layer", "composition/layer.md"),
        ("Concat", "composition/concat.md"),
        ("Repeat", "composition/repeat.md"),
        ("Resolve", "composition/resolve.md"),
    ]),
    ("Parameters", [
        ("Parameter", "parameter/parameter.md"),
        ("Value", "parameter/value.md"),
        ("Bind", "parameter/bind.md"),
        ("Select", "parameter/select.md"),
    ]),
    ("Config", [
        ("Config", "config.md"),
    ]),
    ("Property Types", [
        ("Date Time", "types/datetime.md"),
        ("Gradient", "types/gradient.md"),
        ("Predicate", "types/predicate.md"),
        ("Tooltip", "tooltip.md"),
    ]),
]


@pn.cache(to_disk=True, cache_path=HTTP_CACHE_PATH)
def fetch_markdown(path: str) -> str:
    url = f"{RAW_BASE}/{path}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


@pn.cache(to_disk=True, cache_path=HTTP_CACHE_PATH)
def fetch_example(name: str) -> str | None:
    """Fetch a live-example spec by its `data-name`, or None if it does not exist.

    Only a 404 counts as absence. Everything else raises, because the result is
    cached to disk: swallowing a timeout here would record "this example does not
    exist" permanently, and the run resumes cheaply from cache anyway.
    """
    response = requests.get(f"{EXAMPLES_BASE}/{name}.vl.json", timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text.strip()


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Strip Jekyll frontmatter, returning (meta, body).

    The `permalink` key is the only reliable source of the published URL --
    Vega-Lite flattens nested paths, so `mark/bar.md` lives at `/docs/bar.html`,
    not `/docs/mark-bar.html`.
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[match.end():]


def _expand_include(match: re.Match) -> str:
    """Render `{% include table.html props="a,b" source="Config" %}` as prose.

    On the reference pages the property table *is* the content, so deleting the
    include leaves an empty section and throws away the exact property names a
    retrieval query would match on.
    """
    tag = match.group(0)
    props = re.search(r"props=[\"']([^\"']+)[\"']", tag)
    if not props:
        return ""
    source = re.search(r"source=[\"']([^\"']+)[\"']", tag)
    names = ", ".join(p.strip() for p in props.group(1).split(",") if p.strip())
    prefix = f"{source.group(1)} properties" if source else "Properties"
    return f"{prefix}: {names}."


# Kramdown attribute lists. bar.md contains a malformed `{.#bar-width}` (dot
# instead of colon), so match that shape too. Both forms are narrow enough that
# they cannot collide with JSON, whose keys are always quoted.
KRAMDOWN_ATTR_RE = re.compile(r"\{:[^}\n]*\}|\{\.#?[A-Za-z0-9_\-]*\}")

# Live examples live in separate spec files, referenced from the docs only by
# `data-name`. Carry that name through sectioning as a sentinel so the spec can
# be attached to whichever section it belongs to (see attach_examples).
EXAMPLE_SENTINEL_RE = re.compile(r"\[\[vl-example:([A-Za-z0-9_.\-]+)\]\]")


def _capture_example(match: re.Match) -> str:
    name = re.search(r"data-name=[\"']([^\"']+)[\"']", match.group(0))
    return f"\n[[vl-example:{name.group(1)}]]\n" if name else ""


# Structural HTML that survives example removal: anchor targets, images, wrapper
# divs, and the interactive form controls in the selection docs. Stripped outside
# fences only, so JSON/JS samples stay intact.
RESIDUAL_HTML = re.compile(
    r"</?(?:div|span|a|img|br|p|center|small|select|option|input|button|label|form|textarea)\b[^>]*/?>"
)


def _sub_outside_fences(text: str, pattern: re.Pattern, repl: str = "") -> str:
    out = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(pattern.sub(repl, line))
    return "\n".join(out)


def clean_markdown(text: str, strip_link_urls: bool = True) -> str:
    # Liquid: expand the property tables, then drop every remaining tag
    # ({% assign %}, {% raw %}, {% highlight %}, ...).
    text = re.sub(r"\{%\s*include\s+table\.html\b[^%]*?%\}", _expand_include, text)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{\s*site\.baseurl\s*\}\}", SITE_BASE, text)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    # Live-example embeds render client side. Keep the spec name as a sentinel;
    # the JSON itself is resolved later into metadata, not into the text.
    # The class attribute may carry extras (e.g. "vl-example vl-example-responsive").
    text = re.sub(
        r"<(span|div)\b[^>]*\bclass=\"[^\"]*\bvl-example\b[^\"]*\"[^>]*>.*?</\1>",
        _capture_example,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Inline <script>/<style> bodies are code, not prose; drop tag and contents.
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = _sub_outside_fences(text, RESIDUAL_HTML)

    # Kramdown TOC block and attribute lists ({:#anchor}, {:.no_toc}, ...).
    text = re.sub(r"^[ \t]*-[ \t]*TOC[ \t]*\n[ \t]*\{:toc\}[ \t]*$", "", text, flags=re.MULTILINE)
    text = KRAMDOWN_ATTR_RE.sub("", text)

    text = re.sub(r"\[Edit this page\]\([^)]*\)", "", text)

    if strip_link_urls:
        # URLs are pure token noise in an embedding; the canonical page URL is
        # already carried in metadata.
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _iter_headed_blocks(markdown: str, level: int):
    """Yield (title, body) split on `level` headings, ignoring fenced code.

    The first block has title None (page preamble). Fence tracking prevents a
    `## ` inside a ``` block from triggering a false split.
    """
    prefix = "#" * level + " "
    title = None
    buf: list[str] = []
    in_fence = False
    for line in markdown.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        if not in_fence and line.startswith(prefix):
            yield title, "\n".join(buf).strip()
            title = line[len(prefix):].strip()
            buf = []
        else:
            buf.append(line)
    yield title, "\n".join(buf).strip()


def _pack(body: str, limit: int) -> list[str]:
    """Split an oversized body on blank lines, never mid-fence."""
    if len(body) <= limit:
        return [body]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    open_fence = False
    for para in body.split("\n\n"):
        if current and not open_fence and size + len(para) > limit:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
        if para.count("```") % 2:
            open_fence = not open_fence
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _demote_heading(text: str, page_title: str) -> str:
    """Turn a section's `# Page - Section` heading into a `## Section` subheading."""
    head, _, body = text.partition("\n\n")
    title = head.removeprefix("# ").removeprefix(f"{page_title} - ")
    return f"## {title}\n\n{body}".strip()


def _prose_chars(text: str) -> int:
    """Characters outside headings and code fences -- the actual prose payload.

    A section whose body was entirely live-example embeds reduces to a stack of
    `###` headings. Total length hides that (`mark-trail / Examples` is 127 chars
    of pure heading), so gate merging on prose instead.
    """
    total = 0
    in_fence = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith("#"):
            continue
        total += len(stripped)
    return total


def _merge_short(
    sections: list[dict],
    page_title: str,
    merge_below: int,
    max_chars: int,
) -> list[dict]:
    """Fold prose-starved sections into the preceding section of the same page.

    Two kinds of runt show up: property-table sections ("FoldTransform
    properties: fold, as.") which carry real keywords but are too small to embed
    on their own, and heading-only husks left behind where a live-example embed
    was stripped. Both retrieve better attached to surrounding context, and
    merging never discards content.
    """
    if not merge_below:
        return sections
    out: list[dict] = []
    for section in sections:
        target = out[-1] if out else None
        too_short = _prose_chars(section["text"]) < merge_below
        fits = (
            target is not None
            and len(target["text"]) + len(section["text"]) <= min(max_chars, MERGE_TARGET_MAX)
        )
        if too_short and fits:
            target["text"] += "\n\n" + _demote_heading(section["text"], page_title)
            target["metadata"]["section_title"] += f" / {section['metadata']['section_title']}"
            continue
        out.append(section)
    return out


def split_into_sections(
    markdown: str,
    page_title: str,
    category: str,
    slug: str,
    url: str,
    max_chars: int = MAX_SECTION_CHARS,
    merge_below: int = 0,
) -> list[dict]:
    # Flatten to (title, body) candidates, subdividing on `###` when a `##`
    # section is too large to embed as a single coherent chunk.
    candidates: list[tuple[str, str]] = []
    for raw_title, body in _iter_headed_blocks(markdown, 2):
        title = raw_title or "Introduction"
        if not body or title == "Documentation Overview":
            continue
        if len(body) <= max_chars:
            candidates.append((title, body))
            continue
        subs = [(t, b) for t, b in _iter_headed_blocks(body, 3) if b]
        if len(subs) <= 1:
            candidates.append((title, body))
        else:
            for sub_title, sub_body in subs:
                label = title if sub_title is None else f"{title} - {sub_title}"
                candidates.append((label, sub_body))

    sections = []
    for title, body in candidates:
        parts = _pack(body, max_chars)
        for i, part in enumerate(parts, 1):
            section_title = title if len(parts) == 1 else f"{title} (part {i}/{len(parts)})"
            sections.append({
                "text": f"# {page_title} - {section_title}\n\n{part}",
                "metadata": {
                    "page_title": page_title,
                    "section_title": section_title,
                    "category": category,
                    "slug": slug,
                    "url": url,
                    "kind": "prose",
                },
            })
    return _merge_short(sections, page_title, merge_below, max_chars)


def _placeholder_data(node) -> None:
    """Swap sample datasets for the named-source placeholder, recursively.

    Absolute URLs and anything carrying a `format` are left alone: the choropleth
    examples point at real topojson boundary files, which are genuine content
    rather than stand-in data.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "data" and isinstance(value, dict) and "name" not in value:
                url = value.get("url")
                sample = "values" in value or (
                    isinstance(url, str) and not url.startswith(("http://", "https://"))
                )
                if sample and "format" not in value:
                    node[key] = {"name": TABLE_PLACEHOLDER}
                    continue
            _placeholder_data(value)
    elif isinstance(node, list):
        for item in node:
            _placeholder_data(item)


def normalize_spec(text: str) -> str | None:
    """Rewrite a docs example into the shape VegaLiteAgent asks its LLM to emit.

    The agent's prompt mandates YAML with `data: {name: <table>}` and no
    `$schema` (ChartSpec.yaml_spec says outright: "Skip $schema and data
    fields"), while the docs ship JSON with `$schema` and either a sample URL or
    inline `values`. Handing the raw form to the model would teach the wrong
    shape -- and the inline sample rows are ~58% of the tokens in specs like
    bar_grouped, data the model discards anyway.

    `description` goes too: it restates the docs prose already sitting in the
    item's text. bar_negative's description is near-verbatim with the sentence
    above it, so keeping both means paying twice for one explanation.

    Deliberately NOT done: wrapping in `layer`. The prompt asks for it, but a
    blind wrap breaks any spec using `row`/`column` facet channels, which cannot
    live inside a layer. These specs are here to teach idiom (which channel does
    what), and the prompt's own examples already cover layer structure.
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    obj.pop("$schema", None)
    obj.pop("description", None)
    _placeholder_data(obj)
    return dump_yaml(obj).strip()


# Multi-view composition and faceting keys. Recorded per example spec so the
# agent can drop references that contradict its own prompt: main.jinja2 tells the
# model not to pack several plots into one entry with these, yet retrieval has
# twice injected specs that do (bar_grouped_facet uses `column`, repeat_layer
# uses `repeat`). `row`/`column` are encoding channels rather than top-level
# keys, so the scan is recursive.
SPEC_CONSTRUCT_KEYS = frozenset({
    "facet", "repeat", "concat", "hconcat", "vconcat", "row", "column",
})


def _spec_constructs(node, found: set | None = None) -> list[str]:
    """Every SPEC_CONSTRUCT_KEYS key appearing anywhere in a parsed spec."""
    found = set() if found is None else found
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SPEC_CONSTRUCT_KEYS:
                found.add(key)
            _spec_constructs(value, found)
    elif isinstance(node, list):
        for item in node:
            _spec_constructs(item, found)
    return sorted(found)


def _resolve_specs(names: list[str], report: dict, max_spec_chars: int) -> list[dict]:
    specs = []
    for name in names:
        raw = fetch_example(name)
        if raw is None:
            report["missing_examples"].append(name)
            continue
        spec = normalize_spec(raw)
        if spec is None:
            report["unparsable_examples"].append(name)
            continue
        if len(spec) > max_spec_chars:
            report["oversized_examples"].append((name, len(spec)))
            continue
        entry = {"name": name, "spec": spec, "raw_chars": len(raw)}
        # normalize_spec already proved this parses to a dict.
        if constructs := _spec_constructs(json.loads(raw)):
            entry["constructs"] = constructs
        specs.append(entry)
    return specs


def _humanize(name: str) -> str:
    """bar_grouped -> "bar grouped", so the name is matchable prose."""
    return name.replace("_", " ")


def split_out_examples(
    sections: list[dict],
    report: dict,
    max_spec_chars: int = 4000,
) -> list[dict]:
    """Promote each documented example to its own retrievable item.

    A documented example is already a self-contained unit: a `###` heading, a
    sentence or two of explanation, and one spec. Keeping all of a page's
    examples on one chunk forces retrieval to choose at page granularity when the
    question is at example granularity, and drags every other spec on the page
    into context (mark-bar carried 17 specs / ~8.9k chars on a single chunk).

    So each `###` block that references a spec becomes its own item, carrying only
    its own spec(s); whatever prose is left over stays as the section item. The
    spec still lives in metadata rather than text, so it never reaches the vector.
    """
    out: list[dict] = []
    for section in sections:
        meta = section["metadata"]
        heading, _, body = section["text"].partition("\n\n")
        if not EXAMPLE_SENTINEL_RE.search(body):
            out.append(section)
            continue

        prose_blocks: list[str] = []
        example_items: list[dict] = []
        for sub_title, sub_body in _iter_headed_blocks(body, 3):
            if not sub_body.strip():
                continue
            names = list(dict.fromkeys(EXAMPLE_SENTINEL_RE.findall(sub_body)))
            if not names:
                block = sub_body if sub_title is None else f"### {sub_title}\n\n{sub_body}"
                prose_blocks.append(block.strip())
                continue
            specs = _resolve_specs(names, report, max_spec_chars)
            title = sub_title or meta["section_title"]
            # The subheading becomes the item's own `#` heading, so don't repeat it.
            text = EXAMPLE_SENTINEL_RE.sub(
                lambda m: f"Example: {_humanize(m.group(1))}", sub_body
            ).strip()
            example_items.append({
                "text": f"# {meta['page_title']} - {title}\n\n{text}",
                "metadata": {
                    **meta,
                    "section_title": title,
                    "kind": "example",
                    **({"examples": specs} if specs else {}),
                },
            })

        prose = "\n\n".join(prose_blocks).strip()
        if _prose_chars(prose):
            out.append({"text": f"{heading}\n\n{prose}", "metadata": meta})
        elif not example_items:
            # Nothing salvageable: no prose and every spec failed to resolve.
            report["empty_after_split"].append(f"{meta['slug']} / {meta['section_title']}")
        out.extend(example_items)
    return out


DESCRIPTIONS_CACHE = LUMEN_CACHE_DIR / "vega_lite_docs_descriptions.json"
DESCRIBE_CONCURRENCY = 8
# Every description is echoed in full for the first few items: a bad prompt is
# obvious from two or three samples, and waiting for all 419 to finish before
# seeing any of them has cost several wasted runs. After that, one line per
# batch is enough to show it is still moving.
DESCRIBE_ECHO_FIRST = 5
DESCRIBE_LOG_EVERY = 25
# A one-sentence description under 40 words is ~250 chars. Anything much longer
# is not a description: without a response_model, Llm.invoke returns the raw
# provider response, and str() on that yields ~860 chars of ChatCompletion repr.
MAX_DESCRIPTION_CHARS = 400

# Bumped whenever DESCRIBE_SYSTEM changes: the cache is keyed by body text, so
# without this a prompt change silently reuses every stale description.
DESCRIBE_PROMPT_VERSION = 2

DESCRIBE_SYSTEM = """\
You write one-sentence retrieval descriptions for Vega-Lite documentation sections.

The description is embedded and matched against natural-language chart requests, so
use the words a user would use ("grouped bars side by side", "top N", "stacked",
"rounded corners") rather than documentation jargon.

Describe what the section teaches, not what its example happens to plot. Never
mention the example's sample data, field names or subject matter: write "bar chart
with rounded corners", never "mean precipitation by month".

Name the marks, encoding channels, transforms and properties the section
demonstrates.

Do not judge whether the section is basic, plain, simple, canonical or specialised,
and do not use any of those words. Name the distinguishing feature instead, or say
nothing about it if there is none.

Reply with one sentence under 40 words. No preamble, no markdown, no quotes.
"""

# Section-title words that do not make a section a variant of anything.
TITLE_FILLER = {
    "chart", "charts", "plot", "plots", "graph", "mark", "marks",
    "single", "basic", "simple", "plain", "introduction", "overview",
    "example", "examples", "usage", "a", "an", "the", "of", "for", "with",
}


def _is_base_form(metadata: dict) -> bool:
    """Whether this section is the base form of its page's chart type.

    Kept deterministic because the describe call sees one section at a time and
    so cannot know how it relates to its siblings; asked anyway, it answered
    from the attached example and labelled the pivot transform page a "plain bar
    chart". Only mark-* pages are considered -- elsewhere "which variant" has no
    meaning, and that is where the spurious labels landed.
    """
    if not metadata["slug"].startswith("mark-"):
        return False
    page_words = set(re.findall(r"[a-z0-9]+", metadata["page_title"].lower()))
    words = set(re.findall(r"[a-z0-9]+", metadata["section_title"].lower()))
    return not (words - page_words - TITLE_FILLER)


def _description_key(text: str) -> str:
    payload = f"v{DESCRIBE_PROMPT_VERSION}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clean_description(response) -> str | None:
    """Coerce an LLM response to a one-line description, or None if implausible.

    Applied to fresh responses *and* to cached ones, so a bad run cannot poison
    later rebuilds: every cached value has to survive the same check.
    """
    text = response
    if not isinstance(text, str):
        # Some providers hand back a completion object; dig out the content
        # rather than str()-ing the whole repr.
        try:
            text = response.choices[0].message.content
        except Exception:
            return None
    if not isinstance(text, str):
        return None
    description = " ".join(text.split())
    if not description or len(description) > MAX_DESCRIPTION_CHARS:
        return None
    if description.startswith(("ChatCompletion", "Choice(", "Message(", "{'", '{"')):
        return None
    return description


def load_descriptions() -> dict[str, str]:
    if not DESCRIPTIONS_CACHE.exists():
        return {}
    try:
        cache = json.loads(DESCRIPTIONS_CACHE.read_text())
    except Exception as e:
        print(f"  !! ignoring unreadable description cache: {e}")
        return {}
    clean = {key: value for key, value in cache.items() if _clean_description(value)}
    dropped = len(cache) - len(clean)
    if dropped:
        print(f"  !! discarded {dropped} invalid cached description(s); they will be regenerated")
    return clean


def save_descriptions(cache: dict[str, str]) -> None:
    DESCRIPTIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DESCRIPTIONS_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


async def describe_items(items: list[dict], llm, report: dict) -> None:
    """Prepend an LLM-written description to each item's text.

    This is contextual retrieval, done here rather than through
    DuckDBVectorStore's `situate` for two reasons. First, situate only fires when
    a document splits into several chunks (`len(content_chunks) > 1`), which the
    chunk_size in embed_docs deliberately prevents. Second, its output lands in
    `metadata["llm_context"]`, which the store's default `excluded_metadata`
    keeps out of the embedded text -- so it would never reach a vector.

    Prepending to `text` is unambiguously embedded, visible under --dump, and
    cached to disk keyed by content hash so rebuilds cost nothing.

    Motivation: canonical sections lose to specialised siblings because they are
    terse. "Single Bar Chart" is 139 chars of thin prose and ranked 7th of 8 for
    "create a bar chart", behind four specialisations with more text to match on.
    A description gives it comparable surface area.
    """
    cache = load_descriptions()
    semaphore = asyncio.Semaphore(DESCRIBE_CONCURRENCY)
    counts = Counter()
    total = len(items)

    def log(metadata: dict, description: str | None, problem: str = "") -> None:
        done = sum(counts.values())
        label = f"{metadata['slug']} / {metadata['section_title']}"
        if problem:
            print(f"  [{done}/{total}] {label}: {problem}", flush=True)
        elif description and counts["generated"] <= DESCRIBE_ECHO_FIRST:
            print(f"  [{done}/{total}] {label}\n      {description}", flush=True)
        elif done % DESCRIBE_LOG_EVERY == 0:
            print(f"  [{done}/{total}] {label}", flush=True)

    async def describe(item: dict) -> tuple[str, str | None]:
        key = _description_key(item["text"])
        if key in cache:
            counts["cached"] += 1
            return key, cache[key]
        m = item["metadata"]
        # Titles are supplied so the description is about the section's topic
        # rather than whatever its example happens to render.
        content = f"Page: {m['page_title']}\nSection: {m['section_title']}\n\n{item['text']}"
        async with semaphore:
            try:
                response = await llm.invoke(
                    [{"role": "user", "content": content}],
                    system=DESCRIBE_SYSTEM,
                    response_model=str,
                )
            except Exception as e:
                counts["failed"] += 1
                report["describe_failures"].append(f"{m['slug']} / {m['section_title']}: {e}")
                log(m, None, problem=f"failed ({e})")
                return key, None
        description = _clean_description(response)
        if description is None:
            counts["rejected"] += 1
            report["describe_failures"].append(
                f"{m['slug']} / {m['section_title']}: implausible response "
                f"{str(response)[:70]!r}"
            )
            log(m, None, problem=f"rejected {str(response)[:60]!r}")
            return key, None
        counts["generated"] += 1
        # Cached here rather than after gather() so an interrupt keeps the calls
        # already paid for. Safe to keep even if the batch is abandoned: the key
        # carries DESCRIBE_PROMPT_VERSION, so editing the prompt invalidates it.
        cache[key] = description
        log(m, description)
        return key, description

    print(f"\nDescribing {total} items (concurrency {DESCRIBE_CONCURRENCY})...")
    print("  Ctrl-C to stop; completed descriptions are kept.", flush=True)
    tasks = [asyncio.ensure_future(describe(item)) for item in items]
    try:
        results = await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        for task in tasks:
            task.cancel()
        save_descriptions(cache)
        print(
            f"\nInterrupted: {counts['generated']} generated this run, "
            f"{len(cache)} cached in total -> {DESCRIPTIONS_CACHE}"
        )
        raise
    for item, (key, description) in zip(items, results):
        # Set for every item, so the flag never depends on whether describing
        # happened to succeed. Consumed by the agent as a retrieval flag; it is
        # deliberately NOT written into `text`. A bare "<Page> chart." prefix was
        # tried and made things worse: it did not lift the canonical section it
        # was written for, and on a contentless query ("visualize the data") it
        # became the most query-shaped string in the corpus, so base-form
        # Introduction pages -- which mostly carry no example spec -- swept the
        # results.
        item["metadata"]["base_form"] = _is_base_form(item["metadata"])
        if not description:
            continue
        # Stored for inspection only; it is already inside `text`, so embed_docs
        # must exclude it or it would be embedded twice.
        item["metadata"]["description"] = description
        item["text"] = f"{description}\n\n{item['text']}"
    save_descriptions(cache)
    print(
        f"Descriptions: {counts['cached']} cached, {counts['generated']} generated, "
        f"{counts['rejected']} rejected, {counts['failed']} failed -> {DESCRIPTIONS_CACHE}"
    )


def build_all_items(
    strip_link_urls: bool = True,
    max_chars: int = MAX_SECTION_CHARS,
    min_chars: int = 0,
    merge_below: int = 0,
    examples: bool = True,
) -> tuple[list[dict], dict]:
    """Fetch, clean and section every page. Returns (items, report)."""
    items: list[dict] = []
    seen_slugs: set[str] = set()
    report: dict = {
        "skipped": [],
        "empty_pages": [],
        "dropped_short": [],
        "pages": [],
        "missing_examples": [],
        "oversized_examples": [],
        "unparsable_examples": [],
        "empty_after_split": [],
        "describe_failures": [],
    }

    for category, pages in PAGES:
        for page_title, path in pages:
            slug = path.replace(".md", "").replace("/", "-")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            try:
                raw_md = fetch_markdown(path)
            except Exception as e:
                report["skipped"].append((path, str(e)))
                print(f"  !! SKIPPED {path}: {e}")
                continue

            meta, body = parse_frontmatter(raw_md)
            permalink = meta.get("permalink")
            if permalink:
                url = f"{SITE_BASE}{permalink}" if permalink.startswith("/") else f"{SITE_BASE}/{permalink}"
            else:
                # Vega-Lite permalinks are flat, so fall back to the basename
                # rather than the dashed path.
                url = f"{DOCS_BASE}/{path.rsplit('/', 1)[-1][:-3]}.html"
                report.setdefault("no_permalink", []).append(path)

            cleaned = clean_markdown(body, strip_link_urls=strip_link_urls)
            sections = split_into_sections(
                cleaned, page_title, category, slug, url, max_chars, merge_below
            )

            kept = []
            for section in sections:
                if min_chars and len(section["text"]) < min_chars:
                    report["dropped_short"].append(section)
                    continue
                kept.append(section)
            # Split after merging, so sentinels that moved between sections land
            # on whichever section actually holds them now.
            if examples:
                kept = split_out_examples(kept, report)
            items.extend(kept)

            if not kept:
                report["empty_pages"].append(path)
            n_examples = sum(len(s["metadata"].get("examples", ())) for s in kept)
            n_example_items = sum(1 for s in kept if s["metadata"].get("kind") == "example")
            report["pages"].append({
                "path": path,
                "slug": slug,
                "category": category,
                "page_title": page_title,
                "url": url,
                "raw_chars": len(raw_md),
                "clean_chars": len(cleaned),
                "sections": len(kept),
                "dropped": len(sections) - len(kept),
                "examples": n_examples,
            })
            flag = "  <-- NO SECTIONS" if not kept else ""
            specs = f" ({n_example_items} example items, {n_examples} specs)" if n_examples else ""
            print(
                f"  {category}/{page_title}: {len(kept)} items{specs} "
                f"({len(raw_md)} raw -> {len(cleaned)} clean chars) {url}{flag}"
            )

    return items, report


# Anything matching these in a cleaned section means the cleaner missed something.
RESIDUAL_NOISE = {
    "liquid_tag": re.compile(r"\{%.*?%\}", re.DOTALL),
    "liquid_var": re.compile(r"\{\{.*?\}\}", re.DOTALL),
    "kramdown_attr": KRAMDOWN_ATTR_RE,
    "html_comment": re.compile(r"<!--.*?-->", re.DOTALL),
    "html_tag": re.compile(
        r"</?(?:div|span|img|a|br|p|iframe|script|style|select|option|input|button|label|form)\b[^>]*>"
    ),
    "frontmatter_leak": re.compile(r"^(?:layout|permalink|menu|redirect_from):", re.MULTILINE),
    "edit_this_page": re.compile(r"Edit this page"),
    "empty_link": re.compile(r"\[\]|\]\(\)"),
    "bare_url": re.compile(r"https?://\S+"),
    "toc_leftover": re.compile(r"^\s*-\s*TOC\s*$", re.MULTILINE),
}


def _strip_fences(text: str) -> str:
    """Drop fenced code blocks so the noise scan does not flag legitimate samples."""
    out = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def audit(items: list[dict], report: dict, min_chars: int = MIN_SECTION_CHARS) -> None:
    def rule(label):
        print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")

    lengths = sorted(len(i["text"]) for i in items)
    rule("SUMMARY")
    print(f"pages fetched      : {len(report['pages'])}")
    print(f"pages skipped      : {len(report['skipped'])}")
    print(f"sections           : {len(items)}")
    print(f"total chars        : {sum(lengths):,}")
    if lengths:
        print(
            f"section chars      : min={lengths[0]} p50={statistics.median(lengths):.0f} "
            f"p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]}"
        )

    rule("SECTIONS PER CATEGORY")
    for category, count in Counter(i["metadata"]["category"] for i in items).most_common():
        print(f"  {count:4d}  {category}")

    if report["skipped"]:
        rule("FETCH FAILURES (missing from the index entirely)")
        for path, err in report["skipped"]:
            print(f"  {path}: {err}")

    if report["empty_pages"]:
        rule("PAGES THAT PRODUCED ZERO SECTIONS")
        for path in report["empty_pages"]:
            print(f"  {path}")

    if report.get("no_permalink"):
        rule("PAGES WITH NO permalink IN FRONTMATTER (url is a guess)")
        for path in report["no_permalink"]:
            print(f"  {path}")

    rule(f"SHORT SECTIONS (< {min_chars} prose chars -- likely junk rows)")
    short = [i for i in items if _prose_chars(i["text"]) < min_chars]
    print(f"count: {len(short)}")
    for item in short:
        m = item["metadata"]
        preview = " ".join(item["text"].split())[:110]
        print(
            f"  [{len(item['text']):4d} total / {_prose_chars(item['text']):4d} prose] "
            f"{m['slug']} / {m['section_title']}: {preview}"
        )

    rule("EXAMPLE SPECS (normalized to YAML, in metadata, excluded from embeddings)")
    kinds = Counter(i["metadata"].get("kind", "prose") for i in items)
    print(f"items by kind       : {dict(kinds)}")
    with_examples = [i for i in items if i["metadata"].get("examples")]
    specs = [e for i in with_examples for e in i["metadata"]["examples"]]
    spec_chars = sum(len(e["spec"]) for e in specs)
    raw_chars = sum(e.get("raw_chars", 0) for e in specs)
    per_item = [sum(len(e["spec"]) for e in i["metadata"]["examples"]) for i in with_examples]
    print(f"items with specs    : {len(with_examples)} / {len(items)}")
    print(f"specs attached      : {len(specs)} ({len({e['name'] for e in specs})} distinct)")
    if raw_chars:
        print(
            f"spec chars          : {raw_chars:,} raw -> {spec_chars:,} normalized "
            f"({1 - spec_chars / raw_chars:.0%} smaller)"
        )
    if per_item:
        print(
            f"spec chars per item : median={statistics.median(per_item):.0f} "
            f"max={max(per_item):,}  <-- what one retrieval injects"
        )
    if report.get("unparsable_examples"):
        print(f"\n  UNPARSABLE JSON: {report['unparsable_examples']}")
    if report.get("empty_after_split"):
        print(f"\n  SECTIONS EMPTIED BY THE SPLIT: {len(report['empty_after_split'])}")
        for label in report["empty_after_split"][:10]:
            print(f"      {label}")
    if report.get("missing_examples"):
        missing = Counter(report["missing_examples"])
        print(f"\n  MISSING from {EXAMPLES_BASE}: {len(missing)} name(s)")
        for name, count in missing.most_common(15):
            print(f"      {name}" + (f" (x{count})" if count > 1 else ""))
    if report.get("oversized_examples"):
        print(f"\n  SKIPPED as oversized: {len(report['oversized_examples'])}")
        for name, size in report["oversized_examples"][:10]:
            print(f"      {name}: {size:,} chars")

    rule("10 LONGEST SECTIONS (check these are still coherent chunks)")
    for item in sorted(items, key=lambda i: -len(i["text"]))[:10]:
        m = item["metadata"]
        print(f"  [{len(item['text']):6d}] {m['slug']} / {m['section_title']}")

    rule("RESIDUAL NOISE (cleaner misses; code fences excluded)")
    prose = [(i["metadata"]["slug"], _strip_fences(i["text"])) for i in items]
    clean = True
    for name, pattern in RESIDUAL_NOISE.items():
        hits = []
        for slug, text in prose:
            for match in pattern.findall(text):
                hits.append((slug, match if isinstance(match, str) else match[0]))
        if not hits:
            continue
        clean = False
        print(f"\n  {name}: {len(hits)} hit(s) across {len({h[0] for h in hits})} page(s)")
        for slug, match in hits[:5]:
            print(f"      {slug}: {' '.join(match.split())[:120]!r}")
        if len(hits) > 5:
            print(f"      ... {len(hits) - 5} more")
    if clean:
        print("  none \u2713")

    described = [i for i in items if i["metadata"].get("description")]
    if described or report.get("describe_failures"):
        rule("DESCRIPTIONS (prepended to text, so these ARE embedded)")
        lengths = [len(i["metadata"]["description"]) for i in described]
        print(f"described           : {len(described)} / {len(items)}")
        if lengths:
            print(
                f"description chars   : median={statistics.median(lengths):.0f} "
                f"min={min(lengths)} max={max(lengths)}"
            )
        for item in described[:6]:
            m = item["metadata"]
            print(f"\n  {m['slug']} / {m['section_title']}")
            print(f"    {m['description']}")
        if report.get("describe_failures"):
            print(f"\n  FAILURES: {len(report['describe_failures'])}")
            for failure in report["describe_failures"][:5]:
                print(f"      {failure}")

    rule("DUPLICATE / NEAR-DUPLICATE SECTIONS")
    by_text = Counter(" ".join(i["text"].split()) for i in items)
    dupes = [(t, c) for t, c in by_text.items() if c > 1]
    print(f"exact duplicate bodies: {len(dupes)}")
    for text, count in dupes[:10]:
        print(f"  x{count}: {text[:110]}")

    rule("AMBIGUOUS page_title (same title, different page)")
    titles = {}
    for item in items:
        titles.setdefault(item["metadata"]["page_title"], set()).add(item["metadata"]["slug"])
    for title, slugs in sorted(titles.items()):
        if len(slugs) > 1:
            print(f"  {title!r}: {sorted(slugs)}")


def dump(items: list[dict], path: str) -> None:
    """Write every section verbatim so the actual embedded text can be eyeballed."""
    with open(path, "w") as f:
        for i, item in enumerate(items, 1):
            m = item["metadata"]
            f.write(f"\n{'=' * 78}\n")
            f.write(f"[{i}] {m['category']} | {m['slug']} | {m['section_title']}\n")
            f.write(f"url: {m['url']}  chars: {len(item['text'])}\n")
            f.write(f"{'=' * 78}\n")
            f.write(item["text"] + "\n")
    with open(f"{path}.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"\nWrote {len(items)} sections to {path} and {path}.jsonl")


async def embed_docs(
    items: list[dict],
    embeddings: Embeddings,
    uri: str,
    max_chars: int = MAX_SECTION_CHARS,
):
    # DuckDBVectorStore re-chunks on add() via semchunk. Its bypass guard is
    # `len(text) <= chunk_size`, comparing characters against a token budget, so
    # with the default 512 anything longer than 512 chars gets recut on
    # semchunk's boundaries instead of the heading boundaries computed above.
    # Sizing chunk_size past our largest section keeps our sectioning intact.
    #
    # url/slug are identifiers and `examples` is payload for the LLM, not
    # semantic content -- all three would otherwise be appended to every chunk's
    # embedded text by _join_text_and_metadata.
    vector_store = DuckDBVectorStore(
        uri=uri,
        embeddings=embeddings,
        chunk_size=max_chars + 1,
        # base_form is already expressed inside `text` as a query-shaped phrase;
        # appending it here would embed the literal flag name instead.
        excluded_metadata=[
            "llm_context", "url", "slug", "examples", "kind", "description",
            "base_form",
        ],
    )
    # add() appends rather than upserts, so a rebuild against an existing store
    # would duplicate every section.
    vector_store.clear()
    await vector_store.add(items, situate=False)
    print(f"Embedded {len(items)} sections into {uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Vega-Lite docs vector stores.")
    parser.add_argument("--inspect", action="store_true", help="build and audit only, no embedding")
    parser.add_argument("--dump", metavar="PATH", help="write every section to PATH for review")
    parser.add_argument("--keep-link-urls", action="store_true", help="do not flatten [text](url) to text")
    parser.add_argument("--max-chars", type=int, default=MAX_SECTION_CHARS)
    parser.add_argument(
        "--merge-short",
        type=int,
        default=MIN_SECTION_CHARS,
        help="fold sections shorter than this into the previous one (0 disables)",
    )
    parser.add_argument("--min-chars", type=int, default=0, help="drop sections still shorter than this")
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="skip fetching example specs (avoids a few hundred HTTP requests)",
    )
    parser.add_argument(
        "--situate",
        action="store_true",
        help="prepend an LLM-written retrieval description to each item "
             "(cached to disk, so only the first build pays for it)",
    )
    args = parser.parse_args()

    print("Building items from Vega-Lite docs...")
    items, report = build_all_items(
        strip_link_urls=not args.keep_link_urls,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        merge_below=args.merge_short,
        examples=not args.no_examples,
    )
    if args.situate:
        asyncio.run(describe_items(items, OpenAILLM(), report))
    audit(items, report, min_chars=args.min_chars or MIN_SECTION_CHARS)

    if args.dump:
        dump(items, args.dump)

    if args.inspect:
        print("\n--inspect: skipping embedding.")
        return

    # Build once, embed twice.
    openai_uri = str(LUMEN_CACHE_DIR / VEGA_LITE_DOCS_OPENAI_DB_FILE)
    numpy_uri = str(LUMEN_CACHE_DIR / VEGA_LITE_DOCS_NUMPY_DB_FILE)
    asyncio.run(embed_docs(items, embeddings=OpenAIEmbeddings(), uri=openai_uri, max_chars=args.max_chars))
    asyncio.run(embed_docs(items, embeddings=NumpyEmbeddings(), uri=numpy_uri, max_chars=args.max_chars))


if __name__ == "__main__":
    main()
