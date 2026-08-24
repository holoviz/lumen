"""Config for building Lumen markdown docs and llms.txt.

The zensical nav is the single source of truth for which pages ship and what
they are called: this reads it once and reuses it to build the llms.txt
sections, instead of maintaining a second, parallel page list.
"""

import tomllib

from pathlib import Path

from nbsite.scripts import LlmsBuildConfig, LlmsSection, MarkdownSource

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "docs"
BUILTDOCS_DIR = ROOT / "builtdocs"
OUTPUT_DIR = BUILTDOCS_DIR / "markdown"


def _flatten_nav(nav: list, trail: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str, Path]]:
    """Flatten zensical's nav into (group trail, label, path) triples."""
    pages = []
    for entry in nav:
        for label, target in entry.items():
            if isinstance(target, list):
                pages.extend(_flatten_nav(target, trail + (label,)))
            else:
                pages.append((trail, label, Path(target)))
    return pages


_config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]
NAV_PAGES = _flatten_nav(_config["nav"])
LABELS = {path: label for _, label, path in NAV_PAGES}
TRAILS = {path: trail for trail, _, path in NAV_PAGES}


def _label(path: Path) -> str:
    return LABELS.get(path, path.stem.replace("_", " "))


def _section(title: str, description: str, *trail: str, group: str | None = None, under: bool = False) -> LlmsSection:
    """One LlmsSection for a nav group, matched either exactly (*trail*) or by prefix (*under*)."""
    matches_trail = (lambda path: TRAILS.get(path, ())[: len(trail)] == trail) if under else (lambda path: TRAILS.get(path) == trail)
    return LlmsSection(
        title=title,
        description=description,
        path_prefix=Path("."),
        path_filter=matches_trail,
        label_builder=_label,
        group=group,
    )


CONFIG = LlmsBuildConfig(
    project_title="Lumen",
    project_description=(
        "Lumen is an open-source and extensible agent framework for chatting with data "
        "and for retrieval augmented generation. It turns natural language into SQL, data "
        "transformation pipelines, visualizations and dashboards, and every step stays "
        "inspectable, editable and reproducible."
    ),
    markdown_root=OUTPUT_DIR,
    llms_output_path=BUILTDOCS_DIR / "llms.txt",
    markdown_base_url="/markdown",
    sources=(MarkdownSource(source_dir=DOCS_DIR, output_dir=OUTPUT_DIR),),
    sections=(
        _section("Overview", "Quick start, installation, and top-level pages."),
        _section("Getting Started", "Launching Lumen, navigating the UI, and building your first app.", "Getting Started"),
        _section("Tutorials", "Full walkthroughs building an AI-driven data exploration app end to end.", "Examples", "Tutorials", group="Examples"),
        _section("Gallery", "Short example specs demonstrating individual sources, transforms, and views.", "Examples", "Gallery", group="Examples"),
        _section("Configuration", "Top-level spec reference: sources, transforms, views, agents, and more.", "Configuration"),
        _section("YAML Spec", "Detailed reference for writing a Lumen dashboard spec by hand.", "Configuration", "Specs"),
        _section("API Reference", "Python API reference for Lumen's pipeline, sources, transforms, views, and AI components.", "Reference", under=True),
    ),
)
