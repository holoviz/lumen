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


def _at(*trail: str):
    """Pages whose nav group is exactly *trail*, e.g. Examples/Gallery."""
    return lambda path: TRAILS.get(path) == trail


def _under(*trail: str):
    """Pages whose nav group starts with *trail*, e.g. everything under Reference."""
    return lambda path: TRAILS.get(path, ())[: len(trail)] == trail


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
        LlmsSection(
            title="Overview",
            description="Quick start, installation, and top-level pages.",
            path_prefix=Path("."),
            path_filter=_at(),
            label_builder=_label,
        ),
        LlmsSection(
            title="Getting Started",
            description="Launching Lumen, navigating the UI, and building your first app.",
            path_prefix=Path("."),
            path_filter=_at("Getting Started"),
            label_builder=_label,
        ),
        LlmsSection(
            title="Tutorials",
            description="Full walkthroughs building an AI-driven data exploration app end to end.",
            path_prefix=Path("."),
            path_filter=_at("Examples", "Tutorials"),
            label_builder=_label,
            group="Examples",
        ),
        LlmsSection(
            title="Gallery",
            description="Short example specs demonstrating individual sources, transforms, and views.",
            path_prefix=Path("."),
            path_filter=_at("Examples", "Gallery"),
            label_builder=_label,
            group="Examples",
        ),
        LlmsSection(
            title="Configuration",
            description="Top-level spec reference: sources, transforms, views, agents, and more.",
            path_prefix=Path("."),
            path_filter=_at("Configuration"),
            label_builder=_label,
        ),
        LlmsSection(
            title="YAML Spec",
            description="Detailed reference for writing a Lumen dashboard spec by hand.",
            path_prefix=Path("."),
            path_filter=_at("Configuration", "Specs"),
            label_builder=_label,
        ),
        LlmsSection(
            title="API Reference",
            description="Python API reference for Lumen's pipeline, sources, transforms, views, and AI components.",
            path_prefix=Path("."),
            path_filter=_under("Reference"),
            label_builder=_label,
        ),
    ),
)
