"""
Build the markdown mirror and llms.txt that Lumen serves for LLM consumers.

Runs as the last step of the docs build:
    pixi run -e docs docs-build
"""

import shutil
import tomllib

from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT / "zensical.toml"
DOCS_DIR = ROOT / "docs"
MARKDOWN_URL = "/markdown"

SUMMARY = (
    "Lumen is an open-source and extensible agent framework for chatting with data "
    "and for retrieval augmented generation. It turns natural language into SQL, data "
    "transformation pipelines, visualizations and dashboards, and every step stays "
    "inspectable, editable and reproducible."
)

PREAMBLE = [
    "Lumen is built on Panel and the wider HoloViz stack. Its declarative data model means "
    "anything the language model produces can be serialized to YAML, reopened in a notebook, "
    "or composed into a dashboard.",
    "",
    "Install with `pip install 'lumen[ai-openai]'` and start with `lumen-ai serve data.csv`.",
]

# Nav entries that sit outside any group, e.g. Quick Start and Installation.
UNGROUPED_SECTION = "Overview"


def iter_nav_pages(nav: list, trail: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    """
    Flatten zensical's nav into (section, label, path) triples.

    The nav is the single source of truth here: it decides which pages reach
    llms.txt and what they are called, so no label or grouping is inferred
    from file paths.
    """
    pages = []
    for entry in nav:
        for label, target in entry.items():
            if isinstance(target, list):
                pages.extend(iter_nav_pages(target, trail + (label,)))
            else:
                pages.append((" / ".join(trail) or UNGROUPED_SECTION, label, target))
    return pages


def copy_markdown(pages: list[tuple[str, str, str]], output_dir: Path) -> None:
    """Mirror every listed page into the built site so the links resolve."""
    for _, _, path in pages:
        destination = output_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DOCS_DIR / path, destination)


def render_llms_txt(pages: list[tuple[str, str, str]]) -> str:
    lines = ["# Lumen", "", f"> {SUMMARY}", "", *PREAMBLE, ""]
    for section in dict.fromkeys(section for section, _, _ in pages):
        lines.extend([f"## {section}", ""])
        lines.extend(
            f"- [{label}]({MARKDOWN_URL}/{path})"
            for page_section, label, path in pages
            if page_section == section
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    config = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))["project"]
    site_dir = ROOT / config["site_dir"]
    pages = iter_nav_pages(config["nav"])

    copy_markdown(pages, site_dir / MARKDOWN_URL.strip("/"))
    (site_dir / "llms.txt").write_text(render_llms_txt(pages), encoding="utf-8")
    print(f"Wrote llms.txt and {len(pages)} markdown pages to {site_dir}")


if __name__ == "__main__":
    main()
