"""
The llms.txt index of the Lumen docs, its markdown mirror, and llms-full.txt.

`python -m nbsite build-llms --config scripts/llms_config.py` writes the
mirror and llms.txt; `python scripts/llms_config.py` then joins the indexed
pages into llms-full.txt, for tools that read one file.
"""

import tomllib

from pathlib import Path

from nbsite.scripts import LlmsBuildConfig, LlmsSection, MarkdownSource

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "docs"
BUILTDOCS_DIR = ROOT / "builtdocs"
OUTPUT_DIR = BUILTDOCS_DIR / "markdown"

# Absolute, so the links work wherever the index is read from; the dev
# deploy rewrites site_url to the staging site before building
SITE_URL = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]["site_url"]

SUMMARY = """\
> HoloViz Lumen answers questions about your data in plain English from Python. It writes the SQL, runs it locally on DuckDB, xarray or your database, and returns the answer with its data, SQL and a chart or map. It works with any LLM, including local models, and every step is shown and editable.

```python
import lumen
answer = lumen.read("sales.csv").ask("Which region grew fastest last quarter?")
answer.value, answer.sql, answer.data, answer.chart
```

`pip install lumen` covers OpenAI, OpenAI-compatible providers (Ollama, Groq, OpenRouter, keyless `Kilo()`) and the coding CLIs; other providers have extras such as `lumen[ai-anthropic]`. `lumen.read` takes files, URLs, DataFrames, xarray grids, GeoDataFrames and database connections. `lumen-ai serve data.csv` opens the same agents as a chat app, and YAML specs build dashboards.

Prefer `lumen.read(...).ask(...)` for answering questions about data in code. The pages below are ordered for that. The API reference is in the installed package's docstrings, e.g. `help(lumen.read)` and `help(lumen.ai.answer.Answer)`."""

#: Section title, its description, and its pages as path -> (label, what the page is for).
#: nbsite sorts a section's links by path, so a page that must come first gets its own section.
#: The API reference pages are mkdocstrings directives with no text of their own, so they are left out.
SECTIONS = {
    ("Start here", "Read data and ask questions from a notebook or script."): {
        "getting_started/lumen_in_notebooks.md": (
            "Lumen in notebooks",
            "lumen.read(...).ask(...): answers, SQL, charts, maps, follow-ups, chats, retries, exports and sharing settings",
        ),
        "quick_start.md": ("Quick start", "install, connect an LLM and ask a first question"),
    },
    ("Set up", "Install Lumen, pick an LLM and point it at data."): {
        "installation.md": ("Installation", "pip extras for each LLM provider, local models and coding CLIs"),
        "configuration/llm_providers.md": (
            "LLM providers", "OpenAI, Anthropic, Google, Mistral, Ollama, Llama.cpp, keyless Kilo and others, and choosing models",
        ),
        "configuration/sources.md": (
            "Data sources", "what lumen.read and the app take: files, URLs, DataFrames, xarray, databases, remote Parquet and Zarr",
        ),
    },
    ("Chat app and dashboards", "The browser app built on the same agents, and YAML dashboards."): {
        "getting_started/launching_lumen.md": ("Launching Lumen", "lumen-ai serve from the command line or Python"),
        "getting_started/using_lumen_ai.md": ("Using Lumen AI", "asking, refining and exporting in the chat app"),
        "configuration/cli.md": ("Command line", "every lumen-ai serve option"),
        "configuration/ui.md": ("UI", "configuring the chat app"),
        "getting_started/building_lumen_apps.md": ("Building apps", "embedding Lumen in Panel apps"),
        "configuration/spec/index.md": ("YAML dashboards", "declarative dashboards from a spec"),
    },
    ("Customize and extend", "Add agents, tools, analyses and context for a domain."): {
        "extending.md": ("Extending Lumen", "where custom components plug in"),
        "configuration/agents.md": ("Agents", "built-in agents and writing your own"),
        "configuration/tools.md": ("Tools", "giving agents functions, documents and context"),
        "configuration/analyses.md": ("Analyses", "custom deterministic analyses the LLM can call"),
        "configuration/coordinators.md": ("Coordinators", "how a question is planned across agents"),
        "configuration/context.md": ("Context", "what agents share between steps"),
        "configuration/prompts.md": ("Prompts", "overriding agent prompts"),
    },
    ("Optional", "Background and contribution docs, safe to skip."): {
        "faq.md": ("FAQ", "common questions"),
        "examples/tutorials/index.md": ("Tutorials", "end-to-end example apps"),
        "examples/gallery/index.md": ("Gallery", "example dashboards"),
        "contributing.md": ("Contributing", "repository setup and workflow for changing Lumen itself"),
    },
}


def _section(title: str, description: str, pages: dict[str, tuple[str, str]]) -> LlmsSection:
    pages = {Path(path): page for path, page in pages.items()}
    return LlmsSection(
        title=title,
        description=description,
        path_prefix=Path("."),
        path_filter=pages.__contains__,
        label_builder=lambda path: pages[path][0],
        description_builder=lambda path: pages[path][1],
    )


CONFIG = LlmsBuildConfig(
    project_title="HoloViz Lumen",
    project_description=SUMMARY,
    markdown_root=OUTPUT_DIR,
    llms_output_path=BUILTDOCS_DIR / "llms.txt",
    markdown_base_url=f"{SITE_URL}markdown",
    sources=(MarkdownSource(source_dir=DOCS_DIR, output_dir=OUTPUT_DIR, exclude_files=(Path("releases.md"),)),),
    sections=tuple(_section(title, description, pages) for (title, description), pages in SECTIONS.items()),
)


def write_llms_full() -> Path:
    """The indexed pages, minus the optional ones, as one markdown file in index order."""
    parts = [f"# {CONFIG.project_title}\n\n{SUMMARY}\n"]
    for (title, _), pages in SECTIONS.items():
        if title == "Optional":
            continue
        for path in pages:
            page = OUTPUT_DIR / path
            parts.append(f"\n---\n\nSource: {SITE_URL}markdown/{path}\n\n{page.read_text(encoding='utf-8').strip()}\n")
    output = BUILTDOCS_DIR / "llms-full.txt"
    output.write_text("".join(parts), encoding="utf-8")
    print(f"Generated {output}")
    return output


if __name__ == "__main__":
    write_llms_full()
