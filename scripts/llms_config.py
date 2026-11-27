"""
The llms.txt index of the Lumen docs, its markdown mirror, and the files
that go with it.

`python -m nbsite build-llms --config scripts/llms_config.py` writes the
mirror and llms.txt. `python scripts/llms_config.py` then adds the
Examples and the Optional links to llms.txt, and writes api-list.txt (the
public API, one line each), llms-small.txt (the essentials) and
llms-full.txt (the indexed pages joined), for tools that read one file.
"""

import inspect
import re
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
MARKDOWN_URL = f"{SITE_URL}markdown"
RAW_URL = "https://raw.githubusercontent.com/holoviz/lumen/main"

SUMMARY = f"""\
> HoloViz Lumen answers questions about your data in plain English from Python. It writes the SQL, runs it locally on DuckDB, xarray or your database, and returns the answer with its data, SQL and a chart or map. It works with any LLM, including local models, and every step is shown and editable.

```python
import lumen
answer = lumen.read("sales.csv").ask("Which region grew fastest last quarter?")
answer.value, answer.sql, answer.data, answer.chart
```

Things to remember when writing code with Lumen:

- `ask` needs an LLM: pass `llm=`, set a provider key such as `OPENAI_API_KEY`, or call `lumen.ai.configure(llm=...)`. `lumen.ai.llm.Kilo()` serves free models with no account. Without any of these, `ask` raises "No LLM provider found".
- `pip install lumen` covers OpenAI, OpenAI-compatible providers (Ollama, Groq, OpenRouter, Kilo) and the coding CLIs. Anthropic, Google and Mistral need `lumen[ai-anthropic]`, `lumen[ai-google]` or `lumen[ai-mistralai]`; xarray, NetCDF and Zarr data need `lumen[xarray]`; maps want `lumen[geo]`.
- `lumen.read` takes paths, URLs, DataFrames, pyarrow Tables, xarray Datasets, GeoDataFrames and DuckDB connections. A dict names the tables and puts them in one source, so a question can join them; a ready-made Source cannot go in the dict.
- Check `answer.status` before using the result: "success", "error" (`answer.error` says why) or "unanswerable" (`answer.why` says what is missing).
- Queries run locally; prompts get schema and sample values. `lumen.ai.configure(share="schema")` or `"none"` sends less, and `mask_pii=True` withholds personal-data columns.
- An answer holds at most 1,000,000 rows; `answer.truncated_at` says when it was cut, and `lumen.ai.configure(max_rows=...)` changes the cap.
- In code that already runs an event loop, use `await source.aask(...)`. Follow-ups that build on earlier answers use `chat = source.chat()` and `chat.ask(...)`.
- `answer.code` is Lumen code that rebuilds the answer with no LLM.
- `lumen-ai serve data.csv` (also `lumen serve`) opens the same agents as a chat app. YAML dashboards served with `lumen serve app.yaml` are a separate, older API.

## Agent quick start

- [Lumen for LLMs]({MARKDOWN_URL}/reference/lumen_for_llms.md): the smallest working example, the API at a glance and the errors people hit with their fixes
- [API list]({SITE_URL}api-list.txt): every public function, method and attribute of the notebook API, one line each
- [Essentials]({SITE_URL}llms-small.txt): this summary, the cheat sheet and the API list in one file
- [Complete documentation]({SITE_URL}llms-full.txt): every page indexed here except the optional ones, in one file"""

#: Section title, its description, and its pages as path -> (label, what the page is for).
#: nbsite sorts a section's links by path, so a page that must come first gets its own section.
#: The API reference pages are mkdocstrings directives with no text of their own, so they are
#: left out; api-list.txt gives the notebook API instead.
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

#: Runnable examples, raw so an agent can read them whole
EXAMPLES = {
    "examples/notebooks/ask_in_a_notebook.ipynb": "read data, ask, follow up, chat and export, on the penguins data",
    "examples/notebooks/maps_and_grids.ipynb": "maps, gridded xarray data and spatial questions",
}

#: Links outside the docs tree for the Optional section
OPTIONAL_LINKS = {
    "Release notes": ("https://github.com/holoviz/lumen/releases", "what changed in each version"),
    "What goes to the LLM provider": (
        f"{MARKDOWN_URL}/getting_started/lumen_in_notebooks.md",
        "its last section: what prompts contain, and the share and mask_pii settings that limit it",
    ),
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
    markdown_base_url=MARKDOWN_URL,
    sources=(MarkdownSource(source_dir=DOCS_DIR, output_dir=OUTPUT_DIR, exclude_files=(Path("releases.md"),)),),
    sections=tuple(_section(title, description, pages) for (title, description), pages in SECTIONS.items()),
)


def _first_paragraph(doc: str | None) -> str:
    text = " ".join(inspect.cleandoc(doc or "").split("\n\n")[0].split())
    # Sphinx roles such as :func:`chat` read as plain code
    return re.sub(r":\w+:`([^`]+)`", r"`\1`", text)


def _signature(obj) -> str:
    # Annotations come quoted, and an unset default prints its object address
    return re.sub(r"<object object at 0x[0-9a-f]+>", "unset", str(inspect.signature(obj)).replace("'", ""))


def api_list() -> str:
    """The notebook API, one line per function, method or attribute, from its docstrings."""
    # Deferred: the build-llms step reads this module without needing Lumen importable
    import lumen
    import lumen.ai

    from lumen.ai.answer import Answer, Chat
    from lumen.pipeline import Pipeline
    from lumen.sources import Source

    def entry(owner: str, name: str, obj) -> str:
        if isinstance(obj, property):
            return f"- `{owner}.{name}`: {_first_paragraph(obj.__doc__)}"
        return f"- `{owner}.{name}{_signature(obj)}`: {_first_paragraph(obj.__doc__)}"

    lines = ["# HoloViz Lumen: notebook API", "", "Each public function, method and attribute, from its docstring.", ""]
    groups = {
        "lumen": (lumen, ["read"]),
        "lumen.ai": (lumen.ai, ["ask", "aask", "ask_many", "aask_many", "chat", "configure", "clear_cache"]),
        "Source": (Source, ["ask", "aask", "ask_many", "chat", "describe", "overview"]),
        "Pipeline": (Pipeline, ["ask", "aask", "ask_many"]),
        "Chat": (Chat, ["ask", "aask", "to_html", "to_notebook", "to_report"]),
    }
    for owner, (obj, names) in groups.items():
        lines += [f"## {owner}", ""] + [entry(owner, name, inspect.getattr_static(obj, name)) for name in names] + [""]
    lines += ["## Answer", ""]
    lines += [f"- `Answer.{name}`: {_first_paragraph(Answer.param[name].doc)}" for name in sorted(Answer.param) if name != "name"]
    methods = sorted(
        name for name, obj in vars(Answer).items()
        if not name.startswith("_") and (isinstance(obj, property) or inspect.isfunction(obj))
    )
    lines += [entry("Answer", name, inspect.getattr_static(Answer, name)) for name in methods]
    return "\n".join(lines) + "\n"


def write_extras() -> None:
    """Add the Examples and the Optional links to llms.txt, and write api-list.txt, llms-small.txt and llms-full.txt."""
    llms = CONFIG.llms_output_path
    examples = ["## Examples", "", "Runnable notebooks.", ""] + [
        f"- [{Path(path).stem.replace('_', ' ').capitalize()}]({RAW_URL}/{path}): {description}"
        for path, description in EXAMPLES.items()
    ]
    optional = [f"- [{label}]({url}): {description}" for label, (url, description) in OPTIONAL_LINKS.items()]
    text = llms.read_text(encoding="utf-8").replace(
        f"All documentation is available as markdown files under {MARKDOWN_URL}/.",
        f"Every page below is markdown. Any other docs page is too: take its path under {SITE_URL} "
        f"and fetch it under {MARKDOWN_URL}/ ending in `.md`.",
    )
    head, optional_section = text.split("## Optional", 1)
    llms.write_text(
        head + "\n".join(examples) + "\n\n## Optional" + optional_section.rstrip("\n") + "\n" + "\n".join(optional) + "\n",
        encoding="utf-8",
    )
    print(f"Added examples and optional links to {llms}")

    api = api_list()
    (BUILTDOCS_DIR / "api-list.txt").write_text(api, encoding="utf-8")
    cheat_sheet = (OUTPUT_DIR / "reference" / "lumen_for_llms.md").read_text(encoding="utf-8").strip()
    (BUILTDOCS_DIR / "llms-small.txt").write_text(
        f"# {CONFIG.project_title}\n\n{SUMMARY}\n\n---\n\n{cheat_sheet}\n\n---\n\n{api}", encoding="utf-8",
    )

    parts = [f"# {CONFIG.project_title}\n\n{SUMMARY}\n"]
    for (title, _), pages in SECTIONS.items():
        if title == "Optional":
            continue
        for path in pages:
            page = OUTPUT_DIR / path
            parts.append(f"\n---\n\nSource: {MARKDOWN_URL}/{path}\n\n{page.read_text(encoding='utf-8').strip()}\n")
    (BUILTDOCS_DIR / "llms-full.txt").write_text("".join(parts), encoding="utf-8")
    print(f"Generated api-list.txt, llms-small.txt and llms-full.txt in {BUILTDOCS_DIR}")


if __name__ == "__main__":
    write_extras()
