# AGENTS.md

Lumen is a generalized, open-source agent framework for turning natural
language into SQL, charts, dashboards, and reports. It is not scoped to any
single workflow — the same primitives (Source, Transform, Filter, View)
back everything from ad hoc chat queries to reusable dashboard specs.

## Repo structure

- `lumen/` — core declarative data model (Source, Transform, Filter, View, Pipeline)
- `lumen/ai/` — AI agent framework: LLM providers, agents, coordinator, prompts, tools
- `lumen/sources/` — data connectors (SQL, files, remote APIs)
- `lumen/transforms/` — data transformation primitives
- `lumen/views/` — visualization and output rendering
- `lumen/tests/` — test suite, mirrors package layout
- `docs/` — documentation source (zensical/mkdocs)
- `examples/` — example notebooks and dashboard specs

## Key entry points

- `lumen/dashboard.py` — `Dashboard` class, the primary way to build an app from a YAML spec
- `lumen/pipeline.py` — `Pipeline` class, chains Source → Transform → Filter → View
- `lumen/ai/ui.py` — `ChatUI` and `ExplorerUI`, top-level Panel apps for the AI chat interface
- `lumen/ai/llm.py` — `Llm` base class and provider subclasses; `invoke()`, `stream()`, `model_kwargs`, routing logic
- `lumen/ai/agents/base.py` — `Agent` base class, all agents inherit from this
- `lumen/ai/coordinator/base.py` — `Coordinator`, orchestrates agent selection and execution
- `lumen/sources/base.py` — `Source` base class, all data connectors inherit from this
- `lumen/transforms/base.py` — `Transform` base class
- `lumen/views/base.py` — `View` base class

## Setup

Requires Python >= 3.11.

```bash
pixi install                     # default environment
pixi install -e test-312         # test environment (Python 3.12)
pixi install -e docs             # docs environment
pixi install -e lint             # lint environment
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pixi run -e test-312 test-unit   # full suite, Python 3.12
pixi run -e test-313 test-unit   # full suite, Python 3.13
pixi run -e test-core test-unit  # minimal core tests
pixi run -e test-312 pytest lumen/tests/ai/test_llm.py -x -v  # single file
```

Uses pytest with pytest-asyncio (auto mode) — `async def test_*` runs
without explicit marks. pytest-xdist parallelizes runs.

## Lint & typecheck

```bash
pixi run -e lint lint       # ruff, isort, pygrep pre-commit hooks
pixi run -e lint typecheck  # pyright on lumen/ai/
```

- Ruff line length 165; select B/E/F/FLY/ICN/NPY/PIE/PLC/PLE/PLR/PLW/RUF/T20/UP/W
- Pre-commit blocks `breakpoint()` calls and private keys

## Code conventions

- `param` for declarative class parameters, not dataclasses
- Panel primitives for all views, widgets, and layout
- Pydantic models for structured LLM outputs in `lumen/ai/`
- `__init__.py` files re-export public API symbols

## Key patterns

- **LLM routing**: agents resolve their model via `llm_spec_key`, derived
  from the class name (`SQLAgent` → `"sql"`, `ChatAgent` → `"chat"`),
  mapping to entries in `Llm.model_kwargs`
- **Declarative pipelines**: Source → Transform → Filter → View chains are
  serializable as YAML/JSON
- **Prompt templates**: agent prompts live in `lumen/ai/prompts/` as Jinja2
  templates, referenced via each agent's `prompts` dict
- **Tool integration**: agents call `FunctionTool` / `MCPTool` from
  `lumen/ai/tools/` for LLM-callable functions

## Prompt authoring conventions

Agent prompts live in `lumen/ai/prompts/` as Jinja2 templates. When editing
or adding prompts, follow the conventions in `lumen/ai/prompts/GUIDANCE.md`:

- Use `##` for sections, `###` for subsections; never `#` alone
- Title Case headings, no trailing colons
- Guard conditionals on the value (`memory.get('k')`), not the key
- No emoji or pictographs; at most two emphasis markers per template
- Caveman-compress prose (strip articles, auxiliaries, redundant prepositions;
  preserve content words and code blocks verbatim)
- Extend base templates via `{{ super() }}`, not copy-paste
- Label injected `memory['data']` as a summary, not the dataset
- Cap injected payloads in tokens (`truncate_to_tokens`), not characters

`lumen/tests/ai/test_prompts.py` enforces these mechanically across all templates.

## Extending Lumen

Sources, Agents, Tools, and Analyses are all subclassable Python classes.
For building installable extensions, use the
[Panel Extension Copier Template](https://github.com/panel-extensions/copier-template-panel-extension):

```bash
pixi exec --spec copier --spec ruamel.yaml -- \
  copier copy --trust \
  https://github.com/panel-extensions/copier-template-panel-extension \
  lumen-name-of-extension
```

Choose **Lumen** as the extension type and `py311`+ for minimum Python version.

## PR & commit conventions

- Branch naming: `fix/issue-name`, `feat/feature-name`, or `feat/issue-number`
- Commit style: present tense — `Fix: ...`, `Add: ...`, `feat: ...`, `fix: ...`
- Reference issues: `Closes #123`, `Fixes #456`
- Target `main` on `holoviz/lumen`
- CI runs on Linux, macOS, Windows across Python 3.12 and 3.13
