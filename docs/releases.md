# Releases

This page lists all Lumen releases with links to detailed changelogs.

## Latest

### Version 1.4.0

Lumen AI 1.4 adds keyless and CLI-backed LLM providers (Kilo, Codex CLI, Claude Code, GitHub Copilot CLI), tracks token usage and cost, gives agents more accurate table context, lets reports carry an AI-written story that exports to Word, and supports polars and pyarrow frames via narwhals.

[Release notes →](https://github.com/holoviz/lumen/releases/tag/v1.4.0)

### Version 1.3.0

Lumen AI 1.3 expands support for geospatial and scientific data, makes AI-generated visualizations more capable, and improves the reliability of exploration, validation, reporting, and exports.

[Release notes →](https://github.com/holoviz/lumen/releases/tag/v1.3.0)

### Version 1.2.1

Patch release adding Anthropic prompt caching support, structured prompt guidance across all agents, and several fixes including allowing `temperature=None` on LLM providers and simplifying DbtslAgent context generation.

[Release notes →](https://github.com/holoviz/lumen/releases/tag/v1.2.1)

### Version 1.2.0

Lumen AI 1.2 deepens the tool-calling architecture introduced in 1.1, adds new LLM providers (OpenRouter, MLX, OpenAI Responses API), scientific data sources (XArraySQLSource), and delivers a large batch of agent reliability and UX improvements.

[Full changelog →](https://github.com/holoviz/lumen/blob/main/CHANGELOG.md#version-120)

### Version 1.1.0

Lumen AI 1.1 builds on the stable foundation of 1.0 with first-class LLM tool support (`FunctionTool`, `MCPTool`), stronger retrieval-augmented generation (ChromaDB, `DocumentSummarizerAgent`), and a broad sweep of usability and reliability improvements.

[Full changelog →](https://github.com/holoviz/lumen/blob/main/CHANGELOG.md#version-110)

## Previous

### Version 1.0.0

The first stable release of the Lumen AI agent framework, establishing the core architecture for conversational data exploration with configurable agents, LLM providers, and data pipelines.

[Release notes →](https://github.com/holoviz/lumen/releases/tag/v1.0.0)

### Version 0.10.x

Refined the user experience and improved system robustness with bug fixes and minor enhancements.

### Version 0.9.0

Enhanced integration with various data sources and embedding models, including HuggingFace embeddings, WebLLM, and BigQuery.

### Version 0.8.0

Introduced GraphicWalker integration, configurable chat interfaces, and Snowflake authenticator support.

### Version 0.7.0

Introduced the `lumen.ai` subpackage, re-envisioning Lumen as an extensible agent-based framework for chatting with data.

### Version 0.6.0

Added DuckDB source, auto-generated filters, Panel 1.0 and Param 2 compatibility.

### Version 0.5.0

Major internal overhaul introducing `Pipeline`, `Variable` references, and `SQLTransform` components.

---

For the complete changelog of all releases, see the [CHANGELOG on GitHub](https://github.com/holoviz/lumen/blob/main/CHANGELOG.md).
