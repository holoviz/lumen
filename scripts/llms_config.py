"""Config for building Lumen markdown docs and its developer-facing llms.txt."""

from pathlib import Path

from nbsite.scripts import LlmsBuildConfig, LlmsSection, MarkdownSource

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "docs"
BUILTDOCS_DIR = ROOT / "builtdocs"
OUTPUT_DIR = BUILTDOCS_DIR / "markdown"

DEVELOPMENT_PAGES = {
    Path("contributing.md"): "Contributing",
    Path("extending.md"): "Extending Lumen",
}
ARCHITECTURE_PAGES = {
    Path("configuration/context.md"): "Context",
    Path("configuration/agents.md"): "Agents",
    Path("configuration/coordinators.md"): "Coordinators",
    Path("configuration/tools.md"): "Tools",
    Path("configuration/spec/customization.md"): "Custom Components",
}
API_PAGES = {
    Path("reference/api.md"): "Overview",
    Path("reference/api/ai.md"): "AI",
    Path("reference/api/ai/agents.md"): "AI Agents",
    Path("reference/api/ai/coordinator.md"): "AI Coordinator",
    Path("reference/api/ai/core.md"): "AI Core",
    Path("reference/api/ai/models.md"): "AI Models",
    Path("reference/api/ai/tools.md"): "AI Tools",
    Path("reference/api/pipeline.md"): "Pipeline",
    Path("reference/api/sources.md"): "Sources",
    Path("reference/api/transforms.md"): "Transforms",
    Path("reference/api/views.md"): "Views",
}


def _section(title: str, description: str, pages: dict[Path, str]) -> LlmsSection:
    return LlmsSection(
        title=title,
        description=description,
        path_prefix=Path("."),
        path_filter=pages.__contains__,
        label_builder=pages.__getitem__,
    )


CONFIG = LlmsBuildConfig(
    project_title="Lumen",
    project_description=(
        "Developer documentation for contributing to and extending Lumen, an extensible "
        "framework for building data applications and AI-powered data workflows."
    ),
    markdown_root=OUTPUT_DIR,
    llms_output_path=BUILTDOCS_DIR / "llms.txt",
    markdown_base_url="/markdown",
    sources=(MarkdownSource(
        source_dir=DOCS_DIR,
        output_dir=OUTPUT_DIR,
        exclude_files=(Path("releases.md"),),
    ),),
    sections=(
        _section(
            "Development",
            "Repository setup, contribution workflow, and extension points.",
            DEVELOPMENT_PAGES,
        ),
        _section(
            "Architecture",
            "Core concepts for Lumen's agent and component architecture.",
            ARCHITECTURE_PAGES,
        ),
        _section(
            "API Reference",
            "Python APIs for Lumen pipelines, sources, transforms, views, and AI components.",
            API_PAGES,
        ),
    ),
)
