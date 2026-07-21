import re

from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"
LLMS_TXT = DOCS / "llms.txt"
SITE_URL = "https://lumen.holoviz.org/"

pytestmark = pytest.mark.skipif(
    not LLMS_TXT.is_file(), reason="docs directory is not available"
)


def _page(url):
    """Map a documentation URL back to the Markdown file that builds it."""
    path = url.removeprefix(SITE_URL).strip("/")
    return DOCS / (f"{path}.md" if path else "index.md"), DOCS / path / "index.md"


def test_llms_txt_links_resolve():
    urls = re.findall(rf"\]\(({re.escape(SITE_URL)}[^)]*)\)", LLMS_TXT.read_text())
    assert urls, "llms.txt lists no documentation pages"
    missing = [url for url in urls if not any(p.is_file() for p in _page(url))]
    assert not missing, f"llms.txt links to pages that do not exist: {missing}"


def test_llms_txt_follows_spec():
    lines = [line for line in LLMS_TXT.read_text().splitlines() if line.strip()]
    assert lines[0].startswith("# "), "llms.txt must open with an H1 project name"
    assert lines[1].startswith("> "), "llms.txt must follow the H1 with a summary blockquote"
