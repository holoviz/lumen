import sys
import tomllib

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"
CONFIG_FILE = ROOT / "zensical.toml"
SCRIPTS = ROOT / "scripts"

pytestmark = pytest.mark.skipif(
    not CONFIG_FILE.is_file(), reason="docs directory is not available"
)


@pytest.fixture(scope="module")
def build_llms_txt():
    """Import the build script, which lives outside the installed package."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import build_llms_txt
    finally:
        sys.path.remove(str(SCRIPTS))
    return build_llms_txt


@pytest.fixture(scope="module")
def pages(build_llms_txt):
    nav = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))["project"]["nav"]
    return build_llms_txt.iter_nav_pages(nav)


def test_nav_pages_exist(pages):
    """Every page llms.txt advertises has to be a file the build can copy."""
    assert pages, "zensical nav lists no documentation pages"
    missing = [path for _, _, path in pages if not (DOCS / path).is_file()]
    assert not missing, f"nav lists pages that do not exist: {missing}"


def test_llms_txt_follows_spec(build_llms_txt, pages):
    rendered = build_llms_txt.render_llms_txt(pages)
    lines = [line for line in rendered.splitlines() if line.strip()]
    assert lines[0].startswith("# "), "llms.txt must open with an H1 project name"
    assert lines[1].startswith("> "), "llms.txt must follow the H1 with a summary blockquote"


def test_llms_txt_links_every_nav_page(build_llms_txt, pages):
    rendered = build_llms_txt.render_llms_txt(pages)
    url = build_llms_txt.MARKDOWN_URL
    unlinked = [path for _, _, path in pages if f"({url}/{path})" not in rendered]
    assert not unlinked, f"llms.txt omits nav pages: {unlinked}"
