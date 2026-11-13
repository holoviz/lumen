"""Render public/og-image.png from scripts/og-card.html.

Run once and commit the result; nothing in the Vite build calls this. The card takes the
headline, the tagline and the hero's first question from the prerendered dist/index.html, so the social
preview cannot keep advertising copy the page no longer has:

    pixi run -e homepage homepage-build
    pixi run -e homepage homepage-og-image
"""

from __future__ import annotations

import html
import re
import sys
import tempfile

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
HOMEPAGE = HERE.parent
PRERENDERED = HOMEPAGE / 'dist' / 'index.html'
LOGO = HOMEPAGE / 'src' / 'assets' / 'logo.svg'
CARD = HERE / 'og-card.html'
OUT = HOMEPAGE / 'public' / 'og-image.png'


def from_prerender() -> dict[str, str]:
    """Pull the headline and the first hero question out of the built page."""
    if not PRERENDERED.exists():
        sys.exit(f'{PRERENDERED} is missing; run the homepage build first.')
    page = PRERENDERED.read_text()
    title = re.search(r'<h1[^>]*>(.*?)</h1>', page, re.DOTALL)
    question = re.search(r'role="radio"[^>]*>(.*?)<', page, re.DOTALL)
    tagline = re.search(r'data-hero-tagline=""[^>]*>(.*?)</p>', page, re.DOTALL)
    if not title or not question or not tagline:
        sys.exit('No h1, tagline or hero question in the prerendered output; did the markup change?')
    # React splits text around interpolations with comment markers, which have to come out.
    clean = lambda s: html.unescape(re.sub(r'<!--.*?-->', '', s)).strip()
    return {
        '__TITLE__': clean(title.group(1)),
        '__TAGLINE__': clean(tagline.group(1)),
        '__QUESTION__': clean(question.group(1)),
    }


def main() -> None:
    source = CARD.read_text().replace('__LOGO__', LOGO.read_text())
    for token, value in from_prerender().items():
        source = source.replace(token, html.escape(value))

    with tempfile.NamedTemporaryFile('w', dir=HERE, suffix='.html', delete=False) as f:
        f.write(source)
        filled = Path(f.name)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 1200, 'height': 630}, device_scale_factor=1)
            page.goto(filled.as_uri(), wait_until='networkidle')
            page.wait_for_timeout(400)
            OUT.parent.mkdir(exist_ok=True)
            page.screenshot(path=str(OUT))
            browser.close()
    finally:
        filled.unlink()

    print(f'wrote {OUT.relative_to(HOMEPAGE)} ({OUT.stat().st_size // 1024} kB)')  # noqa: T201


if __name__ == '__main__':
    main()
