"""Playwright checks for the built homepage.

These run against dist/, not the dev server, because the prerendered HTML has to hydrate
without React complaining and the interactive parts have to keep working after it does.

    pixi run -e homepage homepage-build
    pixi run -e homepage homepage-test
"""

from __future__ import annotations

import http.server
import re
import socket
import threading

from pathlib import Path

import pytest

from playwright.sync_api import ConsoleMessage, Page, expect

HOMEPAGE = Path(__file__).resolve().parent.parent
DIST = HOMEPAGE / 'dist'
DOCS = HOMEPAGE.parent / 'docs'


@pytest.fixture(scope='session')
def site() -> str:
    if not (DIST / 'index.html').exists():
        pytest.skip(f'{DIST} has not been built')

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DIST), **kwargs)

        def log_message(self, *args):
            pass

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]

    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{port}'
    finally:
        server.shutdown()


@pytest.fixture
def home(page: Page, site: str) -> Page:
    page.goto(site, wait_until='networkidle')
    return page


def test_prerendered_without_javascript(page: Page, site: str) -> None:
    """The headline must be in the HTML the server sends, not painted in by the bundle."""
    page.route('**/*.js', lambda route: route.abort())
    page.goto(site, wait_until='domcontentloaded')
    expect(page.get_by_role('heading', level=1)).to_contain_text('Illuminate your data')


def test_hydrates_without_console_errors(page: Page, site: str) -> None:
    problems: list[str] = []

    def record(msg: ConsoleMessage) -> None:
        if msg.type in ('error', 'warning'):
            problems.append(msg.text)

    page.on('console', record)
    page.on('pageerror', lambda e: problems.append(str(e)))
    page.goto(site, wait_until='networkidle')
    # A hydration mismatch is reported after the first commit, so give React a beat.
    page.wait_for_timeout(600)
    assert not problems, problems


def test_hero_chat_swaps_the_answer(home: Page) -> None:
    """Each question shows the SQL Lumen wrote and the Vega-Lite chart it drew."""
    chart = home.get_by_role('img', name=re.compile('penguins per island'))
    expect(chart).to_be_visible()
    expect(chart.locator('svg')).to_have_count(1)
    home.get_by_role('radio', name='Which species is the heaviest on average?').click()
    expect(home.get_by_role('img', name=re.compile('average body mass by species'))).to_be_visible()
    expect(home.get_by_label('Generated SQL')).to_contain_text('AVG(')


def test_hero_chart_shows_the_value_on_hover(home: Page) -> None:
    """The live Vega-Lite chart replaces the static one and its tooltips name each bar's value."""
    home.locator('[aria-label*="penguins per island"] .vega-embed').wait_for(timeout=15_000)
    home.locator('[aria-label*="penguins per island"] .mark-rect path').first.hover()
    tooltip = home.locator('#vg-tooltip-element')
    expect(tooltip).to_be_visible()
    expect(tooltip).to_contain_text('Biscoe')
    expect(tooltip).to_contain_text('168')


def test_every_picture_loads_from_the_hashed_prefix(home: Page) -> None:
    # Lazy tiles only fetch near the viewport, so make them all eager and wait for each.
    images = home.evaluate("""async () => {
        const imgs = [...document.querySelectorAll('main img')]
        imgs.forEach((i) => { i.loading = 'eager' })
        await Promise.all(imgs.map((i) => i.decode().catch(() => null)))
        return imgs.map((i) => [i.getAttribute('src'), i.naturalWidth])
    }""")
    assert len(images) == 8
    assert not [src for src, width in images if not width], 'broken pictures'
    assert all(src.startswith(('/_home/', 'data:image/')) for src, _ in images), images


def test_internal_links_point_at_docs_pages(home: Page) -> None:
    """Every docs link has to resolve to a source page in the versioned Zensical build."""
    hrefs = home.eval_on_selector_all('a[href]', 'els => els.map(e => e.getAttribute("href"))')
    internal = {h.split('#')[0] for h in hrefs if h.startswith('/') and h != '/'}
    assert internal, 'expected some docs links'
    outside = [h for h in internal if not re.match(r'/en/docs/[^/]+/', h)]
    assert not outside, f'links outside /en/docs/<version>/: {outside}'
    pages = {re.sub(r'^/en/docs/[^/]+/', '', h).strip('/') for h in internal}
    missing = [
        p for p in pages
        if not (DOCS / f'{p}.md').exists() and not (DOCS / p / 'index.md').exists()
    ]
    assert not missing, missing


@pytest.mark.parametrize('width', [1440, 390])
def test_no_code_scrolls_sideways(page: Page, site: str, width: int) -> None:
    """Snippets wrap rather than hiding the ends of their lines."""
    page.set_viewport_size({'width': width, 'height': 900})
    page.goto(site, wait_until='networkidle')
    clipped = page.eval_on_selector_all(
        'main pre',
        'els => els.filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.textContent.slice(0, 40))',
    )
    assert not clipped, clipped


def test_install_command_copies(home: Page) -> None:
    home.context.grant_permissions(['clipboard-read', 'clipboard-write'])
    home.get_by_role('button', name='Copy').click()
    expect(home.get_by_role('button', name='Copied')).to_be_visible()
    assert home.evaluate('navigator.clipboard.readText()') == 'pip install lumen'


def test_every_focusable_element_shows_a_ring(home: Page) -> None:
    """MUI's ButtonBase sets `outline: 0`, so a missing override would hide keyboard focus."""
    seen = []
    for _ in range(20):
        home.keyboard.press('Tab')
        seen.append(
            home.evaluate("""() => {
                const e = document.activeElement;
                const s = getComputedStyle(e);
                return [(e.getAttribute('aria-label') || e.textContent || '').slice(0, 30),
                        parseFloat(s.outlineWidth)];
            }""")
        )
    ringless = [label for label, width in seen if width < 1]
    assert not ringless, ringless


def test_no_horizontal_overflow_on_a_phone(page: Page, site: str) -> None:
    page.set_viewport_size({'width': 390, 'height': 844})
    page.goto(site, wait_until='networkidle')
    scroll_width, client_width = page.evaluate(
        '() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]'
    )
    assert scroll_width <= client_width + 1


def test_showcase_video_is_silent_and_has_a_poster(home: Page) -> None:
    """The recording autoplays only because it is muted, and needs a still for slow connections."""
    video = home.locator('#showcase video')
    expect(video).to_have_attribute('poster', '/_home/' + (video.get_attribute('poster') or '').split('/_home/')[-1])
    assert video.evaluate('v => v.muted && v.loop && v.playsInline')
    sources = video.locator('source').evaluate_all('els => els.map(e => e.getAttribute("src"))')
    assert len(sources) == 2
    assert all(src.startswith('/_home/') for src in sources), sources


def test_showcase_video_waits_until_visible(home: Page) -> None:
    home.evaluate('window.scrollTo(0, 0)')
    home.wait_for_timeout(500)
    assert home.locator('#showcase video').evaluate('v => v.paused')
    home.locator('#showcase').scroll_into_view_if_needed()
    home.wait_for_timeout(1500)
    assert not home.locator('#showcase video').evaluate('v => v.paused')
