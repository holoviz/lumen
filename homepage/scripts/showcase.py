"""Capture the homepage's showcase pictures from a live Lumen session.

Each shot asks a running Lumen AI server one real question and screenshots the app once the
plan completes, so the pictures show what Lumen actually produces. Run it once and commit the
result; nothing in the Vite build calls this. Start the server with every dataset the shots
ask about, using any provider:

    lumen-ai serve air_temperature.nc penguins.csv earthquakes.csv --provider claude-code
    python homepage/scripts/showcase.py --url http://localhost:5006/lumen_ai

LLM output varies between runs, so look at every picture before committing it.
"""

from __future__ import annotations

import argparse
import io

from pathlib import Path

from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

ASSETS = Path(__file__).resolve().parent.parent / 'src' / 'assets' / 'showcase'

VIEWPORT = {'width': 1440, 'height': 900}

SHOTS = {
    'heatmap': (
        'Using the air table, map the average air temperature over the whole period on a lat/lon grid '
        'as a heatmap, with north at the top.'
    ),
    'earthquakes': 'Map every earthquake on a Deck.gl world map, sized and coloured by magnitude.',
    # VegaLiteAgent answers a request for several charts with one tab each plus an "All" overview.
    'penguins': (
        'Make three charts of the penguins: body mass by species, flipper length against body mass '
        'coloured by species, and penguin counts by island.'
    ),
}


def save(png: bytes, path: Path, width: int = 1600, quality: int = 80) -> None:
    image = Image.open(io.BytesIO(png)).convert('RGB')
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, 'WEBP', quality=quality, method=6)
    print(f'{path.name}  {path.stat().st_size / 1024:.0f} kB')  # noqa: T201


def capture(url: str, names: list[str]) -> None:
    with sync_playwright() as p:
        # Deck.gl draws with WebGL, which headless Chromium only offers through SwiftShader.
        browser = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        for name in names:
            # A fresh page per shot, so each one starts from the splash screen.
            tab = browser.new_page(viewport=VIEWPORT, device_scale_factor=2, color_scheme='light')
            tab.goto(url, wait_until='networkidle', timeout=120_000)
            # Fold the navigation drawer away so the chat and the chart share the width.
            tab.locator('header button').first.click()
            prompt = tab.get_by_placeholder('Ask anything...')
            prompt.click()
            prompt.press_sequentially(SHOTS[name], delay=5)
            prompt.press('Enter')
            try:
                tab.get_by_text('Successfully completed').first.wait_for(timeout=300_000)
                # Steps can report success while the plan still runs its last task
                # (validation), so wait until nothing is running any more.
                for _ in range(300):
                    if not tab.get_by_text('Working on task').count():
                        break
                    tab.wait_for_timeout(1000)
                if tab.get_by_text('Query Validation').count():
                    print(f'{name}: validation reported the answer incomplete')  # noqa: T201
            except PlaywrightTimeout:
                print(f'{name}: no answer within 5 minutes, skipped')  # noqa: T201
                tab.close()
                continue
            # Give the chart most of the width; the chat column only needs the question.
            gutter = tab.locator('.gutter-horizontal').first.bounding_box()
            tab.mouse.move(gutter['x'] + 4, gutter['y'] + 200)
            tab.mouse.down()
            tab.mouse.move(VIEWPORT['width'] * 0.42, gutter['y'] + 200, steps=10)
            tab.mouse.up()
            # Let the chart re-render at its new size, map tiles included.
            tab.wait_for_timeout(8000)
            save(tab.screenshot(), ASSETS / f'{name}.webp')
            tab.close()
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--url', default='http://localhost:5006/lumen_ai')
    parser.add_argument('names', nargs='*', help=f'shots to take from {list(SHOTS)}, default all')
    args = parser.parse_args()
    capture(args.url, args.names or list(SHOTS))


if __name__ == '__main__':
    main()
