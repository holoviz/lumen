"""Capture the homepage's showcase pictures from a live Lumen session.

Each shot asks a running Lumen AI server one real question and screenshots the app once the
plan completes, so the pictures show what Lumen actually produces. Run it once and commit the
result; nothing in the Vite build calls this. Start the server with every dataset the shots
ask about, using any provider:

    lumen-ai serve air_temperature.nc penguins.csv earthquakes.csv --provider claude-code
    python homepage/scripts/showcase.py --url http://localhost:5006/lumen_ai
    python homepage/scripts/showcase.py --video penguins earthquakes

`--video` records one session asking each question in turn, then cuts out the time the model
spends thinking and writes demo.mp4, demo.webm and a poster with ffmpeg.

LLM output varies between runs, so look at every picture before committing it.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
import time

from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ASSETS = Path(__file__).resolve().parent.parent / 'src' / 'assets' / 'showcase'

VIEWPORT = {'width': 1440, 'height': 900}

SHOTS = {
    'heatmap': (
        'Using the air table, map the average air temperature over the whole period on a lat/lon grid '
        'as a heatmap, with north at the top.'
    ),
    'earthquakes': 'Map every earthquake on a Deck.gl world map, sized and coloured by magnitude.',
    # The geospatial shots need airports.csv, flights-airport.csv and cities.geojson loaded too.
    'arcs': (
        'Join flights_airport_csv to airports_csv on origin and destination, and draw the 150 busiest '
        'routes as arcs on a Deck.gl map of the United States, with arc width by flight count.'
    ),
    'hexagons': (
        'Aggregate the earthquakes into large 3D hexagons on a Deck.gl map, each about 300 km across, '
        'extruded tall by the number of quakes with a big elevation scale, coloured yellow to red by '
        'count, with the camera tilted 50 degrees over the Pacific Ring of Fire.'
    ),
    'cities': (
        'Show the world cities as glowing 3D columns on a dark Deck.gl map, each column as tall as the '
        'city population (pop_max) with the tallest about 1,500 km high, coloured from teal to magenta '
        'by population, with the camera tilted 50 degrees over Asia.'
    ),
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


def ask(tab, question: str, typing_delay: int = 5) -> bool:
    """Send one question and wait for the whole plan, validation included, to finish."""
    done = tab.get_by_text('Successfully completed')
    answered = done.count()
    prompt = tab.get_by_placeholder('Ask anything...')
    prompt.click()
    prompt.press_sequentially(question, delay=typing_delay)
    prompt.press('Enter')
    # Earlier answers in the same session are still on screen, so wait for a new one.
    for _ in range(300):
        if done.count() > answered:
            break
        tab.wait_for_timeout(1000)
    else:
        return False
    # Steps can report success while the plan still runs its last task (validation),
    # so wait until nothing is running any more.
    for _ in range(300):
        if not tab.get_by_text('Working on task').count():
            break
        tab.wait_for_timeout(1000)
    if tab.get_by_text('Query Validation').count():
        print('validation reported the answer incomplete')  # noqa: T201
    return True


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
            if not ask(tab, SHOTS[name]):
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


def record(url: str, names: list[str]) -> None:
    """Record one session asking each question, keeping only the typing and the answers."""
    keep = []
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        context = browser.new_context(
            viewport=VIEWPORT, record_video_dir=tmp, record_video_size=VIEWPORT, color_scheme='light'
        )
        tab = context.new_page()
        start = time.monotonic()
        tab.goto(url, wait_until='networkidle', timeout=120_000)
        tab.locator('header button').first.click()
        for name in names:
            typed = time.monotonic() - start
            if not ask(tab, SHOTS[name], typing_delay=35):
                raise SystemExit(f'{name}: no answer within 5 minutes')
            done = time.monotonic() - start
            tab.wait_for_timeout(5000)
            # The typing and the first plan steps, then the finished answer.
            keep += [(max(typed - 1, 0), typed + len(SHOTS[name]) * 0.035 + 4), (done - 1, done + 5)]
        video = Path(tab.video.path())
        context.close()
        browser.close()
        cut(video, keep, ASSETS)


def cut(video: Path, keep: list[tuple[float, float]], out: Path) -> None:
    trims = ''.join(
        f'[0:v]trim=start={a:.2f}:end={b:.2f},setpts=PTS-STARTPTS[v{i}];' for i, (a, b) in enumerate(keep)
    )
    graph = trims + ''.join(f'[v{i}]' for i in range(len(keep))) + f'concat=n={len(keep)}:v=1:a=0[out]'
    out.mkdir(parents=True, exist_ok=True)
    encoders = {
        'demo.mp4': ['-c:v', 'libx264', '-crf', '28', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'],
        'demo.webm': ['-c:v', 'libvpx-vp9', '-crf', '40', '-b:v', '0'],
    }
    for name, codec in encoders.items():
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video), '-filter_complex', graph,
             '-map', '[out]', '-an', *codec, str(out / name)],
            check=True,
        )
        print(f'{name}  {(out / name).stat().st_size / 1024:.0f} kB')  # noqa: T201
    # The poster is the last frame, so a viewer who never plays it still sees an answer.
    frame = subprocess.run(
        ['ffmpeg', '-loglevel', 'error', '-sseof', '-0.5', '-i', str(out / 'demo.mp4'),
         '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
        check=True, capture_output=True,
    ).stdout
    save(frame, out / 'demo-poster.webp')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--url', default='http://localhost:5006/lumen_ai')
    parser.add_argument('--video', action='store_true', help='record the questions as one video')
    parser.add_argument('names', nargs='*', help=f'shots to take from {list(SHOTS)}, default all')
    args = parser.parse_args()
    (record if args.video else capture)(args.url, args.names or list(SHOTS))


if __name__ == '__main__':
    main()
