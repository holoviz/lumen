"""Ask the hero's three questions on a live Lumen and save the SQL and Vega-Lite it wrote.

Run once and commit the result; nothing in the Vite build calls this. Start Lumen on the
penguins dataset first, then render the charts with hero_charts.py:

    lumen-ai serve penguins.csv --provider claude-code
    python homepage/scripts/hero_capture.py
    python homepage/scripts/hero_charts.py

A chart without both axes is retried, because the model now and then answers with a bar that
has no x or y encoding. Look at every chart before committing it.
"""

from __future__ import annotations

import json

from pathlib import Path

import yaml

from playwright.sync_api import sync_playwright
from showcase import VIEWPORT, ask

OUT = Path(__file__).resolve().parent / 'hero'

QUESTIONS = {
    'islands': 'Which islands have the most penguins? Show it as a bar chart.',
    'mass': 'Which species is the heaviest on average? Show a bar chart of average body mass by species.',
    'species': 'How many of each species are there? Show it as a bar chart.',
}

EDITORS = """() => { const out = []; (function walk(root) { for (const el of root.querySelectorAll('*')) {
  if (el.env && el.env.editor && el.env.editor.getValue) out.push(el.env.editor.getValue());
  if (el.shadowRoot) walk(el.shadowRoot); } })(document); return [...new Set(out)]; }"""

CLICK_TAB = """(i) => { const tabs = []; (function walk(root) { for (const el of root.querySelectorAll('*')) {
  const c = (el.className && el.className.toString()) || ''; if (/MuiTab-root/.test(c)) tabs.push(el);
  if (el.shadowRoot) walk(el.shadowRoot); } })(document); tabs[i].click(); }"""


def has_axes(spec_text: str) -> bool:
    spec = yaml.safe_load(spec_text)
    layers = spec.get('layer') or [spec]
    return any({'x', 'y'} <= set(layer.get('encoding', {})) for layer in layers)


def capture(browser, name: str, url: str) -> dict | None:
    tab = browser.new_page(viewport=VIEWPORT)
    try:
        tab.goto(url, wait_until='networkidle', timeout=120_000)
        if not ask(tab, QUESTIONS[name]):
            return None
        tab.wait_for_timeout(30_000)  # chart polishing keeps running after the plan reports success

        def read(index: int, keep):
            tab.evaluate(CLICK_TAB, index)
            tab.wait_for_timeout(2000)
            return next((t for t in tab.evaluate(EDITORS) if keep(t)), None)

        # The tabs are the data source, the SQL query, then the chart.
        sql = read(1, lambda t: 'select' in t.lower() and '$schema' not in t)
        spec = read(2, lambda t: '$schema' in t)
        return {'question': QUESTIONS[name], 'sql': sql, 'spec': spec}
    finally:
        tab.close()


def main(url: str = 'http://localhost:5006/lumen_ai') -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name in QUESTIONS:
            for attempt in range(1, 4):
                result = capture(browser, name, url)
                if result and result['sql'] and result['spec'] and has_axes(result['spec']):
                    (OUT / f'{name}.json').write_text(json.dumps(result, indent=1) + '\n')
                    print(f'{name}: ok on attempt {attempt}')  # noqa: T201
                    break
            else:
                raise SystemExit(f'{name}: no usable chart after 3 attempts')
        browser.close()


if __name__ == '__main__':
    main()
