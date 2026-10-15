"""Render the hero's Vega-Lite charts to SVG, from the specs Lumen wrote.

Run once and commit the result; nothing in the Vite build calls this. It reads the specs and
SQL that hero_capture.py saved in scripts/hero/, runs each query on the penguins dataset,
puts the rows into the spec and compiles it with vl-convert, so every chart on the page is
exactly what Lumen drew, in the dark blue scheme the hero panel uses. It writes the static SVG
the page paints at once and the finished spec the browser embeds for hover tooltips. Needs
duckdb, pyyaml and vl-convert-python:

    python homepage/scripts/hero_charts.py
"""

from __future__ import annotations

import json
import re

from pathlib import Path

import duckdb
import vl_convert as vlc
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'src' / 'assets' / 'hero'
PENGUINS = 'https://datasets.holoviz.org/penguins/v1/penguins.csv'

# Lumen's own blues, recoloured to sit on the dark panel. Bars keep their meaning: the highlighted
# one is the bright blue and the rest are muted.
RECOLOUR = {'#1f77b4': '#64b5f6', '#aec7e8': '#2d5a91'}
INK, MUTED = '#e8f1fb', '#9db8d9'
DARK = {
    'background': '#0c2340',
    'title': {'color': INK, 'subtitleColor': MUTED},
    'axis': {'labelColor': MUTED, 'titleColor': INK, 'domainColor': '#6f8fb5', 'tickColor': '#6f8fb5'},
    'text': {'color': INK},
    'view': {'stroke': None},
}


def recolour(node):
    if isinstance(node, dict):
        return {k: recolour(v) for k, v in node.items()}
    if isinstance(node, list):
        return [recolour(v) for v in node]
    return RECOLOUR.get(node, node)


def main() -> None:
    con = duckdb.connect()
    con.execute('INSTALL httpfs; LOAD httpfs;')
    con.execute(f"CREATE VIEW penguins_csv AS SELECT * FROM read_csv_auto('{PENGUINS}')")
    OUT.mkdir(parents=True, exist_ok=True)
    for capture in sorted((HERE / 'hero').glob('*.json')):
        data = json.loads(capture.read_text())
        rows = con.execute(data['sql']).fetchdf().to_dict('records')
        spec = recolour(yaml.safe_load(data['spec']))
        spec['config'] = {**spec.get('config', {}), **{k: {**spec.get('config', {}).get(k, {}), **v} if isinstance(v, dict) else v for k, v in DARK.items()}}
        # Lumen sizes charts to their container and feeds the data in separately.
        spec['data'] = {'values': rows}
        # Lumen writes v5; the page embeds Vega-Lite 6, which reads the same spec but warns about v5.
        spec['$schema'] = 'https://vega.github.io/schema/vega-lite/v6.json'
        spec['width'], spec['height'] = 520, 190
        (OUT / f'{capture.stem}.json').write_text(json.dumps(spec, default=str, separators=(',', ':')) + '\n')
        svg = vlc.vegalite_to_svg(json.dumps(spec, default=str))
        # Drop the fixed size so the SVG scales with the plate it sits on; its viewBox stays.
        svg = re.sub(r'(<svg[^>]*?) width="[\d.]+" height="[\d.]+"', r'\1', svg, count=1)
        (OUT / f'{capture.stem}.svg').write_text(svg + '\n')
        print(f'{capture.stem}.svg  {len(svg) / 1024:.0f} kB')  # noqa: T201


if __name__ == '__main__':
    main()
