"""Render the hero's Vega-Lite charts to SVG, from the specs Lumen wrote.

Run once and commit the result; nothing in the Vite build calls this. It reads the specs and
SQL that hero_capture.py saved in scripts/hero/, runs each query on the penguins dataset,
puts the rows into the spec and compiles it with vl-convert, so every chart on the page is
exactly what Lumen drew. Needs duckdb, pyyaml and vl-convert-python:

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


def main() -> None:
    con = duckdb.connect()
    con.execute('INSTALL httpfs; LOAD httpfs;')
    con.execute(f"CREATE VIEW penguins_csv AS SELECT * FROM read_csv_auto('{PENGUINS}')")
    OUT.mkdir(parents=True, exist_ok=True)
    for capture in sorted((HERE / 'hero').glob('*.json')):
        data = json.loads(capture.read_text())
        rows = con.execute(data['sql']).fetchdf().to_dict('records')
        spec = yaml.safe_load(data['spec'])
        # Lumen sizes charts to their container and feeds the data in separately.
        spec['data'] = {'values': rows}
        spec['width'], spec['height'] = 520, 190
        svg = vlc.vegalite_to_svg(json.dumps(spec, default=str))
        # Drop the fixed size so the SVG scales with the plate it sits on; its viewBox stays.
        svg = re.sub(r'(<svg[^>]*?) width="[\d.]+" height="[\d.]+"', r'\1', svg, count=1)
        (OUT / f'{capture.stem}.svg').write_text(svg)
        print(f'{capture.stem}.svg  {len(svg) / 1024:.0f} kB')  # noqa: T201


if __name__ == '__main__':
    main()
