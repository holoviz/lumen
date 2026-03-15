"""
Small manual demo for the experimental XarraySource.

This script uses xarray's tutorial dataset, so it requires:
- xarray
- pooch
- network access on first run to download the dataset
"""

from __future__ import annotations

import sys

from pathlib import Path

import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumen.sources.xarray import XarraySource


def main() -> None:
    ds = xr.tutorial.open_dataset("air_temperature")
    source = XarraySource(dataset=ds)

    try:
        table = source.get_tables()[0]

        print("Tables:")
        print(source.get_tables())

        print("\nSchema keys:")
        print(list(source.get_schema(table).keys()))

        print("\nMetadata:")
        print(source.get_metadata(table))

        print("\nFiltered result (lat=(30.0, 60.0), time='2013-01-01'):")
        result = source.get(table, lat=(30.0, 60.0), time="2013-01-01")
        print(result.head())
        print(f"\nReturned {len(result)} rows")
    finally:
        ds.close()
        source.close()


if __name__ == "__main__":
    main()
