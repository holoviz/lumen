"""
Place names to coordinates for the SQL agent, through OpenStreetMap's
Nominatim service. Opt-in, since each place name leaves the machine: off
unless lumen.ai.configure(geocode=True) or LUMEN_AI_GEOCODE=1.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import cache
from typing import Any

from ... import __version__
from ..sharing import sharing
from .base import FunctionTool

#: The place names a question looked up, so its answer can say what was sent
GEOCODED: ContextVar[list[str] | None] = ContextVar("geocoded", default=None)

ATTRIBUTION = "Geocoded with OpenStreetMap Nominatim, © OpenStreetMap contributors"


@cache
def _nominatim() -> Any:
    try:
        # Deferred: geopy is optional, and only an opted-in question needs it
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim
    except ImportError as e:
        raise ImportError("Geocoding needs geopy: pip install geopy") from e
    # Nominatim's policy: an identifying user agent and at most one request a second
    service = Nominatim(user_agent=f"lumen/{__version__} (https://github.com/holoviz/lumen)", timeout=10)
    return RateLimiter(service.geocode, min_delay_seconds=1)


@cache
def _lookup(place: str) -> Any:
    # Cached, so a place asked about twice is sent once
    return _nominatim()(place, exactly_one=True)


def geocode(place: str) -> str:
    """
    Look up a place name, e.g. a city, region or landmark, and return its
    longitude, latitude and bounding box, to use in SQL as
    ST_Point(lon, lat) or a box filter. Only the place name is sent, to
    OpenStreetMap's Nominatim service.
    """
    found = _lookup(place.strip())
    if (places := GEOCODED.get()) is not None and place not in places:
        places.append(place)
    if found is None:
        return f"No place called {place!r} was found; ask for coordinates or a more specific name."
    south, north, west, east = (float(v) for v in found.raw["boundingbox"])
    return (
        f"{found.address}: lon {found.longitude:.6f}, lat {found.latitude:.6f}; "
        f"bounding box west {west}, south {south}, east {east}, north {north}. {ATTRIBUTION}."
    )


def geocode_tools(context: dict[str, Any]) -> list[FunctionTool] | None:
    """An ``llm_tools`` entry: the geocoding tool, once the user has turned geocoding on."""
    return [FunctionTool(geocode)] if sharing.geocode else None
