"""
generate_data.py
-----------------
Capstone step 1-2 (book Ch. 06 Geocoding & Ch. 08 OpenStreetMap):

In production this step would call `geopy` to geocode a list of candidate
addresses, and `osmnx.features_from_place(..., tags={"shop": "coffee"})` to
pull real competitor locations from OpenStreetMap.

Both of those calls need live internet access to Nominatim / the Overpass
API, which this environment does not have. To keep the whole pipeline
runnable end-to-end (and reproducible for anyone cloning the repo), this
module generates a realistic *synthetic* stand-in dataset with the exact
same shape as the real one:

    candidates.geojson   -> POINT geometries, columns: id, name
    competitors.geojson  -> POINT geometries, columns: id, brand

Swapping this module for real geocoding / OSMnx calls (see README.md,
section "Going from synthetic to real data") does not require changing
anything in pipeline.py or visualize.py downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Rough bounding box around central Jakarta (lon, lat / EPSG:4326)
LON_MIN, LON_MAX = 106.74, 106.86
LAT_MIN, LAT_MAX = -6.26, -6.14

# A handful of real, well-known commercial hubs in central Jakarta used as
# cluster centers so the synthetic competitor layer looks like a real city
# (coffee shops cluster around business districts) instead of pure noise.
BUSINESS_HUBS = {
    "SCBD / Sudirman": (106.8090, -6.2250),
    "Senayan": (106.8020, -6.2270),
    "Menteng": (106.8340, -6.1960),
    "Kuningan": (106.8290, -6.2240),
    "Kemang": (106.8140, -6.2600),
}

RNG_SEED = 42


def _random_points_in_bbox(n: int, rng: np.random.Generator) -> np.ndarray:
    lon = rng.uniform(LON_MIN, LON_MAX, n)
    lat = rng.uniform(LAT_MIN, LAT_MAX, n)
    return np.column_stack([lon, lat])


def _in_bbox(pts: np.ndarray) -> np.ndarray:
    return (
        (pts[:, 0] >= LON_MIN) & (pts[:, 0] <= LON_MAX)
        & (pts[:, 1] >= LAT_MIN) & (pts[:, 1] <= LAT_MAX)
    )


def _clustered_points(n: int, rng: np.random.Generator, spread: float = 0.012) -> np.ndarray:
    """Scatter n points around the business hubs (Gaussian clusters).

    Points that fall outside the city bounding box are re-sampled (rejection
    sampling) instead of clipped -- clipping would pile them up on the bbox
    edge and create an obviously fake straight line of points on the map.
    """
    hub_coords = np.array(list(BUSINESS_HUBS.values()))
    out = np.empty((0, 2))
    while len(out) < n:
        k = (n - len(out)) * 2
        centers = hub_coords[rng.integers(0, len(hub_coords), k)]
        pts = centers + rng.normal(0, spread, size=(k, 2))
        out = np.vstack([out, pts[_in_bbox(pts)]])
    return out[:n]


def generate_candidates(n: int = 40, rng: np.random.Generator | None = None) -> gpd.GeoDataFrame:
    """Simulate `n` vacant commercial spots available for rent across the city.

    Stands in for: geocoding a real list of addresses with
    `geopy.geocoders.Nominatim` + `geopy.extra.rate_limiter.RateLimiter`
    (book Ch. 06).
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    coords = _random_points_in_bbox(n, rng)
    gdf = gpd.GeoDataFrame(
        {
            "id": [f"site_{i:03d}" for i in range(n)],
            "name": [f"Ruko Jl. Kandidat No. {i + 1}" for i in range(n)],
            "geometry": [Point(xy) for xy in coords],
        },
        crs="EPSG:4326",
    )
    return gdf


def generate_competitors(n: int = 220, rng: np.random.Generator | None = None) -> gpd.GeoDataFrame:
    """Simulate `n` existing coffee shops, clustered around real business hubs.

    Stands in for:
    `osmnx.features_from_place("Jakarta, Indonesia", tags={"shop": "coffee"})`
    (book Ch. 08).
    """
    rng = rng or np.random.default_rng(RNG_SEED + 1)
    n_clustered = int(n * 0.8)
    n_scattered = n - n_clustered

    clustered = _clustered_points(n_clustered, rng)
    scattered = _random_points_in_bbox(n_scattered, rng)
    coords = np.vstack([clustered, scattered])

    brands = rng.choice(
        ["Independen", "Kopi Kenangan", "Fore Coffee", "Starbucks", "Janji Jiwa", "Tomoro"],
        size=len(coords),
        p=[0.40, 0.14, 0.12, 0.10, 0.14, 0.10],
    )

    gdf = gpd.GeoDataFrame(
        {
            "id": [f"competitor_{i:03d}" for i in range(len(coords))],
            "brand": brands,
            "geometry": [Point(xy) for xy in coords],
        },
        crs="EPSG:4326",
    )
    return gdf


def generate_foot_traffic(candidates: gpd.GeoDataFrame) -> pd.Series:
    """Synthetic foot-traffic proxy: higher near business hubs, with noise.

    Stands in for a real population/footfall raster or H3-aggregated mobility
    data (book Ch. 05 H3 Indexing, Ch. 07 Raster, Ch. 10 Hotspot Analysis) —
    here approximated as inverse distance to the nearest business hub.
    """
    hub_coords = np.array(list(BUSINESS_HUBS.values()))
    cand_coords = np.column_stack([candidates.geometry.x, candidates.geometry.y])

    # distance (in degrees, fine for a relative ranking) to nearest hub
    dists = np.min(
        np.sqrt(((cand_coords[:, None, :] - hub_coords[None, :, :]) ** 2).sum(axis=2)),
        axis=1,
    )
    traffic = 1.0 / (1.0 + dists * 100)  # closer to a hub -> higher index
    rng = np.random.default_rng(RNG_SEED + 2)
    traffic = traffic * rng.uniform(0.85, 1.15, size=len(traffic))  # add realism
    return pd.Series(traffic, index=candidates.index, name="foot_traffic_index")


def build_and_save(data_dir: str = "data") -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    import os

    os.makedirs(data_dir, exist_ok=True)

    candidates = generate_candidates()
    candidates["foot_traffic_index"] = generate_foot_traffic(candidates)
    competitors = generate_competitors()

    candidates.to_file(f"{data_dir}/candidates.geojson", driver="GeoJSON")
    competitors.to_file(f"{data_dir}/competitors.geojson", driver="GeoJSON")

    return candidates, competitors


if __name__ == "__main__":
    cand, comp = build_and_save()
    print(f"Generated {len(cand)} candidate sites -> data/candidates.geojson")
    print(f"Generated {len(comp)} competitor coffee shops -> data/competitors.geojson")
