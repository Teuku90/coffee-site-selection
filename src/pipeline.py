"""
pipeline.py
-----------
The actual geospatial analysis. Every step here maps directly onto a
technique taught in "Geospatial Data Science Essentials" (Milan Janosov):

    1. Reproject to a local metric CRS    -> Ch. 04, Map Projections
       (`estimate_utm_crs()` so buffer radii are real meters, not degrees)
    2. Buffer each candidate site          -> Ch. 01, Buffering a Geometry
    3. Spatial join candidates <-> comps    -> Ch. 02, Spatial Join with GeoPandas
    4. Distance-decay competition pressure  -> Ch. 10, Spatial Feature Generation
    5. Score, rank & return the shortlist

Run directly (`python -m src.pipeline`) or import `run_pipeline()`.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def add_competition_features(
    candidates: gpd.GeoDataFrame,
    competitors: gpd.GeoDataFrame,
    radius_m: float = 400.0,
    max_radius_m: float = 1000.0,
) -> gpd.GeoDataFrame:
    """Add two competition features to every candidate site.

    n_competitors       -- competitors within `radius_m` (walking catchment).
                           Easy to explain, but binary: most sites score 0.
    competition_pressure -- every competitor within `max_radius_m` counts,
                           weighted linearly by distance: a shop next door
                           weighs 1.0, one at `max_radius_m` weighs 0.0.
                           This is what actually drives the ranking, so sites
                           with zero shops *inside* 400 m are still told apart
                           by how crowded their wider neighbourhood is.

    Mirrors the book's CRS + spatial-join recipes:
      - never buffer in degrees (EPSG:4326) -- reproject to a local UTM CRS
        first, exactly like `.estimate_utm_crs()` in Ch. 04.
      - use a real `gpd.sjoin`, not a Python double-loop (Ch. 02 / Ch. 05 --
        sjoin uses GeoPandas' spatial index internally).
    """
    if max_radius_m < radius_m:
        raise ValueError("max_radius_m must be >= radius_m")

    # --- Ch. 04: reproject to meters before any distance/area math ---
    utm_crs = candidates.estimate_utm_crs()
    cand_utm = candidates.to_crs(utm_crs)
    comp_utm = competitors.to_crs(utm_crs)

    # --- Ch. 01: buffer each site by the widest radius we care about ---
    areas = cand_utm[["id", "geometry"]].rename(columns={"id": "candidate_id"})
    areas["site_geom"] = areas.geometry
    areas = areas.set_geometry(areas.geometry.buffer(max_radius_m))

    # --- Ch. 02: real spatial join (uses GeoPandas' built-in R-tree index) ---
    joined = gpd.sjoin(
        comp_utm[["id", "geometry"]].rename(columns={"id": "competitor_id"}),
        areas,
        predicate="within",
        how="inner",
    )
    joined["dist_m"] = joined.geometry.distance(gpd.GeoSeries(joined["site_geom"], crs=utm_crs))

    # --- Ch. 10: spatial feature generation ---
    joined["in_radius"] = joined["dist_m"] <= radius_m
    joined["weight"] = 1.0 - joined["dist_m"] / max_radius_m

    feats = joined.groupby("candidate_id").agg(
        n_competitors=("in_radius", "sum"),
        competition_pressure=("weight", "sum"),
        nearest_competitor_m=("dist_m", "min"),
    )

    out = candidates.merge(feats, left_on="id", right_index=True, how="left")
    out["n_competitors"] = out["n_competitors"].fillna(0).astype(int)
    out["competition_pressure"] = out["competition_pressure"].fillna(0.0).round(3)
    # no competitor within max_radius_m at all -> report the cap itself
    out["nearest_competitor_m"] = out["nearest_competitor_m"].fillna(max_radius_m).round(0)
    return out


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(1.0, index=s.index)
    return (s - s.min()) / rng


def score_candidates(
    candidates: gpd.GeoDataFrame,
    competition_weight: float = 0.6,
    traffic_weight: float = 0.4,
) -> gpd.GeoDataFrame:
    """Combine competition pressure and foot traffic into one 0-1 score.

    Both signals are min-max normalised so the weights mean what they say:
    0.6 / 0.4 = competition matters 1.5x as much as foot traffic.
    """
    if not np.isclose(competition_weight + traffic_weight, 1.0):
        raise ValueError("weights must sum to 1")

    out = candidates.copy()
    out["competition_score"] = (1.0 - _minmax(out["competition_pressure"])).round(3)
    out["traffic_score"] = _minmax(out["foot_traffic_index"]).round(3)
    out["final_score"] = (
        competition_weight * out["competition_score"]
        + traffic_weight * out["traffic_score"]
    ).round(3)
    return out


def run_pipeline(
    candidates: gpd.GeoDataFrame,
    competitors: gpd.GeoDataFrame,
    radius_m: float = 400.0,
    max_radius_m: float = 1000.0,
    competition_weight: float = 0.6,
) -> gpd.GeoDataFrame:
    """Full pipeline: reproject + buffer + spatial join + score + rank.

    Returns all candidates sorted best-first, with a `rank` column; take
    `.head(n)` for the shortlist.
    """
    feats = add_competition_features(candidates, competitors, radius_m, max_radius_m)
    scored = score_candidates(feats, competition_weight, 1.0 - competition_weight)
    scored = scored.sort_values("final_score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    return scored


if __name__ == "__main__":
    candidates = gpd.read_file("data/candidates.geojson")
    competitors = gpd.read_file("data/competitors.geojson")
    result = run_pipeline(candidates, competitors)
    cols = ["rank", "name", "n_competitors", "competition_pressure", "final_score"]
    print(result[cols].head(10).to_string(index=False))
