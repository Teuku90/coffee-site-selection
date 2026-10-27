"""Tests for the capstone pipeline. Run with: pytest -q"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src.generate_data import (
    LAT_MAX, LAT_MIN, LON_MAX, LON_MIN,
    generate_candidates, generate_competitors,
)
from src.pipeline import add_competition_features, run_pipeline, score_candidates


def _gdf(points, ids, **cols):
    return gpd.GeoDataFrame(
        {"id": ids, **cols, "geometry": [Point(p) for p in points]}, crs="EPSG:4326"
    )


# ~0.001 deg of latitude is ~111 m near the equator
DEG_PER_100M = 0.0009


@pytest.fixture
def toy():
    """One candidate at Jakarta, competitors at ~100 m, ~500 m and ~2 km north."""
    lon, lat = 106.8, -6.2
    cand = _gdf([(lon, lat)], ["c0"], foot_traffic_index=[1.0])
    comp = _gdf(
        [(lon, lat + 1 * DEG_PER_100M), (lon, lat + 5 * DEG_PER_100M), (lon, lat + 20 * DEG_PER_100M)],
        ["near", "mid", "far"],
    )
    return cand, comp


def test_counts_are_in_meters_not_degrees(toy):
    cand, comp = toy
    out = add_competition_features(cand, comp, radius_m=400, max_radius_m=1000)
    # only the ~100 m shop is inside 400 m; the 2 km one is outside 1000 m
    assert out.loc[0, "n_competitors"] == 1
    assert 80 < out.loc[0, "nearest_competitor_m"] < 120


def test_distance_decay_weights(toy):
    cand, comp = toy
    out = add_competition_features(cand, comp, radius_m=400, max_radius_m=1000)
    # ~100 m -> weight ~0.9, ~500 m -> ~0.5, 2 km -> excluded  => ~1.4
    assert out.loc[0, "competition_pressure"] == pytest.approx(1.4, abs=0.05)


def test_sjoin_matches_brute_force():
    cand = generate_candidates(25, np.random.default_rng(0))
    cand["foot_traffic_index"] = 1.0
    comp = generate_competitors(150, np.random.default_rng(1))
    out = add_competition_features(cand, comp, radius_m=400, max_radius_m=1000)

    utm = cand.estimate_utm_crs()
    cu, ku = cand.to_crs(utm), comp.to_crs(utm)
    expected = [int((ku.distance(g) <= 400).sum()) for g in cu.geometry]
    assert out["n_competitors"].tolist() == expected


def test_no_competitors_anywhere():
    cand = _gdf([(106.8, -6.2)], ["c0"], foot_traffic_index=[1.0])
    comp = _gdf([(106.0, -7.0)], ["far_away"])
    out = add_competition_features(cand, comp)
    assert out.loc[0, "n_competitors"] == 0
    assert out.loc[0, "competition_pressure"] == 0.0


def test_scores_bounded_and_ranked():
    cand = generate_candidates(30, np.random.default_rng(2))
    cand["foot_traffic_index"] = np.random.default_rng(3).uniform(0, 1, 30)
    comp = generate_competitors(120, np.random.default_rng(4))
    out = run_pipeline(cand, comp)
    assert out["final_score"].between(0, 1).all()
    assert out["final_score"].is_monotonic_decreasing
    assert out["rank"].tolist() == list(range(1, 31))


def test_ranking_is_not_degenerate():
    """Regression: the old binary score gave most sites an identical 1.0."""
    cand = generate_candidates(40, np.random.default_rng(5))
    cand["foot_traffic_index"] = 1.0  # neutralise traffic: competition alone must rank
    comp = generate_competitors(220, np.random.default_rng(6))
    out = run_pipeline(cand, comp)
    assert out["competition_score"].nunique() > 30


def test_weights_must_sum_to_one(toy):
    cand, comp = toy
    feats = add_competition_features(cand, comp)
    with pytest.raises(ValueError):
        score_candidates(feats, 0.7, 0.7)


def test_synthetic_points_stay_in_bbox_without_edge_pileup():
    comp = generate_competitors(500, np.random.default_rng(7))
    x, y = comp.geometry.x, comp.geometry.y
    assert x.between(LON_MIN, LON_MAX).all() and y.between(LAT_MIN, LAT_MAX).all()
    on_edge = x.isin([LON_MIN, LON_MAX]) | y.isin([LAT_MIN, LAT_MAX])
    assert on_edge.sum() == 0
