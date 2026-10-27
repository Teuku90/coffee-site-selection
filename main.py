#!/usr/bin/env python3
"""
Capstone: Cari Lokasi Gerai Kopi Terbaik di Jakarta
====================================================

Companion hands-on project for the slide deck "Geospatial Data Science
Essentials — Ringkasan & Peta Tantangan" (summary of Milan Janosov's book).

End-to-end pipeline:

    1. generate_data.py  -> candidate sites + competitor coffee shops
                             (synthetic stand-in for geocoding + OSMnx,
                              see README.md)
    2. pipeline.py       -> reproject -> buffer -> spatial join -> score
    3. visualize.py      -> static PNG + interactive HTML map

Usage
-----
    python main.py                      # full run, default params
    python main.py --radius 600         # bigger competitor search radius
    python main.py --top 8 --regenerate # regenerate synthetic data first

Outputs land in outputs/:
    scored_candidates.csv
    site_selection_map.png
    interactive_map.html
"""

from __future__ import annotations

import argparse
import os

import geopandas as gpd

from src.generate_data import build_and_save
from src.pipeline import run_pipeline
from src.visualize import plot_static_map, build_interactive_map

DATA_DIR = "data"
OUT_DIR = "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radius", type=float, default=400.0, help="Radius jalan kaki untuk n_competitors, meter (default: 400)")
    p.add_argument("--max-radius", type=float, default=1000.0, help="Radius terjauh kompetitor masih dihitung (bobot jarak), meter (default: 1000)")
    p.add_argument("--competition-weight", type=float, default=0.6, help="Bobot kompetisi 0-1; sisanya untuk foot traffic (default: 0.6)")
    p.add_argument("--top", type=int, default=5, help="Jumlah lokasi terbaik yang direkomendasikan (default: 5)")
    p.add_argument("--regenerate", action="store_true", help="Buat ulang data sintetis sebelum menjalankan pipeline")
    return p.parse_args()


def load_or_generate_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cand_path = f"{DATA_DIR}/candidates.geojson"
    comp_path = f"{DATA_DIR}/competitors.geojson"
    if os.path.exists(cand_path) and os.path.exists(comp_path):
        return gpd.read_file(cand_path), gpd.read_file(comp_path)
    return build_and_save(DATA_DIR)


def main() -> None:
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.regenerate:
        candidates, competitors = build_and_save(DATA_DIR)
    else:
        candidates, competitors = load_or_generate_data()

    print(f"Kandidat lokasi : {len(candidates)}")
    print(f"Kompetitor OSM  : {len(competitors)}")
    print(f"Radius layanan  : {args.radius:.0f} m (pengaruh sampai {args.max_radius:.0f} m)\n")

    scored = run_pipeline(
        candidates,
        competitors,
        radius_m=args.radius,
        max_radius_m=args.max_radius,
        competition_weight=args.competition_weight,
    )

    scored_csv_cols = [
        "rank", "id", "name", "n_competitors", "competition_pressure",
        "nearest_competitor_m", "competition_score", "traffic_score", "final_score",
    ]
    scored[scored_csv_cols].to_csv(f"{OUT_DIR}/scored_candidates.csv", index=False)

    print(f"Top {args.top} lokasi rekomendasi:\n")
    print(scored[scored_csv_cols].head(args.top).to_string(index=False))

    plot_static_map(scored, competitors, top_n=args.top, out_path=f"{OUT_DIR}/site_selection_map.png")
    build_interactive_map(scored, competitors, top_n=args.top, out_path=f"{OUT_DIR}/interactive_map.html")

    print(f"\nTersimpan:")
    print(f"  {OUT_DIR}/scored_candidates.csv")
    print(f"  {OUT_DIR}/site_selection_map.png")
    print(f"  {OUT_DIR}/interactive_map.html")


if __name__ == "__main__":
    main()
