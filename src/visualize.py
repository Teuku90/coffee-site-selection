"""
visualize.py
------------
Two outputs, mirroring the book's visualization chapter (Ch. 03):

  1. A static Matplotlib map (`outputs/site_selection_map.png`)       -- Ch. 03
     Matplotlib categorical/continuous coloring recipes.
  2. An interactive Folium map (`outputs/interactive_map.html`)        -- Ch. 03
     `gdf.explore()` / Folium recipe -- open it in any browser, pan,
     zoom, and click a marker for its score.
"""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_static_map(
    candidates_scored: gpd.GeoDataFrame,
    competitors: gpd.GeoDataFrame,
    top_n: int,
    out_path: str,
) -> None:
    # Light "report" palette: warm white background, amber = competition,
    # green = opportunity. Matches the companion slide deck.
    BG, INK, MUTED = "#FAFAF7", "#1A1F1C", "#5B655F"
    AMBER, GREEN = "#A85F12", "#1F7A4D"

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(BG)

    competitors.plot(ax=ax, markersize=12, color=AMBER, alpha=0.45, zorder=2)

    rest = candidates_scored.iloc[top_n:]
    best = candidates_scored.iloc[:top_n]
    vmin, vmax = candidates_scored["final_score"].min(), candidates_scored["final_score"].max()

    rest.plot(
        ax=ax,
        column="final_score",
        cmap="Greens",
        vmin=vmin,
        vmax=vmax,
        markersize=80,
        edgecolor=MUTED,
        linewidth=0.8,
        zorder=3,
        legend=True,
        legend_kwds={"label": "Skor akhir (0–1)", "orientation": "horizontal", "shrink": 0.45, "pad": 0.03, "aspect": 30},
    )
    best.plot(
        ax=ax,
        color=GREEN,
        markersize=320,
        edgecolor=INK,
        linewidth=1.2,
        marker="*",
        zorder=5,
    )

    for _, row in best.iterrows():
        ax.annotate(
            f"#{row['rank']}  {row['final_score']:.3f}",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(9, 7),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=INK,
            bbox={"boxstyle": "round,pad=0.25", "fc": BG, "ec": "#D9DED9", "lw": 0.8},
            zorder=6,
        )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Kompetitor (coffee shop)", markerfacecolor=AMBER, alpha=0.7, markersize=8, linestyle="None"),
        Line2D([0], [0], marker="o", color="w", label="Kandidat lokasi (warna = skor)", markerfacecolor="#7FBF8E", markeredgecolor=MUTED, markersize=9, linestyle="None"),
        Line2D([0], [0], marker="*", color="w", label=f"Top {top_n} rekomendasi", markerfacecolor=GREEN, markeredgecolor=INK, markersize=15, linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, labelcolor=INK, fontsize=10)

    ax.set_title(
        "Skoring lokasi gerai kopi — Jakarta pusat (data sintetis)",
        color=INK,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=34,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.margins(0.07)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(colors=MUTED, labelsize=9)
    cbar_ax.xaxis.label.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_interactive_map(
    candidates_scored: gpd.GeoDataFrame,
    competitors: gpd.GeoDataFrame,
    top_n: int,
    out_path: str,
) -> None:
    # gdf.explore() (book Ch. 03) returns a folium.Map we can keep layering on
    m = candidates_scored.explore(
        column="final_score",
        cmap="Greens",
        tooltip=["rank", "name", "n_competitors", "final_score"],
        legend=True,
        marker_kwds={"radius": 7},
        name="Kandidat lokasi (skor)",
    )
    competitors.explore(
        m=m,
        color="#F5A524",
        marker_kwds={"radius": 3},
        tooltip=["brand"],
        name="Kompetitor",
    )

    import folium

    for _, row in candidates_scored.iloc[:top_n].iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            tooltip=f"#{row['rank']} {row['name']} — skor {row['final_score']}",
            icon=folium.Icon(color="green", icon="star"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_path)
