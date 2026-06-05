"""
debug_tools.py
==============
Visualization helpers for diagnosing the risk field and angular profile.

Three plots:
  1. arena_risk_plot      — asteroid positions colored by FIS risk score
  2. angular_profile_plot — rho(theta) with corridor and heading overlays
  3. fis_surface_plot     — FIS response surface (tau x distance grid)

All functions return the matplotlib Figure so callers can save or show.
Call debug_snapshot() to render all three for a single game frame.
"""

from __future__ import annotations

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

HEATMAP_DIR = "heatmaps"

from risk_field import RiskField, AsteroidRisk, _fis_risk
from angular_profile import AngularProfile, Corridor

_DARK_BG   = "#0f0f1a"
_PANEL_BG  = "#1a1a2e"
_RISK_CMAP = "RdYlGn_r"   # red = high risk, green = low risk


# ---------------------------------------------------------------------------
# 1. Arena risk plot
# ---------------------------------------------------------------------------

def arena_risk_plot(
    ship_pos: Tuple[float, float],
    asteroid_risks: List[AsteroidRisk],
    map_size: Tuple[float, float],
    resolution: int = 150,
    title: str = "Arena Risk Heatmap",
) -> plt.Figure:
    """
    Top-down spatial heatmap of the arena.

    Each asteroid contributes a Gaussian risk blob centred on its position,
    weighted by its FIS risk score R_i.  Risk peaks at the asteroid and
    dissipates outward, respecting toric (wrap-around) geometry.
    Asteroid outlines and ship position are overlaid on top.
    """
    W, H = map_size

    xs = np.linspace(0, W, resolution)
    ys = np.linspace(0, H, resolution)
    XX, YY = np.meshgrid(xs, ys)

    Z = np.zeros((resolution, resolution))

    for ar in asteroid_risks:
        if ar.risk < 1e-4:
            continue
        ax_x, ax_y = ar.position
        sigma = max(ar.radius * 3.0, 25.0)   # spread a few radii out

        # toric displacement so blobs wrap correctly at map edges
        dx = XX - ax_x
        dy = YY - ax_y
        dx -= W * np.round(dx / W)
        dy -= H * np.round(dy / H)

        Z += ar.risk * np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))

    Z = np.clip(Z, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.set_title(title, color="white")
    ax.set_xlabel("x (px)", color="grey")
    ax.set_ylabel("y (px)", color="grey")
    ax.tick_params(colors="grey")

    img = ax.imshow(
        Z,
        extent=[0, W, H, 0],
        origin="upper",
        cmap=_RISK_CMAP,
        vmin=0, vmax=1,
        aspect="auto",
        interpolation="bilinear",
    )

    # Asteroid outlines
    for ar in asteroid_risks:
        x, y = ar.position
        circle = plt.Circle((x, y), ar.radius, color="white",
                             fill=False, linewidth=1, alpha=0.6)
        ax.add_patch(circle)
        ax.text(x, y, f"{ar.risk:.2f}", color="white", fontsize=6,
                ha="center", va="center")

    # Ship
    sx, sy = ship_pos
    ax.plot(sx, sy, "w+", markersize=14, markeredgewidth=2, label="Ship")

    cbar = fig.colorbar(img, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Risk intensity", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.legend(loc="upper right", facecolor=_PANEL_BG, labelcolor="white")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Angular danger profile plot
# ---------------------------------------------------------------------------

def angular_profile_plot(
    profile: List[float],
    corridors: Optional[List[Corridor]] = None,
    ship_heading: Optional[float] = None,
    rho_threshold: float = 0.38,
    title: str = "Angular Danger Profile ρ(θ)",
) -> plt.Figure:
    """
    Linear plot of the 360-bin angular danger profile.

    - Red fill  = danger (above threshold)
    - Green shading = safe corridors
    - Cyan line = current ship heading
    - Yellow dashed = rho_threshold
    """
    bins = len(profile)
    if bins == 0:
        fig, ax = plt.subplots()
        ax.set_title("No profile data")
        return fig

    angles = list(range(bins))

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.set_title(title, color="white")
    ax.set_xlabel("Bearing (degrees)", color="grey")
    ax.set_ylabel("ρ (danger level)", color="grey")
    ax.tick_params(colors="grey")
    ax.set_xlim(0, bins - 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 361, 45))

    ax.fill_between(angles, profile, alpha=0.7, color="#e74c3c", label="ρ(θ)")
    ax.plot(angles, profile, color="#e74c3c", linewidth=1)

    ax.axhline(rho_threshold, color="yellow", linestyle="--", linewidth=1,
               label=f"threshold = {rho_threshold}")

    if corridors:
        first_corridor_labeled = False
        for c in corridors:
            s = int(c.start_deg) % bins
            e = int(c.end_deg) % bins
            label = "Safe corridor" if not first_corridor_labeled else None
            first_corridor_labeled = True
            if s <= e:
                ax.axvspan(s, e, alpha=0.25, color="#2ecc71", label=label)
            else:
                ax.axvspan(s, bins - 1, alpha=0.25, color="#2ecc71", label=label)
                ax.axvspan(0, e, alpha=0.25, color="#2ecc71")

    if ship_heading is not None:
        h = ship_heading % 360
        ax.axvline(h, color="cyan", linewidth=1.5,
                   label=f"heading = {h:.0f}°")

    ax.legend(facecolor=_PANEL_BG, labelcolor="white")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. FIS response surface
# ---------------------------------------------------------------------------

def fis_surface_plot(
    radius: float = 10.0,
    tau_range: Tuple[float, float] = (0.0, 6.0),
    d_range: Tuple[float, float] = (0.0, 300.0),
    resolution: int = 80,
    title: str = "FIS Response Surface",
) -> plt.Figure:
    """
    2D heatmap of _fis_risk(tau, d, radius) over tau x distance space.

    Use this to validate that the membership function breakpoints produce
    the expected risk gradients.  Run at multiple radius values to check
    the size dimension too.
    """
    taus  = np.linspace(tau_range[0], tau_range[1], resolution)
    dists = np.linspace(d_range[0],   d_range[1],   resolution)

    Z = np.zeros((resolution, resolution))
    for i, d in enumerate(dists):
        for j, tau in enumerate(taus):
            Z[i, j] = _fis_risk(tau, float(d), radius)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.set_title(f"{title}  (radius = {radius:.0f} px)", color="white")
    ax.set_xlabel("Time-to-Collision τ (s)", color="grey")
    ax.set_ylabel("Surface Distance d (px)", color="grey")
    ax.tick_params(colors="grey")

    img = ax.imshow(
        Z,
        extent=[tau_range[0], tau_range[1], d_range[1], d_range[0]],
        aspect="auto",
        cmap=_RISK_CMAP,
        vmin=0,
        vmax=1,
        origin="upper",
    )

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Risk output R_i", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: one-shot debug snapshot for a single game frame
# ---------------------------------------------------------------------------

def debug_snapshot(
    ship_pos: Tuple[float, float],
    ship_heading: float,
    asteroid_risks: List[AsteroidRisk],
    map_size: Tuple[float, float],
    angular_profiler: Optional[AngularProfile] = None,
    save_prefix: Optional[str] = None,
) -> None:
    """
    Render all three debug plots for a single game frame.

    Parameters
    ----------
    save_prefix : if given, saves PNGs as  <save_prefix>_arena.png etc.
                  Otherwise shows interactively via plt.show().
    """
    profile   = None
    corridors = None

    if angular_profiler is not None:
        profile   = angular_profiler.build(asteroid_risks)
        corridors = angular_profiler.extract_corridors(profile)

    fig1 = arena_risk_plot(ship_pos, asteroid_risks, map_size)
    fig2 = angular_profile_plot(
        profile or [],
        corridors=corridors,
        ship_heading=ship_heading,
    )
    fig3 = fis_surface_plot()

    if save_prefix:
        os.makedirs(HEATMAP_DIR, exist_ok=True)
        base = os.path.join(HEATMAP_DIR, save_prefix)
        fig1.savefig(f"{base}_arena.png",   dpi=120, bbox_inches="tight")
        fig2.savefig(f"{base}_angular.png", dpi=120, bbox_inches="tight")
        fig3.savefig(f"{base}_fis.png",     dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"Saved to {HEATMAP_DIR}/: {save_prefix}_arena.png / _angular.png / _fis.png")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Standalone entry point — run with: python debug_tools.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class FakeAsteroid:
        position: tuple
        velocity: tuple
        size: float

    MAP_SIZE   = (1000, 800)
    SHIP_POS   = (400.0, 400.0)
    SHIP_HDG   = 45.0

    asteroids = [
        FakeAsteroid((300, 350), ( 20,  10), 18.0),   # medium, approaching
        FakeAsteroid((500, 200), (-15,  25), 8.0),    # small, crossing
        FakeAsteroid((600, 500), (-30, -10), 25.0),   # large, closing fast
        FakeAsteroid((150, 600), ( 10, -20), 12.0),   # medium, far
        FakeAsteroid((800, 100), (-40,  30), 6.0),    # small, distant
        FakeAsteroid((420, 380), ( 50,  40), 20.0),   # large, very close
    ]

    rf      = RiskField(map_size=MAP_SIZE)
    ap      = AngularProfile()
    risks   = rf.compute_all(SHIP_POS, (0.0, 0.0), asteroids)
    profile = ap.build(risks)
    corridors = ap.extract_corridors(profile)

    print(f"Global risk: {rf.aggregate(risks):.3f}")
    print(f"Corridors found: {len(corridors)}")
    for c in corridors:
        print(f"  [{c.start_deg:.0f}° – {c.end_deg:.0f}°]  width={c.width_deg:.0f}°  freedom={c.freedom:.2f}")

    os.makedirs(HEATMAP_DIR, exist_ok=True)

    figs = [
        ("arena",        arena_risk_plot(SHIP_POS, risks, MAP_SIZE)),
        ("angular",      angular_profile_plot(profile, corridors=corridors, ship_heading=SHIP_HDG)),
        ("fis_small",    fis_surface_plot(radius=10.0)),
        ("fis_large",    fis_surface_plot(radius=25.0, title="FIS Response Surface (large asteroid)")),
    ]

    for name, fig in figs:
        path = os.path.join(HEATMAP_DIR, f"{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
