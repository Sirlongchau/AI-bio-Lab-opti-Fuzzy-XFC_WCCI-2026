"""
angular_field.py
================
Transforms asteroid risk points into a continuous angular danger profile
and extracts safe navigation corridors.

This module is purely geometric + aggregation logic:
    - no fuzzy inference
    - no MPC
    - no tactical decisions

It is the bridge between RiskField (F2) and MPC (F5).
"""

from __future__ import annotations

import math
from typing import List, Tuple

from risk_field import AsteroidRisk
from toric_utils import angular_diff

Vec2 = Tuple[float, float]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINS = 360                      # angular resolution
RHO_THRESHOLD = 0.35           # free-space threshold
SIGMA_BASE = 18.0              # angular spread baseline (deg)


# ---------------------------------------------------------------------------
# Angular projection
# ---------------------------------------------------------------------------

def build_angular_profile(
    ship_heading: float,
    asteroid_risks: List[AsteroidRisk],
) -> list:
    """
    Build discrete angular risk profile rho[0..359].
    """
    rho = [0.0 for _ in range(BINS)]

    for ar in asteroid_risks:
        phi = ar.bearing

        # angular spread scales with urgency
        sigma = SIGMA_BASE + (1.0 - ar.risk) * 10.0

        for b in range(BINS):
            angle = float(b)

            diff = angular_diff(angle, phi)
            weight = math.exp(-(diff * diff) / (2 * sigma * sigma))

            rho[b] += ar.risk * weight

    # clamp
    rho = [min(1.0, r) for r in rho]
    return rho


# ---------------------------------------------------------------------------
# Corridor extraction
# ---------------------------------------------------------------------------

def extract_corridors(rho: list) -> List[Tuple[int, int]]:
    """
    Extract contiguous safe angular intervals where rho < threshold.
    Returns list of (start_bin, end_bin).
    """
    corridors = []

    in_free = False
    start = 0

    for i in range(BINS):
        if rho[i] < RHO_THRESHOLD:
            if not in_free:
                in_free = True
                start = i
        else:
            if in_free:
                in_free = False
                corridors.append((start, i - 1))

    # wrap-around case
    if in_free:
        corridors.append((start, BINS - 1))

    return corridors


# ---------------------------------------------------------------------------
# Corridor scoring
# ---------------------------------------------------------------------------

def score_corridor(
    corridor: Tuple[int, int],
    rho: list,
    current_heading: float,
    target_bearing: float,
) -> float:
    """
    Score a corridor for MPC selection.
    """
    start, end = corridor
    center = (start + end) * 0.5

    # alignment with target
    target_align = abs(angular_diff(center, target_bearing))

    # inertia penalty
    inertia_penalty = abs(angular_diff(center, current_heading))

    # safety (mean risk)
    mean_rho = sum(rho[start:end + 1]) / max(1, (end - start + 1))

    return (
        1.5 * (1.0 - mean_rho)
        - 0.8 * (target_align / 180.0)
        - 0.5 * (inertia_penalty / 180.0)
    )


def best_corridor(
    corridors,
    rho,
    current_heading,
    target_bearing,
):
    if not corridors:
        return None

    return max(
        corridors,
        key=lambda c: score_corridor(
            c, rho, current_heading, target_bearing
        )
    )