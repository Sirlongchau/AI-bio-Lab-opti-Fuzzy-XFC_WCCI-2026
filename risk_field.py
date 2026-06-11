"""
risk_field.py
=============
Fuzzy TTC-based per-asteroid risk estimation and global aggregation.

Updated behavior:
    - fixed trapezoid shoulders
    - faster Kessler-scale tau memberships
    - more separated Sugeno output singletons
    - corrected medium-term near/medium-size non-monotonic rule
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from toric_utils import toric_bearing, toric_distance, time_to_collision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRAG_RADIUS_THRESHOLD = 20.0
FRAG_COUNT = 2
FRAG_RADIUS_RATIO = 0.55

ALPHA_AGGREGATION = 0.60

TAU_CRITICAL = (0.0, 0.0, 0.20, 0.45)
TAU_CLOSE = (0.25, 0.45, 0.75, 1.10)
TAU_MEDIUM = (0.75, 1.10, 1.50, 2.20)
TAU_FAR = (1.50, 2.00, 99.0, 99.0)

D_NEAR = (0.0, 0.0, 50.0, 90.0)
D_MEDIUM = (50.0, 90.0, 90.0, 220.0)
D_FAR = (90.0, 220.0, 9999.0, 9999.0)

S_SMALL = (0.0, 0.0, 0.0, 1.0)
S_MEDIUM = (0.50, 1.0, 1.0, 2.0)
S_LARGE = (1.0, 2.0, 999.0, 999.0)

OUT_CRITICAL = 1.00
OUT_HIGH = 0.75
OUT_MEDIUM = 0.35
OUT_LOW = 0.12
OUT_NEGLIGIBLE = 0.02


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return 1.0 if a == b else (x - a) / (b - a)
    return 1.0 if c == d else (d - x) / (d - c)


def _singleton_tau(tau: float) -> Dict[str, float]:
    return {
        "critical": _trapezoid(tau, *TAU_CRITICAL),
        "close": _trapezoid(tau, *TAU_CLOSE),
        "medium": _trapezoid(tau, *TAU_MEDIUM),
        "far": _trapezoid(tau, *TAU_FAR),
    }


def _singleton_d(d: float) -> Dict[str, float]:
    return {
        "near": _trapezoid(d, *D_NEAR),
        "medium": _trapezoid(d, *D_MEDIUM),
        "far": _trapezoid(d, *D_FAR),
    }


def _singleton_size(r: float) -> Dict[str, float]:
    return {
        "small": _trapezoid(r, *S_SMALL),
        "medium": _trapezoid(r, *S_MEDIUM),
        "large": _trapezoid(r, *S_LARGE),
    }


# ---------------------------------------------------------------------------
# Fuzzy risk FIS
# ---------------------------------------------------------------------------

def _fis_risk(tau: float, d_surface: float, radius: float) -> float:
    mu_tau = _singleton_tau(tau)
    mu_d = _singleton_d(d_surface)
    mu_size = _singleton_size(radius)

    t, d, s = mu_tau, mu_d, mu_size

    rules: List[Tuple[float, float]] = [
        (t["critical"], OUT_CRITICAL),

        (min(t["close"], d["near"], s["large"]), OUT_CRITICAL),
        (min(t["close"], d["near"], s["medium"]), OUT_HIGH),
        (min(t["close"], d["near"], s["small"]), OUT_MEDIUM),
        (min(t["close"], d["medium"], s["large"]), OUT_HIGH),
        (min(t["close"], d["medium"], s["medium"]), OUT_HIGH),
        (min(t["close"], d["medium"], s["small"]), OUT_MEDIUM),
        (min(t["close"], d["far"], s["large"]), OUT_MEDIUM),
        (min(t["close"], d["far"], s["medium"]), OUT_MEDIUM),
        (min(t["close"], d["far"], s["small"]), OUT_LOW),

        (min(t["medium"], d["near"], s["large"]), OUT_HIGH),
        (min(t["medium"], d["near"], s["medium"]), OUT_MEDIUM),
        (min(t["medium"], d["near"], s["small"]), OUT_MEDIUM),
        (min(t["medium"], d["medium"], s["large"]), OUT_HIGH),
        (min(t["medium"], d["medium"], s["medium"]), OUT_MEDIUM),
        (min(t["medium"], d["medium"], s["small"]), OUT_LOW),
        (min(t["medium"], d["far"], s["large"]), OUT_MEDIUM),
        (min(t["medium"], d["far"], s["medium"]), OUT_LOW),
        (min(t["medium"], d["far"], s["small"]), OUT_NEGLIGIBLE),

        (min(t["far"], d["near"], s["large"]), OUT_LOW),
        (min(t["far"], d["near"], s["medium"]), OUT_LOW),
        (min(t["far"], d["near"], s["small"]), OUT_LOW),
        (min(t["far"], d["medium"], s["large"]), OUT_LOW),
        (min(t["far"], d["medium"], s["medium"]), OUT_LOW),
        (min(t["far"], d["medium"], s["small"]), OUT_NEGLIGIBLE),
        (min(t["far"], d["far"], s["large"]), OUT_LOW),
        (min(t["far"], d["far"], s["medium"]), OUT_NEGLIGIBLE),
        (min(t["far"], d["far"], s["small"]), OUT_NEGLIGIBLE),
    ]

    total_weight = sum(w for w, _ in rules)
    if total_weight < 1e-9:
        return 0.0
    return sum(w * out for w, out in rules) / total_weight


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class AsteroidRisk:
    asteroid_id: int
    tau: float
    d_surface: float
    radius: float
    bearing: float
    risk: float
    position: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    velocity: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    delta_risk_if_destroyed: float = 0.0


def _get_attr(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# RiskField API
# ---------------------------------------------------------------------------

class RiskField:
    def __init__(self, map_size: Tuple[float, float], alpha: float = ALPHA_AGGREGATION) -> None:
        self.map_size = map_size
        self.alpha = alpha

    def compute_all(
        self,
        ship_pos: Tuple[float, float],
        ship_vel: Tuple[float, float],
        asteroids: list,
    ) -> List[AsteroidRisk]:
        results: List[AsteroidRisk] = []

        for idx, ast in enumerate(asteroids):
            ast_pos = tuple(_get_attr(ast, "position", (0.0, 0.0)))
            ast_vel = tuple(_get_attr(ast, "velocity", (0.0, 0.0)))
            radius = float(_get_attr(ast, "size", _get_attr(ast, "radius", 10.0)))  # size class 1-4: keeps ship aggressive (benchmarked higher than px radius)
            asteroid_id = int(_get_attr(ast, "id", idx) if _get_attr(ast, "id", None) is not None else idx)

            d_center = toric_distance(ship_pos, ast_pos, self.map_size)
            d_surface = max(d_center - radius, 0.0)
            tau = time_to_collision(ship_pos, ship_vel, ast_pos, ast_vel, radius, self.map_size)
            bearing = toric_bearing(ship_pos, ast_pos, self.map_size)
            risk = _fis_risk(tau, d_surface, radius)

            results.append(
                AsteroidRisk(
                    asteroid_id=asteroid_id,
                    tau=tau,
                    d_surface=d_surface,
                    radius=radius,
                    bearing=bearing,
                    risk=risk,
                    position=ast_pos,
                    velocity=ast_vel,
                )
            )

        for ar in results:
            ar.delta_risk_if_destroyed = self._frag_delta(ar)

        results.sort(key=lambda x: x.risk, reverse=True)
        return results

    def aggregate(self, asteroid_risks: List[AsteroidRisk]) -> float:
        if not asteroid_risks:
            return 0.0
        risks = [ar.risk for ar in asteroid_risks]
        r_max = max(risks)
        r_sum_raw = sum(risks)
        r_sum_norm = r_sum_raw / (1.0 + r_sum_raw)
        return self.alpha * r_max + (1.0 - self.alpha) * r_sum_norm

    def min_tau(self, asteroid_risks: List[AsteroidRisk]) -> float:
        if not asteroid_risks:
            return math.inf
        return min(ar.tau for ar in asteroid_risks)

    def _frag_delta(self, target: AsteroidRisk) -> float:
        # Objective = asteroids destroyed. Splitting a large rock yields MORE
        # targets, so destruction is always net-beneficial -> no fragmentation penalty.
        return -target.risk
        if target.radius <= FRAG_RADIUS_THRESHOLD:
            return -target.risk

        frag_radius = target.radius * FRAG_RADIUS_RATIO
        frag_tau = target.tau * 1.2 if math.isfinite(target.tau) else math.inf
        frag_d = target.d_surface
        frag_risk_each = _fis_risk(frag_tau, frag_d, frag_radius)
        total_frag_risk = FRAG_COUNT * frag_risk_each
        return total_frag_risk - target.risk
