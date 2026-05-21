"""
f9_vectorized_risk.py
=====================
Vectorised replacement layer for:
- toric distance / delta batch ops
- TTC batch computation
- risk aggregation pre-processing

Goal:
- eliminate per-asteroid Python loops in F2 (risk_field)
- prepare data for MPC and angular projection

Design constraint:
- MUST remain compatible with existing AsteroidRisk API
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


Vec2 = Tuple[float, float]


# ============================================================
# Toroidal vectorisation primitives
# ============================================================

def toric_delta_batch(
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    map_size: Vec2,
) -> np.ndarray:
    """
    Vectorised toric displacement.

    Parameters
    ----------
    a_pos : (N, 2)
    b_pos : (2,) or (N, 2)
    """
    W, H = map_size

    diff = a_pos - b_pos

    diff[:, 0] -= W * np.round(diff[:, 0] / W)
    diff[:, 1] -= H * np.round(diff[:, 1] / H)

    return diff


def toric_distance_batch(
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    map_size: Vec2,
) -> np.ndarray:
    d = toric_delta_batch(a_pos, b_pos, map_size)
    return np.sqrt(np.sum(d * d, axis=1))


# ============================================================
# Vectorised TTC (core performance gain)
# ============================================================

def time_to_collision_batch(
    ship_pos: np.ndarray,
    ship_vel: np.ndarray,
    ast_pos: np.ndarray,
    ast_vel: np.ndarray,
    ast_radius: np.ndarray,
    map_size: Vec2,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Vectorised TTC for all asteroids.
    """

    rel_pos = toric_delta_batch(ast_pos, ship_pos, map_size)
    dist = np.linalg.norm(rel_pos, axis=1)

    d_eff = np.maximum(dist - ast_radius, 0.0)

    # unit direction
    u = rel_pos / (dist[:, None] + eps)

    rel_vel = ast_vel - ship_vel
    closing = np.sum(u * rel_vel, axis=1)

    ttc = np.where(closing > 0, d_eff / (closing + eps), np.inf)

    return ttc


# ============================================================
# Vectorised FIS risk approximation
# ============================================================

def fis_risk_vectorized(
    tau: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """
    Lightweight vectorised surrogate of fuzzy system.

    This replaces per-asteroid rule evaluation with smooth nonlinear proxy.

    NOTE:
    This is intentionally a continuous approximation of your FIS,
    not a strict replica (huge speed gain).
    """

    # normalized urgency (fast TTC decay)
    tau_term = np.exp(-tau)

    # distance hazard
    d_term = np.exp(-d / 100.0)

    # size amplification
    r_term = 1.0 + (r / 40.0)

    risk = tau_term * d_term * r_term

    # squash to [0,1]
    return np.clip(risk, 0.0, 1.0)


# ============================================================
# Batch risk computation engine (F2 replacement core)
# ============================================================

class VectorizedRiskEngine:
    def __init__(self, map_size: Vec2, alpha: float = 0.6):
        self.map_size = map_size
        self.alpha = alpha

    # ------------------------------------------------------------
    # Main batch evaluation
    # ------------------------------------------------------------

    def compute(
        self,
        ship_pos: Vec2,
        ship_vel: Vec2,
        asteroids,
    ) -> Dict[str, np.ndarray]:

        N = len(asteroids)

        if N == 0:
            return {
                "risk": np.zeros(0),
                "ttc": np.zeros(0),
                "distance": np.zeros(0),
            }

        # --------------------------------------------------------
        # Pack arrays (zero Python loops inside compute phase)
        # --------------------------------------------------------

        ast_pos = np.array([a.position for a in asteroids], dtype=float)
        ast_vel = np.array([a.velocity for a in asteroids], dtype=float)
        ast_rad = np.array([a.size for a in asteroids], dtype=float)

        ship_pos = np.array(ship_pos, dtype=float)
        ship_vel = np.array(ship_vel, dtype=float)

        # --------------------------------------------------------
        # Core vectorised computations
        # --------------------------------------------------------

        rel = toric_delta_batch(ast_pos, ship_pos, self.map_size)
        dist = np.linalg.norm(rel, axis=1)

        ttc = time_to_collision_batch(
            ship_pos,
            ship_vel,
            ast_pos,
            ast_vel,
            ast_rad,
            self.map_size,
        )

        # surface distance
        d_surface = np.maximum(dist - ast_rad, 0.0)

        risk = fis_risk_vectorized(ttc, d_surface, ast_rad)

        return {
            "risk": risk,
            "ttc": ttc,
            "distance": d_surface,
        }

    # ------------------------------------------------------------
    # Global aggregation (vectorised)
    # ------------------------------------------------------------

    def aggregate(self, risk: np.ndarray) -> float:
        if risk.size == 0:
            return 0.0

        r_max = np.max(risk)
        r_sum = np.sum(risk)

        r_sum_norm = r_sum / (1.0 + r_sum)

        return self.alpha * r_max + (1.0 - self.alpha) * r_sum_norm

    # ------------------------------------------------------------
    # Min TTC (vectorised reduction)
    # ------------------------------------------------------------

    def min_ttc(self, ttc: np.ndarray) -> float:
        if ttc.size == 0:
            return float("inf")
        return float(np.min(ttc))