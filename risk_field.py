"""
risk_field.py
=============
Fuzzy TTC-based per-asteroid risk estimation and global aggregation.

Architecture
------------
Each asteroid is evaluated by a two-input Fuzzy Inference System (FIS)
that maps (tau_i, d_i, size_i) -> R_i in [0, 1].

The global risk is then a weighted combination of max and sum, giving:
    R_global = alpha * max(R_i) + (1 - alpha) * normalised_sum(R_i)

This module intentionally avoids scikit-fuzzy to stay dependency-light.
Membership functions and rule aggregation are implemented from scratch
using simple trapezoidal/triangular shapes, which are:
    - fully explainable (each MF is a named linguistic value)
    - fast (pure Python arithmetic, no overhead)
    - easy to tune by editing the MF breakpoints below

Fragmentation
-------------
Large asteroids (radius > FRAG_RADIUS_THRESHOLD) that are destroyed
produce fragments.  The module exposes a helper to estimate the
post-destruction net risk delta (Section 5, Caveat C2 of the synthesis).

Usage
-----
    from risk_field import RiskField
    rf = RiskField(map_size=(1000, 800))
    risks = rf.compute_all(ship_state, game_state)
    R_global = rf.aggregate(risks)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from toric_utils import toric_distance, time_to_collision

# ---------------------------------------------------------------------------
# Configuration — all tuneable hyperparameters in one place
# ---------------------------------------------------------------------------

# Fragmentation model
FRAG_RADIUS_THRESHOLD = 20.0   # px — asteroids above this split on destruction
FRAG_COUNT            = 2      # number of fragments produced
FRAG_RADIUS_RATIO     = 0.55   # fragment radius = parent * ratio

# Global aggregation
ALPHA_AGGREGATION = 0.6        # weight on max(R_i) vs normalised sum

# FIS membership function breakpoints — (tau in seconds, d in pixels)
# Recalibrated: less conservative so FUZZY mode activates more often.
# Key insight: Kessler asteroids at typical speeds cross ~200px in ~2s.
# "Critical" should mean imminent (< 1s), not anything under 1.5s.

# Tau linguistic values
TAU_CRITICAL = (0.0,  0.0,  0.25,  0.5)   # truly imminent: < 1s
TAU_CLOSE    = (0.25,  0.5,  0.5,  1.5)   # approaching: 1-2.5s
TAU_MEDIUM   = (0.6,  1.5,  1.5,  5.5)   # watch: 2.5-5.5s
TAU_FAR      = (1.5,  5.5, 99.0, 99.0)   # safe: > 5s

# Distance linguistic values (surface distance in pixels)
# Tightened: "near" < 80px, "far" > 200px (not 320px)
D_NEAR   = (0.0,   0.0,  50.0,  90.0)
D_MEDIUM = (50.0,  90.0, 90.0, 220.0)
D_FAR    = (90.0, 220.0, 9999., 9999.)

# Size linguistic values (radius in pixels)
S_SMALL  = (0.0,  0.0,  0.0,  1.0)
S_MEDIUM = (0.50, 1.0,  1.0,  2.0)
S_LARGE  = (1.0, 2.0, 999.0, 999.0)

# Output singletons for Sugeno-style defuzzification
# Recalibrated: spread the outputs further apart so gradations matter
OUT_CRITICAL   = 1.00
OUT_HIGH       = 0.75
OUT_MEDIUM     = 0.20
OUT_LOW        = 0.15
OUT_NEGLIGIBLE = 0.02
FFS = 0.0
FFM = 0.0
FFL = 0.0
FMS = 0.0
FMM = 0.0
FML = 0.0
FNS = 0.0
FNM = 0.0
FNL = 0.0
CMS = 0.0
CMM = 0.0
CML = 0.0
CNS = 0.0
CNM = 0.0
CNL = 0.0
CFS = 0.0
CFM = 0.0
CFL = 0.0
MNS = 0.0
MNM = 0.0
MNL = 0.0
MMS = 0.0
MMM = 0.0
MML = 0.0
MFL = 0.0
MFM = 0.0
MFS = 0.0



# ---------------------------------------------------------------------------
# Membership function primitives
# ---------------------------------------------------------------------------

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.

        1         ___________
                 /           \
        0 ______/             \______
              a  b           c  d

    Returns mu in [0, 1].
    """
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def _singleton_tau(tau: float) -> Dict[str, float]:
    """Fuzzify tau into linguistic grades."""
    return {
        "critical": _trapezoid(tau, *TAU_CRITICAL),
        "close":    _trapezoid(tau, *TAU_CLOSE),
        "medium":   _trapezoid(tau, *TAU_MEDIUM),
        "far":      _trapezoid(tau, *TAU_FAR),
    }


def _singleton_d(d: float) -> Dict[str, float]:
    """Fuzzify surface distance into linguistic grades."""
    return {
        "near":   _trapezoid(d, *D_NEAR),
        "medium": _trapezoid(d, *D_MEDIUM),
        "far":    _trapezoid(d, *D_FAR),
    }


def _singleton_size(r: float) -> Dict[str, float]:
    """Fuzzify asteroid radius into linguistic grades."""
    return {
        "small":  _trapezoid(r, *S_SMALL),
        "medium": _trapezoid(r, *S_MEDIUM),
        "large":  _trapezoid(r, *S_LARGE),
    }


# ---------------------------------------------------------------------------
# Fuzzy rule base
# ---------------------------------------------------------------------------

def _fis_risk(tau: float, d_surface: float, radius: float) -> float:
    """
    Mamdani-like rule base with Sugeno singleton outputs.

    Returns R_i in [0, 1].

    Rules are listed as (antecedent_strength, output_singleton) pairs.
    The antecedent strength is min() of the individual MF grades (AND).
    Defuzzification: weighted average of singletons.
    """
    mu_tau  = _singleton_tau(tau)
    mu_d    = _singleton_d(d_surface)
    mu_size = _singleton_size(radius)

    # Shorthand
    t = mu_tau
    d = mu_d
    s = mu_size

    rules: List[Tuple[float, float]] = [
        # (rule_strength, output_singleton)
        ## any imminent collision is critical risk
        # --- Immediate threat rules ---
        (t["critical"], OUT_CRITICAL),  # truly imminent, any distance, any size

        # --- Close approach ---
        (min(t["close"], d["near"], s["large"]),              CNL),
        (min(t["close"], d["near"], s["medium"]),             CNM),
        (min(t["close"], d["near"], s["small"]),             CNS),
        (min(t["close"], d["medium"], s["large"]),            CML),
        (min(t["close"], d["medium"], s["medium"]),           CMM),
        (min(t["close"], d["medium"], s["small"]),           CMS),
        (min(t["close"], d["far"], s["large"]),             CFL),
        (min(t["close"], d["far"], s["medium"]),            CFM),
        (min(t["close"], d["far"], s["small"]),            CFS),

        # --- Medium term ---
        (min(t["medium"], d["near"], s["large"]),              MNL),
        (min(t["medium"], d["near"], s["medium"]),             MNM),
        (min(t["medium"], d["near"], s["small"]),             MNS),
        (min(t["medium"], d["medium"], s["large"]),            MML),
        (min(t["medium"], d["medium"], s["medium"]),           MMM),
        (min(t["medium"], d["medium"], s["small"]),           MMS),
        (min(t["medium"], d["far"], s["large"]),             MFL),
        (min(t["medium"], d["far"], s["medium"]),            MFM),
        (min(t["medium"], d["far"], s["small"]),            MFS),

        # --- Far / receding ---
        (min(t["far"], d["near"], s["large"]),              FNL),
        (min(t["far"], d["near"], s["medium"]),             FNM),
        (min(t["far"], d["near"], s["small"]),             FNS),
        (min(t["far"], d["medium"], s["large"]),            FML),
        (min(t["far"], d["medium"], s["medium"]),           FMM),
        (min(t["far"], d["medium"], s["small"]),           FMS),
        (min(t["far"], d["far"], s["large"]),             FFL),
        (min(t["far"], d["far"], s["medium"]),            FFM),
        (min(t["far"], d["far"], s["small"]),            FFS),
    ]

    total_weight = sum(w for w, _ in rules)
    if total_weight < 1e-9:
        return 0.0

    weighted_sum = sum(w * o for w, o in rules)
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Per-asteroid result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AsteroidRisk:
    """Risk evaluation for a single asteroid."""
    asteroid_id:   int          # index in game_state.asteroids list
    tau:           float        # TTC in seconds
    d_surface:     float        # surface distance in pixels
    radius:        float        # asteroid radius
    bearing:       float        # bearing from ship (degrees, [0, 360))
    risk:          float        # R_i in [0, 1]
    position:      Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    velocity:      Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    # Post-destruction net risk delta (negative = good to shoot)
    delta_risk_if_destroyed: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RiskField:
    """
    Stateless risk field computer.  Instantiate once per controller init,
    call compute_all() each frame.

    Parameters
    ----------
    map_size  : (width, height) of the toroidal arena
    alpha     : weight on max(R_i) in global aggregation [0, 1]
    """

    def __init__(self, map_size, alpha=None):
        self.map_size = map_size
        self.alpha = ALPHA_AGGREGATION if alpha is None else alpha  # lu à l'appel

    # ------------------------------------------------------------------
    # Frame-level API
    # ------------------------------------------------------------------

    def compute_all(
        self,
        ship_pos: Tuple[float, float],
        ship_vel: Tuple[float, float],
        asteroids: list,          # list of asteroid dicts/objects from game_state
    ) -> List[AsteroidRisk]:
        """
        Evaluate risk for every asteroid in the current frame.

        Parameters
        ----------
        ship_pos    : (x, y) ship position
        ship_vel    : (vx, vy) ship velocity
        asteroids   : list of objects with attributes:
                        .position  -> (x, y)
                        .velocity  -> (vx, vy)
                        .size      -> radius in pixels

        Returns
        -------
        List of AsteroidRisk, sorted by descending risk.
        """
        results: List[AsteroidRisk] = []

        for idx, ast in enumerate(asteroids):
            ast_pos = tuple(ast.position)
            ast_vel = tuple(ast.velocity)

            # In the kessler engine `.size` is the category (1..4) while the true
            # collision radius is `.radius` (= size * 8, i.e. 8..32 px).
            #   - size_cat feeds the FIS, whose membership functions are calibrated
            #     on the category. Kept as-is so risk/tau values are unchanged
            #     (no drift in the Supervisor's R_lo/R_hi thresholds).
            #   - radius_px is stored on AsteroidRisk so downstream geometry
            #     consumers (targeting, MPC collision radius) get real pixels.
            size_cat  = ast.size
            radius_px = ast.radius

            # --- Geometry (FIS inputs frozen on size_cat, see note above) ---
            d_center  = toric_distance(ship_pos, ast_pos, self.map_size)
            d_surface = max(d_center - size_cat, 0.0)

            tau = time_to_collision(
                ship_pos, ship_vel,
                ast_pos, ast_vel,
                size_cat, self.map_size,
            )

            # Bearing from ship to asteroid (degrees)
            from toric_utils import toric_bearing
            bearing = toric_bearing(ship_pos, ast_pos, self.map_size)

            # --- FIS ---
            risk = _fis_risk(tau, d_surface, size_cat)

            ar = AsteroidRisk(
                asteroid_id=idx,
                tau=tau,
                d_surface=d_surface,
                radius=radius_px,        # true collision radius in px
                bearing=bearing,
                risk=risk,
                position=ast_pos,
                velocity=ast_vel,
            )
            results.append(ar)

        # --- Fragmentation delta (requires full results list) ---
        for ar in results:
            ar.delta_risk_if_destroyed = self._frag_delta(ar, results)

        results.sort(key=lambda x: x.risk, reverse=True)
        return results

    def aggregate(self, asteroid_risks: List[AsteroidRisk]) -> float:
        """
        Combine per-asteroid risks into a single global risk scalar.

            R_global = alpha * max(R_i) + (1 - alpha) * normalised_sum

        The normalised_sum uses a soft saturation so that a large number
        of low-risk asteroids does not dominate the max term.

        Returns
        -------
        float in [0, 1]
        """
        if not asteroid_risks:
            return 0.0

        risks = [ar.risk for ar in asteroid_risks]
        r_max = max(risks)

        # Soft normalisation: sum / (1 + sum) keeps it in [0, 1)
        r_sum_raw = sum(risks)
        r_sum_norm = r_sum_raw / (1.0 + r_sum_raw)

        return self.alpha * r_max + (1.0 - self.alpha) * r_sum_norm

    def min_tau(self, asteroid_risks: List[AsteroidRisk]) -> float:
        """Return the minimum TTC across all asteroids (inf if empty)."""
        if not asteroid_risks:
            return math.inf
        return min(ar.tau for ar in asteroid_risks)

    # ------------------------------------------------------------------
    # Fragmentation model
    # ------------------------------------------------------------------

    def _frag_delta(
        self,
        target: AsteroidRisk,
        all_risks: List[AsteroidRisk],
    ) -> float:
        """
        Estimate the change in total risk if `target` is destroyed.

        delta > 0  : destroying it makes things worse (fragmentation hazard)
        delta < 0  : destroying it reduces risk (good shot)
        delta = 0  : small asteroid, no fragments

        Fragment model:
            - Parent radius <= FRAG_RADIUS_THRESHOLD  → no fragments
            - Parent radius >  FRAG_RADIUS_THRESHOLD  → FRAG_COUNT fragments,
              each with radius = parent * FRAG_RADIUS_RATIO

        Fragments are assumed to appear at the same position with similar
        tau (conservative: tau_frag = tau_parent * 1.2).
        """
        if target.radius <= FRAG_RADIUS_THRESHOLD:
            # Small asteroid: simply removes its risk
            return -target.risk

        # Estimate fragment risks
        frag_radius = target.radius * FRAG_RADIUS_RATIO
        frag_tau    = target.tau * 1.2   # fragments slightly less urgent
        frag_d      = target.d_surface   # same position

        frag_risk_each = _fis_risk(frag_tau, frag_d, frag_radius)
        total_frag_risk = FRAG_COUNT * frag_risk_each

        # Net delta = sum_fragments - parent
        return total_frag_risk - target.risk