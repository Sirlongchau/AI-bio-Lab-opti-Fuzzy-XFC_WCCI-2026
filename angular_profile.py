"""
angular_profile.py
==================
Angular danger profile (ρ) and free-corridor extraction.

Conceptual foundation
---------------------
Instead of summing per-asteroid turn-rate commands — which causes
destructive cancellation when two asteroids are in opposite directions —
we project each asteroid's risk onto the directional circle S¹ as a
Gaussian lobe.  The result is a scalar danger field ρ(θ) ∈ [0, 1]
defined over all bearings θ ∈ [0°, 360°).

Free corridors are contiguous arcs where ρ < ρ_threshold.  The turn-rate
command becomes "steer toward the centre of the best-scored corridor",
which intrinsically resolves multi-asteroid conflicts including:
  - Face-to-face asteroids   → two red zones, two green corridors between them
  - Pincer (3×120°)          → three lobes, corridors in the gaps
  - Full saturation          → no corridor → escalation trigger for MPC

Corridor scoring
----------------
Each corridor is scored on three criteria:
  1. Internal freedom   : mean(1 - ρ(b)) over the corridor arc
  2. Alignment bonus    : proximity of corridor centre to the attract target
  3. Inertia bonus      : proximity of corridor centre to current heading

This naturally prefers corridors that let the ship keep momentum and
remain offensive when safe.

Usage
-----
    from angular_profile import AngularProfile
    ap = AngularProfile(bins=360)
    profile = ap.build(ship_state, asteroid_risks)
    corridors = ap.extract_corridors(profile)
    best = ap.best_corridor(corridors, target_bearing, current_heading)
    # best is None → full saturation → escalate to MPC
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from toric_utils import angular_diff

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINS            = 360        # angular resolution (1 bin = 1 degree)
RHO_THRESHOLD   = 0.38       # below this → corridor (free)
MIN_CORRIDOR_WIDTH = 10      # degrees — narrower corridors are ignored

# Corridor scoring weights (must sum to 1 for interpretability)
W_FREEDOM   = 0.40
W_ALIGNMENT = 0.35
W_INERTIA   = 0.25

# Gaussian spread parameters
SIGMA_BASE_DEG   = 15.0   # minimum angular spread regardless of asteroid
SIGMA_SIZE_SCALE  = 1.4   # extra degrees per unit of asteroid radius
SIGMA_TAU_SCALE   = 20.0  # extra degrees for very close asteroids (1/tau)

# Alignment and inertia falloff (lower = sharper preference)
KAPPA_ALIGNMENT = 55.0    # degrees
MU_INERTIA      = 80.0    # degrees


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Corridor:
    """A contiguous free arc in the angular danger profile."""
    start_deg:   float          # first free bin (degrees)
    end_deg:     float          # last free bin (degrees)
    centre_deg:  float          # geometric centre of the arc
    width_deg:   float          # angular width
    freedom:     float          # mean internal freedom [0, 1]
    score:       float = 0.0    # composite score (set by best_corridor)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AngularProfile:
    """
    Stateless angular danger profile computer.

    Parameters
    ----------
    bins        : angular resolution (default 360 → 1 bin per degree)
    rho_th      : free/danger threshold on ρ
    min_width   : minimum corridor width in degrees
    """

    def __init__(
        self,
        bins: int = None,
        rho_th: float = None,
        min_width: float = None,
    ) -> None:
        self.bins      = BINS if bins is None else bins
        self.rho_th    = RHO_THRESHOLD if rho_th is None else rho_th
        self.min_width = MIN_CORRIDOR_WIDTH if min_width is None else min_width
        self._bin_deg  = 360.0 / self.bins   # degrees per bin
        self._bins_deg = np.arange(self.bins) * self._bin_deg
        self._cos_bins = np.cos(np.radians(self._bins_deg))
        self._sin_bins = np.sin(np.radians(self._bins_deg))

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------

    def build(self, asteroid_risks: list) -> List[float]:
        """
        Build the angular danger profile from a list of AsteroidRisk objects.

        Each asteroid projects a Gaussian lobe centred on its bearing,
        scaled by its risk intensity R_i and angular spread σ_i.

        Parameters
        ----------
        asteroid_risks : list of AsteroidRisk (from risk_field.py)

        Returns
        -------
        profile : list of BINS floats in [0, 1]
        """
        prof = np.zeros(self.bins)

        for ar in asteroid_risks:
            if ar.risk < 1e-4:
                continue   # negligible contribution — skip for speed

            # Angular spread: larger for big/close asteroids
            tau_clamped = max(ar.tau, 0.1)
            sigma = (
                SIGMA_BASE_DEG
                + SIGMA_SIZE_SCALE * ar.radius
                + SIGMA_TAU_SCALE / tau_clamped
            )
            sigma = min(sigma, 60.0)   # cap: don't let one asteroid block everything

            # Vectorised Gaussian lobe over all bins (wrapped angular diff)
            diff = (self._bins_deg - ar.bearing + 180.0) % 360.0 - 180.0
            prof += ar.risk * np.exp(-(diff * diff) / (2.0 * sigma * sigma))

        np.minimum(prof, 1.0, out=prof)
        return prof.tolist()   # list: extract_corridors relies on `profile + profile`

    # ------------------------------------------------------------------
    # Corridor extraction
    # ------------------------------------------------------------------

    def extract_corridors(self, profile: List[float]) -> List[Corridor]:
        """
        Find all contiguous arcs where ρ(b) < rho_th.

        Handles the wrap-around (0°/360° boundary) by doubling the
        profile array and de-duplicating.

        Returns
        -------
        List of Corridor objects, sorted by width descending.
        An empty list means full saturation → escalate to MPC.
        """
        # Double the array to handle wrap-around
        doubled = profile + profile
        n       = len(doubled)

        corridors: List[Corridor] = []
        in_free  = False
        start_b  = 0

        for b in range(n):
            free = doubled[b] < self.rho_th

            if free and not in_free:
                in_free = True
                start_b = b

            elif not free and in_free:
                in_free = False
                end_b   = b - 1

                # Only consider arcs that start in the first copy
                if start_b >= self.bins:
                    break

                width_bins = end_b - start_b + 1
                width_deg  = width_bins * self._bin_deg

                if width_deg < self.min_width:
                    continue

                # Clip end to first copy for centre calculation
                end_clipped  = min(end_b, self.bins - 1)
                centre_deg   = ((start_b + end_b) / 2.0 * self._bin_deg) % 360.0
                freedom_mean = 1.0 - (
                    sum(doubled[start_b:end_b + 1]) / width_bins
                )

                corridors.append(Corridor(
                    start_deg  = (start_b * self._bin_deg) % 360.0,
                    end_deg    = (end_b   * self._bin_deg) % 360.0,
                    centre_deg = centre_deg,
                    width_deg  = width_deg,
                    freedom    = freedom_mean,
                ))

        # Handle case where free arc reaches the end of doubled array
        if in_free and start_b < self.bins:
            end_b      = n - 1
            width_bins = end_b - start_b + 1
            width_deg  = width_bins * self._bin_deg
            if width_deg >= self.min_width:
                centre_deg  = ((start_b + end_b) / 2.0 * self._bin_deg) % 360.0
                freedom_mean = 1.0 - (
                    sum(doubled[start_b:]) / width_bins
                )
                corridors.append(Corridor(
                    start_deg  = (start_b * self._bin_deg) % 360.0,
                    end_deg    = 359.0,
                    centre_deg = centre_deg,
                    width_deg  = width_deg,
                    freedom    = freedom_mean,
                ))

        # Remove duplicates (wrap-around can create identical corridors)
        seen: set = set()
        unique: List[Corridor] = []
        for c in corridors:
            key = round(c.centre_deg, 1)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda c: c.width_deg, reverse=True)
        return unique

    # ------------------------------------------------------------------
    # Corridor scoring and selection
    # ------------------------------------------------------------------

    def score_corridors(
        self,
        corridors: List[Corridor],
        target_bearing: float,    # bearing of the selected attack target (degrees)
        current_heading: float,   # ship's current heading (degrees)
    ) -> List[Corridor]:
        """
        Attach a composite score to each corridor in-place and return
        the list sorted by descending score.

        score = W_FREEDOM   * freedom
              + W_ALIGNMENT * exp(-|Δ_target|  / KAPPA)
              + W_INERTIA   * exp(-|Δ_heading| / MU)

        The alignment bonus pulls the ship toward a corridor that lines it
        up for a shot.  The inertia bonus avoids unnecessary course changes.
        """
        for c in corridors:
            delta_target  = abs(angular_diff(c.centre_deg, target_bearing))
            delta_heading = abs(angular_diff(c.centre_deg, current_heading))

            align_bonus   = math.exp(-delta_target  / KAPPA_ALIGNMENT)
            inertia_bonus = math.exp(-delta_heading / MU_INERTIA)

            c.score = (
                W_FREEDOM   * c.freedom
                + W_ALIGNMENT * align_bonus
                + W_INERTIA   * inertia_bonus
            )

        corridors.sort(key=lambda c: c.score, reverse=True)
        return corridors

    def best_corridor(
        self,
        corridors: List[Corridor],
        target_bearing: float,
        current_heading: float,
    ) -> Optional[Corridor]:
        """
        Score all corridors and return the highest-scoring one, or None
        if the list is empty (full saturation → MPC escalation).
        """
        if not corridors:
            return None
        scored = self.score_corridors(corridors, target_bearing, current_heading)
        return scored[0]

    # ------------------------------------------------------------------
    # Saturation diagnostics
    # ------------------------------------------------------------------

    def saturation_fraction(self, profile: List[float]) -> float:
        """
        Fraction of bins above rho_th in [0, 1].
        1.0 = fully saturated (no corridor anywhere).
        Used as an auxiliary input to the modal supervisor.
        """
        danger_bins = sum(1 for rho in profile if rho >= self.rho_th)
        return danger_bins / self.bins

    def repulse_direction(self, profile: List[float]) -> float:
        """
        Compute the gradient-descent escape direction from the danger profile.

        Returns the bearing (degrees) that minimises ρ(θ), weighted by
        angular proximity.  Used as fallback when no corridor exists.

        This is the direction the ship should face if forced into MPC evasion
        and needing a quick best-guess heading before the optimiser runs.
        """
        # Vector field: each bin repels with strength ρ(b)
        rho = np.asarray(profile, dtype=float)
        fx = float(-np.dot(rho, self._cos_bins))
        fy = float(-np.dot(rho, self._sin_bins))

        if math.hypot(fx, fy) < 1e-6:
            return 0.0   # uniform field, no preferred direction

        return math.degrees(math.atan2(fy, fx)) % 360.0
