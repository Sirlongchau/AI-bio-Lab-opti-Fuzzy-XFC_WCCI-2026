"""
target_selector.py
==================
Multi-criteria target selection with commitment mechanism.

Problem statement
-----------------
Greedy selection (always attack the highest-risk asteroid) is suboptimal
because it ignores:
  - Acquisition cost: rotating 150° to face the biggest asteroid while
    two others converge is worse than taking the easy shot.
  - Fragmentation: destroying a large asteroid may worsen net risk.
  - Oscillation: if two targets have close scores, the selector would
    switch every frame, causing the ship to spin without firing.

Solution
--------
Each candidate asteroid is scored:

    score(k) = ΔR(k) · (1 − D_acq(k)) · (1 − P_frag(k))

    ΔR(k)    : estimated global risk reduction if k is destroyed
    D_acq(k) : acquisition difficulty [0,1] via FIS(heading_err, distance)
    P_frag(k): fragmentation penalty [0,1] — non-zero only for large asteroids

Commitment
----------
Once a target is selected it is held for at least COMMIT_FRAMES frames.
Early override is allowed only if:
  - The committed target is destroyed (no longer in the asteroid list).
  - A new asteroid has tau < TAU_URGENT_OVERRIDE (imminent collision).

Weapon alignment
----------------
The module also exposes `fire_decision()`: given the current heading and
the selected target's bearing, decide whether to fire this frame.

Usage
-----
    from target_selector import TargetSelector
    ts = TargetSelector()
    target = ts.select(asteroid_risks, current_heading, r_global)
    fire = ts.fire_decision(current_heading, target)
    ts.tick()   # call once per frame to advance commitment counter
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from toric_utils import angular_diff

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMMIT_FRAMES        = 45      # ~0.75s at 60 fps — minimum frames to hold target
TAU_URGENT_OVERRIDE  = 0.8     # seconds — override commitment if a new asteroid
                                #           is this close
FIRE_ANGLE_THRESHOLD = 8.0     # degrees — max heading error to fire
MIN_SCORE_TO_TARGET  = 0.001   # very low — let the fallback raw-risk path always produce a target

# Acquisition FIS breakpoints
# Input 1: |heading_error| in degrees
HE_SMALL  = (0.0,  0.0, 15.0, 30.0)    # easy shot already lined up
HE_MEDIUM = (15.0, 30.0, 60.0, 90.0)
HE_LARGE  = (60.0, 90.0, 180.0, 180.0) # expensive rotation needed

# Input 2: distance in pixels (surface distance)
D_CLOSE  = (0.0,    0.0,  100.0, 200.0)
D_MEDIUM = (100.0, 200.0, 400.0, 600.0)
D_FAR    = (400.0, 600.0, 9999., 9999.)

# Output singletons for D_acq (difficulty)
DIFF_EASY   = 0.10
DIFF_MEDIUM = 0.45
DIFF_HARD   = 0.85

# Fragmentation model (must match risk_field.py constants)
FRAG_RADIUS_THRESHOLD = 20.0
FRAG_PENALTY_LARGE    = 0.40   # penalty applied when asteroid is large


# ---------------------------------------------------------------------------
# Membership function helper (shared shape with risk_field.py)
# ---------------------------------------------------------------------------

def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


# ---------------------------------------------------------------------------
# Acquisition difficulty FIS
# ---------------------------------------------------------------------------

def _acquisition_difficulty(heading_error_abs: float, d_surface: float) -> float:
    """
    FIS estimating how hard it is to acquire a target from the current
    heading and distance.

    Returns D_acq ∈ [0, 1] — 0 = trivial, 1 = nearly impossible.
    """
    mu_he = {
        "small":  _trap(heading_error_abs, *HE_SMALL),
        "medium": _trap(heading_error_abs, *HE_MEDIUM),
        "large":  _trap(heading_error_abs, *HE_LARGE),
    }
    mu_d = {
        "close":  _trap(d_surface, *D_CLOSE),
        "medium": _trap(d_surface, *D_MEDIUM),
        "far":    _trap(d_surface, *D_FAR),
    }

    rules = [
        # (strength, difficulty_singleton)
        (min(mu_he["small"],  mu_d["close"]),  DIFF_EASY),
        (min(mu_he["small"],  mu_d["medium"]), DIFF_EASY),
        (min(mu_he["small"],  mu_d["far"]),    DIFF_MEDIUM),
        (min(mu_he["medium"], mu_d["close"]),  DIFF_MEDIUM),
        (min(mu_he["medium"], mu_d["medium"]), DIFF_MEDIUM),
        (min(mu_he["medium"], mu_d["far"]),    DIFF_HARD),
        (min(mu_he["large"],  mu_d["close"]),  DIFF_HARD),
        (min(mu_he["large"],  mu_d["medium"]), DIFF_HARD),
        (min(mu_he["large"],  mu_d["far"]),    DIFF_HARD),
    ]

    total_w = sum(w for w, _ in rules)
    if total_w < 1e-9:
        return 0.5
    return sum(w * o for w, o in rules) / total_w


# ---------------------------------------------------------------------------
# Fragmentation penalty
# ---------------------------------------------------------------------------

def _fragmentation_penalty(radius: float, delta_risk: float) -> float:
    """
    Penalty ∈ [0, 1] that discourages shooting large asteroids when doing
    so worsens the net risk field (delta_risk_if_destroyed > 0).
    """
    if radius <= FRAG_RADIUS_THRESHOLD:
        return 0.0   # small asteroid — no fragments, no penalty

    if delta_risk <= 0.0:
        return 0.0   # destroying it still reduces risk — no penalty

    # Linear penalty: worse fragmentation delta → higher penalty
    # Capped at FRAG_PENALTY_LARGE
    return min(FRAG_PENALTY_LARGE, delta_risk * 0.5)


# ---------------------------------------------------------------------------
# Target score
# ---------------------------------------------------------------------------

@dataclass
class TargetScore:
    """Scored candidate target for one frame."""
    asteroid_id:   int
    bearing:       float    # degrees
    tau:           float    # TTC seconds
    risk:          float    # R_i
    delta_risk:    float    # estimated global risk reduction
    d_acq:         float    # acquisition difficulty
    p_frag:        float    # fragmentation penalty
    score:         float    # composite score


def _compute_score(ar, heading_error_abs: float) -> TargetScore:
    """Compute the composite target score for one AsteroidRisk."""
    # delta_risk_if_destroyed convention (from risk_field.py):
    #   negative value = destroying reduces net risk  (GOOD)
    #   positive value = destroying increases net risk via fragments (BAD)
    #
    # We want delta_risk as "benefit of shooting" — positive = good.
    # For small asteroids: delta_risk_if_destroyed = -R_i  → benefit = R_i
    # For large asteroids: may be positive (fragments worse) → benefit < R_i
    #
    # Benefit = how much risk we remove. Clamp so we never get negative benefit
    # from a fragmentation-positive asteroid — we simply get 0 benefit there.
    benefit = max(0.0, -ar.delta_risk_if_destroyed)

    # Fallback: if benefit is 0 (e.g. all large asteroids with bad fragments),
    # use raw risk so the ship still has a target to face.
    # This prevents the selector from returning None in dense fields.
    if benefit < 1e-3:
        benefit = ar.risk * 0.3   # reduced weight — not a great shot, but valid

    d_acq  = _acquisition_difficulty(heading_error_abs, ar.d_surface)
    p_frag = _fragmentation_penalty(ar.radius, ar.delta_risk_if_destroyed)

    # Composite score — all factors in [0, 1]
    score = benefit * (1.0 - d_acq) * (1.0 - p_frag)

    return TargetScore(
        asteroid_id=ar.asteroid_id,
        bearing=ar.bearing,
        tau=ar.tau,
        risk=ar.risk,
        delta_risk=benefit,
        d_acq=d_acq,
        p_frag=p_frag,
        score=score,
    )


# ---------------------------------------------------------------------------
# Selector with commitment
# ---------------------------------------------------------------------------

class TargetSelector:
    """
    Stateful target selector.

    State is kept between frames to implement the commitment mechanism.

    Parameters
    ----------
    commit_frames       : minimum frames to hold a target
    tau_urgent_override : TTC threshold for overriding commitment
    fire_threshold_deg  : heading error threshold to fire (degrees)
    """

    def __init__(
        self,
        commit_frames: int         = COMMIT_FRAMES,
        tau_urgent_override: float = TAU_URGENT_OVERRIDE,
        fire_threshold_deg: float  = FIRE_ANGLE_THRESHOLD,
    ) -> None:
        self.commit_frames       = commit_frames
        self.tau_urgent_override = tau_urgent_override
        self.fire_threshold_deg  = fire_threshold_deg

        # Commitment state
        self._committed_id:     Optional[int]   = None
        self._committed_target: Optional[TargetScore] = None
        self._frames_held:      int             = 0

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def select(
        self,
        asteroid_risks: list,       # List[AsteroidRisk] from risk_field.py
        current_heading: float,     # ship heading in degrees
        r_global: float,            # global risk scalar (unused in scoring,
                                    # reserved for future threshold logic)
    ) -> Optional[TargetScore]:
        """
        Select the best target for this frame.

        Returns None if there are no actionable targets (all asteroids have
        negligible risk or no shot is beneficial).
        """
        if not asteroid_risks:
            self._reset_commitment()
            return None

        # --- Score all candidates ---
        candidates: List[TargetScore] = []
        for ar in asteroid_risks:
            he_abs = abs(angular_diff(current_heading, ar.bearing))
            ts = _compute_score(ar, he_abs)
            if ts.score >= MIN_SCORE_TO_TARGET:
                candidates.append(ts)

        if not candidates:
            self._reset_commitment()
            return None

        candidates.sort(key=lambda t: t.score, reverse=True)
        best = candidates[0]

        # --- Commitment logic ---
        # Check if committed target still exists
        active_ids = {ar.asteroid_id for ar in asteroid_risks}
        committed_still_alive = (
            self._committed_id is not None
            and self._committed_id in active_ids
        )

        # Urgent override: any asteroid with very small tau
        urgent_asteroid = next(
            (ar for ar in asteroid_risks if ar.tau < self.tau_urgent_override),
            None,
        )
        if urgent_asteroid:
            # Redirect to the most urgent threat immediately
            he_abs = abs(angular_diff(current_heading, urgent_asteroid.bearing))
            urgent_target = _compute_score(urgent_asteroid, he_abs)
            self._set_commitment(urgent_target)
            return urgent_target

        # If we're within the commitment window and target is alive, hold it
        if (
            committed_still_alive
            and self._frames_held < self.commit_frames
        ):
            # Return the current frame's view of the committed target
            committed_ar = next(
                (ar for ar in asteroid_risks
                 if ar.asteroid_id == self._committed_id),
                None,
            )
            if committed_ar is not None:
                he_abs = abs(angular_diff(current_heading, committed_ar.bearing))
                held = _compute_score(committed_ar, he_abs)
                self._frames_held += 1
                return held

        # Commitment expired or target gone — select fresh
        self._set_commitment(best)
        return best

    def fire_decision(
        self,
        current_heading: float,
        target: Optional[TargetScore],
        can_fire: bool,
    ) -> bool:
        """
        Return True if the ship should fire this frame.

        Conditions (all must hold):
          - A target is selected
          - The ship's cooldown allows firing (can_fire)
          - Heading error to target is within fire_threshold_deg
          - The shot is expected to be beneficial (delta_risk > 0)
        """
        if target is None or not can_fire:
            return False
        heading_error = abs(angular_diff(current_heading, target.bearing))
        return (
            heading_error <= self.fire_threshold_deg
            and target.delta_risk > 0.0
        )

    def tick(self) -> None:
        """
        Advance the frame counter.  Must be called exactly once per frame
        at the end of actions(), after select() and fire_decision().
        """
        # Intentionally no-op here; frame counting is handled inside select()
        # This method exists as a hook for future per-frame bookkeeping.
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_commitment(self, target: TargetScore) -> None:
        self._committed_id     = target.asteroid_id
        self._committed_target = target
        self._frames_held      = 1

    def _reset_commitment(self) -> None:
        self._committed_id     = None
        self._committed_target = None
        self._frames_held      = 0