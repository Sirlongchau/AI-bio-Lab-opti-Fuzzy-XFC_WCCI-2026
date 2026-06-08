"""
target_selector.py
==================
Crisp multi-criteria target selection for XFC Kessler.

Updated behavior:
    - short-horizon lead / predicted aim bearing
    - closing-speed urgency term
    - shorter target commitment
    - urgent override chooses minimum tau, not first risk-sorted item
    - fixed trapezoid shoulders
    - fire decision uses aim_bearing, not current asteroid bearing

This module is intentionally crisp.  The fuzzy controller consumes the
selected target and decides how to move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from toric_utils import angular_diff, toric_bearing

Vec2 = Tuple[float, float]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREDICT_DT = 1.0 / 3.0
BULLET_SPEED = 800.0   # px/s (kesslergame bullet speed)
# Fixed 1/3s lead benchmarked higher than full intercept (51.6 vs 48.6 hits,
# fewer deaths) because the 180 deg/s turn cap can't track a perfect intercept.
# Flip True for very fast / long-range fields where lead error dominates.
USE_INTERCEPT_LEAD = False

COMMIT_FRAMES = 12
COMMIT_SWITCH_RATIO = 1.35
TAU_URGENT_OVERRIDE = 0.50

FIRE_ANGLE_THRESHOLD = 12.0
FIRE_ANGLE_PERFECT = 0.5
MIN_SCORE_TO_TARGET = 0.001

AIM_FEASIBILITY_KAPPA = 35.0
URGENCY_TAU_CAP = 2.0
URGENCY_SPEED_SCALE = 180.0
URGENCY_TAU_WEIGHT = 0.65
URGENCY_SPEED_WEIGHT = 0.35

# Acquisition FIS breakpoints, input 1: |heading error to aim_bearing| in deg.
HE_TINY = (0.0, 0.0, 2.0, 6.0)
HE_SMALL = (2.0, 6.0, 12.0, 25.0)
HE_MEDIUM = (15.0, 30.0, 60.0, 90.0)
HE_LARGE = (60.0, 90.0, 180.0, 180.0)

# Acquisition FIS breakpoints, input 2: surface distance in px.
D_CLOSE = (0.0, 0.0, 90.0, 180.0)
D_MEDIUM = (120.0, 220.0, 400.0, 600.0)
D_FAR = (450.0, 650.0, 9999.0, 9999.0)

DIFF_EASY = 0.05
DIFF_MEDIUM = 0.40
DIFF_HARD = 0.85

FRAG_RADIUS_THRESHOLD = 20.0
FRAG_PENALTY_LARGE = 0.0    # kill-max: do not avoid large (splitting) asteroids


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid with correct shoulder behavior for a==b and c==d."""
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return 1.0 if a == b else (x - a) / (b - a)
    return 1.0 if c == d else (d - x) / (d - c)


def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def asteroid_signature(ar: Any) -> Tuple[float, float, float, float, float]:
    """Stable-ish fallback identity when the engine has no asteroid ID."""
    px, py = ar.position
    vx, vy = ar.velocity
    return (
        round(px / 10.0) * 10.0,
        round(py / 10.0) * 10.0,
        round(vx / 10.0) * 10.0,
        round(vy / 10.0) * 10.0,
        round(float(ar.radius), 1),
    )


def predicted_aim_bearing(
    ship_pos: Vec2,
    asteroid_pos: Vec2,
    asteroid_vel: Vec2,
    map_size: Vec2,
    predict_dt: float = PREDICT_DT,
) -> float:
    """Bearing to a short-horizon predicted asteroid position."""
    ax, ay = asteroid_pos
    vx, vy = asteroid_vel
    width, height = map_size

    pred_pos = (
        (ax + vx * predict_dt) % width,
        (ay + vy * predict_dt) % height,
    )
    return toric_bearing(ship_pos, pred_pos, map_size)


def closing_speed(
    ship_pos: Vec2,
    ship_vel: Vec2,
    asteroid_pos: Vec2,
    asteroid_vel: Vec2,
    map_size: Vec2,
) -> float:
    """Return positive px/s when asteroid is closing on the ship."""
    sx, sy = ship_pos
    ax, ay = asteroid_pos
    svx, svy = ship_vel
    avx, avy = asteroid_vel
    width, height = map_size

    dx = ax - sx
    dy = ay - sy
    dx -= width * round(dx / width)
    dy -= height * round(dy / height)

    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0

    ux, uy = dx / dist, dy / dist
    rvx, rvy = avx - svx, avy - svy

    return max(0.0, -(rvx * ux + rvy * uy))

def intercept_aim_bearing(
    ship_pos: Vec2,
    ship_vel: Vec2,
    asteroid_pos: Vec2,
    asteroid_vel: Vec2,
    map_size: Vec2,
    bullet_speed: float = BULLET_SPEED,
) -> float:
    """Bearing at which a bullet (speed `bullet_speed`, fired along heading) and
    the asteroid arrive at the same point. Solves |r + v*t| = bullet_speed*t for
    the soonest t>0; falls back to the direct bearing if no intercept exists."""
    sx, sy = ship_pos
    ax, ay = asteroid_pos
    vx, vy = asteroid_vel
    width, height = map_size

    rx = ax - sx
    ry = ay - sy
    rx -= width * round(rx / width)
    ry -= height * round(ry / height)

    a = vx * vx + vy * vy - bullet_speed * bullet_speed
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry

    t = None
    if abs(a) < 1e-6:
        if abs(b) > 1e-9:
            cand = -c / b
            if cand > 1e-4:
                t = cand
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sq = math.sqrt(disc)
            roots = ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a))
            pos = [r for r in roots if r > 1e-4]
            if pos:
                t = min(pos)

    if t is None:
        return math.degrees(math.atan2(ry, rx)) % 360.0
    return math.degrees(math.atan2(ry + vy * t, rx + vx * t)) % 360.0



def _urgency_score(tau: float, closing: float) -> float:
    if not math.isfinite(tau):
        tau_term = 0.0
    else:
        tau_term = 1.0 - min(max(tau, 0.0), URGENCY_TAU_CAP) / URGENCY_TAU_CAP
    speed_term = min(1.0, max(0.0, closing) / URGENCY_SPEED_SCALE)
    raw = URGENCY_TAU_WEIGHT * tau_term + URGENCY_SPEED_WEIGHT * speed_term
    return _clamp(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Acquisition difficulty FIS
# ---------------------------------------------------------------------------

def _acquisition_difficulty(heading_error_abs: float, d_surface: float) -> float:
    """Sugeno-style acquisition difficulty in [0, 1]."""
    mu_he = {
        "tiny": _trap(heading_error_abs, *HE_TINY),
        "small": _trap(heading_error_abs, *HE_SMALL),
        "medium": _trap(heading_error_abs, *HE_MEDIUM),
        "large": _trap(heading_error_abs, *HE_LARGE),
    }
    mu_d = {
        "close": _trap(d_surface, *D_CLOSE),
        "medium": _trap(d_surface, *D_MEDIUM),
        "far": _trap(d_surface, *D_FAR),
    }

    rules = [
        (min(mu_he["tiny"], mu_d["close"]), DIFF_EASY),
        (min(mu_he["tiny"], mu_d["medium"]), DIFF_EASY),
        (min(mu_he["tiny"], mu_d["far"]), DIFF_MEDIUM),
        (min(mu_he["small"], mu_d["close"]), DIFF_EASY),
        (min(mu_he["small"], mu_d["medium"]), DIFF_MEDIUM),
        (min(mu_he["small"], mu_d["far"]), DIFF_MEDIUM),
        (min(mu_he["medium"], mu_d["close"]), DIFF_MEDIUM),
        (min(mu_he["medium"], mu_d["medium"]), DIFF_MEDIUM),
        (min(mu_he["medium"], mu_d["far"]), DIFF_HARD),
        (mu_he["large"], DIFF_HARD),
    ]

    total_w = sum(w for w, _ in rules)
    if total_w < 1e-9:
        return 0.5
    return sum(w * out for w, out in rules) / total_w


# ---------------------------------------------------------------------------
# Fragmentation penalty
# ---------------------------------------------------------------------------

def _fragmentation_penalty(radius: float, delta_risk: float, lives_remaining: int = 3) -> float:
    if radius <= FRAG_RADIUS_THRESHOLD:
        penalty = 0.0
    elif delta_risk <= 0.0:
        penalty = 0.0
    else:
        penalty = min(FRAG_PENALTY_LARGE, delta_risk * 0.5)

    if lives_remaining <= 1:
        penalty *= 1.25
    elif lives_remaining >= 3:
        penalty *= 0.85

    return _clamp(penalty, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Target score
# ---------------------------------------------------------------------------

@dataclass
class TargetScore:
    asteroid_id: int
    signature: Tuple[float, float, float, float, float]

    bearing: float
    aim_bearing: float
    aim_error_abs: float

    tau: float
    risk: float
    delta_risk: float
    d_surface: float
    radius: float

    closing_speed: float
    approach_angle: float

    d_acq: float
    p_frag: float
    urgency: float
    aim_feasibility: float
    score: float


def _compute_score(
    ar: Any,
    current_heading: float,
    ship_state: Any = None,
    game_state: Any = None,
    lives_remaining: int = 3,
) -> TargetScore:
    ship_pos = _obj_get(ship_state, "position", None)
    ship_vel = _obj_get(ship_state, "velocity", (0.0, 0.0))
    map_size = _obj_get(game_state, "map_size", None)

    if ship_pos is not None and map_size is not None:
        if USE_INTERCEPT_LEAD:
            aim_bearing = intercept_aim_bearing(ship_pos, ship_vel, ar.position, ar.velocity, map_size)
        else:
            aim_bearing = predicted_aim_bearing(ship_pos, ar.position, ar.velocity, map_size)
        c_speed = closing_speed(ship_pos, ship_vel, ar.position, ar.velocity, map_size)
    else:
        aim_bearing = ar.bearing
        c_speed = 0.0

    aim_error_abs = abs(angular_diff(current_heading, aim_bearing))

    benefit = max(0.0, -ar.delta_risk_if_destroyed)
    if benefit < 1e-3:
        benefit = ar.risk * 0.3

    d_acq = _acquisition_difficulty(aim_error_abs, ar.d_surface)
    p_frag = _fragmentation_penalty(ar.radius, ar.delta_risk_if_destroyed, lives_remaining)
    urgency = _urgency_score(ar.tau, c_speed)
    urgency_factor = 0.6 + 0.4 * urgency
    aim_feasibility = math.exp(-aim_error_abs / AIM_FEASIBILITY_KAPPA)

    score = benefit * (1.0 - d_acq) * (1.0 - p_frag) * urgency_factor * aim_feasibility

    return TargetScore(
        asteroid_id=ar.asteroid_id,
        signature=asteroid_signature(ar),
        bearing=ar.bearing,
        aim_bearing=aim_bearing,
        aim_error_abs=aim_error_abs,
        tau=ar.tau,
        risk=ar.risk,
        delta_risk=benefit,
        d_surface=ar.d_surface,
        radius=ar.radius,
        closing_speed=c_speed,
        approach_angle=0.0,
        d_acq=d_acq,
        p_frag=p_frag,
        urgency=urgency,
        aim_feasibility=aim_feasibility,
        score=score,
    )


# ---------------------------------------------------------------------------
# Stateful selector
# ---------------------------------------------------------------------------

class TargetSelector:
    def __init__(
        self,
        commit_frames: int = None,
        tau_urgent_override: float = None,
        fire_threshold_deg: float = None,
        commit_switch_ratio: float = None,
    ) -> None:
        # Read module globals at construction time (not as default args) so that
        # params.apply() / the optimizer takes effect on freshly-built instances.
        self.commit_frames = COMMIT_FRAMES if commit_frames is None else commit_frames
        self.tau_urgent_override = TAU_URGENT_OVERRIDE if tau_urgent_override is None else tau_urgent_override
        self.fire_threshold_deg = FIRE_ANGLE_THRESHOLD if fire_threshold_deg is None else fire_threshold_deg
        self.commit_switch_ratio = COMMIT_SWITCH_RATIO if commit_switch_ratio is None else commit_switch_ratio

        self._committed_id: Optional[int] = None
        self._committed_signature: Optional[Tuple[float, float, float, float, float]] = None
        self._committed_target: Optional[TargetScore] = None
        self._frames_held = 0

    def select(
        self,
        asteroid_risks: list,
        current_heading: float,
        r_global: float = 0.0,
        ship_state: Any = None,
        game_state: Any = None,
    ) -> Optional[TargetScore]:
        if not asteroid_risks:
            self._reset_commitment()
            return None

        lives_remaining = int(_obj_get(ship_state, "lives_remaining", 3) or 3)

        candidates: List[TargetScore] = []
        by_id = {}
        by_sig = {}
        for ar in asteroid_risks:
            ts = _compute_score(ar, current_heading, ship_state, game_state, lives_remaining)
            by_id[ts.asteroid_id] = ts
            by_sig[ts.signature] = ts
            if ts.score >= MIN_SCORE_TO_TARGET:
                candidates.append(ts)

        if not candidates:
            self._reset_commitment()
            return None

        candidates.sort(key=lambda t: t.score, reverse=True)
        best = candidates[0]

        urgent_risks = [ar for ar in asteroid_risks if ar.tau < self.tau_urgent_override]
        if urgent_risks:
            urgent_ar = min(urgent_risks, key=lambda ar: ar.tau)
            urgent_target = _compute_score(
                urgent_ar, current_heading, ship_state, game_state, lives_remaining
            )
            self._set_commitment(urgent_target)
            return urgent_target

        held = self._current_committed(by_id, by_sig)
        if held is not None and self._frames_held < self.commit_frames:
            if best.score <= held.score * self.commit_switch_ratio:
                self._frames_held += 1
                self._committed_target = held
                return held

        self._set_commitment(best)
        return best

    def fire_decision(
        self,
        current_heading: float,
        target: Optional[TargetScore],
        can_fire: bool,
        emergency: bool = False,
    ) -> bool:
        if target is None or not can_fire:
            return False  # fire even while dodging (offense-as-defense)
        if target.delta_risk <= 0.0:
            return False

        heading_error = abs(angular_diff(current_heading, target.aim_bearing))
        return heading_error <= self.fire_threshold_deg

    def tick(self) -> None:
        pass

    def _current_committed(self, by_id: dict, by_sig: dict) -> Optional[TargetScore]:
        if self._committed_id is not None and self._committed_id in by_id:
            return by_id[self._committed_id]
        if self._committed_signature is not None and self._committed_signature in by_sig:
            return by_sig[self._committed_signature]
        return None

    def _set_commitment(self, target: TargetScore) -> None:
        self._committed_id = target.asteroid_id
        self._committed_signature = target.signature
        self._committed_target = target
        self._frames_held = 1

    def _reset_commitment(self) -> None:
        self._committed_id = None
        self._committed_signature = None
        self._committed_target = None
        self._frames_held = 0
