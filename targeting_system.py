"""
targeting_system.py — Aggressive turret + sacrifice-mine controller.

Role in the architecture
-------------------------
The Supervisor runs this module as the **aggressive** mode. Its job is to
destroy as many asteroids as possible, for two reasons:
  1. Score.
  2. Keep the live asteroid count under the MPC's N_AST_MAX (=50) budget, so
     the CasADi evasion solver always sees the *full* threat set.

Behaviour: a **stationary turret**. The ship does not translate (thrust = 0);
it only rotates (bang-bang at OMEGA_MAX) and fires. Evasion is delegated to
the MPC — mixing translation in here would only risk flying into asteroids.

Targeting philosophy — minimise rotation, not asteroids-in-front
----------------------------------------------------------------
Rotating is the expensive operation; firing is free and instantaneous. So the
turret is **sticky on bearing**: it strongly prefers whatever good target sits
closest to the current heading and commits to it.

This is exactly what you want against fragmentation. Destroying a size>=2
asteroid spawns 3 children *at the same spot* (asteroid.py), i.e. at almost the
same bearing. Holding the heading lets the turret rake the whole 3->2->1
cascade with near-zero slew — "suppression fire in the direction of the
splitting asteroid". It also removes the oscillation that a "jump to the next
target then get pulled back by the fresh fragments" policy produces: we never
jump away, so nothing pulls us back. Only when the local cluster is exhausted
does the best-scoring target move elsewhere and the turret sweeps on.

Three engine realities this controller respects
-----------------------------------------------
1. Bullets do NOT wrap. `bullet.update` integrates without modulo and the
   collision pass "does not consider wrapping" (collisions.py), so a shot at a
   toroidal image across the seam always misses. => All firing geometry uses
   the DIRECT (un-wrapped) relative vector; an asteroid only reachable across
   the seam is not a firing candidate. Target *identity* is still tracked
   torically, so an asteroid that wraps doesn't drop the lock.

2. The ship keeps drifting at thrust = 0 (inertia + slow drag), and the engine
   fires the bullet AFTER moving the ship this frame (ship.update: move ->
   turn -> fire). => The intercept is solved from the predicted MUZZLE position
   `pos + velocity*DT`, target propagated by the same DT. Removes drift error.

3. AsteroidRisk.radius is the true collision radius in px (risk_field contract),
   used for the angular fire tolerance.

Conventions: heading deg CCW, 0 = +x; bullet velocity = BULLET_SPEED*(cos,sin),
so the intercept heading is atan2(dy, dx) on the direct relative vector. The
engine applies heading += turn_rate*DT then fires, so the fire gate checks the
alignment that will hold AFTER this frame's turn.
"""

import math
from typing import List, Optional, Tuple

# --- Engine-derived constants -------------------------------------------------
BULLET_SPEED = 800.0        # px/s  (bullet.py)
OMEGA_MAX    = 180.0        # deg/s (ship.turn_rate_range)
DT           = 1.0 / 30.0   # s     (engine frame period)

# --- Firing envelope ----------------------------------------------------------
MAX_DIST_FIRE = 700.0       # px — direct (un-wrapped) range cap
HIT_FRACTION  = 0.85        # aim within this fraction of the radius

# --- Target scoring -----------------------------------------------------------
# Risk is primary; small/close are mild tie-breakers. Bearing is a BONUS, not a
# multiplier that crushes off-bearing risk, so a splitting cluster is favoured
# yet a clearly higher-risk asteroid elsewhere can still steal the lock:
#     score = risk * (1 + W_SMALL*small + W_CLOSE*close) * (1 + STICK_GAIN*stick)
#     stick = exp(-|dtheta| / STICK_DEG)   in [0, 1]
# STICK_GAIN = 0 ignores bearing (pure risk priority); large STICK_GAIN glues
# the turret to its cluster. STICK_DEG sets how wide the "near heading" lobe is.
MAX_RADIUS_PX = 32.0        # size-4 asteroid (radius = size*8)
W_SMALL       = 0.30
W_CLOSE       = 0.25
STICK_DEG     = 50.0        # bearing-bonus falloff in degrees
STICK_GAIN    = 1.0         # bearing-bonus weight (0 = none, high = very sticky)

# --- Lock hysteresis (anti-dither) -------------------------------------------
TRACK_TOL_PX = 28.0         # re-identify the locked asteroid within this radius (toric)
SWITCH_RATIO = 1.30         # leave the lock only for a challenger >30% better


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ang_diff(a: float, b: float) -> float:
    """Signed smallest angle a - b in (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def _toric_rel(fx, fy, tx, ty, map_size) -> Tuple[float, float]:
    """Shortest relative vector (tx,ty)-(fx,fy) on the torus (identity tracking)."""
    w, h = map_size
    dx = tx - fx; dx -= w * round(dx / w)
    dy = ty - fy; dy -= h * round(dy / h)
    return dx, dy


def _intercept(rx, ry, vx, vy) -> Tuple[float, float, bool]:
    """
    Earliest bullet-intercept for a target at relative position (rx,ry) moving
    at (vx,vy), bullet leaving the (fixed) muzzle at BULLET_SPEED:
        (|v|^2 - Vb^2) t^2 + 2(r.v) t + |r|^2 = 0.
    Returns (aim_heading_deg, time_to_hit, ok).
    """
    a = (vx * vx + vy * vy) - BULLET_SPEED * BULLET_SPEED
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry

    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return 0.0, 0.0, False
        t = -c / b
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return 0.0, 0.0, False
        sq = math.sqrt(disc)
        roots = [r for r in ((-b + sq) / (2 * a), (-b - sq) / (2 * a)) if r > 1e-4]
        if not roots:
            return 0.0, 0.0, False
        t = min(roots)

    return math.degrees(math.atan2(ry + vy * t, rx + vx * t)) % 360.0, t, True


# A firing solution, cached so selection and the final command agree.
class _Shot:
    __slots__ = ("ar", "dist", "aim", "score")

    def __init__(self, ar, dist, aim, score):
        self.ar = ar; self.dist = dist; self.aim = aim; self.score = score


# ---------------------------------------------------------------------------
# TargetingController — the aggressive turret
# ---------------------------------------------------------------------------

class TargetingController:

    def __init__(self) -> None:
        # Locked target, tracked by predicted position (indices aren't stable).
        self._locked: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    # -- public API ---------------------------------------------------------

    def compute(self, ship_state, game_state, asteroid_risks, now=None
                ) -> Tuple[float, float, bool]:
        """Return (thrust, turn_rate, fire). thrust is always 0 (pure turret)."""
        map_size = game_state.map_size

        if not asteroid_risks:
            self._locked = None
            return 0.0, 0.0, False

        heading = ship_state.heading
        # Muzzle = where the bullet will actually spawn next frame (inertia comp).
        sx, sy   = ship_state.position
        svx, svy = ship_state.velocity
        mx, my   = sx + svx * DT, sy + svy * DT

        shot = self._select(mx, my, heading, asteroid_risks, map_size)
        if shot is None:
            self._locked = None
            return 0.0, 0.0, False

        # Bang-bang slew toward the (lead-corrected) aim point.
        err       = _ang_diff(shot.aim, heading)
        turn_rate = max(-OMEGA_MAX, min(OMEGA_MAX, err / DT))

        # Fire gate: residual error after this frame's turn within the target's
        # angular half-size, in (direct) range, engine ready.
        aim_tol   = math.degrees(math.atan2(shot.ar.radius * HIT_FRACTION, max(shot.dist, 1.0)))
        err_after = err - turn_rate * DT
        fire = (abs(err_after) <= aim_tol
                and shot.dist <= MAX_DIST_FIRE
                and ship_state.can_fire)

        return 0.0, turn_rate, fire

    # -- target selection ---------------------------------------------------

    def _shot_for(self, mx, my, heading, ar) -> Optional["_Shot"]:
        """Direct-vector firing solution for one asteroid (None if unreachable)."""
        ax = ar.position[0] + ar.velocity[0] * DT      # target at fire time
        ay = ar.position[1] + ar.velocity[1] * DT
        rx, ry = ax - mx, ay - my                      # DIRECT, no wrap -> no seam shots
        dist = math.hypot(rx, ry)
        if dist > MAX_DIST_FIRE:
            return None
        aim, _t, ok = _intercept(rx, ry, ar.velocity[0], ar.velocity[1])
        if not ok:
            return None
        return _Shot(ar, dist, aim, self._score(ar, dist, abs(_ang_diff(aim, heading))))

    def _select(self, mx, my, heading, risks, map_size) -> Optional["_Shot"]:
        # 1. Best reachable candidate (bearing-sticky score).
        best: Optional[_Shot] = None
        for ar in risks:
            s = self._shot_for(mx, my, heading, ar)
            if s is not None and (best is None or s.score > best.score):
                best = s

        # 2. Re-identify the locked target (None if destroyed or unreachable).
        locked = self._reacquire_lock(mx, my, heading, risks, map_size)

        # 3. Commit to the lock unless a clearly better challenger appears.
        if locked is None:
            chosen = best
        elif best is not None and best.ar is not locked.ar and best.score >= locked.score * SWITCH_RATIO:
            chosen = best
        else:
            chosen = locked
        if chosen is None:
            chosen = best

        self._locked = (chosen.ar.position, chosen.ar.velocity) if chosen else None
        return chosen

    def _reacquire_lock(self, mx, my, heading, risks, map_size) -> Optional["_Shot"]:
        """Match the lock to this frame's asteroids by predicted position (toric)."""
        if self._locked is None:
            return None
        (lx, ly), (lvx, lvy) = self._locked
        px, py = lx + lvx * DT, ly + lvy * DT

        best_d, found = TRACK_TOL_PX, None
        for ar in risks:
            dx, dy = _toric_rel(px, py, ar.position[0], ar.position[1], map_size)
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d, found = d, ar
        return self._shot_for(mx, my, heading, found) if found is not None else None

    # -- scoring ------------------------------------------------------------

    def _score(self, ar, dist: float, ang_err: float) -> float:
        small = max(0.0, 1.0 - ar.radius / MAX_RADIUS_PX)
        close = 1.0 - min(dist, MAX_DIST_FIRE) / MAX_DIST_FIRE
        stick = 1.0 + STICK_GAIN * math.exp(-ang_err / STICK_DEG)   # bounded bearing bonus
        return ar.risk * (1.0 + W_SMALL * small + W_CLOSE * close) * stick


# ---------------------------------------------------------------------------
# SacrificeController — clear the field on the way down
# ---------------------------------------------------------------------------

# Mine parameters (mines.py).
MINE_FUSE_S    = 3.0        # s  — detonation delay after the drop (NOT instant)
MINE_BLAST_R   = 150.0      # px — blast radius
MINE_MIN_CATCH = 3          # min asteroids predicted in the blast to bother dropping
MINE_MAX_HOLD  = 15         # frames — eagerness: drop by now regardless once active


class SacrificeController:
    """
    Activated by the Supervisor when the MPC is infeasible for several frames in
    a row (no escape exists). Same turret as TargetingController, plus a single
    mine dropped to take out as many asteroids as possible.

    The fuse is 3 s and the mine stays where it is dropped, so we count the
    asteroids that will be inside the blast radius AT DETONATION (t = fuse),
    around the current position, and drop as soon as that is worthwhile — or by
    MINE_MAX_HOLD frames at the latest, since the ship is going down anyway.
    """

    def __init__(self) -> None:
        self._turret        = TargetingController()
        self._mine_used     = False
        self._active_frames = 0

    def compute(self, ship_state, game_state, asteroid_risks, now=None
                ) -> Tuple[float, float, bool, bool]:
        thrust, turn_rate, fire = self._turret.compute(
            ship_state, game_state, asteroid_risks, now)
        drop_mine = self._decide_mine(ship_state, game_state, asteroid_risks)
        return thrust, turn_rate, fire, drop_mine

    def reset(self) -> None:
        """Called by the Supervisor when the MPC becomes feasible again."""
        self._mine_used     = False
        self._active_frames = 0
        self._turret._locked = None

    # -- mine logic ---------------------------------------------------------

    def _decide_mine(self, ship_state, game_state, risks) -> bool:
        if self._mine_used or not risks:
            return False
        if ship_state.mines_remaining == 0 or not getattr(ship_state, 'can_deploy_mine', True):
            return False

        self._active_frames += 1
        catch = self._predicted_catch(ship_state, game_state, risks)
        if catch >= MINE_MIN_CATCH or self._active_frames >= MINE_MAX_HOLD:
            self._mine_used = True
            return True
        return False

    @staticmethod
    def _predicted_catch(ship_state, game_state, risks) -> int:
        """Asteroids whose surface is inside the blast radius at detonation."""
        mx, my   = ship_state.position
        map_size = game_state.map_size
        t = MINE_FUSE_S
        n = 0
        for ar in risks:
            ax = ar.position[0] + ar.velocity[0] * t
            ay = ar.position[1] + ar.velocity[1] * t
            dx, dy = _toric_rel(mx, my, ax, ay, map_size)
            if math.hypot(dx, dy) - ar.radius <= MINE_BLAST_R:
                n += 1
        return n