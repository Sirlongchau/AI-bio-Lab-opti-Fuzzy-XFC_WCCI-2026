"""
targeting_system.py — Aggressive turret + sacrifice-mine controller.

CORRECTION (v2) — contrôle de vitesse (V -> 0)
----------------------------------------------
Le turret n'est plus un "pure turret" qui suppose thrust = 0 : il prenait pour
acquis que le vaisseau était immobile, alors qu'il hérite de toute la vitesse
accumulée par le MPC au moment du handoff. Résultat : le vaisseau dérivait hors
de la zone sûre tout en visant/tirant, et fonçait sur un astéroïde.

Le turret applique maintenant un freinage bang-bang temps-optimal en 1D pour
amener la vitesse scalaire s -> 0 le plus vite possible, tout en continuant à
tourner (bang-bang sur le heading) et à tirer. Cela maximise le temps passé
dans la zone sûre où le MPC l'a déposé.

Modèle physique (confirmé) :
  - La vitesse est SCALAIRE, portée par le heading : s_next = s + (u_T - drag)*DT
    et la position avance le long du heading. Donc le freinage est un problème
    1D le long du nez du vaisseau.
  - La vitesse scalaire signée se récupère depuis le vecteur 2D par projection
    sur le heading : s = svx*cos(h) + svy*sin(h).
  - Bang-bang temps-optimal : thrust maximal opposé au signe de s, sauf près de
    zéro où l'on calcule le thrust exact qui annule s sans la dépasser (sinon
    oscillation autour de 0).

[... le reste du docstring d'origine, inchangé, plus bas ...]

Role in the architecture
-------------------------
The Supervisor runs this module as the **aggressive** mode. Its job is to
destroy as many asteroids as possible, for two reasons:
  1. Score.
  2. Keep the live asteroid count under the MPC's N_AST_MAX budget, so the
     CasADi evasion solver always sees the *full* threat set.

Targeting philosophy — minimise rotation, not asteroids-in-front
----------------------------------------------------------------
Rotating is the expensive operation; firing is free and instantaneous. The
turret is sticky on bearing: it prefers the good target closest to the current
heading and commits to it (rakes fragmentation cascades with near-zero slew).

Three engine realities this controller respects
-----------------------------------------------
1. Bullets do NOT wrap -> all firing geometry uses the DIRECT (un-wrapped)
   relative vector; identity is still tracked torically.
2. The ship keeps drifting at thrust = 0 (inertia + slow drag), and the engine
   fires the bullet AFTER moving the ship this frame. The intercept is solved
   from the predicted MUZZLE position pos + velocity*DT.
3. AsteroidRisk.radius is the true collision radius in px.
"""

import math
from typing import List, Optional, Tuple

# --- Engine-derived constants -------------------------------------------------
BULLET_SPEED = 800.0        # px/s  (bullet.py)
OMEGA_MAX    = 180.0        # deg/s (ship.turn_rate_range)
DT           = 1.0 / 30.0   # s     (engine frame period)

# --- Ship translation dynamics (ship.py) — needed for velocity control --------
THRUST_MAX   =  480.0       # px/s^2 — max forward thrust
THRUST_MIN   = -480.0       # px/s^2 — max reverse thrust (braking)
DRAG         =   80.0       # px/s^2 — passive drag magnitude (opposes motion)
STOP_EPS     =    2.0       # px/s   — |s| below this is treated as stopped

# --- Firing envelope ----------------------------------------------------------
MAX_DIST_FIRE = 700.0       # px — direct (un-wrapped) range cap
HIT_FRACTION  = 0.85        # aim within this fraction of the radius

# --- Target scoring -----------------------------------------------------------
MAX_RADIUS_PX = 32.0        # size-4 asteroid (radius = size*8)
W_SMALL       = 0.30
W_CLOSE       = 0.25
STICK_DEG     = 50.0        # bearing-bonus falloff in degrees
STICK_GAIN    = 1.0         # bearing-bonus weight (0 = none, high = very sticky)

# --- Lock hysteresis (anti-dither) -------------------------------------------
TRACK_TOL_PX = 28.0
SWITCH_RATIO = 1.30


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
    at (vx,vy), bullet leaving the (fixed) muzzle at BULLET_SPEED.
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


# ---------------------------------------------------------------------------
# Velocity control — bang-bang time-optimal braking to s -> 0
# ---------------------------------------------------------------------------

def _scalar_speed(velocity, heading_deg) -> float:
    """
    Vitesse scalaire SIGNÉE le long du heading.

    L'engine porte la vitesse par le heading (modèle scalaire), mais expose un
    vecteur 2D (svx, svy). On récupère le scalaire signé par projection :
        s = svx*cos(h) + svy*sin(h)
    s > 0  : le vaisseau avance dans le sens du nez.
    s < 0  : le vaisseau recule.
    """
    h = math.radians(heading_deg)
    svx, svy = velocity
    return svx * math.cos(h) + svy * math.sin(h)


def _braking_thrust(s: float) -> float:
    """
    Thrust bang-bang temps-optimal pour amener s -> 0 le plus vite possible.

    Dynamique : s_next = s + (u_T - drag_signed)*DT
    où drag_signed = DRAG * sign(s) (le drag s'oppose toujours au mouvement).

    - |s| <= STOP_EPS : on est arrêté, thrust nul (laisser le drag finir).
    - Sinon, near-zero handling : si un freinage à pleine puissance ferait
      DÉPASSER zéro (changer le signe de s), on calcule le thrust exact qui
      amène s pile à 0 :
          0 = s + (u_T - drag_signed)*DT  =>  u_T = drag_signed - s/DT
      clampé à [THRUST_MIN, THRUST_MAX]. Sinon, pleine puissance opposée.
    """
    if abs(s) <= STOP_EPS:
        return 0.0

    sign_s      = math.copysign(1.0, s)
    drag_signed = DRAG * sign_s    # le drag aide déjà au freinage

    # Thrust qui annulerait EXACTEMENT s en une frame (avant clamp).
    u_exact = drag_signed - s / DT

    # Pleine puissance opposée au mouvement.
    u_full  = THRUST_MIN if s > 0 else THRUST_MAX

    # On choisit le plus DOUX des deux en magnitude quand u_exact est dans
    # l'enveloppe : si |u_exact| < |u_full|, c'est qu'on est assez proche de 0
    # pour s'arrêter pile cette frame -> on prend u_exact (anti-oscillation).
    if THRUST_MIN <= u_exact <= THRUST_MAX:
        # u_exact réalisable : il amène s à 0 sans dépasser.
        return u_exact
    # Sinon, s est trop grand pour s'arrêter en une frame -> pleine puissance.
    return u_full


# A firing solution, cached so selection and the final command agree.
class _Shot:
    __slots__ = ("ar", "dist", "aim", "score")

    def __init__(self, ar, dist, aim, score):
        self.ar = ar; self.dist = dist; self.aim = aim; self.score = score


# ---------------------------------------------------------------------------
# TargetingController — the aggressive turret (now with velocity control)
# ---------------------------------------------------------------------------

class TargetingController:

    def __init__(self) -> None:
        self._locked: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    # -- public API ---------------------------------------------------------

    def compute(self, ship_state, game_state, asteroid_risks, now=None
                ) -> Tuple[float, float, bool]:
        """
        Return (thrust, turn_rate, fire).

        thrust n'est plus toujours 0 : c'est désormais le freinage bang-bang
        qui amène la vitesse scalaire s -> 0 pour rester dans la zone sûre.
        Le pointage (turn_rate) et le tir (fire) sont inchangés.
        """
        map_size = game_state.map_size

        # --- Contrôle de vitesse : freinage indépendant du fait qu'on ait
        #     une cible ou non. On veut TOUJOURS annuler la dérive. ---
        s_scalar = _scalar_speed(ship_state.velocity, ship_state.heading)
        brake    = _braking_thrust(s_scalar)

        if not asteroid_risks:
            self._locked = None
            # Pas de cible mais on freine quand même pour rester sur place.
            return brake, 0.0, False

        heading = ship_state.heading
        sx, sy   = ship_state.position
        svx, svy = ship_state.velocity
        mx, my   = sx + svx * DT, sy + svy * DT

        shot = self._select(mx, my, heading, asteroid_risks, map_size)
        if shot is None:
            self._locked = None
            return brake, 0.0, False

        err       = _ang_diff(shot.aim, heading)
        turn_rate = max(-OMEGA_MAX, min(OMEGA_MAX, err / DT))

        aim_tol   = math.degrees(math.atan2(shot.ar.radius * HIT_FRACTION, max(shot.dist, 1.0)))
        err_after = err - turn_rate * DT
        fire = (abs(err_after) <= aim_tol
                and shot.dist <= MAX_DIST_FIRE
                and ship_state.can_fire)

        # Freinage + pointage + tir simultanés.
        return brake, turn_rate, fire

    # -- target selection ---------------------------------------------------

    def _shot_for(self, mx, my, heading, ar) -> Optional["_Shot"]:
        ax = ar.position[0] + ar.velocity[0] * DT
        ay = ar.position[1] + ar.velocity[1] * DT
        rx, ry = ax - mx, ay - my
        dist = math.hypot(rx, ry)
        if dist > MAX_DIST_FIRE:
            return None
        aim, _t, ok = _intercept(rx, ry, ar.velocity[0], ar.velocity[1])
        if not ok:
            return None
        return _Shot(ar, dist, aim, self._score(ar, dist, abs(_ang_diff(aim, heading))))

    def _select(self, mx, my, heading, risks, map_size) -> Optional["_Shot"]:
        best: Optional[_Shot] = None
        for ar in risks:
            s = self._shot_for(mx, my, heading, ar)
            if s is not None and (best is None or s.score > best.score):
                best = s

        locked = self._reacquire_lock(mx, my, heading, risks, map_size)

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
        stick = 1.0 + STICK_GAIN * math.exp(-ang_err / STICK_DEG)
        return ar.risk * (1.0 + W_SMALL * small + W_CLOSE * close) * stick


# ---------------------------------------------------------------------------
# SacrificeController — clear the field on the way down
# ---------------------------------------------------------------------------

MINE_FUSE_S        = 3.0
MINE_BLAST_R       = 150.0

# GA-tunable mine policy:
# Drop when TTC is medium and risk is high, especially if the mine is predicted
# to catch enough asteroids or the nearest medium/high-risk object is close.
MINE_MIN_CATCH     = 2
MINE_MAX_HOLD      = 18
MINE_TAU_MIN       = 0.70
MINE_TAU_MAX       = 3.00
MINE_RISK_TH       = 0.55
MINE_NEAR_SURFACE  = 210.0
MINE_COOLDOWN_FRAMES = 30


class SacrificeController:
    """
    Activated by the Supervisor when the MPC is infeasible for several frames.
    Same turret (now with velocity control), plus a single mine.
    """

    def __init__(self) -> None:
        self._turret        = TargetingController()
        self._mine_used     = False  # legacy flag; not used by new policy
        self._last_mine_frame = -9999
        self._active_frames = 0

    def compute(self, ship_state, game_state, asteroid_risks, now=None
                ) -> Tuple[float, float, bool, bool]:
        thrust, turn_rate, fire = self._turret.compute(
            ship_state, game_state, asteroid_risks, now)
        drop_mine = self._decide_mine(ship_state, game_state, asteroid_risks)
        return thrust, turn_rate, fire, drop_mine

    def reset(self) -> None:
        self._mine_used     = False  # legacy flag; not used by new policy
        self._last_mine_frame = -9999
        self._active_frames = 0
        self._turret._locked = None

    def _decide_mine(self, ship_state, game_state, risks) -> bool:
        """
        Mine policy:
        - Uses all available mines, not just one.
        - Drops only when at least one asteroid has medium TTC and high risk.
        - Requires either predicted mine catch OR close surface distance.
        - Tuned by GA through globals patched from ga_optimizer.py.
        """
        if not risks:
            return False
        if getattr(ship_state, "respawn_time_left", 0.0) > 0:
            return False
        if ship_state.mines_remaining == 0 or not getattr(ship_state, "can_deploy_mine", True):
            return False

        self._active_frames += 1

        if not hasattr(self, "_last_mine_frame"):
            self._last_mine_frame = -9999

        if (self._active_frames - self._last_mine_frame) < MINE_COOLDOWN_FRAMES:
            return False

        medium_high = [
            ar for ar in risks
            if (
                math.isfinite(ar.tau)
                and MINE_TAU_MIN <= ar.tau <= MINE_TAU_MAX
                and ar.risk >= MINE_RISK_TH
            )
        ]

        if not medium_high:
            return False

        nearest_medium_high = min((ar.d_surface for ar in medium_high), default=9999.0)
        catch = self._predicted_catch(ship_state, game_state, risks)

        should_drop = (
            catch >= MINE_MIN_CATCH
            or nearest_medium_high <= MINE_NEAR_SURFACE
            or self._active_frames >= MINE_MAX_HOLD
        )

        if should_drop:
            self._last_mine_frame = self._active_frames
            self._active_frames = 0
            return True

        return False

    @staticmethod
    def _predicted_catch(ship_state, game_state, risks) -> int:
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