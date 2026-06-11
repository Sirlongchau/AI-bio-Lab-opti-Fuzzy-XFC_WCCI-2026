"""
targeting_system.py — version corrigée + mode sacrifice
Convention: kessler-game standard → y monde vers le HAUT, heading CCW, 0° = +x.
Si ton fork est réellement screen-down, mets Y_SIGN = -1.0 (un seul endroit).
"""

import math
from typing import Optional, Tuple

BULLET_SPEED  = 800.0
OMEGA_MAX     = 180.0
DT            = 1.0 / 30.0

Y_SIGN        = 1.0      # 1.0 = convention monde (correct pour kessler-game)
                         # -1.0 = screen-down (ancien comportement, bugué)

W_SMALL       = 0.3      # bonus multiplicatif petite taille
MAX_RADIUS    = 40.0
MAX_DIST_FIRE = 700.0
HIT_FRACTION  = 0.8      # viser dans 80% du rayon → marge de sécurité

LOCK_MIN_FRAMES = 4      # snap : ~0.13 s de lock minimum seulement
SWITCH_RATIO    = 1.3    # switch dès qu'un challenger est 30% meilleur
TRACK_TOL       = 30.0   # px — tolérance pour ré-identifier la cible

# ── Mode sacrifice ──────────────────────────────────────────────────────────
MINE_LEAD_S      = 0.20  # s — larguer la mine ce délai avant l'impact prédit
MINE_BLAST_R     = 150.0 # px — rayon d'explosion mine Kessler (défaut moteur)
MINE_MIN_THREATS = 1     # nb min d'astéroïdes dans le rayon pour valider le drop


def _angle_diff(a: float, b: float) -> float:
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def _toric_rel(sx, sy, ax, ay, map_size):
    """Vecteur relatif le plus court (torique)."""
    W, H = map_size
    dx = ax - sx;  dx -= W * round(dx / W)
    dy = ay - sy;  dy -= H * round(dy / H)
    return dx, dy


def _lead_solution(rx, ry, vx, vy):
    """
    Intercept en coordonnées RELATIVES.
    Retourne (heading_deg, t, ok) en convention monde standard.
    """
    a_q = BULLET_SPEED**2 - (vx*vx + vy*vy)
    b_q = 2.0 * (rx*vx + ry*vy)
    c_q = -(rx*rx + ry*ry)

    if abs(a_q) < 1e-6:
        if abs(b_q) < 1e-6:
            return 0.0, 0.0, False
        t = -c_q / b_q
    else:
        disc = b_q*b_q - 4.0*a_q*c_q
        if disc < 0.0:
            return 0.0, 0.0, False
        sq = math.sqrt(disc)
        ts = [(-b_q + sq)/(2*a_q), (-b_q - sq)/(2*a_q)]
        ts = [t for t in ts if t > 1e-4]
        if not ts:
            return 0.0, 0.0, False
        t = min(ts)

    ix = rx + vx * t          # point d'intercept relatif
    iy = ry + vy * t
    heading = math.degrees(math.atan2(Y_SIGN * iy, ix)) % 360.0
    return heading, t, True


# ---------------------------------------------------------------------------
# TargetingController
# ---------------------------------------------------------------------------

class TargetingController:

    def __init__(self) -> None:
        self._tgt_pos: Optional[Tuple[float, float]] = None
        self._tgt_vel: Optional[Tuple[float, float]] = None
        self._lock_ttl = 0

    # ------------------------------------------------------------------

    def compute(self, ship_state, game_state, asteroid_risks, now=None):
        sx, sy   = ship_state.position
        heading  = ship_state.heading
        map_size = game_state.map_size

        if not asteroid_risks:
            self._drop_lock()
            return 0.0, 0.0, False

        target = self._track_or_acquire(sx, sy, asteroid_risks, map_size)
        if target is None:
            self._drop_lock()
            return 0.0, 0.0, False

        # ── Lead en torique relatif ───────────────────────────────────
        rx, ry = _toric_rel(sx, sy, target.position[0], target.position[1], map_size)
        lead_deg, t_hit, ok = _lead_solution(rx, ry,
                                             target.velocity[0],
                                             target.velocity[1])
        if not ok:
            self._drop_lock()
            return 0.0, 0.0, False

        # ── Bang-bang pur : pleine vitesse jusqu'à alignement ─────────
        error = _angle_diff(lead_deg, heading)
        turn_rate = max(-OMEGA_MAX, min(OMEGA_MAX, error / DT))

        # ── Précision ADAPTATIVE : tolérance = taille angulaire cible ─
        dist = math.hypot(rx, ry)
        fire_tol = math.degrees(math.atan2(target.radius * HIT_FRACTION,
                                           max(dist, 1.0)))
        error_after = error - turn_rate * DT
        fire = (abs(error_after) <= fire_tol
                and dist <= MAX_DIST_FIRE
                and ship_state.can_fire)

        return 0.0, turn_rate, fire

    # ------------------------------------------------------------------

    def _drop_lock(self):
        self._tgt_pos = None
        self._tgt_vel = None
        self._lock_ttl = 0

    def _score(self, ar):
        size_bonus = max(0.0, 1.0 - ar.radius / MAX_RADIUS)
        return ar.risk * (1.0 + W_SMALL * size_bonus)

    def _track_or_acquire(self, sx, sy, asteroid_risks, map_size):
        """
        Ré-identifie la cible verrouillée par prédiction de position
        (nearest-neighbor sur pos + vel·dt), pas par signature quantifiée.
        """
        # 1. Retrouver la cible lockée
        locked = None
        if self._tgt_pos is not None:
            px = self._tgt_pos[0] + self._tgt_vel[0] * DT
            py = self._tgt_pos[1] + self._tgt_vel[1] * DT
            best_d = TRACK_TOL
            for ar in asteroid_risks:
                dx, dy = _toric_rel(px, py, ar.position[0], ar.position[1], map_size)
                d = math.hypot(dx, dy)
                if d < best_d:
                    best_d = d
                    locked = ar

        # 2. Meilleur candidat global (cibles atteignables seulement)
        best, best_score = None, -1.0
        for ar in asteroid_risks:
            rx, ry = _toric_rel(sx, sy, ar.position[0], ar.position[1], map_size)
            if math.hypot(rx, ry) > MAX_DIST_FIRE:
                continue
            _, _, ok = _lead_solution(rx, ry, ar.velocity[0], ar.velocity[1])
            if not ok:
                continue
            s = self._score(ar)
            if s > best_score:
                best_score, best = s, ar

        if best is None and locked is None:
            return None

        # 3. Décision de switch
        if locked is None:                       # cible détruite → snap immédiat
            chosen = best
            self._lock_ttl = LOCK_MIN_FRAMES
        elif self._lock_ttl > 0:                 # hold court
            self._lock_ttl -= 1
            chosen = locked
        elif best is not None and best is not locked \
                and best_score >= self._score(locked) * SWITCH_RATIO:
            chosen = best                        # challenger nettement meilleur
            self._lock_ttl = LOCK_MIN_FRAMES
        else:
            chosen = locked if locked is not None else best

        if chosen is not None:
            self._tgt_pos = chosen.position
            self._tgt_vel = chosen.velocity
        return chosen


# ---------------------------------------------------------------------------
# SacrificeController
# ---------------------------------------------------------------------------

class SacrificeController:
    """
    Activé par le Supervisor quand le MPC est infaisable N frames d'affilée.
    Même ciblage que TargetingController + un largage de mine au bon moment.

    Stratégie mine :
      - On largue quand l'impact le plus proche est imminent (TTC ≤ MINE_LEAD_S)
      - ET que la mine touchera au moins MINE_MIN_THREATS astéroïdes
        au moment de l'impact prédit (sinon on la garde, elle est précieuse).
      - Une seule mine par activation ; le Supervisor appelle reset()
        quand le MPC redevient faisable.
    """

    def __init__(self) -> None:
        self._targeting  = TargetingController()
        self._mine_armed = True
        self._mine_fired = False

    # ------------------------------------------------------------------

    def compute(self, ship_state, game_state, asteroid_risks, now=None):
        """Retourne (thrust, turn_rate, fire, drop_mine)."""
        thrust, turn_rate, fire = self._targeting.compute(
            ship_state, game_state, asteroid_risks, now
        )
        drop_mine = self._should_drop_mine(ship_state, game_state, asteroid_risks)
        return thrust, turn_rate, fire, drop_mine

    # ------------------------------------------------------------------

    def _should_drop_mine(self, ship_state, game_state, asteroid_risks) -> bool:
        if self._mine_fired or not self._mine_armed or not asteroid_risks:
            return False

        # Plus de mines disponibles → inutile d'essayer
        if getattr(ship_state, 'mines_remaining', 0) <= 0:
            self._mine_armed = False
            return False

        # Impact imminent ?
        min_ttc = min(ar.tau for ar in asteroid_risks)
        if min_ttc > MINE_LEAD_S:
            return False

        # La mine vaut-elle le coup ? Compter les astéroïdes qui seront
        # dans le rayon d'explosion au moment min_ttc (position prédite,
        # géométrie torique).
        sx, sy   = ship_state.position
        map_size = game_state.map_size
        threats  = 0
        for ar in asteroid_risks:
            ax = ar.position[0] + ar.velocity[0] * min_ttc
            ay = ar.position[1] + ar.velocity[1] * min_ttc
            dx, dy = _toric_rel(sx, sy, ax, ay, map_size)
            if math.hypot(dx, dy) - ar.radius <= MINE_BLAST_R:
                threats += 1

        if threats >= MINE_MIN_THREATS:
            self._mine_fired = True
            return True
        return False

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Appelé par le Supervisor quand le MPC redevient faisable."""
        self._mine_armed = True
        self._mine_fired = False
        self._targeting._drop_lock()