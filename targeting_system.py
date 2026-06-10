"""
targeting_system.py  —  Système de visée XFC Kessler
=====================================================

Expose deux contrôleurs autonomes, sans couplage MPC :

  TargetingController
  -------------------
  Objectif : destruction maximale d'astéroïdes.
  - Sélectionne la cible prioritaire via un score multi-critère
    (risque FIS, urgence TTC, kill-chain, pénalité fragmentation).
  - Calcule l'angle de lead shot (interception ballistique).
  - Contrôle le heading via un PID pur (entrée : erreur angulaire,
    sortie : turn_rate). Thrust = 0, vitesse non commandée.
  - Tire dès que l'erreur angulaire est dans le cône de tir.

  SacrificeController
  -------------------
  Mode sacrificiel : activé par le Supervisor quand le MPC est infaisable
  pendant N frames consécutives (collision inévitable imminente).

  Comportement
  ~~~~~~~~~~~~
  - Continue de viser et tirer comme TargetingController (destruction max).
  - Pose une mine (drop_mine = True) au moment optimal, défini comme :

        t_mine = t_impact - MINE_LEAD_S

    où t_impact est estimé à partir du min(TTC) courant.
    La mine n'est posée qu'une seule fois par activation du mode sacrificiel
    (flag interne remis à zéro lors du reset()).

  Interface de retour
  ~~~~~~~~~~~~~~~~~~~
  actions() retourne (turn_rate, thrust, fire, drop_mine) — 4-tuple.
  Le Supervisor doit lire le 4e élément et appeler l'API mine de Kessler.

Utilisation dans le Supervisor
-------------------------------
    from targeting_system import TargetingController, SacrificeController

    class Supervisor:
        def __init__(self):
            self.targeting    = TargetingController()
            self.sacrifice    = SacrificeController()
            self._mpc_infeasible_streak = 0
            ...

        def compute(self, ship_state, game_state):
            ...
            if mpc_result.feasible:
                self._mpc_infeasible_streak = 0
            else:
                self._mpc_infeasible_streak += 1

            if self._mpc_infeasible_streak >= N_INFEASIBLE_TRIGGER:
                turn_rate, thrust, fire, drop_mine = self.sacrifice.actions(
                    ship_state, game_state, asteroid_risks
                )
            else:
                self.sacrifice.reset()   # réarme la mine pour la prochaine activation
                turn_rate, thrust, fire = self.targeting.actions(
                    ship_state, game_state, asteroid_risks
                )
                drop_mine = False
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constantes partagées
# ---------------------------------------------------------------------------

BULLET_SPEED       = 800.0   # px/s
OMEGA_MAX          = 180.0   # degrés/s  (limite physique Kessler)
FIRE_CONE_DEG      =   4.0   # demi-angle de tir (degrés) — erreur max pour tirer
MIN_DIST_FIRE      =  10.0   # px — trop proche pour tirer
MAX_DIST_FIRE      = 650.0   # px — trop loin pour tirer avec précision
FIRE_COOLDOWN_S    =   0.10  # s  — cadence maximale

MIN_RISK_TO_TARGET = 0.05    # ignorer les astéroïdes en dessous de ce risque
TTC_REF            = 3.0     # s — TTC de référence pour la normalisation d'urgence
CHAIN_RADIUS       = 55.0    # px — épaisseur du couloir pour le kill-chain
CHAIN_MAX_DIST     = 700.0   # px — portée max du kill-chain

# Poids du score composite
W_RISK        = 2.0
W_TTC         = 1.5
W_KILL_EFF    = 1.2
W_FRAG        = 1.8   # pénalité si delta_risk > 0 (fragmentation défavorable)
W_DIST        = 0.4
W_SMALL       = 0.8   # bonus pour les petits astéroïdes (stoppe la croissance géométrique)

# Target lock — hysteresis
# Un challenger doit dépasser la cible verrouillée d'au moins ce ratio
# pour provoquer un switch. 1.0 = pas d'hysteresis, 1.3 = +30% requis.
SWITCH_RATIO  = 1.25


# ---------------------------------------------------------------------------
# Dataclasses de résultat (utiles pour le debug / Supervisor)
# ---------------------------------------------------------------------------

@dataclass
class TargetScore:
    """Score détaillé pour un astéroïde candidat."""
    asteroid_idx:   int
    risk:           float    # risque FIS individuel [0, 1]
    ttc:            float    # TTC (s)
    dist_surface:   float    # distance surface (px)
    lead_angle_deg: float    # heading bullet requis (degrés Kessler)
    angular_err:    float    # |heading_ship - lead_angle| (degrés)
    kill_chain:     int      # nb d'astéroïdes estimés détruits par ce bullet
    frag_delta:     float    # delta risque post-destruction
    raw_score:      float    # score composite
    in_cone:        bool     # True → tir immédiatement possible


@dataclass
class TargetingResult:
    """Résultat complet d'une frame TargetingController."""
    best_score:   Optional[TargetScore]
    all_scores:   List[TargetScore]   # triés par score décroissant
    fire_heading: float               # heading optimal pour viser
    should_fire:  bool
    fire_reason:  str                 # 'snap' | 'cooldown' | 'out_of_cone' | 'no_target'


# ---------------------------------------------------------------------------
# Géométrie
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    """Différence signée a − b dans (−180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def _toric_wrap(dx: float, dy: float, W: float, H: float) -> Tuple[float, float]:
    dx -= W * round(dx / W)
    dy -= H * round(dy / H)
    return dx, dy


def _lead_angle_kessler(
    shooter_xy: Tuple[float, float],
    target_xy:  Tuple[float, float],
    target_vel: Tuple[float, float],
    map_size:   Tuple[float, float],
    bullet_spd: float = BULLET_SPEED,
) -> Tuple[float, bool]:
    """
    Calcule le heading Kessler (degrés) à adopter pour intercepter la cible.

    Prend en compte le wrapping toroïdal : choisit l'image de la cible
    (parmi les 9 copies de la map) qui donne le temps d'interception le plus court.

    Retourne (heading_deg, ok).  ok=False si aucune solution réelle.
    """
    W, H   = map_size
    sx, sy = shooter_xy
    ax, ay = target_xy
    vx, vy = target_vel

    best_t   = math.inf
    best_hd  = 0.0
    found    = False

    for kx in (-1, 0, 1):
        for ky in (-1, 0, 1):
            rx = (ax + kx * W) - sx
            ry = (ay + ky * H) - sy

            Vb2 = bullet_spd ** 2
            v2  = vx*vx + vy*vy
            rv  = rx*vx + ry*vy
            r2  = rx*rx + ry*ry

            a_q = Vb2 - v2
            b_q = 2.0 * rv
            c_q = -r2

            if abs(a_q) < 1e-6:
                if abs(b_q) < 1e-6:
                    continue
                t_int = -c_q / b_q
            else:
                disc = b_q*b_q - 4*a_q*c_q
                if disc < 0:
                    continue
                sq = math.sqrt(disc)
                t1 = (-b_q + sq) / (2*a_q)
                t2 = (-b_q - sq) / (2*a_q)
                candidates = [t for t in (t1, t2) if t > 0]
                if not candidates:
                    continue
                t_int = min(candidates)

            if t_int < best_t:
                best_t = t_int
                ix = ax + kx * W + vx * t_int
                iy = ay + ky * H + vy * t_int
                # Convention Kessler : atan2(-dy, dx), y écran vers le bas
                best_hd = math.degrees(math.atan2(-(iy - sy), ix - sx)) % 360.0
                found   = True

    return best_hd, found


def _estimate_kill_chain(
    shooter_xy:     Tuple[float, float],
    heading_kessler: float,
    all_risks:      list,
    map_size:       Tuple[float, float],
) -> int:
    """
    Nombre d'astéroïdes sur la trajectoire du bullet (ligne droite).
    Projection perpendiculaire + seuil rayon+CHAIN_RADIUS.
    """
    W, H   = map_size
    sx, sy = shooter_xy

    # Direction du bullet en coords écran (y vers le bas)
    angle_rad = math.radians(heading_kessler)
    dx_dir    =  math.cos(angle_rad)
    dy_dir    = -math.sin(angle_rad)   # inversion y

    count = 0
    for ar in all_risks:
        ax, ay   = ar.position
        rdx, rdy = _toric_wrap(ax - sx, ay - sy, W, H)

        proj = rdx * dx_dir + rdy * dy_dir
        if proj < 0 or proj > CHAIN_MAX_DIST:
            continue

        perp = abs(rdx * (-dy_dir) + rdy * dx_dir)
        if perp <= ar.radius + CHAIN_RADIUS:
            count += 1

    return max(count, 0)


# ---------------------------------------------------------------------------
# Repulsion parameters
# ---------------------------------------------------------------------------

REPULSION_THRUST    = 80.0   # px/s² — poussée de répulsion (légère, < THRUST_MAX)
REPULSION_TTC_MAX   =  2.0   # s — en dessous de cette TTC, la répulsion s'active
REPULSION_TTC_FULL  =  0.5   # s — en dessous, répulsion à pleine puissance


def _repulsion_thrust(
    ship_pos:       Tuple[float, float],
    ship_heading:   float,               # degrés Kessler
    asteroid_risks: list,
    map_size:       Tuple[float, float],
) -> float:
    """
    Calcule une composante de poussée le long de l'axe du ship
    pour s'éloigner de l'astéroïde avec le TTC le plus court.

    La poussée est projetée sur le heading courant du ship :
      - positive (avant) si l'astéroïde est derrière
      - negative (arrière) si l'astéroïde est devant

    L'intensité est proportionnelle à l'urgence (interpolation
    linéaire entre REPULSION_TTC_MAX et REPULSION_TTC_FULL).

    Retourne une valeur dans [-REPULSION_THRUST, REPULSION_THRUST].
    """
    if not asteroid_risks:
        return 0.0

    # Astéroïde avec le TTC minimal
    closest = min(asteroid_risks, key=lambda ar: ar.tau)

    if closest.tau >= REPULSION_TTC_MAX:
        return 0.0

    W, H   = map_size
    sx, sy = ship_pos
    ax, ay = closest.position

    # Vecteur ship → astéroïde (toroïdal)
    dx, dy = _toric_wrap(ax - sx, ay - sy, W, H)

    # Direction de heading du ship en coords écran (y vers le bas)
    h_rad  = math.radians(ship_heading)
    hx     =  math.cos(h_rad)
    hy     = -math.sin(h_rad)   # inversion y Kessler

    # Projection de la direction vers l'astéroïde sur l'axe du ship
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    proj = (dx / dist) * hx + (dy / dist) * hy
    # proj > 0 → astéroïde devant → reculer (thrust négatif)
    # proj < 0 → astéroïde derrière → avancer (thrust positif)

    # Intensité selon l'urgence
    alpha = 1.0 - max(0.0, min(1.0,
        (closest.tau - REPULSION_TTC_FULL) / (REPULSION_TTC_MAX - REPULSION_TTC_FULL)
    ))
    thrust_cmd = -proj * alpha * REPULSION_THRUST

    return max(-REPULSION_THRUST, min(REPULSION_THRUST, thrust_cmd))


# ---------------------------------------------------------------------------
# Cooldown interne
# ---------------------------------------------------------------------------

class _FireGate:
    def __init__(self) -> None:
        self._last: float = -999.0

    def can_fire(self, now: float) -> bool:
        return (now - self._last) >= FIRE_COOLDOWN_S

    def register(self, now: float) -> None:
        self._last = now


# ---------------------------------------------------------------------------
# TargetingController — contrôleur pur destruction
# ---------------------------------------------------------------------------

class TargetingController:
    """
    Contrôleur autonome de visée et destruction d'astéroïdes.

    Commandes produites
    -------------------
    turn_rate : float  — degrés/s, sortie d'un PD sur l'erreur angulaire
    thrust    : float  — légère répulsion opposée à l'astéroïde min-TTC
    fire      : bool   — True si l'erreur angulaire ≤ FIRE_CONE_DEG

    Paramètres PD
    -------------
    Kp : gain proportionnel (deg/s par deg d'erreur).
         Kessler tourne à max 180°/s, donc Kp=5 → 180°/s dès 36° d'erreur.
    Kd : gain dérivé — amortit les oscillations en fin de convergence.
         Garder petit (< 1) pour ne pas freiner la rotation sur grands écarts.
    """

    def __init__(
        self,
        Kp: float = 5.0,
        Ki: float = 0.0,
        Kd: float = 0.3,
    ) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self._integral:   float = 0.0
        self._prev_error: Optional[float] = None
        self._prev_time:  Optional[float] = None
        self._fire_gate   = _FireGate()

        # ── Target lock — hysteresis pour éviter l'oscillation de cible ──
        self._locked_sig:   Optional[Tuple[int,int,int,int]] = None  # signature physique
        self._locked_score: float = 0.0   # score au moment du lock

    # ------------------------------------------------------------------
    # Interface principale (compatible Kessler)
    # ------------------------------------------------------------------

    def compute(
        self,
        ship_state,
        game_state,
        asteroid_risks: list,   # List[AsteroidRisk] depuis RiskField.compute_all()
        now: Optional[float] = None,
    ) -> Tuple[float, float, bool]:
        """
        Retourne (turn_rate, thrust, fire).

        Parameters
        ----------
        ship_state      : .position, .speed, .heading
        game_state      : .map_size
        asteroid_risks  : liste AsteroidRisk triée par risque décroissant
        now             : timestamp courant (s)
        """
        if now is None:
            now = time.perf_counter()

        if not asteroid_risks:
            self._reset_pid()
            return 0.0, 0.0, False

        map_size     = game_state.map_size
        ship_pos     = ship_state.position
        ship_heading = ship_state.heading

        # --- Sélection de la cible et heading optimal ---
        result = self._evaluate(ship_pos, ship_heading, asteroid_risks, map_size, now)

        if result.best_score is None:
            self._reset_pid()
            return 0.0, 0.0, False

        # --- PD sur l'erreur angulaire ---
        # error > 0 → cible à gauche (CCW) → turn_rate > 0 correct en Kessler
        error = _angle_diff(result.fire_heading, ship_heading)

        if self._prev_time is None:
            # Première frame : pas de dérivée fiable, terme P seulement
            derivative = 0.0
        else:
            dt = max(now - self._prev_time, 1e-4)
            derivative = (error - self._prev_error) / dt

        # Mise à jour état
        if self.Ki > 0:
            self._integral = max(-OMEGA_MAX, min(OMEGA_MAX,
                                 self._integral + self.Ki * error *
                                 (max(now - self._prev_time, 1e-4) if self._prev_time else 0.0)))

        self._prev_error = error
        self._prev_time  = now

        raw_cmd   = self.Kp * error + self._integral + self.Kd * derivative
        turn_rate = max(-OMEGA_MAX, min(OMEGA_MAX, raw_cmd))

        # --- Répulsion légère (axe heading uniquement — physique non holonomique) ---
        # thrust > 0 = avancer,  thrust < 0 = reculer
        thrust = _repulsion_thrust(ship_pos, ship_heading, asteroid_risks, map_size)

        fire = result.should_fire

        # Kessler attend (thrust, turn_rate, fire)
        return thrust, turn_rate, fire

    # ------------------------------------------------------------------
    # Scoring + sélection de cible
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        ship_pos:       Tuple[float, float],
        ship_heading:   float,
        asteroid_risks: list,
        map_size:       Tuple[float, float],
        now:            float,
    ) -> TargetingResult:
        W, H   = map_size
        scores: List[TargetScore] = []

        for ar in asteroid_risks:
            if ar.risk < MIN_RISK_TO_TARGET:
                continue

            ax, ay   = ar.position
            rdx, rdy = _toric_wrap(ax - ship_pos[0], ay - ship_pos[1], W, H)
            dist_c   = math.hypot(rdx, rdy)
            dist_s   = max(dist_c - ar.radius, 0.0)

            if dist_s < MIN_DIST_FIRE or dist_s > MAX_DIST_FIRE:
                continue

            lead_deg, ok = _lead_angle_kessler(
                ship_pos, ar.position, ar.velocity, map_size
            )
            if not ok:
                continue

            ang_err = abs(_angle_diff(lead_deg, ship_heading))
            in_cone = ang_err <= FIRE_CONE_DEG

            # --- Sous-scores ---
            ttc_score   = math.exp(-ar.tau / TTC_REF)
            dist_score  = 1.0 - min(dist_s / MAX_DIST_FIRE, 1.0)
            chain       = _estimate_kill_chain(ship_pos, lead_deg, asteroid_risks, map_size)
            frag_pen    = max(0.0, ar.delta_risk_if_destroyed)

            # Bonus petits astéroïdes — normalisé sur la plage réelle des rayons Kessler.
            # Petits ≈ 8 px, grands ≈ 40 px. On normalise entre SIZE_SMALL_MAX (bonus nul)
            # et SIZE_SMALL_MIN (bonus plein) pour couvrir toute la plage [0, 1].
            SIZE_SMALL_MIN = 8.0    # px — rayon minimum Kessler (bonus = 1.0)
            SIZE_SMALL_MAX = 40.0   # px — rayon maximum Kessler (bonus = 0.0)
            small_score = 1.0 - (ar.radius - SIZE_SMALL_MIN) / (SIZE_SMALL_MAX - SIZE_SMALL_MIN)
            small_score = max(0.0, min(1.0, small_score))

            # Facteur angulaire — ne descend jamais à zéro pour garder les cibles
            # hors-cône dans le classement (le ship doit s'y orienter).
            # Plafond à 1.0 dans le cône (ang_err ≤ FIRE_CONE_DEG), décroît ensuite.
            angle_factor = min(1.0, FIRE_CONE_DEG / max(ang_err, FIRE_CONE_DEG)) \
                           if ang_err > FIRE_CONE_DEG \
                           else 1.0
            # → 1.0 si dans le cône, ~0.36 à 11°, ~0.13 à 30°, jamais 0

            raw_score = (
                W_RISK    * ar.risk
                + W_TTC   * ttc_score
                + W_KILL_EFF * math.log1p(chain)
                + W_DIST  * dist_score
                + W_SMALL * small_score
                - W_FRAG  * frag_pen
            ) * angle_factor

            scores.append(TargetScore(
                asteroid_idx   = ar.asteroid_id,
                risk           = ar.risk,
                ttc            = ar.tau,
                dist_surface   = dist_s,
                lead_angle_deg = lead_deg,
                angular_err    = ang_err,
                kill_chain     = chain,
                frag_delta     = ar.delta_risk_if_destroyed,
                raw_score      = raw_score,
                in_cone        = in_cone,
            ))

        if not scores:
            return TargetingResult(
                best_score=None, all_scores=[],
                fire_heading=ship_heading,
                should_fire=False, fire_reason='no_target',
            )

        # ── Target lock avec hysteresis ───────────────────────────────────
        # On ne peut pas locker sur asteroid_id : c'est un index dans la liste
        # courante de Kessler, réindexée à chaque destruction. On utilise une
        # signature physique (position + vitesse arrondies) qui reste stable
        # tant que l'astéroïde n'est pas détruit ou fragmenté.

        def _sig(s: TargetScore) -> Tuple[int, int, int, int]:
            """Signature stable : position et vitesse quantifiées à 4 px/px/s."""
            ar = next((a for a in asteroid_risks if a.asteroid_id == s.asteroid_idx), None)
            if ar is None:
                return (0, 0, 0, 0)
            return (
                int(ar.position[0] / 4),
                int(ar.position[1] / 4),
                int(ar.velocity[0] / 4),
                int(ar.velocity[1] / 4),
            )

        all_sorted = sorted(scores, key=lambda s: s.raw_score, reverse=True)
        best_candidate = all_sorted[0]
        best_sig = _sig(best_candidate)

        # Retrouver la cible verrouillée par signature (pas par index)
        locked_current: Optional[TargetScore] = None
        if self._locked_sig is not None:
            for s in scores:
                if _sig(s) == self._locked_sig:
                    locked_current = s
                    break

        if locked_current is None:
            # Cible verrouillée absente (détruite, fragmentée, hors portée)
            # → switch immédiat, PID conservé (heading probablement encore bon)
            self._locked_sig   = best_sig
            self._locked_score = best_candidate.raw_score
            best = best_candidate

        elif best_sig == self._locked_sig:
            # La cible verrouillée est toujours la meilleure → on reste
            self._locked_score = locked_current.raw_score
            best = locked_current

        else:
            # Challenger présent : switch seulement s'il dépasse SWITCH_RATIO
            threshold = self._locked_score * SWITCH_RATIO
            if best_candidate.raw_score >= threshold:
                self._locked_sig   = best_sig
                self._locked_score = best_candidate.raw_score
                self._reset_pid()
                best = best_candidate
            else:
                # Hysteresis : garder la cible verrouillée
                self._locked_score = locked_current.raw_score
                best = locked_current

        # --- Décision de tir ---
        can_fire    = self._fire_gate.can_fire(now)
        should_fire = best.in_cone and can_fire

        if should_fire:
            self._fire_gate.register(now)
            reason = 'snap'
        elif not can_fire:
            reason = 'cooldown'
        elif not best.in_cone:
            reason = 'out_of_cone'
        else:
            reason = 'no_target'

        return TargetingResult(
            best_score   = best,
            all_scores   = all_sorted,
            fire_heading = best.lead_angle_deg,
            should_fire  = should_fire,
            fire_reason  = reason,
        )

    def _reset_pid(self) -> None:
        self._integral     = 0.0
        self._prev_error   = None
        self._prev_time    = None
        self._locked_sig   = None
        self._locked_score = 0.0


# ---------------------------------------------------------------------------
# SacrificeController — mode sacrificiel (MPC infaisable pendant N frames)
# ---------------------------------------------------------------------------

# Avance de pose de mine par rapport au TTC estimé (secondes).
# On pose la mine MINE_LEAD_S avant l'impact prédit pour laisser
# le temps au nuage de fragmentation de se déployer.
MINE_LEAD_S = 0.20


class SacrificeController:
    """
    Mode sacrificiel : activé par le Supervisor quand le MPC est infaisable
    pendant N frames consécutives (collision inévitable imminente).

    Stratégie
    ---------
    1. Continue de viser et tirer avec la même logique que TargetingController
       (destruction maximale pendant le temps restant).
    2. Pose une mine une seule fois par activation, au moment où :

           min(TTC) <= MINE_LEAD_S

       c'est-à-dire à MINE_LEAD_S secondes de l'impact prédit, pour maximiser
       la zone de fragmentation autour du ship au moment de la collision.

    Interface
    ---------
    actions() retourne un 4-tuple : (turn_rate, thrust, fire, drop_mine)
      - turn_rate : float degrés/s — PID heading hérité de TargetingController
      - thrust    : 0.0            — vitesse non commandée
      - fire      : bool           — tir si dans le cône
      - drop_mine : bool           — True une seule fois par activation

    reset() doit être appelé par le Supervisor quand le mode sacrificiel
    se termine (MPC redevient faisable) pour réarmer la mine.
    """

    def __init__(self, Kp: float = 6.0, Ki: float = 0.0, Kd: float = 0.4) -> None:
        # Délègue la logique de visée au TargetingController
        self._targeting   = TargetingController(Kp=Kp, Ki=Ki, Kd=Kd)
        self._mine_armed  = True    # True = mine pas encore posée cette activation
        self._mine_fired  = False   # True = mine posée, ne plus la poser

    # ------------------------------------------------------------------

    def compute(
        self,
        ship_state,
        game_state,
        asteroid_risks: list,          # List[AsteroidRisk]
        now: Optional[float] = None,
    ) -> Tuple[float, float, bool, bool]:
        """
        Retourne (turn_rate, thrust, fire, drop_mine).

        Parameters
        ----------
        ship_state      : .position, .speed, .heading
        game_state      : .map_size
        asteroid_risks  : List[AsteroidRisk] depuis RiskField.compute_all()
        now             : timestamp courant (s) ; si None, time.perf_counter()
        """
        if now is None:
            now = time.perf_counter()

        # --- Visée + tir hérités du TargetingController ---
        turn_rate, thrust, fire = self._targeting.compute(
            ship_state, game_state, asteroid_risks, now=now
        )

        # --- Décision de pose de mine ---
        drop_mine = self._should_drop_mine(asteroid_risks)

        return turn_rate, thrust, fire, drop_mine

    # ------------------------------------------------------------------

    def _should_drop_mine(self, asteroid_risks: list) -> bool:
        """
        Retourne True une seule fois par activation, quand min(TTC) <= MINE_LEAD_S.

        Logique
        -------
        - Si la mine a déjà été posée (_mine_fired), retourne False.
        - Si min(TTC) > MINE_LEAD_S, on attend encore (impact pas assez proche).
        - Dès que min(TTC) <= MINE_LEAD_S, on pose la mine et on arme le verrou.
        """
        if self._mine_fired or not self._mine_armed:
            return False

        if not asteroid_risks:
            return False

        min_ttc = min(ar.tau for ar in asteroid_risks)

        if min_ttc <= MINE_LEAD_S:
            self._mine_fired = True   # verrou : une seule mine par activation
            return True

        return False

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Réarme le contrôleur pour la prochaine activation du mode sacrificiel.
        Appeler depuis le Supervisor quand le MPC redevient faisable.
        """
        self._mine_armed  = True
        self._mine_fired  = False
        self._targeting._reset_pid()