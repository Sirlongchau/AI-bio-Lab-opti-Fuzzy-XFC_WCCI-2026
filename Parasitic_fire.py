"""
parasitic_fire.py  —  Tir parasite pour MPCController / XFC Kessler
=====================================================================

Principe
--------
Le MPC optimise la trajectoire d'esquive sans tirer (trop lourd à 60 fps).
Ce module analyse la trajectoire optimale *déjà calculée* (u_opt) et
décide de tirer si un astéroïde peut être touché sans dégradation
significative de la trajectoire.

Trois niveaux de décision, du moins coûteux au plus coûteux :

  1. SNAP SHOT  — astéroïde dans le cône de tir immédiat (heading actuel)
                  → tir instant, coût O(n_ast)

  2. LEAD SHOT  — astéroïde hittable dans les prochains steps de la
                  trajectoire optimale (compensation de vitesse)
                  → tir immédiat si le bullet peut l'atteindre en temps T_k
                  → coût O(N × n_ast)

  3. OPPORTUNISTIC — astéroïde hittable si on applique une légère
                  correction angulaire (< FIRE_CONE_DEG) au premier step
                  → décision de tourner + tirer si le risque est faible
                  → coût O(n_ast), optionnel

Seuls 1 et 2 sont activés par défaut (microsecondaire).

Intégration dans MPCResult
--------------------------
La fonction `evaluate_fire` reçoit le MPCResult et les asteroid_risks
(déjà calculés par risk_field) et retourne un FireDecision.

Utilisation dans le contrôleur principal :
    result  = mpc.compute(...)
    fire_d  = evaluate_fire(result, asteroid_risks, ship_state)
    fire    = fire_d.should_fire
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paramètres du tir
# ---------------------------------------------------------------------------

BULLET_SPEED      = 800.0    # px/s  — à ajuster selon Kessler
FIRE_CONE_DEG     =  3.0     # demi-angle du cône de tir "snap" (degrés)
LEAD_CONE_DEG     = 6.0     # demi-angle élargi pour lead shot
MIN_RISK_TO_FIRE  =  0.0    # ne pas gaspiller des bullets sur < 10 % risk
MAX_DIST_FIRE     = 600.0    # px — au-delà, trop imprécis
MIN_DIST_FIRE     = 10.0     # px — trop proche, risque de rater + dangereux
FIRE_COOLDOWN_S   =  0.1    # s  — cadence max (évite le spam)

# Paramètres de la trajectoire (doivent correspondre à mpc_director.py)
HORIZON_STEPS = 8
DT            = 0.12         # s/step


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FireDecision:
    should_fire:  bool  = False
    reason:       str   = 'no_shot'       # 'snap', 'lead', 'opportunistic'
    target_idx:   int   = -1              # index dans asteroid_risks
    lead_angle:   float = 0.0            # correction angulaire suggérée (deg)
    confidence:   float = 0.0           # [0, 1]


@dataclass
class FireState:
    """État persistant entre les frames (cooldown, historique)."""
    last_fire_time: float = -999.0       # timestamp (s) du dernier tir

    def can_fire(self, now: float) -> bool:
        return (now - self.last_fire_time) >= FIRE_COOLDOWN_S

    def register_fire(self, now: float) -> None:
        self.last_fire_time = now


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _angle_diff_deg(a: float, b: float) -> float:
    """Différence angulaire signée a-b dans [-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def _kessler_to_math_rad(heading_deg: float) -> float:
    """Convertit le heading Kessler (degrés, sens trigo ?) en rad mathématique."""
    return math.radians(heading_deg)


def _predict_ship_trajectory(
    ship_pos:     Tuple[float, float],
    ship_speed:   float,
    ship_heading: float,       # degrés Kessler
    u_opt:        Optional[np.ndarray],  # (2, N) physique : [thrust, omega_rad/s]
) -> List[Tuple[float, float, float]]:
    """
    Retourne la liste des (x, y, heading_deg) du ship aux steps 0..N
    en simulant la trajectoire optimale (ou trajectoire libre si u_opt=None).

    Note : simulation dans le repère absolu pour simplifier le calcul
    de l'interception bullet.
    """
    N     = HORIZON_STEPS
    DRAG  = 80.0        # px/s²  (même que mpc_director.py)

    traj: List[Tuple[float, float, float]] = []

    x, y      = ship_pos
    s         = ship_speed
    th_rad    = _kessler_to_math_rad(ship_heading)

    traj.append((x, y, ship_heading))

    for k in range(N):
        if u_opt is not None and k < u_opt.shape[1]:
            uT = float(u_opt[0, k])
            om = float(u_opt[1, k])     # rad/s
        else:
            uT, om = 0.0, 0.0

        drag = DRAG * math.copysign(1.0, s) if abs(s) > 0.5 else 0.0
        s    = s + (uT - drag) * DT
        x    = x + s * math.cos(th_rad) * DT
        y    = y + s * math.sin(th_rad) * DT
        th_rad = th_rad + om * DT

        traj.append((x, y, math.degrees(th_rad) % 360.0))

    return traj


def _compute_lead_angle(
    shooter_pos: Tuple[float, float],
    target_pos:  Tuple[float, float],
    target_vel:  Tuple[float, float],
    bullet_speed: float,
) -> Tuple[float, bool]:
    """
    Calcule l'angle de tir avec compensation de vitesse (lead shot).
    Résout l'équation quadratique d'interception.

    Retourne (angle_deg, success).
    angle_deg : heading mathématique (degrés) vers le point d'interception.
    """
    rx = target_pos[0] - shooter_pos[0]
    ry = target_pos[1] - shooter_pos[1]
    vx, vy = target_vel

    # ||v||² t² - 2(r·v) t - ||r||² + Vb² t² = 0
    # → (Vb² - ||v||²) t² + 2(r·v) t - ||r||² = 0
    Vb2 = bullet_speed ** 2
    v2  = vx*vx + vy*vy
    rv  = rx*vx + ry*vy
    r2  = rx*rx + ry*ry

    a = Vb2 - v2
    b = 2.0 * rv
    c = -r2

    if abs(a) < 1e-6:
        # Cas limite : bullet speed ≈ target speed
        if abs(b) < 1e-6:
            return 0.0, False
        t_int = -c / b
    else:
        disc = b*b - 4*a*c
        if disc < 0:
            return 0.0, False
        t1 = (-b + math.sqrt(disc)) / (2*a)
        t2 = (-b - math.sqrt(disc)) / (2*a)
        # Choisir la solution positive la plus petite
        candidates = [t for t in (t1, t2) if t > 0]
        if not candidates:
            return 0.0, False
        t_int = min(candidates)

    # Point d'interception
    ix = target_pos[0] + vx * t_int
    iy = target_pos[1] + vy * t_int

    angle_rad = math.atan2(iy - shooter_pos[1], ix - shooter_pos[0])
    return math.degrees(angle_rad) % 360.0, True


# ---------------------------------------------------------------------------
# Évaluateur principal
# ---------------------------------------------------------------------------

def evaluate_fire(
    mpc_result,            # MPCResult (duck-typed, pas d'import circulaire)
    asteroid_risks: list,  # liste de AsteroidRisk depuis risk_field.py
    ship_state,            # duck-typed : .position, .speed, .heading
    fire_state:   FireState,
    now:          float,   # timestamp courant en secondes
    map_size:     Tuple[float, float] = (1000.0, 800.0),
    u_opt:        Optional[np.ndarray] = None,   # (2, N) si disponible
    opportunistic: bool = False,
) -> FireDecision:
    """
    Évalue si le ship doit tirer maintenant.

    Paramètres
    ----------
    mpc_result     : résultat du MPC (utilisé pour theta_star, feasible)
    asteroid_risks : liste triée par risque décroissant
    ship_state     : état du ship (.position, .speed, .heading)
    fire_state     : état persistant (cooldown)
    now            : temps courant (s) pour cooldown
    map_size       : (W, H) de la map
    u_opt          : contrôles optimaux (2, N) en unités physiques
    opportunistic  : activer le mode opportuniste (légère correction d'angle)
    """
    if not asteroid_risks:
        return FireDecision(reason='no_asteroids')

    if not fire_state.can_fire(now):
        return FireDecision(reason='cooldown')

    ship_pos     = ship_state.position
    ship_heading = ship_state.heading   # degrés Kessler
    ship_speed   = ship_state.speed
    W, H         = map_size

    # Trajectoire optimale dans le repère absolu
    traj = _predict_ship_trajectory(ship_pos, ship_speed, ship_heading, u_opt)

    best_decision = FireDecision()
    best_priority = -1.0  # risque × confiance
    max_risk = max((ar.risk for ar in asteroid_risks), default=0.0)
    effective_min_risk = min(MIN_RISK_TO_FIRE, max_risk * 0.3)
    for idx, ar in enumerate(asteroid_risks):
        if ar.risk < effective_min_risk:
            continue

        ax, ay = ar.position
        vx, vy = ar.velocity
        radius = ar.radius

        # ---------------------------------------------------------------
        # Parcourir les steps de la trajectoire optimale
        # ---------------------------------------------------------------
        for step, (sx, sy, sh_deg) in enumerate(traj):
            t_k = step * DT

            # Position de l'astéroïde à t_k (sans modulo pour le NLP,
            # mais ici on reste en absolu + correction toroïdale ponctuelle)
            ax_k = ax + vx * t_k
            ay_k = ay + vy * t_k

            # Distance toroïdale
            dx = ax_k - sx;  dx -= W * round(dx / W)
            dy = ay_k - sy;  dy -= H * round(dy / H)
            dist = math.hypot(dx, dy)

            raw_dist = max(dist - radius, 0.0)
            if raw_dist < MIN_DIST_FIRE or raw_dist > MAX_DIST_FIRE:
                continue

            # ------ Lead angle depuis cette position du ship ------
            lead_deg, lead_ok = _compute_lead_angle(
                (sx, sy), (ax_k, ay_k), (vx, vy), BULLET_SPEED
            )
            if not lead_ok:
                continue

            # Écart angulaire entre heading du ship à ce step et lead angle
            angular_err = abs(_angle_diff_deg(lead_deg, sh_deg))

            if step == 0:
                cone = FIRE_CONE_DEG
            else:
                cone = LEAD_CONE_DEG

            if angular_err <= cone:
                # Tir possible depuis ce step
                confidence  = (1.0 - angular_err / cone) * ar.risk
                if step == 0:
                    reason = 'snap'
                    bonus  = 1.2    # privilégier le tir immédiat
                else:
                    reason = 'lead'
                    bonus  = 1.0

                priority = confidence * bonus
                if priority > best_priority:
                    best_priority = priority
                    best_decision = FireDecision(
                        should_fire = True,
                        reason      = reason,
                        target_idx  = idx,
                        lead_angle  = lead_deg,
                        confidence  = confidence,
                    )

        # ---------------------------------------------------------------
        # Mode opportuniste : légère correction angulaire au step 0
        # ---------------------------------------------------------------
        if opportunistic and not best_decision.should_fire:
            lead_deg, lead_ok = _compute_lead_angle(
                traj[0][:2], (ax, ay), (vx, vy), BULLET_SPEED
            )
            if lead_ok:
                angular_err = abs(_angle_diff_deg(lead_deg, ship_heading))
                if FIRE_CONE_DEG < angular_err <= 2 * FIRE_CONE_DEG:
                    # Proposer une légère correction sans forcer le tir
                    confidence = (1.0 - angular_err / (2 * FIRE_CONE_DEG)) * ar.risk
                    if confidence > effective_min_risk and confidence > best_priority:
                        best_priority = confidence
                        best_decision = FireDecision(
                            should_fire = True,
                            reason      = 'opportunistic',
                            target_idx  = idx,
                            lead_angle  = lead_deg,
                            confidence  = confidence,
                        )

    if best_decision.should_fire:
        fire_state.register_fire(now)

    return best_decision


# ---------------------------------------------------------------------------
# Intégration dans MPCResult — patch monkey
# ---------------------------------------------------------------------------

def patch_mpc_result_with_fire(mpc_result, fire_decision: FireDecision):
    """
    Ajoute fire_decision au MPCResult existant sans modifier sa dataclass.
    Utiliser comme :
        result.fire_decision = patch_mpc_result_with_fire(result, fire_d)
    Ou simplement stocker fire_d à côté.
    """
    mpc_result.fire_decision = fire_decision
    return fire_decision