from __future__ import annotations

"""
mpc_director.py  —  CasADi/IPOPT MPC for XFC Kessler  (convergence fix)
=========================================================================

Fixes de convergence vs version précédente
-------------------------------------------
A. [CRITIQUE]  Suppression de hessian_approximation='limited-memory'
               L-BFGS est incompatible avec des contraintes d'égalité denses
               (4×N équations de dynamique).  IPOPT avec Hessien exact converge
               en 5-15 itérations là où L-BFGS diverge sur 100+.

B. [CRITIQUE]  FAR_AWAY 1e5 → masquage par risk=0 UNIQUEMENT
               Une position à 1e5 px crée un gradient ‖diff‖ ~ 1e10 qui
               corrompt le conditionnement du Hessien.  Les astéroïdes fictifs
               sont maintenant placés à une position proche (centre de la map)
               MAIS avec risk=0, de sorte que leur contribution à J est nulle
               sans polluer les dérivées.

C. [CRITIQUE]  Scaling explicite des variables
               px/py  ~ 800 px,  s ~ 200 px/s,  theta ~ π rad,  uT ~ 480 px/s²
               → ordres de grandeur très différents → mauvais conditionnement.
               On normalise dans le NLP (variables adimensionnées) et on
               redimensionne à la sortie.

D. [IMPORTANT] Reformulation de fmax(0, x)² → softplus lisse
               ca.fmax crée une non-dérivabilité en 0 qui gêne IPOPT.
               Remplacé par une approximation C∞ : sp(x) = log(1+exp(x/ε))·ε

Paramètres gardés des versions précédentes
-------------------------------------------
- NLP taille fixe (N_AST_MAX) compilé une seule fois
- Vectorisation MX sur les astéroïdes
- Pénalités effort hors boucle i
- Warm-start par décalage temporel
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from risk_field import RiskField

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZON_STEPS    = 8
DT               = 0.12        # s/step  →  ~1 s d'horizon

THRUST_MAX       =  480.0      # px/s²
THRUST_MIN       = -480.0
OMEGA_MAX        =  180.0      # deg/s
DRAG             =   80.0      # px/s²

LAMBDA_D         = 0.25
LAMBDA_D_RESPAWN = 0.0

W_THRUST     = 1e-5
W_OMEGA      = 5e-3
W_SLOW       = 1e-2            # pénalise s < S_MIN_TARGET
S_MIN_TARGET = 80.0            # px/s

DANGER_SCALE    = 120.0        # px
COLLISION_SOFT  =  25.0        # px
COLLISION_W     =   8.0

MAX_SOLVER_MS              = 25.0
IPOPT_MAX_ITER             = 50
N_AST_MAX                  = 50
INFEASIBILITY_COST_CEILING = 2000.0

# Scaling (pour normaliser les variables dans le NLP)
S_PX    = 800.0   # échelle positions  (px)
S_SPD   = 200.0   # échelle vitesse    (px/s)
S_TH    = math.pi # échelle angle      (rad)
S_UT    = 480.0   # échelle thrust     (px/s²)
S_OM    = math.radians(OMEGA_MAX)  # échelle omega (rad/s)

# Softplus lissage (remplace fmax(0,x)²)
SP_EPS  = 2.0     # px — demi-largeur de transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softplus_sq(expr, eps: float = SP_EPS):
    """Approximation C∞ de max(0, expr)²  utilisable avec AD CasADi."""
    # sp(x) = eps * log(1 + exp(x/eps))  →  dérivable partout
    # on veut sp(x)² avec x = (COLLISION_SOFT - d_sur)
    sp = eps * ca.log1p(ca.exp(expr / eps))
    return sp ** 2


def _step_np(
    px: float, py: float, s: float, theta_rad: float,
    u_T: float, omega_rad: float,
    map_W: float, map_H: float,
) -> Tuple[float, float, float, float]:
    """Pas de dynamique numpy (theta en RADIANS ici, contrairement à l'ancienne version)."""
    drag   = DRAG * math.copysign(1.0, s) if abs(s) > 0.5 else 0.0
    new_s  = s + (u_T - drag) * DT
    new_px = (px + s * math.cos(theta_rad) * DT) % map_W
    new_py = (py + s * math.sin(theta_rad) * DT) % map_H
    new_th = theta_rad + omega_rad * DT      # pas de wrap pour le NLP
    return new_px, new_py, new_s, new_th


# ---------------------------------------------------------------------------
# CasADi NLP  —  taille fixe, Hessien exact, variables scalées
# ---------------------------------------------------------------------------

class _CasADiSolver:

    def __init__(self, map_size: Tuple[float, float]) -> None:
        self.map_W, self.map_H = map_size
        self._solver: Optional[object] = None
        self._n_xw   = 0
        self._n_uw   = 0
        self._lbw: Optional[list] = None
        self._ubw: Optional[list] = None
        self._u_prev: Optional[np.ndarray] = None  # (2, N) — unscaled rad/s, px/s²

    # ------------------------------------------------------------------
    def _build(self, lambda_d: float) -> None:
        """
        Construit et compile le NLP symbolique.
        Variables internes NORMALISÉES (ordre ~1) pour un bon conditionnement.

        État normalisé  xn = [px/S_PX, py/S_PX, s/S_SPD, th/S_TH]
        Contrôle normalisé un = [uT/S_UT, om/S_OM]
        """
        N = HORIZON_STEPS
        n = N_AST_MAX

        # Variables de décision normalisées
        XN = ca.MX.sym('XN', 4, N + 1)   # état normalisé
        UN = ca.MX.sym('UN', 2, N)        # contrôle normalisé

        # Paramètres (en unités physiques — convertis dans le NLP)
        x0_p      = ca.MX.sym('x0',      4)        # [px, py, s, th_rad]
        ast_p0_p  = ca.MX.sym('ast_p0',  2 * n)
        ast_v_p   = ca.MX.sym('ast_v',   2 * n)
        ast_r_p   = ca.MX.sym('ast_r',   n)
        ast_rsk_p = ca.MX.sym('ast_rsk', n)
        p_all     = ca.vertcat(x0_p, ast_p0_p, ast_v_p, ast_r_p, ast_rsk_p)

        # ------ Contraintes d'égalité (dynamique normalisée) ------
        g, lbg, ubg = [], [], []

        # Condition initiale normalisée
        x0_norm = ca.vertcat(
            x0_p[0] / S_PX,
            x0_p[1] / S_PX,
            x0_p[2] / S_SPD,
            x0_p[3] / S_TH,
        )
        g   += [XN[:, 0] - x0_norm]
        lbg += [0.0] * 4
        ubg += [0.0] * 4

        for k in range(N):
            # Dénormalisation pour la dynamique physique
            px_k = XN[0, k] * S_PX
            py_k = XN[1, k] * S_PX
            s_k  = XN[2, k] * S_SPD
            th_k = XN[3, k] * S_TH
            uT_k = UN[0, k] * S_UT
            om_k = UN[1, k] * S_OM

            drag_k = DRAG * ca.tanh(s_k / 5.0)   # tanh lisse, saturation douce

            px_next = px_k + s_k * ca.cos(th_k) * DT
            py_next = py_k + s_k * ca.sin(th_k) * DT
            s_next  = s_k  + (uT_k - drag_k) * DT
            th_next = th_k + om_k * DT

            g += [
                XN[0, k+1] - px_next / S_PX,
                XN[1, k+1] - py_next / S_PX,
                XN[2, k+1] - s_next  / S_SPD,
                XN[3, k+1] - th_next / S_TH,
            ]
            lbg += [0.0] * 4
            ubg += [0.0] * 4

        # ------ Objectif (vectorisé, unités physiques) ------
        J = ca.MX(0)
        ast_p0_mat = ca.reshape(ast_p0_p, 2, n)
        ast_v_mat  = ca.reshape(ast_v_p,  2, n)

        for k in range(N + 1):
            t_k  = k * DT

            # Dénormalisation état
            px_k = XN[0, k] * S_PX
            py_k = XN[1, k] * S_PX
            th_k = XN[3, k] * S_TH
            s_k  = XN[2, k] * S_SPD

            # Contrôle (0 au dernier step)
            if k < N:
                uT_k = UN[0, k] * S_UT
                om_k = UN[1, k] * S_OM
            else:
                uT_k = ca.MX(0)
                om_k = ca.MX(0)

            # Positions astéroïdes à t_k  — (2, n)
            ast_pos_k  = ast_p0_mat + ast_v_mat * t_k
            ship_pos_k = ca.vertcat(px_k, py_k)

            diff   = ca.repmat(ship_pos_k, 1, n) - ast_pos_k   # (2, n)
            dist   = ca.sqrt(diff[0, :] ** 2 + diff[1, :] ** 2 + 1.0)  # +1 px² évite grad infini
            d_sur  = ca.fmax(dist - ast_r_p.T, 0.0)            # (1, n)

            # [FIX B] risk=0 sur les slots fictifs → contribution nulle même si pos ~ centre
            danger = ast_rsk_p.T * ca.exp(-d_sur / DANGER_SCALE)
            J = J + ca.sum2(danger)

            # [FIX D] Collision soft avec softplus (C∞) au lieu de fmax²
            coll_arg = (COLLISION_SOFT - d_sur) / SP_EPS   # >0 quand trop proche
            coll_sp  = SP_EPS * ca.log1p(ca.exp(coll_arg))
            J = J + COLLISION_W * ca.sum2(coll_sp ** 2) / (COLLISION_SOFT ** 2)

            # Destruction term
            if lambda_d > 0:
                hx     =  ca.cos(th_k)
                hy     = -ca.sin(th_k)
                to_ast = (ast_pos_k - ca.repmat(ship_pos_k, 1, n)) / ca.repmat(dist, 2, 1)
                align  = ca.fmax(0.0, hx * to_ast[0, :] + hy * to_ast[1, :])
                J = J - lambda_d * ca.sum2(ast_rsk_p.T * align * ca.exp(-d_sur / DANGER_SCALE))

            # Pénalités effort — hors boucle i, normalisées
            if k < N:
                J = J + W_THRUST * UN[0, k] ** 2
                J = J + W_OMEGA  * UN[1, k] ** 2
                J = J + W_SLOW   * ca.fmax(0.0, S_MIN_TARGET / S_SPD - XN[2, k]) ** 2

        # ------ Bornes sur UN (normalisées → [-1, 1]) ------
        N_xw = 4 * (N + 1)
        N_uw = 2 * N
        wN   = ca.vertcat(ca.reshape(XN, -1, 1), ca.reshape(UN, -1, 1))

        lbw = ([-1e6] * N_xw + [THRUST_MIN / S_UT, -1.0] * N)
        ubw = ([ 1e6] * N_xw + [THRUST_MAX / S_UT,  1.0] * N)

        # ------ Options IPOPT ------
        # [FIX A]  PAS de limited-memory — Hessien exact (ma57/mumps)
        nlp  = {'x': wN, 'f': J, 'g': ca.vertcat(*g), 'p': p_all}
        opts = {
            'ipopt.max_iter':               IPOPT_MAX_ITER,
            'ipopt.tol':                    1e-3,
            'ipopt.acceptable_tol':         5e-3,
            'ipopt.acceptable_iter':        3,
            # PAS de hessian_approximation  → Hessien exact  [FIX A]
            'ipopt.warm_start_init_point':  'yes',
            'ipopt.warm_start_bound_push':  1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.nlp_scaling_method':     'gradient-based',  # scaling auto IPOPT
            'ipopt.print_level':            0,
            'ipopt.sb':                     'yes',
            'print_time':                   False,
        }
        self._solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._n_xw   = N_xw
        self._n_uw   = N_uw
        self._lbw    = lbw
        self._ubw    = ubw
        self._u_prev = None

    # ------------------------------------------------------------------
    def _pad_asteroids(self, asteroid_risks: list, map_cx: float, map_cy: float):
        """
        Remplit jusqu'à N_AST_MAX.
        [FIX B] Les slots vides sont au CENTRE de la map avec risk=0
                (gradient faible au lieu de 1e10).
        """
        n   = N_AST_MAX
        p0  = np.array([map_cx, map_cy] * n, dtype=float)
        vel = np.zeros(2 * n)
        rad = np.ones(n) * 10.0
        rsk = np.zeros(n)

        for i, ar in enumerate(asteroid_risks[:n]):
            p0[2*i:2*i+2]  = ar.position
            vel[2*i:2*i+2] = ar.velocity
            rad[i]          = ar.radius
            rsk[i]          = ar.risk
        return p0, vel, rad, rsk

    # ------------------------------------------------------------------
    def solve(
        self,
        ship_pos:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,   # degrés Kessler → converti en rad ici
        asteroid_risks: list,
        r_global:       float,
        lambda_d:       float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Retourne (u_opt shape (2,N), cost) ou (None, inf).
        u_opt[0,:] = thrust  (px/s²,  non normalisé)
        u_opt[1,:] = omega   (rad/s,  non normalisé)
        """
        if len(asteroid_risks) == 0:
            return None, 0.0

        if self._solver is None:
            self._build(lambda_d)

        N = HORIZON_STEPS
        th_rad = math.radians(ship_heading)
        x0_val = np.array([ship_pos[0], ship_pos[1], ship_speed, th_rad])

        map_cx = self.map_W / 2.0
        map_cy = self.map_H / 2.0
        p0, vel, rad, rsk = self._pad_asteroids(asteroid_risks, map_cx, map_cy)
        p_val = np.concatenate([x0_val, p0, vel, rad, rsk])

        # Warm-start normalisé
        if self._u_prev is not None and self._u_prev.shape == (2, N):
            u_init_phys = np.hstack([self._u_prev[:, 1:], self._u_prev[:, -1:]])
        else:
            u_init_phys = np.zeros((2, N))

        u_init_norm = np.zeros_like(u_init_phys)
        u_init_norm[0, :] = u_init_phys[0, :] / S_UT
        u_init_norm[1, :] = u_init_phys[1, :] / S_OM

        # Trajectoire initiale normalisée
        XN_init = np.zeros((4, N + 1))
        XN_init[:, 0] = [x0_val[0]/S_PX, x0_val[1]/S_PX,
                          x0_val[2]/S_SPD, x0_val[3]/S_TH]
        for k in range(N):
            px_k = XN_init[0, k] * S_PX
            py_k = XN_init[1, k] * S_PX
            s_k  = XN_init[2, k] * S_SPD
            th_k = XN_init[3, k] * S_TH
            uT_k = u_init_norm[0, k] * S_UT
            om_k = u_init_norm[1, k] * S_OM
            px_n, py_n, s_n, th_n = _step_np(px_k, py_k, s_k, th_k,
                                               uT_k, om_k,
                                               self.map_W, self.map_H)
            XN_init[0, k+1] = px_n / S_PX
            XN_init[1, k+1] = py_n / S_PX
            XN_init[2, k+1] = s_n  / S_SPD
            XN_init[3, k+1] = th_n / S_TH

        w0 = np.concatenate([
            XN_init.flatten(order='F'),
            u_init_norm.flatten(order='F'),
        ])

        try:
            sol   = self._solver(
                x0=w0, p=p_val,
                lbx=self._lbw, ubx=self._ubw,
                lbg=0.0,       ubg=0.0,
            )
            w_opt = np.array(sol['x']).flatten()
            cost  = float(sol['f'])
            stats = self._solver.stats()
            ok    = stats.get('success', False) or stats.get(
                'return_status', '') in ('Solve_Succeeded',
                                         'Solved_To_Acceptable_Level')
            if not ok:
                self._u_prev = None
                return None, math.inf

            # Dénormalisation des contrôles
            u_opt_norm   = w_opt[self._n_xw:].reshape(2, N, order='F')
            u_opt_phys   = np.zeros_like(u_opt_norm)
            u_opt_phys[0, :] = u_opt_norm[0, :] * S_UT
            u_opt_phys[1, :] = u_opt_norm[1, :] * S_OM
            self._u_prev = u_opt_phys
            return u_opt_phys, cost

        except Exception as e:
            self._u_prev = None
            return None, math.inf


# ---------------------------------------------------------------------------
# Contrôleur public
# ---------------------------------------------------------------------------

class MPCController:

    def __init__(self, prefer_casadi: bool = True) -> None:
        self._use_casadi = CASADI_AVAILABLE and prefer_casadi
        self._casadi_slv: Optional[_CasADiSolver] = None
        self._map_size:   Optional[Tuple[float, float]] = None
        self.risk_field:  Optional[RiskField]  = None

    def _ensure_solver(self, map_size: Tuple[float, float]) -> None:
        if self._casadi_slv is None or self._map_size != map_size:
            self._map_size   = map_size
            self.risk_field  = RiskField(map_size=map_size)
            self._casadi_slv = _CasADiSolver(map_size) if self._use_casadi else None

    def compute(
        self,
        ship_state,
        game_state,
        tau_min:     float = 0.0,
        mode:        str   = 'active',
        repulse_dir: float = 0.0,
    ) -> 'MPCResult':

        map_size = game_state.map_size
        self._ensure_solver(map_size)

        asteroid_risks = self.risk_field.compute_all(
            ship_state.position, ship_state.velocity, game_state.asteroids
        )
        r_global = self.risk_field.aggregate(asteroid_risks)

        ship_pos     = ship_state.position
        ship_speed   = ship_state.speed
        ship_heading = ship_state.heading
        lambda_d     = LAMBDA_D if mode == 'active' else LAMBDA_D_RESPAWN

        t0 = time.perf_counter()

        if self._use_casadi and len(asteroid_risks) > 0:
            u_opt, cost_cas = self._casadi_slv.solve(
                ship_pos, ship_speed, ship_heading,
                asteroid_risks, r_global, lambda_d,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if u_opt is not None:
                uT_0   = float(np.clip(u_opt[0, 0], THRUST_MIN, THRUST_MAX))
                om_0   = float(np.clip(u_opt[1, 0], -S_OM, S_OM))
                om_deg = math.degrees(om_0)
                theta_star = (ship_heading + om_deg * DT) % 360.0
                feasible   = cost_cas < INFEASIBILITY_COST_CEILING

                return MPCResult(
                    feasible         = feasible,
                    theta_star       = theta_star,
                    thrust           = uT_0,
                    turn_rate        = om_deg,
                    best_cost        = cost_cas,
                    fallback_heading = repulse_dir,
                    solver_used      = 'casadi',
                    solve_time_ms    = elapsed,
                )

            return MPCResult(
                feasible         = False,
                theta_star       = 0.0,
                thrust           = 0.0,
                turn_rate        = 0.0,
                best_cost        = math.inf,
                fallback_heading = repulse_dir,
                solver_used      = 'casadi_failed',
                solve_time_ms    = elapsed,
            )

        return MPCResult(
            feasible         = True,
            theta_star       = ship_heading,
            thrust           = 0.0,
            turn_rate        = 0.0,
            best_cost        = 0.0,
            fallback_heading = repulse_dir,
            solver_used      = 'none',
            solve_time_ms    = 0.0,
        )

    def viability_check(
        self,
        ship_pos:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,          # degrés Kessler
        asteroid_risks: list,
        tau_safe:       float = 0.5,
        n_probes:       int   = 8,
    ) -> bool:
        if self._map_size is None:
            return True
        W, H  = self._map_size
        steps = int(3.0 / DT)
        for j in range(n_probes):
            th_rad = math.radians(360.0 * j / n_probes)
            px, py, s, th = ship_pos[0], ship_pos[1], ship_speed, th_rad
            safe = True
            for k in range(steps):
                t_k = k * DT
                px, py, s, th = _step_np(px, py, s, th, THRUST_MIN, 0.0, W, H)
                for ar in asteroid_risks:
                    ax = (ar.position[0] + ar.velocity[0] * t_k) % W
                    ay = (ar.position[1] + ar.velocity[1] * t_k) % H
                    dx, dy = px - ax, py - ay
                    dx -= W * round(dx / W)
                    dy -= H * round(dy / H)
                    if max(math.hypot(dx, dy) - ar.radius, 0.0) < tau_safe * 50:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                return True
        return False

    @property
    def solver_backend(self) -> str:
        return 'casadi+ipopt' if self._use_casadi else 'none'


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MPCResult:
    feasible:         bool
    theta_star:       float
    thrust:           float
    turn_rate:        float
    best_cost:        float
    fallback_heading: float
    solver_used:      str
    solve_time_ms:    float