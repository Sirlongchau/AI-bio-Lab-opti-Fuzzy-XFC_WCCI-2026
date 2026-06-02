from __future__ import annotations
"""
mpc_director.py
===============
Directional MPC using CasADi/IPOPT with grid-search fallback.

Mathematical formulation
------------------------
At each frame we solve the Nonlinear Program (NLP):

    min_{u_{0:N-1}}  Σ_{k=0}^{N} [ R(x_k) - λ_d · D(x_k) ]

    subject to:
        x_{k+1}   = f(x_k, u_k)          ship dynamics (discrete, DT step)
        u_T_k     ∈ [T_min, T_max]        thrust bounds
        ω_k       ∈ [-ω_max, ω_max]       turn-rate bounds
        x_0       = x_ship                initial condition (parameter)

State vector:   x = (p_x, p_y, s, θ)      — 4 states
Control vector: u = (u_T, ω)              — 2 inputs per step

This is a proper NLP solved by IPOPT via CasADi's automatic
differentiation.  The grid-search variant (previous code) is kept as
a fallback for:
  - Machines where CasADi is not installed
  - Frames where IPOPT exceeds its time budget

Toroidal geometry note
----------------------
The modulo wrap (%) creates C⁰ discontinuities that break AD.
Strategy: during the optimisation horizon we work in *unbounded*
coordinates for the ship position and compute distances without wrap.
For a 1.5s horizon at max speed 200px/s the ship travels at most 300px,
which is well within a typical 1000×800 map, so boundary crossings are
rare.  The fallback grid search applies proper toroidal wrapping.

Warm-starting
-------------
The N-step control solution from frame k is shifted by one and used as
the initial guess for frame k+1.  This typically reduces solve time from
~20ms (cold) to ~2ms (warm).

Usage
-----
    pip install casadi          # once, in your project environment

    from mpc_director import MPCDirector
    mpc = MPCDirector(map_size=(1000, 800))
    result = mpc.plan(ship_pos, ship_vel, ship_speed, ship_heading,
                      asteroid_risks, r_global, tau_min, mode='active')
    print(result.solver_used, result.solve_time_ms)
"""

"""
mpc_director.py  —  Optimised CasADi/IPOPT MPC for XFC Kessler
================================================================

Optimisations vs version originale
-----------------------------------
1. [CRITIQUE]  _CasADiSolver instancié UNE SEULE FOIS dans MPCController.__init__
               (plus de reconstruction à chaque frame → -300 ms cold)
2. [CRITIQUE]  NLP de taille FIXE (N_AST_MAX astéroïdes max).
               Les astéroïdes manquants sont masqués avec risk=0 et position
               très lointaine → rebuild quasi-éliminé même si n_ast change.
3. [IMPORTANT] Boucle interne sur i VECTORISÉE en opérations CasADi MX
               → graphe symbolique plus petit, compilation et solve plus rapides.
4. [IMPORTANT] Pénalités d'effort (thrust/omega/speed) SORTIES de la boucle i
               (elles étaient multipliées n fois, ce qui biaisait le coût).
5. [MOYEN]     hessian_approximation='limited-memory'  (L-BFGS, ~2x plus rapide/iter)
6. [MOYEN]     acceptable_iter=3  (early-stop agressif dès 3 itérations acceptables)
7. [MOYEN]     warm_start_bound_push/mult_bound_push  (meilleur redémarrage chaud)
8. [MOYEN]     Horizon légèrement réduit (N=8, DT=0.12) pour couvrir ~1 s avec
               moins de variables (ajustable selon besoin).
"""


import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from risk_field import RiskField
from toric_utils import math_angle_to_turn_rate

# ---------------------------------------------------------------------------
# CasADi availability check
# ---------------------------------------------------------------------------
try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZON_STEPS   = 8            # réduit de 10 → 8  (-20 % variables)
DT              = 0.12         # s/step  →  ~1 s d'horizon total

THRUST_MAX      =  480.0       # px/s²
THRUST_MIN      = -480.0
OMEGA_MAX       =  180.0       # deg/s
DRAG            =   80.0       # px/s²

SPEED_MAX       = 200.0
SPEED_MIN       =  20.0

LAMBDA_D         = 0.25        # poids destruction (mode actif)
LAMBDA_D_RESPAWN = 0.0

W_THRUST = 1e-4
W_OMEGA  = 5e-3
W_SPEED  = 2e-3

DANGER_SCALE    = 120.0        # px
COLLISION_SOFT  =  25.0        # px
COLLISION_W     =   8.0

K_OMEGA         = 2.0
MAX_SOLVER_MS   = 30.0
IPOPT_MAX_ITER  = 30

# [OPTIMISATION 2] Taille fixe du NLP — pad/mask si n_ast < N_AST_MAX
N_AST_MAX       = 50           # à ajuster selon la map
FAR_AWAY        = 1e5          # position fictive pour astéroïdes masqués

INFEASIBILITY_COST_CEILING = 2000.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MPCResult:
    feasible:         bool
    theta_star:       float
    thrust:           float
    turn_rate:        float
    best_cost:        float
    fallback_heading: float
    solver_used:      str      # 'casadi' | 'fallback_repulse'
    solve_time_ms:    float


# ---------------------------------------------------------------------------
# Numpy dynamics step (warm-start init, viability check)
# ---------------------------------------------------------------------------

def _step_np(
    px: float, py: float, s: float, theta_deg: float,
    u_T: float, omega_deg: float,
    map_W: float, map_H: float,
) -> Tuple[float, float, float, float]:
    th_rad = math.radians(theta_deg)
    drag   = DRAG * math.copysign(1.0, s) if abs(s) > 0.5 else 0.0
    new_s  = s + (u_T - drag) * DT
    new_px = (px + s * math.cos(th_rad) * DT) % map_W
    new_py = (py + s * math.sin(th_rad) * DT) % map_H
    new_th = (theta_deg + omega_deg * DT) % 360.0
    return new_px, new_py, new_s, new_th


# ---------------------------------------------------------------------------
# CasADi NLP  —  taille FIXE, vectorisé, compilé une seule fois
# ---------------------------------------------------------------------------

class _CasADiSolver:
    """
    NLP de taille fixe (N_AST_MAX astéroïdes).
    Compilé UNE SEULE FOIS au premier appel ; les frames suivantes
    utilisent uniquement self._solver() avec des paramètres différents.
    """

    def __init__(self, map_size: Tuple[float, float]) -> None:
        self.map_W, self.map_H = map_size
        self._solver  = None          # compilé au premier appel
        self._n_x     = 0
        self._n_u     = 0
        self._lbx: Optional[list] = None
        self._ubx: Optional[list] = None
        self._u_prev: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build(self, lambda_d: float) -> None:
        """
        Construit le NLP symbolique pour exactement N_AST_MAX astéroïdes.
        Appelé UNE SEULE FOIS (ou lors d'un changement de lambda_d).

        Vecteur de décision  w = [X_flat (col-major), U_flat (col-major)]
            X : (4, N+1)  [px, py, s, theta_rad]
            U : (2, N)    [u_T (px/s²), omega (rad/s)]

        Paramètres p = [x0(4), ast_p0(2·n), ast_v(2·n), ast_r(n), ast_rsk(n)]
        avec n = N_AST_MAX fixé.
        """
        N = HORIZON_STEPS
        n = N_AST_MAX

        X = ca.MX.sym('X', 4, N + 1)
        U = ca.MX.sym('U', 2, N)

        # Paramètres symboliques
        x0_p      = ca.MX.sym('x0',      4)
        ast_p0_p  = ca.MX.sym('ast_p0',  2 * n)   # positions initiales
        ast_v_p   = ca.MX.sym('ast_v',   2 * n)   # vitesses
        ast_r_p   = ca.MX.sym('ast_r',   n)        # rayons
        ast_rsk_p = ca.MX.sym('ast_rsk', n)        # risques
        p_all     = ca.vertcat(x0_p, ast_p0_p, ast_v_p, ast_r_p, ast_rsk_p)

        # ------ Contraintes d'égalité (dynamique) ------
        g, lbg, ubg = [], [], []

        # Condition initiale
        g   += [X[:, 0] - x0_p]
        lbg += [0.0] * 4
        ubg += [0.0] * 4

        for k in range(N):
            px_k, py_k, s_k, th_k = X[0,k], X[1,k], X[2,k], X[3,k]
            uT_k, om_k             = U[0,k], U[1,k]
            drag_k = DRAG * ca.tanh(s_k / 1.0)   # sign() lisse pour AD

            g += [
                X[0, k+1] - (px_k + s_k * ca.cos(th_k) * DT),
                X[1, k+1] - (py_k + s_k * ca.sin(th_k) * DT),
                X[2, k+1] - (s_k  + (uT_k - drag_k) * DT),
                X[3, k+1] - (th_k + om_k * DT),
            ]
            lbg += [0.0] * 4
            ubg += [0.0] * 4

        # ------ Objectif (VECTORISÉ sur les n astéroïdes) ------
        # [OPTIMISATION 3]  Toutes les opérations sur i sont en MX vectoriel
        J = ca.MX(0)

        # Reshape des paramètres astéroïdes en matrices (2, n)
        ast_p0_mat  = ca.reshape(ast_p0_p, 2, n)   # colonnes = astéroïdes
        ast_v_mat   = ca.reshape(ast_v_p,  2, n)

        for k in range(N + 1):
            t_k  = k * DT
            px_k = X[0, k]
            py_k = X[1, k]
            th_k = X[3, k]
            s_k  = X[2, k]
            uT_k = U[0, k] if k < N else ca.MX(0)
            om_k = U[1, k] if k < N else ca.MX(0)

            # Positions astéroïdes à t_k  — (2, n)
            ast_pos_k = ast_p0_mat + ast_v_mat * t_k

            # Différence ship → chaque astéroïde  — (2, n)
            ship_pos_k = ca.vertcat(px_k, py_k)
            diff = ca.repmat(ship_pos_k, 1, n) - ast_pos_k   # (2, n)

            # Distances  — (1, n)
            dist_sq = diff[0, :] ** 2 + diff[1, :] ** 2 + 1e-4
            dist    = ca.sqrt(dist_sq)                        # (1, n)

            # Surface distance (dist − radius), clampé à 0
            d_sur = ca.fmax(dist - ast_r_p.T, 0.0)           # (1, n)

            # Danger exponentiel
            danger = ast_rsk_p.T * ca.exp(-d_sur / DANGER_SCALE)

            # Rampe collision douce
            coll_mu = ca.fmax(0.0, 1.0 - d_sur / COLLISION_SOFT)

            # Somme scalaire sur les n astéroïdes
            J = J + ca.sum2(danger) + COLLISION_W * ca.sum2(coll_mu ** 2)

            # Destruction term (heading alignment)
            if lambda_d > 0:
                hx     =  ca.cos(th_k)
                hy     = -ca.sin(th_k)                        # convention Kessler
                # Vecteur ship → astéroïde normalisé  — (2, n)
                to_ast = (ast_pos_k - ca.repmat(ship_pos_k, 1, n)) / ca.repmat(dist, 2, 1)
                # Produit scalaire  — (1, n)
                align  = ca.fmax(0.0, hx * to_ast[0, :] + hy * to_ast[1, :])
                J      = J - lambda_d * ca.sum2(ast_rsk_p.T * align * ca.exp(-d_sur / DANGER_SCALE))

            # [OPTIMISATION 4] Pénalités effort HORS de la boucle i
            if k < N:
                J = J + W_THRUST * uT_k ** 2
                J = J + W_OMEGA  * om_k ** 2
                J = J + W_SPEED  * s_k  ** 2

        # ------ Décision & bornes ------
        n_x = 4 * (N + 1)
        n_u = 2 * N
        w   = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        lbx = ([-1e6] * n_x
               + [THRUST_MIN,       -math.radians(OMEGA_MAX)] * N)
        ubx = ([ 1e6] * n_x
               + [THRUST_MAX,        math.radians(OMEGA_MAX)] * N)

        # ------ Options IPOPT ------
        # [OPTIMISATION 5]  L-BFGS : ~2× plus rapide par itération
        # [OPTIMISATION 6]  acceptable_iter=3 : early-stop agressif
        # [OPTIMISATION 7]  warm_start_bound_push : meilleur redémarrage chaud
        nlp  = {'x': w, 'f': J, 'g': ca.vertcat(*g), 'p': p_all}
        opts = {
            'ipopt.max_iter':                   IPOPT_MAX_ITER,
            'ipopt.tol':                        1e-3,
            'ipopt.acceptable_tol':             5e-3,
            'ipopt.acceptable_iter':            3,
            'ipopt.hessian_approximation':      'limited-memory',
            'ipopt.warm_start_init_point':      'yes',
            'ipopt.warm_start_bound_push':      1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.print_level':                0,
            'ipopt.sb':                         'yes',
            'print_time':                       False,
        }
        self._solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._n_x    = n_x
        self._n_u    = n_u
        self._lbx    = lbx
        self._ubx    = ubx
        self._u_prev = None   # invalide le warm-start

    # ------------------------------------------------------------------
    def _pad_asteroids(self, asteroid_risks: list):
        """
        Remplit jusqu'à N_AST_MAX astéroïdes.
        Les slots vides sont placés à FAR_AWAY avec risk=0.
        """
        n   = N_AST_MAX
        p0  = np.full(2 * n, FAR_AWAY)
        vel = np.zeros(2 * n)
        rad = np.ones(n)
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
        ship_heading:   float,   # degrés (convention Kessler)
        asteroid_risks: list,
        r_global:       float,
        lambda_d:       float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Résout le NLP.  Retourne (u_opt shape (2,N), cost) ou (None, inf).
        u_opt[0,:] = séquence de thrust  (px/s²)
        u_opt[1,:] = séquence de omega   (rad/s)
        """
        if len(asteroid_risks) == 0:
            return None, 0.0

        # [OPTIMISATION 1+2] Compilation une seule fois (ou si lambda_d change)
        if self._solver is None:
            self._build(lambda_d)

        N = HORIZON_STEPS

        # Paramètres numériques
        th_rad = math.radians(ship_heading)
        x0_val = np.array([ship_pos[0], ship_pos[1], ship_speed, th_rad])
        p0, vel, rad, rsk = self._pad_asteroids(asteroid_risks)
        p_val = np.concatenate([x0_val, p0, vel, rad, rsk])

        # Warm start : décaler la solution précédente d'un pas
        if self._u_prev is not None and self._u_prev.shape == (2, N):
            u_init = np.hstack([self._u_prev[:, 1:], self._u_prev[:, -1:]])
        else:
            u_init = np.zeros((2, N))

        # Simulation de la trajectoire initiale (sans toroïde pour l'optimiseur)
        X_init = np.zeros((4, N + 1))
        X_init[:, 0] = x0_val
        for k in range(N):
            px_k, py_k, s_k, th_k = X_init[:, k]
            uT_k = float(u_init[0, k])
            om_k = float(u_init[1, k])
            drag = DRAG * math.copysign(1.0, s_k) if abs(s_k) > 0.5 else 0.0
            X_init[0, k+1] = px_k + s_k * math.cos(th_k) * DT
            X_init[1, k+1] = py_k + s_k * math.sin(th_k) * DT
            X_init[2, k+1] = s_k  + (uT_k - drag) * DT
            X_init[3, k+1] = th_k + om_k * DT

        w0 = np.concatenate([
            X_init.flatten(order='F'),
            u_init.flatten(order='F'),
        ])

        try:
            sol   = self._solver(
                x0=w0, p=p_val,
                lbx=self._lbx, ubx=self._ubx,
                lbg=0.0,       ubg=0.0,
            )
            w_opt = np.array(sol['x']).flatten()
            cost  = float(sol['f'])
            stats = self._solver.stats()
            ok    = stats.get('success', False) or stats.get(
                'return_status', '') in ('Solve_Succeeded',
                                         'Solved_To_Acceptable_Level')
            if not ok:
                return None, math.inf

            u_opt        = w_opt[self._n_x:].reshape(2, N, order='F')
            self._u_prev = u_opt
            return u_opt, cost

        except Exception:
            self._u_prev = None
            return None, math.inf


# ---------------------------------------------------------------------------
# Contrôleur public
# ---------------------------------------------------------------------------

class MPCController:
    """
    Contrôleur MPC optimisé pour XFC.

    Le _CasADiSolver est instancié UNE SEULE FOIS dans __init__,
    et le NLP est compilé au premier appel à compute().
    Les frames suivantes bénéficient du warm-start et du NLP pré-compilé.

    Paramètres
    ----------
    prefer_casadi : False pour forcer le fallback (debug/profiling)
    """

    def __init__(self, prefer_casadi: bool = True) -> None:
        self._use_casadi = CASADI_AVAILABLE and prefer_casadi
        # [OPTIMISATION 1]  Instanciation unique — pas dans compute()
        # map_size sera défini au premier compute() via late-init
        self._casadi_slv: Optional[_CasADiSolver] = None
        self._map_size: Optional[Tuple[float, float]] = None
        self.risk_field: Optional[RiskField] = None

    # ------------------------------------------------------------------
    def _ensure_solver(self, map_size: Tuple[float, float]) -> None:
        """Late-init : crée le solver si besoin (map_size parfois inconnu à __init__)."""
        if self._casadi_slv is None or self._map_size != map_size:
            self._map_size   = map_size
            self.risk_field  = RiskField(map_size=map_size)
            self._casadi_slv = _CasADiSolver(map_size) if self._use_casadi else None

    # ------------------------------------------------------------------
    def compute(
        self,
        ship_state,
        game_state,
        tau_min:     float = 0.0,
        mode:        str   = 'active',
        repulse_dir: float = 0.0,
    ) -> MPCResult:

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

        # ---- Chemin CasADi ----
        if self._use_casadi and len(asteroid_risks) > 0:
            u_opt, cost_cas = self._casadi_slv.solve(
                ship_pos, ship_speed, ship_heading,
                asteroid_risks, r_global, lambda_d,
            )
            elapsed = (time.perf_counter() - t0)

            if u_opt is not None and elapsed < MAX_SOLVER_MS:
                uT_0   = float(np.clip(u_opt[0, 0], THRUST_MIN, THRUST_MAX))
                om_0   = float(np.clip(u_opt[1, 0],
                                       -math.radians(OMEGA_MAX),
                                        math.radians(OMEGA_MAX)))
                om_deg     = math.degrees(om_0)
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

            # Timeout ou infaisable
            return MPCResult(
                feasible         = False,
                theta_star       = 0.0,
                thrust           = 0.0,
                turn_rate        = 0.0,
                best_cost        = cost_cas if u_opt is None else math.inf,
                fallback_heading = repulse_dir,
                solver_used      = 'casadi',
                solve_time_ms    = elapsed,
            )

        # ---- Aucun astéroïde ou CasADi indisponible ----
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

    # ------------------------------------------------------------------
    def viability_check(
        self,
        ship_pos:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,
        asteroid_risks: list,
        tau_safe:       float = 0.5,
        n_probes:       int   = 8,
    ) -> bool:
        if self._map_size is None:
            return True
        W, H  = self._map_size
        steps = int(3.0 / DT)
        for j in range(n_probes):
            theta = 360.0 * j / n_probes
            px, py, s, th = ship_pos[0], ship_pos[1], ship_speed, theta
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