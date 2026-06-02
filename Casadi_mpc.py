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

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from risk_field import RiskField
from toric_utils import math_angle_to_turn_rate

# ---------------------------------------------------------------------------
# CasADi availability check — graceful degradation
# ---------------------------------------------------------------------------
try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZON_STEPS   = 10
DT              = 0.10         # s/step → 1.5 s horizon

THRUST_MAX      =  480.0       # px/s²
THRUST_MIN      = -480.0
OMEGA_MAX       =  180.0       # deg/s
DRAG            =   80.0       # px/s²

SPEED_MAX       = 200.0
SPEED_MIN       =  20.0

LAMBDA_D        = 0.25         # destruction weight (active mode)
LAMBDA_D_RESPAWN = 0.0

W_THRUST = 1e-4      # thrust effort penalty
W_OMEGA  = 5e-3      # turn-rate penalty
W_SPEED  = 2e-3      # speed magnitude penalty

DANGER_SCALE    = 120.0        # px — exponential falloff characteristic length
COLLISION_SOFT  =  25.0        # px — soft collision ramp radius
COLLISION_W     =   8.0        # collision ramp weight

K_OMEGA         = 2.0          # proportional heading gain
MAX_SOLVER_MS   = 20.0          # ms — CasADi budget before fallback
IPOPT_MAX_ITER  = 30

N_THETA_GRID    = 36           # fallback grid resolution
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
    solver_used:      str      # 'casadi' | 'grid' | 'fallback_repulse'
    solve_time_ms:    float





# ---------------------------------------------------------------------------
# Numpy dynamics step (used by grid search and warm-start init)
# ---------------------------------------------------------------------------

def _step_np(
    px: float, py: float, s: float, theta_deg: float,
    u_T: float, omega_deg: float,
    map_W: float, map_H: float,
) -> Tuple[float, float, float, float]:
    th_rad  = math.radians(theta_deg)
    drag    = DRAG * math.copysign(1.0, s) if abs(s) > 0.5 else 0.0
    new_s   = s + (u_T - drag) * DT
    new_px  = (px + s * math.cos(th_rad) * DT) % map_W
    new_py  = (py + s * math.sin(th_rad) * DT) % map_H
    new_th  = (theta_deg + omega_deg * DT) % 360.0
    return new_px, new_py, new_s, new_th


# ---------------------------------------------------------------------------
# Numpy cost at one state (used by grid search)
# ---------------------------------------------------------------------------

def _step_cost_np(
    px: float, py: float, theta_deg: float,
    asteroid_risks: list,
    t_elapsed: float,
    lambda_d: float,
    map_W: float, map_H: float,
) -> float:
    W, H      = map_W, map_H
    th_rad    = math.radians(theta_deg)
    cost      = 0.0

    for ar in asteroid_risks:
        ax = (ar.position[0] + ar.velocity[0] * t_elapsed) % W
        ay = (ar.position[1] + ar.velocity[1] * t_elapsed) % H

        dx = px - ax
        dy = py - ay
        dx -= W * round(dx / W)
        dy -= H * round(dy / H)
        dist   = math.hypot(dx, dy)
        d_surf = max(dist - ar.radius, 0.0)

        danger = ar.risk * math.exp(-d_surf / DANGER_SCALE)
        if d_surf < COLLISION_SOFT:
            danger += COLLISION_W * (1.0 - d_surf / COLLISION_SOFT) ** 2
        cost += danger

        if lambda_d > 0 and dist > 1e-3:
            # Kessler convention: heading vector is (cos θ, -sin θ) in screen coords
            hx =  math.cos(th_rad)
            hy = -math.sin(th_rad)
            to_ax = (ax - px) / dist
            to_ay = (ay - py) / dist
            align = max(0.0, hx * to_ax + hy * to_ay)
            cost -= lambda_d * ar.risk * align * math.exp(-d_surf / DANGER_SCALE)

    return cost


# ---------------------------------------------------------------------------
# CasADi NLP solver (rebuilt only when asteroid count changes)
# ---------------------------------------------------------------------------

class _CasADiSolver:

    def __init__(self, map_size: Tuple[float, float]) -> None:
        self.map_W, self.map_H = map_size
        self._solver  = None
        self._n_ast   = -1
        self._n_x     = 0
        self._n_u     = 0
        self._lbx     = None
        self._ubx     = None
        self._u_prev: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build(self, n_ast: int, lambda_d: float) -> None:
        """
        Construct the CasADi NLP symbolically.

        Decision variable w = [X_flat (col-major), U_flat (col-major)]
            X : (4, N+1)  state trajectory  [px, py, s, theta_rad]
            U : (2, N)    control sequence   [u_T (px/s²), omega (rad/s)]

        Parameters p = [x0(4), ast_p0(2n), ast_v(2n), ast_r(n), ast_rsk(n)]
        """
        N = HORIZON_STEPS
        n = n_ast

        X = ca.MX.sym('X', 4, N + 1)
        U = ca.MX.sym('U', 2, N)

        x0_p      = ca.MX.sym('x0',      4)
        ast_p0_p  = ca.MX.sym('ast_p0',  2 * n)
        ast_v_p   = ca.MX.sym('ast_v',   2 * n)
        ast_r_p   = ca.MX.sym('ast_r',   n)
        ast_rsk_p = ca.MX.sym('ast_rsk', n)
        p_all     = ca.vertcat(x0_p, ast_p0_p, ast_v_p, ast_r_p, ast_rsk_p)

        # Dynamics equality constraints
        g, lbg, ubg = [], [], []

        # Initial condition
        g   += [X[:, 0] - x0_p]
        lbg += [0.0] * 4
        ubg += [0.0] * 4

        for k in range(N):
            px_k, py_k, s_k, th_k = X[0,k], X[1,k], X[2,k], X[3,k]
            
            uT_k, om_k             = U[0,k], U[1,k]
            drag_k = DRAG * ca.tanh(s_k / 1.0)   # smooth sign approximation

            g += [
                X[0, k+1] - (px_k + s_k * ca.cos(th_k) * DT),
                X[1, k+1] - (py_k + s_k * ca.sin(th_k) * DT),
                X[2, k+1] - (s_k  + (uT_k - drag_k) * DT),
                X[3, k+1] - (th_k + om_k * DT),
            ]
            lbg += [0.0] * 4
            ubg += [0.0] * 4

        # Objective
        J = ca.MX(0)
        for k in range(N + 1):
            t_k  = k * DT
            px_k = X[0, k]
            py_k = X[1, k]
            th_k = X[3, k]   # radians (math convention)

            for i in range(n):
                ax = ast_p0_p[2*i]     + ast_v_p[2*i]     * t_k
                ay = ast_p0_p[2*i + 1] + ast_v_p[2*i + 1] * t_k

                dx    = px_k - ax
                dy    = py_k - ay
                dist  = ca.sqrt(dx**2 + dy**2 + 1e-4)
                d_sur = ca.fmax(dist - ast_r_p[i], 0.0)

                danger  = ast_rsk_p[i] * ca.exp(-d_sur / DANGER_SCALE)
                coll_mu = ca.fmax(0.0, 1.0 - d_sur / COLLISION_SOFT)
                J       = J + danger + COLLISION_W * coll_mu**2
                J += W_THRUST * uT_k**2
                J += W_OMEGA  * om_k**2
                J += W_SPEED  * s_k**2

                if lambda_d > 0:
                    # Kessler heading: x component = cos θ, y component = -sin θ
                    hx     =  ca.cos(th_k)
                    hy     = -ca.sin(th_k)
                    to_ax  = (ax - px_k) / dist
                    to_ay  = (ay - py_k) / dist
                    align  = ca.fmax(0.0, hx * to_ax + hy * to_ay)
                    J      = J - lambda_d * ast_rsk_p[i] * align * ca.exp(-d_sur / DANGER_SCALE)

        # Decision variable vector and bounds
        n_x = 4 * (N + 1)
        n_u = 2 * N
        w   = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        lbx = ([-1e6] * n_x
               + [THRUST_MIN,       -math.radians(OMEGA_MAX)] * N)
        ubx = ([ 1e6] * n_x
               + [THRUST_MAX,        math.radians(OMEGA_MAX)] * N)

        nlp  = {'x': w, 'f': J, 'g': ca.vertcat(*g), 'p': p_all}
        opts = {
            'ipopt.max_iter':               IPOPT_MAX_ITER,
            'ipopt.tol':                    1e-4,
            'ipopt.acceptable_tol':         5e-3,
            'ipopt.print_level':            0,
            'ipopt.sb':                     'yes',
            'ipopt.warm_start_init_point':  'yes',
            'print_time':                   False,
        }
        self._solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._n_ast  = n_ast
        self._n_x    = n_x
        self._n_u    = n_u
        self._lbx    = lbx
        self._ubx    = ubx
        self._u_prev = None   # invalidate warm start

    # ------------------------------------------------------------------
    def solve(
        self,
        ship_pos:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,   # degrees (Kessler)
        asteroid_risks: list,
        r_global:       float,
        lambda_d:       float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Returns (u_opt shape (2,N), cost) or (None, inf) on failure.
        u_opt[0,:] = thrust sequence (px/s²)
        u_opt[1,:] = omega sequence  (rad/s)
        """
        n_ast = len(asteroid_risks)
        if n_ast == 0:
            return None, 0.0

        if n_ast != self._n_ast:
            self._build(n_ast, lambda_d)

        N = HORIZON_STEPS

        # Build parameter vector
        th_rad = math.radians(ship_heading)
        x0_val = np.array([ship_pos[0], ship_pos[1], ship_speed, th_rad])

        ast_p0  = np.array([c for ar in asteroid_risks for c in ar.position])
        ast_v   = np.array([c for ar in asteroid_risks for c in ar.velocity])
        ast_r   = np.array([ar.radius for ar in asteroid_risks])
        ast_rsk = np.array([ar.risk   for ar in asteroid_risks])
        p_val   = np.concatenate([x0_val, ast_p0, ast_v, ast_r, ast_rsk])

        # Warm start: shift previous solution
        if self._u_prev is not None and self._u_prev.shape == (2, N):
            u_init = np.hstack([self._u_prev[:, 1:], self._u_prev[:, -1:]])
        else:
            u_init = np.zeros((2, N))

        # Simulate to get X_init
        X_init = np.zeros((4, N + 1))
        X_init[:, 0] = x0_val
        W, H = self.map_W, self.map_H
        for k in range(N):
            px_k, py_k, s_k, th_k = X_init[:, k]
            uT_k = float(u_init[0, k])
            om_k = float(u_init[1, k])
            drag = DRAG * math.copysign(1.0, s_k) if abs(s_k) > 0.5 else 0.0
            X_init[0, k+1] = px_k + s_k * math.cos(th_k) * DT
            X_init[1, k+1] = py_k + s_k * math.sin(th_k) * DT
            X_init[2, k+1] = s_k  + (uT_k - drag) * DT
            X_init[3, k+1] = th_k + om_k * DT

        w0 = np.concatenate([X_init.flatten(order='F'), u_init.flatten(order='F')])

        try:
            sol  = self._solver(
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
# Grid search fallback
# ---------------------------------------------------------------------------

# def _grid_search(
#     ship_pos:       Tuple[float, float],
#     ship_speed:     float,
#     ship_heading:   float,
#     asteroid_risks: list,
#     r_global:       float,
#     lambda_d:       float,
#     map_size:       Tuple[float, float],
#     repulse_dir:    float,
# ) -> Tuple[float, float]:
#     W, H   = map_size
#     s_star = _speed_target(r_global)
#     cands  = [360.0 * j / N_THETA_GRID for j in range(N_THETA_GRID)]
#     if repulse_dir not in cands:
#         cands.append(repulse_dir)

#     best_cost    = math.inf
#     best_heading = ship_heading

#     for theta_j in cands:
#         px, py, s, th = ship_pos[0], ship_pos[1], ship_speed, theta_j
#         tau_loc = min((ar.tau for ar in asteroid_risks), default=math.inf)
#         total   = 0.0

#         for k in range(HORIZON_STEPS):
#             t_el = k * DT
#             u_T  = _fuzzy_thrust(s_star - s, r_global, tau_loc)
#             px, py, s, th = _step_np(px, py, s, th, u_T, 0.0, W, H)
#             total += _step_cost_np(px, py, th, asteroid_risks, t_el,
#                                    lambda_d, W, H)

#         if total < best_cost:
#             best_cost    = total
#             best_heading = theta_j

#     return best_heading, best_cost


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MPCController:
    """
    MPC planner.  Uses CasADi/IPOPT when available, grid search otherwise.

    Parameters
    ----------
    map_size        : (width, height) of the arena
    prefer_casadi   : set False to force grid search (for profiling/debug)
    """

    def __init__(
        self,
        prefer_casadi: bool = True,
    ) -> None:
        
        self._use_casadi = CASADI_AVAILABLE and prefer_casadi
        

    # ------------------------------------------------------------------
    def compute(
        self,
        ship_state,
        game_state,
        tau_min:        float = 0.0,
        mode:           str   = 'active',
        repulse_dir:    float = 0.0,
    ) -> MPCResult:

        ship_pos=ship_state.position
        ship_vel = ship_state.velocity
        asteroids=game_state.asteroids
        self.risk_field  = RiskField(map_size=game_state.map_size)

        asteroid_risks = self.risk_field.compute_all(ship_pos,ship_vel, asteroids)
        r_global = self.risk_field.aggregate(asteroid_risks)
#game data
        self.map_size    = game_state.map_size
        self._casadi_slv = _CasADiSolver(self.map_size) if self._use_casadi else None
        map_size=self.map_size
# ship states
        ship_pos=ship_state.position
        ship_vel=ship_state.velocity
        ship_speed=ship_state.speed
        ship_heading=ship_state.heading
        

        lambda_d = LAMBDA_D if mode == 'active' else LAMBDA_D_RESPAWN
        # s_star   = _speed_target(r_global)
        t0       = time.perf_counter()

        # ---- CasADi path ----
        if self._use_casadi and len(asteroid_risks) > 0:
            u_opt, cost_cas = self._casadi_slv.solve(
                ship_pos, ship_speed, ship_heading,
                asteroid_risks, r_global, lambda_d,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if u_opt is not None and elapsed < MAX_SOLVER_MS:
                uT_0  = float(np.clip(u_opt[0, 0], THRUST_MIN, THRUST_MAX))
                om_0  = float(np.clip(u_opt[1, 0],
                                      -math.radians(OMEGA_MAX),
                                       math.radians(OMEGA_MAX)))
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
            else:
                return MPCResult(
                    feasible         = False,
                    theta_star       = 0,
                    
                    thrust           = 0,
                    turn_rate        = 0,
                    best_cost        = cost_cas,
                    fallback_heading = repulse_dir,
                    solver_used      = 'casadi',
                    solve_time_ms    = elapsed,
                )

        # # ---- Grid search fallback ----
        # t1 = time.perf_counter()
        # theta_star, best_cost = _grid_search(
        #     ship_pos, ship_speed, ship_heading,
        #     asteroid_risks, r_global, lambda_d,
        #     self.map_size, repulse_dir,
        # )
        # elapsed = (time.perf_counter() - t1) * 1000

        # u_T       = _fuzzy_thrust(s_star - ship_speed, r_global, tau_min)
        # turn_rate = math_angle_to_turn_rate(
        #     ship_heading, theta_star, k_omega=K_OMEGA, omega_max=OMEGA_MAX
        # )
        # feasible  = best_cost < INFEASIBILITY_COST_CEILING

        # return MPCResult(
        #     feasible         = feasible,
        #     theta_star       = theta_star,
        #     s_star           = s_star,
        #     thrust           = u_T,
        #     turn_rate        = turn_rate,
        #     best_cost        = best_cost,
        #     fallback_heading = repulse_dir,
        #     solver_used      = 'grid',
        #     solve_time_ms    = elapsed,
        # )

    # ------------------------------------------------------------------
    def plan_respawn(
        self,
        ship_pos: Tuple[float, float], ship_vel: Tuple[float, float],
        ship_speed: float, ship_heading: float,
        asteroid_risks: list, r_global: float, tau_min: float,
        repulse_dir: float,
    ) -> MPCResult:
        return self.plan(
            ship_pos, ship_vel, ship_speed, ship_heading,
            asteroid_risks, r_global, tau_min,
            mode='respawn', repulse_dir=repulse_dir,
        )

    # ------------------------------------------------------------------
    def viability_check(
        self,
        ship_pos: Tuple[float, float], ship_speed: float,
        ship_heading: float, asteroid_risks: list,
        tau_safe: float = 0.5, n_probes: int = 8,
    ) -> bool:
        W, H   = self.map_size
        steps  = int(3.0 / DT)
        for j in range(n_probes):
            theta = 360.0 * j / n_probes
            px, py, s, th = ship_pos[0], ship_pos[1], ship_speed, theta
            safe  = True
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
        return 'casadi+ipopt' if self._use_casadi else 'grid_search'
    
    
    
# @dataclass
# class MPCController:

#     def __init__(self):
#         self.director = MPCDirector()

#     def compute(self, ship_state, game_state):

#         result = self.director.plan(
#             ship_pos=ship_state.position,
#             ship_vel=ship_state.velocity,
#             ship_speed=ship_state.speed,
#             ship_heading=ship_state.heading,
#             asteroid_risks=[],
#             r_global=0.0,
#             tau_min=0.0,
#         )

#         return result