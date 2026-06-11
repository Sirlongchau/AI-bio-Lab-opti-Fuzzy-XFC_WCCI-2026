from __future__ import annotations

"""
Casadi_mpc.py  —  CasADi/IPOPT MPC for XFC Kessler
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from risk_field import RiskField
from Parasitic_fire import FireDecision, FireState, evaluate_fire

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZON_STEPS    = 8
DT               = 0.12
THRUST_MAX       =  480.0
THRUST_MIN       = -480.0
OMEGA_MAX        =  180.0
DRAG             =   80.0

LAMBDA_D         = 0.25
LAMBDA_D_RESPAWN = 0.0

W_THRUST     = 1e-5
W_OMEGA      = 5e-3
W_SLOW       = 1e-2
S_MIN_TARGET = 80.0

DANGER_SCALE    = 120.0
COLLISION_SOFT  =  25.0
COLLISION_W     =   8.0
SP_EPS          =   2.0

MAX_SOLVER_MS              = 25.0
IPOPT_MAX_ITER             = 50
N_AST_MAX                  = 50
INFEASIBILITY_COST_CEILING = 2000.0

S_REL   = 400.0
S_SPD   = 200.0
S_TH    = math.pi
S_UT    = 480.0
S_OM    = math.radians(OMEGA_MAX)


# ---------------------------------------------------------------------------
# Result dataclass — défini EN PREMIER pour être disponible dans compute()
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
    fire_decision:    bool


# ---------------------------------------------------------------------------
# Numpy dynamics
# ---------------------------------------------------------------------------

def _step_np_rel(
    rx: float, ry: float, s: float, th_rad: float,
    u_T: float, om_rad: float,
) -> Tuple[float, float, float, float]:
    drag   = DRAG * math.copysign(1.0, s) if abs(s) > 0.5 else 0.0
    new_s  = s  + (u_T - drag) * DT
    new_rx = rx + s * math.cos(th_rad) * DT
    new_ry = ry + s * math.sin(th_rad) * DT
    new_th = th_rad + om_rad * DT
    return new_rx, new_ry, new_s, new_th


# ---------------------------------------------------------------------------
# CasADi NLP
# ---------------------------------------------------------------------------

class _CasADiSolver:

    def __init__(self, map_size: Tuple[float, float]) -> None:
        self.map_W, self.map_H = map_size
        self._solver: Optional[object] = None
        self._n_xw   = 0
        self._lbw: Optional[list] = None
        self._ubw: Optional[list] = None
        self._u_prev: Optional[np.ndarray] = None

    def _build(self, lambda_d: float) -> None:
        N = HORIZON_STEPS
        n = N_AST_MAX

        XN = ca.MX.sym('XN', 4, N + 1)
        UN = ca.MX.sym('UN', 2, N)

        s0_p       = ca.MX.sym('s0')
        th0_p      = ca.MX.sym('th0')
        ast_rp0_p  = ca.MX.sym('ast_rp0', 2*n)
        ast_v_p    = ca.MX.sym('ast_v',   2*n)
        ast_r_p    = ca.MX.sym('ast_r',   n)
        ast_rsk_p  = ca.MX.sym('ast_rsk', n)
        p_all = ca.vertcat(s0_p, th0_p, ast_rp0_p, ast_v_p, ast_r_p, ast_rsk_p)

        g, lbg, ubg = [], [], []
        x0_norm = ca.vertcat(0.0, 0.0, s0_p / S_SPD, th0_p / S_TH)
        g   += [XN[:, 0] - x0_norm]
        lbg += [0.0] * 4
        ubg += [0.0] * 4

        for k in range(N):
            rx_k = XN[0, k] * S_REL
            ry_k = XN[1, k] * S_REL
            s_k  = XN[2, k] * S_SPD
            th_k = XN[3, k] * S_TH
            uT_k = UN[0, k] * S_UT
            om_k = UN[1, k] * S_OM

            drag_k  = DRAG * ca.tanh(s_k / 5.0)
            rx_next = rx_k + s_k * ca.cos(th_k) * DT
            ry_next = ry_k + s_k * ca.sin(th_k) * DT
            s_next  = s_k  + (uT_k - drag_k) * DT
            th_next = th_k + om_k * DT

            g += [
                XN[0, k+1] - rx_next / S_REL,
                XN[1, k+1] - ry_next / S_REL,
                XN[2, k+1] - s_next  / S_SPD,
                XN[3, k+1] - th_next / S_TH,
            ]
            lbg += [0.0] * 4
            ubg += [0.0] * 4

        J = ca.MX(0)
        ast_rp0_mat = ca.reshape(ast_rp0_p, 2, n)
        ast_v_mat   = ca.reshape(ast_v_p,   2, n)

        for k in range(N + 1):
            t_k  = k * DT
            rx_k = XN[0, k] * S_REL
            ry_k = XN[1, k] * S_REL
            th_k = XN[3, k] * S_TH
            s_k  = XN[2, k] * S_SPD
            uT_k = UN[0, k] * S_UT if k < N else ca.MX(0)
            om_k = UN[1, k] * S_OM if k < N else ca.MX(0)

            ship_rel  = ca.vertcat(rx_k, ry_k)
            ast_rel_k = ast_rp0_mat + ast_v_mat * t_k - ca.repmat(ship_rel, 1, n)

            dist  = ca.sqrt(ast_rel_k[0,:]**2 + ast_rel_k[1,:]**2 + 1.0)
            d_sur = ca.fmax(dist - ast_r_p.T, 0.0)

            danger = ast_rsk_p.T * ca.exp(-d_sur / DANGER_SCALE)
            J = J + ca.sum2(danger)

            coll_arg = (COLLISION_SOFT - d_sur) / SP_EPS
            coll_sp  = SP_EPS * ca.log1p(ca.exp(coll_arg))
            J = J + COLLISION_W * ca.sum2(coll_sp**2) / (COLLISION_SOFT**2)

            if lambda_d > 0:
                hx    =  ca.cos(th_k)
                hy    = -ca.sin(th_k)
                norm  = ca.repmat(dist, 2, 1)
                to_a  = ast_rel_k / norm
                to_a  = -to_a
                align = ca.fmax(0.0, hx * to_a[0,:] + hy * to_a[1,:])
                J = J - lambda_d * ca.sum2(
                    ast_rsk_p.T * align * ca.exp(-d_sur / DANGER_SCALE)
                )

            if k < N:
                J = J + W_THRUST * UN[0, k]**2
                J = J + W_OMEGA  * UN[1, k]**2
                J = J + W_SLOW   * ca.fmax(0.0, S_MIN_TARGET/S_SPD - XN[2, k])**2

        N_xw = 4 * (N + 1)
        wN   = ca.vertcat(ca.reshape(XN, -1, 1), ca.reshape(UN, -1, 1))

        lbw = [-1e6] * N_xw + [THRUST_MIN/S_UT, -1.0] * N
        ubw = [ 1e6] * N_xw + [THRUST_MAX/S_UT,  1.0] * N

        nlp  = {'x': wN, 'f': J, 'g': ca.vertcat(*g), 'p': p_all}
        opts = {
            'ipopt.max_iter':                   IPOPT_MAX_ITER,
            'ipopt.tol':                        1e-3,
            'ipopt.acceptable_tol':             5e-3,
            'ipopt.acceptable_iter':            3,
            'ipopt.warm_start_init_point':      'yes',
            'ipopt.warm_start_bound_push':      1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.nlp_scaling_method':         'gradient-based',
            'ipopt.print_level':                0,
            'ipopt.sb':                         'yes',
            'print_time':                       False,
        }
        self._solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._n_xw   = N_xw
        self._lbw    = lbw
        self._ubw    = ubw
        self._u_prev = None

    def _build_params(
        self,
        ship_pos:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,
        asteroid_risks: list,
    ) -> np.ndarray:
        W, H   = self.map_W, self.map_H
        sx, sy = ship_pos
        th_rad = math.radians(ship_heading)
        n      = N_AST_MAX

        rel_p0  = np.zeros(2 * n)
        ast_vel = np.zeros(2 * n)
        ast_r   = np.ones(n) * 10.0
        ast_rsk = np.zeros(n)

        for i, ar in enumerate(asteroid_risks[:n]):
            ax, ay = ar.position
            vx, vy = ar.velocity
            dx = ax - sx
            dy = ay - sy
            dx -= W * round(dx / W)
            dy -= H * round(dy / H)
            rel_p0[2*i]     = dx
            rel_p0[2*i + 1] = dy
            ast_vel[2*i]    = vx
            ast_vel[2*i+1]  = vy
            ast_r[i]        = ar.radius
            ast_rsk[i]      = ar.risk

        return np.concatenate([
            [ship_speed, th_rad],
            rel_p0, ast_vel, ast_r, ast_rsk,
        ])

    def _init_trajectory(
        self,
        ship_speed: float,
        ship_heading: float,
        u_init_norm: np.ndarray,
    ) -> np.ndarray:
        N      = HORIZON_STEPS
        th_rad = math.radians(ship_heading)
        XN     = np.zeros((4, N + 1))
        XN[:, 0] = [0.0, 0.0, ship_speed / S_SPD, th_rad / S_TH]

        rx, ry, s, th = 0.0, 0.0, ship_speed, th_rad
        for k in range(N):
            uT = u_init_norm[0, k] * S_UT
            om = u_init_norm[1, k] * S_OM
            rx, ry, s, th = _step_np_rel(rx, ry, s, th, uT, om)
            XN[0, k+1] = rx / S_REL
            XN[1, k+1] = ry / S_REL
            XN[2, k+1] = s  / S_SPD
            XN[3, k+1] = th / S_TH

        return XN

    def solve(
        self,
        ship_pos:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,
        asteroid_risks: list,
        r_global:       float,
        lambda_d:       float,
    ) -> Tuple[Optional[np.ndarray], float]:
        if not asteroid_risks:
            return None, 0.0

        if self._solver is None:
            self._build(lambda_d)

        N = HORIZON_STEPS

        if self._u_prev is not None and self._u_prev.shape == (2, N):
            u_phys = np.hstack([self._u_prev[:, 1:], self._u_prev[:, -1:]])
        else:
            u_phys = np.zeros((2, N))

        u_norm = np.vstack([u_phys[0] / S_UT, u_phys[1] / S_OM])

        XN_init = self._init_trajectory(ship_speed, ship_heading, u_norm)
        p_val   = self._build_params(ship_pos, ship_speed, ship_heading, asteroid_risks)

        w0 = np.concatenate([
            XN_init.flatten(order='F'),
            u_norm.flatten(order='F'),
        ])

        try:
            sol   = self._solver(
                x0=w0, p=p_val,
                lbx=self._lbw, ubx=self._ubw,
                lbg=0.0, ubg=0.0,
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

            u_opt_norm = w_opt[self._n_xw:].reshape(2, N, order='F')
            u_opt_phys = np.vstack([
                u_opt_norm[0] * S_UT,
                u_opt_norm[1] * S_OM,
            ])
            self._u_prev = u_opt_phys
            return u_opt_phys, cost

        except Exception:
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
        self.risk_field:  Optional[RiskField] = None
        self._fire_state  = FireState()

    def _ensure_solver(self, map_size: Tuple[float, float]) -> None:
        if self._casadi_slv is None or self._map_size != map_size:
            self._map_size  = map_size
            self.risk_field = RiskField(map_size=map_size)
            if self._use_casadi:
                self._casadi_slv = _CasADiSolver(map_size)

    def compute(
        self,
        ship_state,
        game_state,
        tau_min:     float = 0.0,
        mode:        str   = 'active',
        repulse_dir: float = 0.0,
    ) -> 'MPCResult':

        self._ensure_solver(game_state.map_size)

        asteroid_risks = self.risk_field.compute_all(
            ship_state.position, ship_state.velocity, game_state.asteroids
        )
        r_global = self.risk_field.aggregate(asteroid_risks)

        ship_pos     = ship_state.position
        ship_speed   = ship_state.speed
        ship_heading = ship_state.heading
        lambda_d     = LAMBDA_D if mode == 'active' else LAMBDA_D_RESPAWN

        t0 = time.perf_counter()

        if self._use_casadi and asteroid_risks:
            u_opt, cost = self._casadi_slv.solve(
                ship_pos, ship_speed, ship_heading,
                asteroid_risks, r_global, lambda_d,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if u_opt is not None:
                uT_0       = float(np.clip(u_opt[0, 0], THRUST_MIN, THRUST_MAX))
                om_0       = float(np.clip(u_opt[1, 0], -S_OM, S_OM))
                om_deg     = math.degrees(om_0)
                theta_star = (ship_heading + om_deg * DT) % 360.0

                # ── Construire le résultat partiel AVANT evaluate_fire ──────
                partial = MPCResult(
                    feasible         = cost < INFEASIBILITY_COST_CEILING,
                    theta_star       = theta_star,
                    thrust           = uT_0,
                    turn_rate        = om_deg,
                    best_cost        = cost,
                    fallback_heading = repulse_dir,
                    solver_used      = 'casadi',
                    solve_time_ms    = elapsed,
                    fire_decision    = False,
                )

                # ── Parasitic fire ──────────────────────────────────────────
                u_opt_for_fire = self._casadi_slv._u_prev if (
                    self._use_casadi and self._casadi_slv is not None
                ) else None

                fire_d = evaluate_fire(
                    mpc_result     = partial,          # ← instance, pas la classe
                    asteroid_risks = asteroid_risks,
                    ship_state     = ship_state,
                    fire_state     = self._fire_state,
                    now            = time.perf_counter(),
                    map_size       = game_state.map_size,
                    u_opt          = u_opt_for_fire,
                    opportunistic  = False,
                )

                if ship_state.respawn_time_left > 0:
                    fire_decision = False
                else:
                    fire_decision = fire_d.should_fire

                # Retourner le résultat final avec la vraie décision de feu
                partial.fire_decision = fire_decision
                return partial

            return MPCResult(
                feasible         = False,
                theta_star       = 0.0,
                thrust           = 0.0,
                turn_rate        = 0.0,
                best_cost        = math.inf,
                fallback_heading = repulse_dir,
                solver_used      = 'casadi_failed',
                solve_time_ms    = (time.perf_counter() - t0) * 1000,
                fire_decision    = False,
            )

        # Pas d'astéroïdes ou CasADi indisponible
        return MPCResult(
            feasible         = True,
            theta_star       = ship_heading,
            thrust           = 0.0,
            turn_rate        = 0.0,
            best_cost        = 0.0,
            fallback_heading = repulse_dir,
            solver_used      = 'none',
            solve_time_ms    = 0.0,
            fire_decision    = False,
        )

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
            th = math.radians(360.0 * j / n_probes)
            rx, ry, s = 0.0, 0.0, ship_speed
            safe = True
            for k in range(steps):
                t_k = k * DT
                rx, ry, s, th = _step_np_rel(rx, ry, s, th, THRUST_MIN, 0.0)
                for ar in asteroid_risks:
                    ax0, ay0 = ar.position
                    sx, sy   = ship_pos
                    dvx, dvy = ar.velocity
                    dx0 = ax0 - sx;  dx0 -= W * round(dx0 / W)
                    dy0 = ay0 - sy;  dy0 -= H * round(dy0 / H)
                    adx = dx0 + dvx * t_k - rx
                    ady = dy0 + dvy * t_k - ry
                    if max(math.hypot(adx, ady) - ar.radius, 0.0) < tau_safe * 50:
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