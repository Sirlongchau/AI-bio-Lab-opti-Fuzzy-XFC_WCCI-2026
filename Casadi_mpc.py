from __future__ import annotations

"""
Casadi_mpc.py — CasADi/IPOPT receding-horizon evasion MPC for XFC Kessler.

The solver works in the ship-relative frame (origin = ship, so the ship state
collapses to [0, 0, speed, heading]). Asteroids are propagated ballistically
over the horizon and enter the cost as:
  * a smooth danger field   (risk-weighted exp() of surface distance), and
  * a soft collision barrier (squared softplus inside COLLISION_SOFT px).

Firing is *not* optimised here (too heavy at frame rate); a separate parasitic
pass (Parasitic_fire.evaluate_fire) reads the already-computed optimal controls
and decides whether a free shot is available.

Public API (consumed by the Supervisor):
    MPCController(prefer_casadi=True)
        .compute(ship_state, game_state, ...) -> MPCResult
        .viability_check(...)                 -> bool
        .solver_backend                       -> str
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from risk_field import RiskField
from Parasitic_fire import evaluate_fire, FireState

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Horizon
HORIZON_STEPS = 8
DT            = 0.12          # s / step

# Ship limits (ship.py)
THRUST_MAX =  480.0
THRUST_MIN = -480.0
OMEGA_MAX  =  180.0           # deg/s
DRAG       =   80.0

# Cost weights
LAMBDA_D         = 0.25       # "point at threats" alignment bonus (active mode)
LAMBDA_D_RESPAWN = 0.0        # disabled during respawn invulnerability
W_THRUST     = 1e-5
W_OMEGA      = 5e-3
W_SLOW       = 1e-2
S_MIN_TARGET = 80.0           # px/s — gently discourage crawling to a stop

DANGER_SCALE   = 120.0        # px — danger field length scale
COLLISION_SOFT =  25.0        # px — soft barrier kicks in inside this surface distance
COLLISION_W    =   8.0
SP_EPS         =   2.0        # softplus sharpness

# Solver budget
MAX_SOLVER_MS              = 25.0
IPOPT_MAX_ITER             = 50
N_AST_MAX                  = 200       # computational cap (the turret keeps us under it)
INFEASIBILITY_COST_CEILING = 2000.0   # cost above this => Supervisor treats it as "no escape"

# Inactive asteroid slots are parked here so they contribute zero cost.
FAR_AWAY = 1.0e5              # px

# Decision-variable scaling (keeps the NLP well-conditioned)
S_REL = 400.0
S_SPD = 200.0
S_TH  = math.pi
S_UT  = 480.0
S_OM  = math.radians(OMEGA_MAX)


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
    fire_decision:    bool


# ---------------------------------------------------------------------------
# Plain-numpy dynamics (used by the warm-start guess and viability_check)
# ---------------------------------------------------------------------------

def _step_np_rel(rx, ry, s, th_rad, u_T, om_rad):
    drag   = DRAG * math.copysign(1.0, s) if abs(s) > 0.5 else 0.0
    new_s  = s + (u_T - drag) * DT
    new_rx = rx + s * math.cos(th_rad) * DT
    new_ry = ry + s * math.sin(th_rad) * DT
    new_th = th_rad + om_rad * DT
    return new_rx, new_ry, new_s, new_th


# ---------------------------------------------------------------------------
# CasADi NLP wrapper
# ---------------------------------------------------------------------------

class _CasADiSolver:

    def __init__(self, map_size: Tuple[float, float]) -> None:
        self.map_W, self.map_H = map_size
        self._solver: Optional[object] = None
        self._n_xw   = 0
        self._lbw:    Optional[list] = None
        self._ubw:    Optional[list] = None
        self._u_prev: Optional[np.ndarray] = None   # last solution, for warm start

    # -- one-time symbolic build -------------------------------------------

    def _build(self, lambda_d: float) -> None:
        N = HORIZON_STEPS
        n = N_AST_MAX

        XN = ca.MX.sym('XN', 4, N + 1)   # [rx, ry, s, th] (scaled), per knot
        UN = ca.MX.sym('UN', 2, N)       # [thrust, omega]  (scaled), per step

        # Parameters
        s0_p      = ca.MX.sym('s0')
        th0_p     = ca.MX.sym('th0')
        ast_rp0_p = ca.MX.sym('ast_rp0', 2 * n)   # initial relative positions
        ast_v_p   = ca.MX.sym('ast_v',   2 * n)   # velocities
        ast_r_p   = ca.MX.sym('ast_r',   n)       # radii (px)
        ast_rsk_p = ca.MX.sym('ast_rsk', n)       # per-asteroid risk
        p_all = ca.vertcat(s0_p, th0_p, ast_rp0_p, ast_v_p, ast_r_p, ast_rsk_p)

        # --- Dynamics defects (multiple shooting) ---
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

            drag_k  = DRAG * ca.tanh(s_k / 5.0)            # smooth signum
            rx_next = rx_k + s_k * ca.cos(th_k) * DT
            ry_next = ry_k + s_k * ca.sin(th_k) * DT
            s_next  = s_k  + (uT_k - drag_k) * DT
            th_next = th_k + om_k * DT

            g += [
                XN[0, k + 1] - rx_next / S_REL,
                XN[1, k + 1] - ry_next / S_REL,
                XN[2, k + 1] - s_next  / S_SPD,
                XN[3, k + 1] - th_next / S_TH,
            ]
            lbg += [0.0] * 4
            ubg += [0.0] * 4

        # --- Cost ---
        J = ca.MX(0)
        ast_rp0_mat = ca.reshape(ast_rp0_p, 2, n)
        ast_v_mat   = ca.reshape(ast_v_p,   2, n)

        for k in range(N + 1):
            t_k  = k * DT
            rx_k = XN[0, k] * S_REL
            ry_k = XN[1, k] * S_REL
            s_k  = XN[2, k] * S_SPD
            th_k = XN[3, k] * S_TH

            ship_rel  = ca.vertcat(rx_k, ry_k)
            ast_rel_k = ast_rp0_mat + ast_v_mat * t_k - ca.repmat(ship_rel, 1, n)

            dist  = ca.sqrt(ast_rel_k[0, :] ** 2 + ast_rel_k[1, :] ** 2 + 1.0)
            d_sur = ca.fmax(dist - ast_r_p.T, 0.0)

            # Danger field (risk-weighted proximity).
            J = J + ca.sum2(ast_rsk_p.T * ca.exp(-d_sur / DANGER_SCALE))

            # Soft collision barrier (squared softplus inside COLLISION_SOFT).
            coll_sp = SP_EPS * ca.log1p(ca.exp((COLLISION_SOFT - d_sur) / SP_EPS))
            J = J + COLLISION_W * ca.sum2(coll_sp ** 2) / (COLLISION_SOFT ** 2)

            # Optional "face the threats" alignment bonus (helps parasitic fire).
            if lambda_d > 0:
                hx, hy = ca.cos(th_k), -ca.sin(th_k)
                to_a   = -ast_rel_k / ca.repmat(dist, 2, 1)
                align  = ca.fmax(0.0, hx * to_a[0, :] + hy * to_a[1, :])
                J = J - lambda_d * ca.sum2(
                    ast_rsk_p.T * align * ca.exp(-d_sur / DANGER_SCALE)
                )

            # Control effort + anti-stall (steps only, not the terminal knot).
            if k < N:
                J = J + W_THRUST * UN[0, k] ** 2
                J = J + W_OMEGA  * UN[1, k] ** 2
                J = J + W_SLOW   * ca.fmax(0.0, S_MIN_TARGET / S_SPD - XN[2, k]) ** 2

        # --- Assemble ---
        N_xw = 4 * (N + 1)
        wN   = ca.vertcat(ca.reshape(XN, -1, 1), ca.reshape(UN, -1, 1))
        lbw  = [-1e6] * N_xw + [THRUST_MIN / S_UT, -1.0] * N
        ubw  = [ 1e6] * N_xw + [THRUST_MAX / S_UT,  1.0] * N

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

    # -- parameter vector ---------------------------------------------------

    def _build_params(self, ship_pos, ship_speed, ship_heading, asteroid_risks) -> np.ndarray:
        W, H   = self.map_W, self.map_H
        sx, sy = ship_pos
        n      = N_AST_MAX

        # Inactive slots parked at FAR_AWAY with zero radius/risk -> zero cost.
        rel_p0  = np.full(2 * n, FAR_AWAY)
        ast_vel = np.zeros(2 * n)
        ast_r   = np.zeros(n)
        ast_rsk = np.zeros(n)

        for i, ar in enumerate(asteroid_risks[:n]):
            ax, ay = ar.position
            dx = ax - sx; dx -= W * round(dx / W)
            dy = ay - sy; dy -= H * round(dy / H)
            rel_p0[2 * i]     = dx
            rel_p0[2 * i + 1] = dy
            ast_vel[2 * i]    = ar.velocity[0]
            ast_vel[2 * i + 1] = ar.velocity[1]
            ast_r[i]          = ar.radius        # true px radius (risk_field contract)
            ast_rsk[i]        = ar.risk

        return np.concatenate([
            [ship_speed, math.radians(ship_heading)],
            rel_p0, ast_vel, ast_r, ast_rsk,
        ])

    def _init_trajectory(self, ship_speed, ship_heading, u_init_norm) -> np.ndarray:
        N      = HORIZON_STEPS
        th_rad = math.radians(ship_heading)
        XN     = np.zeros((4, N + 1))
        XN[:, 0] = [0.0, 0.0, ship_speed / S_SPD, th_rad / S_TH]

        rx, ry, s, th = 0.0, 0.0, ship_speed, th_rad
        for k in range(N):
            uT = u_init_norm[0, k] * S_UT
            om = u_init_norm[1, k] * S_OM
            rx, ry, s, th = _step_np_rel(rx, ry, s, th, uT, om)
            XN[0, k + 1] = rx / S_REL
            XN[1, k + 1] = ry / S_REL
            XN[2, k + 1] = s  / S_SPD
            XN[3, k + 1] = th / S_TH
        return XN

    # -- solve --------------------------------------------------------------

    def solve(self, ship_pos, ship_speed, ship_heading, asteroid_risks, lambda_d
              ) -> Tuple[Optional[np.ndarray], float]:
        if not asteroid_risks:
            return None, 0.0
        if self._solver is None:
            self._build(lambda_d)

        N = HORIZON_STEPS

        # Warm start: shift the previous optimal controls forward one step.
        if self._u_prev is not None and self._u_prev.shape == (2, N):
            u_phys = np.hstack([self._u_prev[:, 1:], self._u_prev[:, -1:]])
        else:
            u_phys = np.zeros((2, N))
        u_norm = np.vstack([u_phys[0] / S_UT, u_phys[1] / S_OM])

        XN_init = self._init_trajectory(ship_speed, ship_heading, u_norm)
        p_val   = self._build_params(ship_pos, ship_speed, ship_heading, asteroid_risks)
        w0      = np.concatenate([XN_init.flatten(order='F'), u_norm.flatten(order='F')])

        try:
            sol = self._solver(x0=w0, p=p_val, lbx=self._lbw, ubx=self._ubw,
                               lbg=0.0, ubg=0.0)
            cost  = float(sol['f'])
            stats = self._solver.stats()
            ok = stats.get('success', False) or stats.get('return_status', '') in (
                'Solve_Succeeded', 'Solved_To_Acceptable_Level')
            if not ok:
                self._u_prev = None
                return None, math.inf

            w_opt      = np.array(sol['x']).flatten()
            u_opt_norm = w_opt[self._n_xw:].reshape(2, N, order='F')
            u_opt_phys = np.vstack([u_opt_norm[0] * S_UT, u_opt_norm[1] * S_OM])
            self._u_prev = u_opt_phys
            return u_opt_phys, cost

        except Exception:
            self._u_prev = None
            return None, math.inf


# ---------------------------------------------------------------------------
# Public controller
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

    def compute(self, ship_state, game_state, tau_min: float = 0.0,
                mode: str = 'active', repulse_dir: float = 0.0) -> MPCResult:
        self._ensure_solver(game_state.map_size)

        asteroid_risks = self.risk_field.compute_all(
            ship_state.position, ship_state.velocity, game_state.asteroids
        )
        lambda_d = LAMBDA_D if mode == 'active' else LAMBDA_D_RESPAWN

        t0 = time.perf_counter()

        if not (self._use_casadi and asteroid_risks):
            # No asteroids or CasADi unavailable: nothing to dodge.
            return MPCResult(True, ship_state.heading, 0.0, 0.0, 0.0,
                             repulse_dir, 'none', 0.0, False)

        u_opt, cost = self._casadi_slv.solve(
            ship_state.position, ship_state.speed, ship_state.heading,
            asteroid_risks, lambda_d,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0

        if u_opt is None:
            return MPCResult(False, 0.0, 0.0, 0.0, math.inf,
                             repulse_dir, 'casadi_failed', elapsed, False)

        uT_0   = float(np.clip(u_opt[0, 0], THRUST_MIN, THRUST_MAX))
        om_0   = float(np.clip(u_opt[1, 0], -S_OM, S_OM))
        om_deg = math.degrees(om_0)

        result = MPCResult(
            feasible         = cost < INFEASIBILITY_COST_CEILING,
            theta_star       = (ship_state.heading + om_deg * DT) % 360.0,
            thrust           = uT_0,
            turn_rate        = om_deg,
            best_cost        = cost,
            fallback_heading = repulse_dir,
            solver_used      = 'casadi',
            solve_time_ms    = elapsed,
            fire_decision    = False,
        )

        # Parasitic fire: a free shot along the already-chosen trajectory.
        fire_d = evaluate_fire(
            mpc_result     = result,
            asteroid_risks = asteroid_risks,
            ship_state     = ship_state,
            fire_state     = self._fire_state,
            now            = time.perf_counter(),
            map_size       = game_state.map_size,
            u_opt          = self._casadi_slv._u_prev,
            opportunistic  = False,
        )
        result.fire_decision = (ship_state.respawn_time_left <= 0) and fire_d.should_fire
        return result

    # -- escape-existence oracle -------------------------------------------

    def viability_check(self, ship_pos, ship_speed, ship_heading, asteroid_risks,
                        tau_safe: float = 0.5, n_probes: int = 8) -> bool:
        """
        Cheap "does any escape exist?" probe: fan out n_probes headings, run
        full-reverse-thrust open loop for 3 s, and report True if any probe
        stays clear. Independent of the NLP; usable to gate the sacrifice mode.
        """
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
                    dx0 = ax0 - ship_pos[0]; dx0 -= W * round(dx0 / W)
                    dy0 = ay0 - ship_pos[1]; dy0 -= H * round(dy0 / H)
                    adx = dx0 + ar.velocity[0] * t_k - rx
                    ady = dy0 + ar.velocity[1] * t_k - ry
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