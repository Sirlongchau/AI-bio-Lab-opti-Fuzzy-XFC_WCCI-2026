"""
mpc_director.py
===============
Directional Model Predictive Control (MPC) by heading sampling.

Design rationale
----------------
A full continuous MPC over (x, y, s, θ) would require a nonlinear
optimiser at every frame — infeasible at 60 fps in Python.

Instead we use *directional sampling*: evaluate N_THETA candidate headings
over a short horizon, simulate the ship's trajectory under each (including
the fuzzy speed policy and true drag dynamics — Caveat C2 fix), and pick
the heading that minimises the cumulative cost.

Cost function
-------------
    J(θ_j) = Σ_k [ R(x_k) - λ_d · D(x_k) ]

    R(x_k) : risk at predicted position x_k (interpolated from risk field)
    D(x_k) : destruction potential at x_k (aligned shots on asteroids)
    λ_d    : trade-off weight (0 during respawn, positive during active mode)

The MPC is considered infeasible if ALL candidate headings produce a
trajectory that enters a collision zone within the horizon.  The modal
supervisor handles this case (sacrifice mode).

Speed policy
------------
Speed is regulated independently of heading by a risk-aware policy:
    s* = s_max · (1 - R_global) + s_min · R_global

During high risk the ship brakes; during low risk it can accelerate.
The fuzzy thrust FIS converts (s* - s) into a thrust command u_T.

Simulation model
----------------
Each candidate trajectory integrates:
    p_{k+1} = p_k + s_k · [cos(θ_j), sin(θ_j)] · dt
    s_{k+1} = s_k + (u_T_k - d · sign(s_k)) · dt
using the true drag constant d (see DRAG below).
Turn rate during the planning horizon is not simulated — the MPC assumes
the ship is already flying at θ_j.  The turn-rate command to reach θ_j
is computed separately by the proportional heading controller.

Usage
-----
    from mpc_director import MPCDirector
    mpc = MPCDirector(map_size=(1000, 800))
    result = mpc.plan(ship_state, asteroid_risks, r_global, mode='active')
    # result.feasible    — False → escalate to sacrifice
    # result.theta_star  — best heading (degrees)
    # result.s_star      — target speed
    # result.thrust      — FIS thrust command
    # result.turn_rate   — proportional heading command
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from toric_utils import (
    toric_distance,
    toric_delta,
    math_angle_to_turn_rate,
    angular_diff,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Candidate headings: sampled uniformly over [0°, 360°)
N_THETA         = 36           # angular resolution: 10° steps
HORIZON_STEPS   = 35           # number of simulation steps
DT              = 0.10         # seconds per step → 2 s horizon

# Ship physics
DRAG            = 80.0         # drag deceleration in px/s² (tune to match Kessler)
THRUST_MAX      = 480.0        # px/s²  (Kessler spec)
THRUST_MIN      = -480.0
OMEGA_MAX       = 180.0        # degrees/s
K_OMEGA         = 2.0          # proportional turn-rate gain

# Speed policy
SPEED_MAX       = 200.0        # px/s target at R=0
SPEED_MIN       = 20.0         # px/s target at R=1 (nearly stopped)

# Cost weights
LAMBDA_D        = 0.30         # destruction bonus weight (active mode)
LAMBDA_D_RESPAWN = 0.0         # no shooting during respawn

# Collision penalty: added to J when a predicted position is inside an asteroid
COLLISION_PENALTY = 10000.0

# Infeasibility: a heading is feasible if its cost is below this ceiling
# (set high enough that a clean trajectory always stays below it)
INFEASIBILITY_COST_CEILING = COLLISION_PENALTY * HORIZON_STEPS * 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MPCResult:
    feasible:    bool          # False → no safe heading found → sacrifice
    theta_star:  float         # best heading (degrees)
    s_star:      float         # target speed (px/s)
    thrust:      float         # thrust command u_T (px/s²)
    turn_rate:   float         # turn rate command ω (degrees/s)
    best_cost:   float         # J(θ*) for diagnostics
    fallback_heading: float    # repulse direction if infeasible


# ---------------------------------------------------------------------------
# Fuzzy thrust FIS (inline, kept simple)
# ---------------------------------------------------------------------------

def _fuzzy_thrust(
    speed_error: float,       # s* - s  (px/s)
    r_global: float,          # [0, 1]
    tau_min: float,           # seconds
) -> float:
    """
    Minimal Sugeno FIS for thrust control.

    Priority rules (evaluated in order, first match wins for clarity;
    in production all rules contribute via weighted average):

    R1: IF tau_min is critical (< 1s) THEN brake hard
    R2: IF speed_error large positive THEN accelerate
    R3: IF speed_error large negative THEN brake
    R4: IF r_global high THEN moderate brake bias
    """
    rules: List[Tuple[float, float]] = []   # (weight, singleton)

    # Emergency brake override
    tau_critical_mu = max(0.0, min(1.0, (1.5 - tau_min) / 1.5))
    rules.append((tau_critical_mu, THRUST_MIN * 0.8))

    # Speed tracking rules
    e_pos_mu = max(0.0, min(1.0,  speed_error / 150.0))   # large positive error
    e_neg_mu = max(0.0, min(1.0, -speed_error / 150.0))   # large negative error
    rules.append((e_pos_mu, THRUST_MAX * 0.8))
    rules.append((e_neg_mu, THRUST_MIN * 0.8))

    # Risk-modulated deceleration
    risk_mu = max(0.0, r_global - 0.5) * 2.0   # active above R=0.5
    rules.append((risk_mu, THRUST_MIN * 0.4))

    total_w = sum(w for w, _ in rules)
    if total_w < 1e-9:
        return 0.0

    thrust = sum(w * o for w, o in rules) / total_w
    return max(THRUST_MIN, min(THRUST_MAX, thrust))


# ---------------------------------------------------------------------------
# Speed policy
# ---------------------------------------------------------------------------

def _speed_target(r_global: float) -> float:
    """
    Risk-aware speed target.
    High risk → slow down for manoeuvrability.
    Low risk  → accelerate for coverage.
    """
    return SPEED_MAX * (1.0 - r_global) + SPEED_MIN * r_global


# ---------------------------------------------------------------------------
# Single-step dynamics integration
# ---------------------------------------------------------------------------

def _step(
    px: float, py: float,
    s: float,
    heading_rad: float,
    thrust: float,
    map_size: Tuple[float, float],
) -> Tuple[float, float, float]:
    """
    Integrate one DT step of the nonholonomic ship dynamics.

        p_{k+1} = p_k + s_k · [cos θ, sin θ] · dt
        s_{k+1} = s_k + (u_T - d · sign(s)) · dt

    Returns (px, py, s) after the step, with toroidal wrap.
    """
    W, H = map_size
    drag_force = DRAG * math.copysign(1.0, s) if abs(s) > 1e-3 else 0.0

    new_s  = s + (thrust - drag_force) * DT
    new_px = (px + s * math.cos(heading_rad) * DT) % W
    new_py = (py + s * math.sin(heading_rad) * DT) % H

    return new_px, new_py, new_s


# ---------------------------------------------------------------------------
# Trajectory cost evaluation
# ---------------------------------------------------------------------------

def _trajectory_cost(
    init_px: float, init_py: float,
    init_s: float,
    heading_deg: float,
    asteroid_risks: list,         # List[AsteroidRisk]
    r_global: float,
    s_star: float,
    lambda_d: float,
    map_size: Tuple[float, float],
) -> float:
    """
    Simulate HORIZON_STEPS steps at constant heading heading_deg and
    compute the cumulative cost J.

    Asteroid positions are extrapolated linearly at each step (constant
    velocity model — accurate for Kessler where asteroids move in straight
    lines).  This fixes Caveat C2: the cost now reflects the actual future
    geometry, not a time-decayed approximation.

    Cost per step:
        step_cost = Σ_i  risk_proxy(d_i_k)  -  λ_d · destruction_proxy(d_i_k, aligned_i_k)

    risk_proxy uses a soft distance-based danger: closer = higher cost.
    destruction_proxy rewards being close AND roughly facing the asteroid.
    """
    W, H   = map_size
    px, py = init_px, init_py
    s      = init_s
    h_rad  = math.radians(heading_deg)
    total_cost = 0.0

    for k in range(HORIZON_STEPS):
        t_elapsed = k * DT   # seconds into the future

        # Fuzzy thrust recomputed each step (uses initial tau as proxy;
        # tau decreases as we advance — conservative approximation)
        tau_min_local = min(
            (max(ar.tau - t_elapsed, 0.01) for ar in asteroid_risks),
            default=math.inf,
        )
        u_T = _fuzzy_thrust(s_star - s, r_global, tau_min_local)

        px, py, s = _step(px, py, s, h_rad, u_T, map_size)

        # --- Step cost: evaluate against extrapolated asteroid positions ---
        step_risk = 0.0
        step_dest = 0.0

        for ar in asteroid_risks:
            # Extrapolate asteroid position linearly
            ast_px = (ar.position[0] + ar.velocity[0] * t_elapsed) % W
            ast_py = (ar.position[1] + ar.velocity[1] * t_elapsed) % H

            # Toroidal distance from predicted ship pos to predicted asteroid pos
            dx = px - ast_px
            dy = py - ast_py
            dx -= W * round(dx / W)
            dy -= H * round(dy / H)
            dist = math.hypot(dx, dy)

            # Surface distance (0 if overlapping)
            d_surf = max(dist - ar.radius, 0.0)

            # Risk proxy: high when close, scaled by asteroid's base risk
            # Uses a soft exponential so the cost is smooth (gradient-friendly)
            DANGER_SCALE = 100.0   # pixels — characteristic danger radius
            danger = ar.risk * math.exp(-d_surf / DANGER_SCALE)

            # Hard collision penalty
            if d_surf < 5.0:
                danger += COLLISION_PENALTY

            step_risk += danger

            # Destruction proxy: reward being close AND heading toward the asteroid
            #if lambda_d > 0 and dist > 1e-3:
                # Unit vector from ship to asteroid
             #   to_ast_x = dx / dist
              #  to_ast_y = dy / dist
                # Alignment: dot product of heading vector and to-asteroid vector
               # alignment = (
                #    math.cos(h_rad) * to_ast_x
                 #   + math.sin(h_rad) * to_ast_y
                #)
                #alignment = max(0.0, alignment)   # only reward forward alignment
                #step_dest += ar.risk * alignment * math.exp(-d_surf / DANGER_SCALE)

        total_cost += step_risk #- lambda_d * step_dest

    return total_cost


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MPCDirector:
    """
    Directional MPC planner.

    Evaluates N_THETA candidate headings, simulates trajectories, and
    selects the optimal heading θ*.

    Parameters
    ----------
    map_size    : (width, height) of the arena
    n_theta     : number of candidate headings
    horizon     : number of simulation steps
    dt          : simulation time step in seconds
    """

    def __init__(
        self,
        map_size: Tuple[float, float],
        n_theta:  int   = N_THETA,
        horizon:  int   = HORIZON_STEPS,
        dt:       float = DT,
    ) -> None:
        self.map_size = map_size
        self.n_theta  = n_theta
        self.horizon  = horizon
        self.dt       = dt

        # Pre-compute candidate headings
        self._candidates = [
            360.0 * j / n_theta for j in range(n_theta)
        ]

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def plan(
        self,
        ship_pos:       Tuple[float, float],
        ship_vel:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,
        asteroid_risks: list,          # List[AsteroidRisk]
        r_global:       float,
        tau_min:        float,
        mode:           str = 'active',  # 'active' | 'respawn'
        repulse_dir:    float = 0.0,     # fallback from angular profile
    ) -> MPCResult:
        """
        Run one frame of directional MPC.

        Parameters
        ----------
        ship_pos       : (x, y) current position
        ship_vel       : (vx, vy) current velocity
        ship_speed     : scalar speed (px/s)
        ship_heading   : current heading (degrees)
        asteroid_risks : per-asteroid risk list from RiskField
        r_global       : aggregated global risk
        tau_min        : minimum TTC across all asteroids
        mode           : 'active' or 'respawn' (changes λ_d)
        repulse_dir    : best escape direction from angular profile
                         (used as first candidate to bias search)

        Returns
        -------
        MPCResult — see dataclass definition above
        """
        lambda_d = LAMBDA_D if mode == 'active' else LAMBDA_D_RESPAWN
        s_star   = _speed_target(r_global)

        # Include repulse direction as an extra candidate
        candidates = list(self._candidates)
        if repulse_dir not in candidates:
            candidates.append(repulse_dir)

        best_cost    = math.inf
        best_heading = ship_heading   # default: hold current heading
        min_cost_seen = math.inf

        heading_costs: List[Tuple[float, float]] = []

        for theta_j in candidates:
            cost = _trajectory_cost(
                ship_pos[0], ship_pos[1],
                ship_speed,
                theta_j,
                asteroid_risks,
                r_global,
                s_star,
                lambda_d,
                self.map_size,
            )
            heading_costs.append((theta_j, cost))
            if cost < best_cost:
                best_cost    = cost
                best_heading = theta_j

        # --- Feasibility check ---
        # Infeasible if every candidate produces collision-level cost
        feasible = best_cost < INFEASIBILITY_COST_CEILING

        # --- Low-level commands ---
        u_T       = _fuzzy_thrust(s_star - ship_speed, r_global, tau_min)
        turn_rate = math_angle_to_turn_rate(
            ship_heading, best_heading,
            k_omega=K_OMEGA, omega_max=OMEGA_MAX,
        )

        return MPCResult(
            feasible         = feasible,
            theta_star       = best_heading,
            s_star           = s_star,
            thrust           = u_T,
            turn_rate        = turn_rate,
            best_cost        = best_cost,
            fallback_heading = repulse_dir,
        )

    # ------------------------------------------------------------------
    # Respawn-only planner (pure survival, no destruction term)
    # ------------------------------------------------------------------

    def plan_respawn(
        self,
        ship_pos:       Tuple[float, float],
        ship_vel:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,
        asteroid_risks: list,
        r_global:       float,
        tau_min:        float,
        repulse_dir:    float,
    ) -> MPCResult:
        """
        Convenience wrapper for the respawn mode (λ_d = 0, pure survival).
        Enforces the mode-dependent constraint set from the synthesis
        (Section 7: J_respawn = Σ R(x_k)).
        """
        return self.plan(
            ship_pos, ship_vel, ship_speed, ship_heading,
            asteroid_risks, r_global, tau_min,
            mode='respawn',
            repulse_dir=repulse_dir,
        )

    # ------------------------------------------------------------------
    # Viability forward check (Section 8 of synthesis — Caveat C5 approx)
    # ------------------------------------------------------------------

    def viability_check(
        self,
        ship_pos:       Tuple[float, float],
        ship_speed:     float,
        ship_heading:   float,
        asteroid_risks: list,
        tau_safe:       float = 0.5,    # minimum acceptable TTC at each step
        n_probes:       int   = 8,      # number of headings to probe
    ) -> bool:
        """
        Approximate viability kernel check for the respawn window.

        Simulates n_probes headings over the respawn horizon (3s) and
        returns True if at least one keeps TTC > tau_safe at every step.

        This replaces the exact viability kernel (computationally infeasible)
        with a forward-simulation safety guard.
        """
        respawn_horizon = int(3.0 / DT)   # steps for 3s respawn window
        probe_headings  = [360.0 * j / n_probes for j in range(n_probes)]

        for theta in probe_headings:
            px, py = ship_pos
            s      = ship_speed
            h_rad  = math.radians(theta)
            safe   = True

            for k in range(respawn_horizon):
                u_T    = THRUST_MIN   # conservative: always braking
                px, py, s = _step(px, py, s, h_rad, u_T, self.map_size)

                # Check minimum distance from predicted position to extrapolated asteroid positions
                t_step = k * DT
                for ar in asteroid_risks:
                    ast_px = (ar.position[0] + ar.velocity[0] * t_step) % self.map_size[0]
                    ast_py = (ar.position[1] + ar.velocity[1] * t_step) % self.map_size[1]
                    dx = px - ast_px
                    dy = py - ast_py
                    dx -= self.map_size[0] * round(dx / self.map_size[0])
                    dy -= self.map_size[1] * round(dy / self.map_size[1])
                    d_surf = max(math.hypot(dx, dy) - ar.radius, 0.0)
                    if d_surf < tau_safe * 50:   # proxy: tau_safe seconds at ~50px/s
                        safe = False
                        break

                if not safe:
                    break

            if safe:
                return True   # at least one viable heading found

        return False   # no viable heading → truly infeasible