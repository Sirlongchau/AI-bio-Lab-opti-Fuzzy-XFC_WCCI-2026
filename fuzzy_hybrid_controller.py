"""
fuzzy_hybrid_controller.py
==========================
Top-level KesslerController implementation.

This is the only file that imports from Kessler's API.  All other modules
are pure Python and agnostic to the game engine.

Pipeline (executed every frame in actions())
-------------------------------------------
1.  Parse ship_state and game_state into clean Python structures.
2.  Compute per-asteroid risk field (RiskField).
3.  Build angular danger profile and extract free corridors (AngularProfile).
4.  Select attack target with commitment (TargetSelector).
5.  Run directional MPC planner (MPCDirector).
6.  Compute fuzzy heading command from best corridor (proportional control).
7.  Compute fuzzy thrust from speed policy + FIS.
8.  Compute fire decision (TargetSelector.fire_decision).
9.  Run modal supervisor to select active mode and enforce constraints.
10. Return (thrust, turn_rate, fire, drop_mine) to Kessler.

Adapter notes
-------------
Kessler passes ship_state and game_state as objects with attributes.
The exact attribute names used below match the KesslerGame v2.x API:

    ship_state.position        -> (x, y) tuple
    ship_state.velocity        -> (vx, vy) tuple
    ship_state.speed           -> scalar float
    ship_state.heading         -> float (degrees, 0=right, CCW positive)
    ship_state.lives_remaining -> int
    ship_state.can_fire        -> bool
    ship_state.fire_rate       -> float (not used directly)

    game_state.asteroids       -> list of asteroid objects
        asteroid.position      -> (x, y)
        asteroid.velocity      -> (vx, vy)
        asteroid.size          -> radius float

    game_state.map_size        -> (width, height)
    game_state.time            -> float seconds

Respawn detection
-----------------
Kessler does not expose a respawn_timer directly.  We detect the respawn
window by monitoring lives_remaining: when it decreases, we start a 3s
internal timer.  This is conservative but safe.

Debug logging
-------------
Set DEBUG = True to print a one-line diagnostic per frame to stdout.
Useful for tuning thresholds but should be False during competition.
"""

from __future__ import annotations

import math
import time
from typing import Tuple
import numpy as np

# Kessler API (imported only here)
try:
    from kesslergame import KesslerController
except ImportError:
    # Fallback stub for offline development / unit tests
    class KesslerController:  # type: ignore
        def actions(self, ship_state, game_state):
            raise NotImplementedError
        @property
        def name(self) -> str:
            return "stub"

# Our modules
from toric_utils       import math_angle_to_turn_rate, angular_diff
from risk_field        import RiskField
from angular_profile   import AngularProfile
from target_selector   import TargetSelector
from Casadi_mpc      import MPCDirector
from modal_supervisor  import ModalSupervisor, ControlOutput
from debug_tools       import FrameDebugger, RiskHeatmap

# ---------------------------------------------------------------------------
# Debug configuration
# ---------------------------------------------------------------------------
DEBUG            = True    # one-line console log every DEBUG_EVERY frames
DEBUG_EVERY      = 30      # frames between log lines (1 = every frame)
HEATMAP_ENABLED  = False   # set True to open live matplotlib heatmap window
HEATMAP_EVERY    = 60      # frames between heatmap updates (perf cost ~5ms)

# ---------------------------------------------------------------------------
# Respawn detection parameters
# ---------------------------------------------------------------------------
RESPAWN_DURATION = 3.0   # seconds — Kessler's invulnerability window
ASSUMED_FPS      = 60.0  # used to convert frames to seconds


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class FuzzyHybridController(KesslerController):
    """
    Explainable hybrid fuzzy-MPC controller for the Kessler Game.

    Implements the architecture described in:
        "Explainable Hybrid TTC-MPC Control for Toroidal Asteroid Combat"
        (XFC competition submission, Thales / NAFIPS 2024)

    Three-level escalation:
        FUZZY       — nominal, full offence + avoidance
        MPC_EVASION — high risk, optimal evasion, no fire
        SACRIFICE   — infeasible, mine + voluntary death + respawn recovery
    """

    # ------------------------------------------------------------------
    # Kessler requires a name property
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "FuzzyHybridController"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Map size is not available at __init__ time in Kessler;
        # modules requiring it are initialised lazily on first frame.
        self._initialised      = False
        self._map_size: Tuple[float, float] = (1000.0, 800.0)  # placeholder

        # Sub-modules (initialised in _lazy_init)
        self._risk_field:    RiskField
        self._angular_prof:  AngularProfile
        self._target_sel:    TargetSelector
        self._mpc:           MPCDirector
        self._supervisor:    ModalSupervisor

        # Respawn tracking
        self._lives_prev:       int   = 3
        self._respawn_timer:    float = 0.0
        self._last_time:        float = 0.0

        # Frame counter for debug output
        self._frame: int = 0

        # ---- Debug tools ----
        self._dbg = FrameDebugger(enabled=DEBUG)
        # Register default conditions — edit freely:
        # self._dbg.break_at_frame(100)
        # self._dbg.break_when(lambda s: s['r_global'] > 0.85, 'high risk')
        # self._dbg.break_on_mode_change()
        # self._dbg.break_every(300, 'periodic')

        self._heatmap = RiskHeatmap(
            resolution=8,   # res=8 → 6ms/frame, good balance
            live_display=HEATMAP_ENABLED,
        ) if HEATMAP_ENABLED else None

    def _lazy_init(self, map_size: Tuple[float, float]) -> None:
        """Initialise all sub-modules once map_size is known."""
        self._map_size    = map_size
        self._risk_field  = RiskField(map_size=map_size)
        self._angular_prof = AngularProfile()
        self._target_sel  = TargetSelector()
        self._mpc         = MPCDirector(map_size=map_size)
        self._supervisor  = ModalSupervisor()
        self._initialised = True

    # ------------------------------------------------------------------
    # Main entry point — called by Kessler every frame
    # ------------------------------------------------------------------

    def actions(self, ship_state, game_state) -> Tuple[float, float, bool, bool]:
        """
        Compute control actions for the current frame.

        Returns
        -------
        (thrust, turn_rate, fire, drop_mine) as required by Kessler.
            thrust    : float in [-480, 480]  px/s²
            turn_rate : float in [-180, 180]  degrees/s
            fire      : bool
            drop_mine : bool
        """
        t_start = time.perf_counter()   # frame timing guard

        # ----------------------------------------------------------------
        # 0. Lazy initialisation
        # ----------------------------------------------------------------
        map_size = tuple(game_state.map_size)
        if not self._initialised:
            self._lazy_init(map_size)

        self._frame += 1

        # ----------------------------------------------------------------
        # 1. Parse state
        # ----------------------------------------------------------------
        ship_pos     = tuple(ship_state.position)
        ship_vel     = tuple(ship_state.velocity)
        ship_speed   = float(ship_state.speed)
        ship_heading = float(ship_state.heading)
        lives        = int(ship_state.lives_remaining)
        can_fire     = bool(ship_state.can_fire)
        game_time    = float(game_state.time)

        # Detect frame dt (capped to avoid huge jumps on first frame)
        dt_real = min(game_time - self._last_time, 0.1) if self._frame > 1 else 1.0 / ASSUMED_FPS
        self._last_time = game_time

        # ----------------------------------------------------------------
        # 2. Respawn detection
        # ----------------------------------------------------------------
        if lives < self._lives_prev:
            # Life lost — start respawn timer
            self._respawn_timer = RESPAWN_DURATION
        self._lives_prev = lives

        if self._respawn_timer > 0.0:
            self._respawn_timer = max(0.0, self._respawn_timer - dt_real)

        respawn_active = self._respawn_timer > 0.0

        # ----------------------------------------------------------------
        # 3. Risk field
        # ----------------------------------------------------------------
        asteroid_risks = self._risk_field.compute_all(
            ship_pos, ship_vel, game_state.asteroids
        )
        r_global = self._risk_field.aggregate(asteroid_risks)
        tau_min  = self._risk_field.min_tau(asteroid_risks)

        # ----------------------------------------------------------------
        # 4. Angular danger profile + corridors
        # ----------------------------------------------------------------
        profile        = self._angular_prof.build(asteroid_risks)
        corridors      = self._angular_prof.extract_corridors(profile)
        saturation     = self._angular_prof.saturation_fraction(profile)
        repulse_dir    = self._angular_prof.repulse_direction(profile)
        corridors_exist = len(corridors) > 0

        # ----------------------------------------------------------------
        # 5. Target selection
        # ----------------------------------------------------------------
        target = self._target_sel.select(asteroid_risks, ship_heading, r_global)

        # Determine the target bearing for corridor scoring
        # (fall back to repulse direction if no target)
        target_bearing = target.bearing if target is not None else repulse_dir

        # Score corridors using target bearing + inertia
        best_corridor = self._angular_prof.best_corridor(
            corridors, target_bearing, ship_heading
        )

        # ----------------------------------------------------------------
        # 6. Fuzzy heading command
        # ----------------------------------------------------------------
        # DESIGN NOTE: turn_rate and evasion are now separated.
        #
        # - turn_rate aims at the selected TARGET (offensive / fire-aligned).
        #   When there is no target, it steers toward the best corridor centre
        #   (pure evasion fallback).
        #
        # - The corridor information is used by the MPC and by the supervisor
        #   to decide whether to escalate, but it does NOT override the turn_rate
        #   in FUZZY mode when a target exists.  This prevents the ship from
        #   perpetually steering away from asteroids instead of shooting them.
        #
        # - Thrust (forward/backward) handles the separation from asteroids.

        if target is not None:
            # Primary: face the attack target so we can shoot it
            fuzzy_desired_heading = target.bearing
        elif best_corridor is not None:
            # No target — steer into the safest free corridor
            fuzzy_desired_heading = best_corridor.centre_deg
        else:
            # Fully saturated — follow the repulse gradient
            fuzzy_desired_heading = repulse_dir

        fuzzy_turn_rate = math_angle_to_turn_rate(
            ship_heading, fuzzy_desired_heading,
            k_omega=2.0, omega_max=180.0,
        )

        # ----------------------------------------------------------------
        # 7. Fuzzy thrust (speed policy + FIS)
        # ----------------------------------------------------------------
        from typing import List, Optional, Tuple
        THRUST_MAX      =  480.0       # px/s²
        THRUST_MIN      = -480.0
        OMEGA_MAX       =  180.0       # deg/s
        DRAG            =   80.0       # px/s²

        SPEED_MAX       = 200.0
        SPEED_MIN       =  20.0
        def _speed_target(r_global: float) -> float:
            return SPEED_MAX * (1.0 - r_global) + SPEED_MIN * r_global


        def _fuzzy_thrust(speed_error: float, r_global: float, tau_min: float) -> float:
            """
            Sugeno FIS for thrust control.

            Four rules — weighted average defuzzification.

            Fixes vs original:
            - Emergency brake activates smoothly from tau=1s (not 1.5s)
            - Asymmetric speed tracking: faster to accelerate than brake
            - Risk deceleration only kicks in above R=0.5
            - All rule weights balanced so ship can actually reverse when needed
            """
            rules: List[Tuple[float, float]] = []

            # R1 Emergency brake (tau < 1s → full weight)
            w_emg = float(np.clip((1.0 - tau_min) / 1.0, 0.0, 1.0))
            rules.append((w_emg * 1.5, THRUST_MIN * 0.9))

            # R2 Accelerate to reach s*
            w_acc = float(np.clip( speed_error / 100.0, 0.0, 1.0))
            rules.append((w_acc, THRUST_MAX * 0.7))

            # R3 Decelerate (gentler — avoids over-braking near zero)
            w_dec = float(np.clip(-speed_error / 150.0, 0.0, 1.0))
            rules.append((w_dec, THRUST_MIN * 0.5))

            # R4 Risk-modulated deceleration (above R=0.5)
            w_rsk = float(np.clip((r_global - 0.5) * 2.0, 0.0, 1.0)) * 0.6
            rules.append((w_rsk, THRUST_MIN * 0.3))

            total_w = sum(w for w, _ in rules)
            if total_w < 1e-9:
                return 0.0
            thrust = sum(w * o for w, o in rules) / total_w
            return float(np.clip(thrust, THRUST_MIN, THRUST_MAX))
        
        s_star = _speed_target(r_global)

        # When no asteroids are present tau_min=inf and r_global≈0,
        # s_star = SPEED_MAX.  If the ship is already at speed the error
        # is ~0 and all FIS rules produce ≈0 thrust — ship drifts to a
        # stop due to drag.  We add a drag-compensation floor so the ship
        # actively maintains its target speed even in a clear field.
        from Casadi_mpc import DRAG as _DRAG
        speed_error  = s_star - ship_speed
        fuzzy_thrust = _fuzzy_thrust(speed_error, r_global, tau_min)

        # Drag compensation: if FIS output is near zero but we need speed,
        # add just enough thrust to overcome drag at current speed.
        if abs(fuzzy_thrust) < 20.0 and abs(speed_error) > 10.0:
            drag_comp    = _DRAG * (1.0 if speed_error > 0 else -1.0)
            fuzzy_thrust = float(max(-480.0, min(480.0,
                                    fuzzy_thrust + drag_comp * 0.5)))

        # ----------------------------------------------------------------
        # 8. MPC planning
        # ----------------------------------------------------------------
        if respawn_active:
            mpc_result = self._mpc.plan_respawn(
                ship_pos, ship_vel, ship_speed, ship_heading,
                asteroid_risks, r_global, tau_min, repulse_dir,
            )
        else:
            mpc_result = self._mpc.plan(
                ship_pos, ship_vel, ship_speed, ship_heading,
                asteroid_risks, r_global, tau_min,
                mode='active',
                repulse_dir=repulse_dir,
            )

        # ----------------------------------------------------------------
        # 9. Fire decision
        # ----------------------------------------------------------------
        fire_decision = self._target_sel.fire_decision(
            ship_heading, target, can_fire
        )

        # ----------------------------------------------------------------
        # 10. Sacrifice heading: centroid of highest-risk asteroids
        # ----------------------------------------------------------------
        sacrifice_heading = self._compute_sacrifice_heading(
            asteroid_risks, ship_heading
        )

        # ----------------------------------------------------------------
        # 11. Modal supervisor
        # ----------------------------------------------------------------
        output: ControlOutput = self._supervisor.update(
            r_global         = r_global,
            tau_min          = tau_min,
            corridors_exist  = corridors_exist,
            saturation       = saturation,
            repulse_dir      = repulse_dir,
            mpc_feasible     = mpc_result.feasible,
            mpc_thrust       = mpc_result.thrust,
            mpc_turn_rate    = mpc_result.turn_rate,
            fuzzy_thrust     = fuzzy_thrust,
            fuzzy_turn_rate  = fuzzy_turn_rate,
            fire_decision    = fire_decision,
            lives_remaining  = lives,
            can_fire         = can_fire,
            respawn_timer    = self._respawn_timer,
            sacrifice_heading = sacrifice_heading,
        )

        # ----------------------------------------------------------------
        # 12. Tick selectors
        # ----------------------------------------------------------------
        self._target_sel.tick()

        # ----------------------------------------------------------------
        # Debug output
        # ----------------------------------------------------------------
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # --- Conditional breakpoint system ---
        debug_state = dict(
            frame          = self._frame,
            r_global       = r_global,
            tau_min        = tau_min,
            mode           = output.mode,
            ship_pos       = ship_pos,
            ship_speed     = ship_speed,
            ship_heading   = ship_heading,
            corridors      = corridors,
            asteroid_risks = asteroid_risks,
            target         = target,
            mpc_result     = mpc_result,
            output         = output,
            respawn_active = respawn_active,
            respawn_timer  = self._respawn_timer,
        )
        self._dbg.tick(debug_state)

        # --- One-line console log ---
        if DEBUG and self._frame % DEBUG_EVERY == 0:
            solver = getattr(mpc_result, 'solver_used', '?')
            print(
                f"[{self._frame:05d}] mode={output.mode:13s} "
                f"R={r_global:.3f} tau={tau_min:5.2f}s "
                f"cor={len(corridors):02d} "
                f"tgt={'#'+str(target.asteroid_id) if target else 'none':6s} "
                f"fire={str(output.fire):5s} mine={str(output.drop_mine):5s} "
                f"solver={solver:6s} "
                f"respawn={self._respawn_timer:.1f}s "
                f"dt={elapsed_ms:.1f}ms"
            )

        # --- Risk heatmap (parallel Tkinter window) ---
        if (HEATMAP_ENABLED
                and self._heatmap is not None
                and self._frame % HEATMAP_EVERY == 0):
            self._heatmap.update(
                ship_pos        = ship_pos,
                ship_heading    = ship_heading,
                asteroid_risks  = asteroid_risks,
                map_size        = self._map_size,
                r_global        = r_global,
                tau_min         = tau_min,
                mode            = output.mode,
                frame           = self._frame,
                target          = target,
                angular_profile = profile,
                mpc_trajectory  = None,
                solver_used     = getattr(mpc_result, 'solver_used', '?'),
            )

        # ----------------------------------------------------------------
        # 13. Return to Kessler
        # ----------------------------------------------------------------
        return (
            output.thrust,
            output.turn_rate,
            output.fire,
            output.drop_mine,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sacrifice_heading(
        self,
        asteroid_risks: list,
        current_heading: float,
    ) -> float:
        """
        Compute the heading toward the centroid of the three highest-risk
        asteroids.  Used by the supervisor to aim the sacrifice collision.

        Falls back to current_heading if no asteroids are present.
        """
        if not asteroid_risks:
            return current_heading

        # Take top-3 by risk
        top = asteroid_risks[:3]

        # Weighted average bearing (using risk as weight)
        # Convert to unit vectors to handle wrap-around correctly
        fx, fy = 0.0, 0.0
        for ar in top:
            b_rad = math.radians(ar.bearing)
            fx += ar.risk * math.cos(b_rad)
            fy += ar.risk * math.sin(b_rad)

        if math.hypot(fx, fy) < 1e-6:
            return current_heading

        return math.degrees(math.atan2(fy, fx)) % 360.0
