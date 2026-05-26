"""
modal_supervisor.py
===================
Hybrid modal supervisor implementing the three-level escalation ladder
and the sacrifice/respawn finite state machine.

Architecture
------------
The supervisor is the single entity that decides which controller is
active on any given frame.  It consumes the outputs of the risk field,
the angular profile, and the MPC planner, and emits:
  - active_mode : str — 'fuzzy' | 'mpc_evasion' | 'sacrifice' | 'respawn'
  - control_mask: dict — which actions are permitted this frame

Escalation ladder
-----------------
Level 1 — FUZZY (nominal)
    Condition  : R_global < R_LO and at least one free corridor
    Controller : Fuzzy FIS (visée + déplacement + tir)

Level 2 — MPC_EVASION
    Condition  : R_global >= R_LO or no corridor
    Controller : Directional MPC, fire disabled
    Exit up    : R_global < R_LO sustained for HOLD_FRAMES, corridor exists

Level 3 — SACRIFICE
    Condition  : MPC infeasible (no safe heading)
    Actions    : drop_mine = True, thrust toward asteroid cluster (voluntary death)
    Exit       : death detected → enter RESPAWN

RESPAWN (modal, not a level)
    Condition  : respawn_timer > 0 (set by Kessler on death)
    Controller : MPC pure survival (λ_d = 0), fire = 0 hard constraint
    Exit       : timer reaches 0 → return to FUZZY

Hysteresis
----------
To prevent chattering between FUZZY and MPC_EVASION:
  - Escalate up   when R_global > R_HI  (upper trigger)
  - De-escalate   when R_global < R_LO  sustained for HOLD_FRAMES frames

Sacrifice guard
---------------
Sacrifice is a costly last resort.  Additional guards:
  - Minimum lives remaining > 0 (obviously)
  - MPC has been infeasible for at least SACRIFICE_CONFIRM_FRAMES consecutive
    frames (avoids triggering on a single bad solve)
  - Not already in RESPAWN

Usage
-----
    from modal_supervisor import ModalSupervisor, ControlOutput
    sup = ModalSupervisor()
    out = sup.update(
        r_global, tau_min, corridors_exist,
        mpc_feasible, lives_remaining,
        respawn_timer, current_heading,
        mpc_result, target, can_fire
    )
    # out.mode, out.thrust, out.turn_rate, out.fire, out.drop_mine
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration — all thresholds in one place for easy tuning
# ---------------------------------------------------------------------------

# Risk thresholds (hysteresis band)
# Raised to give FUZZY mode more room to operate.
# FUZZY is active when R < R_LO; MPC kicks in above R_HI.
R_LO = 0.45    # de-escalation threshold (must be < R_HI)
R_HI = 0.70    # escalation threshold — only truly dangerous situations

# Minimum frames to stay in a mode before de-escalating
HOLD_FRAMES_MPC       = 24     # ~0.4s at 60 fps — faster de-escalation
HOLD_FRAMES_FUZZY     = 18     # shorter — fuzzy can re-escalate quickly

# Opportunistic fire in MPC mode: fire if a non-dangerous asteroid is
# well-aligned with the current heading (free shot, no manoeuvre needed)
MPC_FIRE_ANGLE_THRESH = 12.0   # degrees — max heading error to fire opportunistically
MPC_FIRE_RISK_THRESH  = 0.30   # only fire at asteroids below this individual risk
MPC_FIRE_TAU_MIN      = 1.5    # only fire if asteroid TTC > this (not imminent)

# Consecutive MPC-infeasible frames before triggering sacrifice
SACRIFICE_CONFIRM_FRAMES = 10

# Turn-rate gain for sacrifice heading (aim toward asteroid cluster)
K_SACRIFICE = 3.0

# Respawn: after respawn we force FUZZY with a brief warmup
RESPAWN_WARMUP_FRAMES = 30   # ~0.5s of pure survival before re-enabling tir


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

MODE_FUZZY    = 'fuzzy'
MODE_MPC      = 'mpc_evasion'
MODE_SACRIFICE = 'sacrifice'
MODE_RESPAWN  = 'respawn'


# ---------------------------------------------------------------------------
# Control output bundle
# ---------------------------------------------------------------------------

@dataclass
class ControlOutput:
    """All four Kessler actions plus diagnostics."""
    mode:       str     # current mode name
    thrust:     float   # u_T in [-480, 480] px/s²
    turn_rate:  float   # ω in [-180, 180] degrees/s
    fire:       bool    # fire bullet
    drop_mine:  bool    # drop mine
    # Diagnostics (not sent to Kessler)
    r_global:   float   = 0.0
    tau_min:    float   = math.inf
    note:       str     = ''


# ---------------------------------------------------------------------------
# Main supervisor
# ---------------------------------------------------------------------------

class ModalSupervisor:
    """
    Stateful modal supervisor.

    Instantiate once in the controller's __init__ and call update()
    every frame.

    Parameters
    ----------
    r_lo, r_hi         : hysteresis band thresholds
    hold_frames_mpc    : frames to hold MPC before de-escalating
    sacrifice_confirm  : frames of MPC infeasibility before sacrifice
    """

    def __init__(
        self,
        r_lo:              float = R_LO,
        r_hi:              float = R_HI,
        hold_frames_mpc:   int   = HOLD_FRAMES_MPC,
        sacrifice_confirm: int   = SACRIFICE_CONFIRM_FRAMES,
    ) -> None:
        self.r_lo             = r_lo
        self.r_hi             = r_hi
        self.hold_frames_mpc  = hold_frames_mpc
        self.sacrifice_confirm = sacrifice_confirm

        # Internal state
        self._mode:                  str = MODE_FUZZY
        self._frames_in_mode:        int = 0
        self._mpc_infeasible_streak: int = 0
        self._respawn_timer_prev:    float = 0.0
        self._post_respawn_frames:   int = 0
        self._sacrifice_armed:       bool = False
        self._in_respawn_prev:       bool = False   # edge detection for respawn

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def update(
        self,
        # Risk field outputs
        r_global:          float,
        tau_min:           float,
        # Angular profile outputs
        corridors_exist:   bool,
        saturation:        float,      # fraction of profile saturated [0,1]
        repulse_dir:       float,      # escape heading from angular profile
        # MPC outputs
        mpc_feasible:      bool,
        mpc_thrust:        float,
        mpc_turn_rate:     float,
        # Fuzzy outputs (pre-computed by caller)
        fuzzy_thrust:      float,
        fuzzy_turn_rate:   float,
        # Weapons
        fire_decision:     bool,       # from TargetSelector
        # Ship state
        lives_remaining:   int,
        can_fire:          bool,
        respawn_timer:     float,      # seconds remaining in respawn window
        current_heading:   float = 0.0,     # ship heading for opportunistic fire
        asteroid_risks_ref: list = None,    # List[AsteroidRisk] for opp. fire
        # Sacrifice heading (bearing of asteroid cluster centroid)
        sacrifice_heading: float = 0.0,
    ) -> ControlOutput:
        """
        Decide the active mode and emit control outputs for this frame.

        The caller is responsible for:
          - Computing fuzzy_thrust, fuzzy_turn_rate (from angular profile + FIS)
          - Computing mpc_thrust, mpc_turn_rate (from MPCDirector)
          - Computing fire_decision (from TargetSelector)
          - Providing respawn_timer > 0 during the invulnerability window

        Returns
        -------
        ControlOutput with all four Kessler actions.
        """
        self._frames_in_mode += 1

        # ----------------------------------------------------------------
        # 0. Respawn handling
        # ----------------------------------------------------------------
        in_respawn = respawn_timer > 0.0

        # Detect the rising edge: respawn just started this frame
        if in_respawn and not self._in_respawn_prev:
            self._set_mode(MODE_RESPAWN)
            self._post_respawn_frames = RESPAWN_WARMUP_FRAMES
            self._mpc_infeasible_streak = 0
            self._sacrifice_armed = False

        # Detect the falling edge: respawn just ended this frame
        # _post_respawn_frames was already set on the rising edge,
        # so it will naturally count down in the warmup block below.
        self._in_respawn_prev = in_respawn

        if in_respawn:
            return self._respawn_control(
                mpc_thrust, mpc_turn_rate, r_global, tau_min, respawn_timer
            )

        # Post-respawn warmup: stay in FUZZY without firing
        if self._post_respawn_frames > 0:
            self._post_respawn_frames -= 1
            return self._emit(
                MODE_FUZZY, fuzzy_thrust, fuzzy_turn_rate,
                fire=False, drop_mine=False,
                r_global=r_global, tau_min=tau_min,
                note=f'post-respawn warmup ({self._post_respawn_frames} frames left)',
            )

        # ----------------------------------------------------------------
        # 1. Sacrifice mode: MPC infeasible for N consecutive frames
        # ----------------------------------------------------------------
        if not mpc_feasible:
            self._mpc_infeasible_streak += 1
        else:
            self._mpc_infeasible_streak = max(
                0, self._mpc_infeasible_streak - 1
            )

        sacrifice_triggered = (
            self._mpc_infeasible_streak >= self.sacrifice_confirm
            and lives_remaining > 1
            and self._mode != MODE_SACRIFICE
        )

        if sacrifice_triggered or self._mode == MODE_SACRIFICE:
            return self._sacrifice_control(
                r_global, tau_min, sacrifice_heading, lives_remaining
            )

        # ----------------------------------------------------------------
        # 2. Escalation ladder: FUZZY ↔ MPC_EVASION
        # ----------------------------------------------------------------
        self._update_mode(r_global, corridors_exist, mpc_feasible)

        # ----------------------------------------------------------------
        # 3. Emit controls based on active mode
        # ----------------------------------------------------------------
        if self._mode == MODE_FUZZY:
            return self._emit(
                MODE_FUZZY,
                thrust    = fuzzy_thrust,
                turn_rate = fuzzy_turn_rate,
                fire      = fire_decision and can_fire,
                drop_mine = False,
                r_global  = r_global,
                tau_min   = tau_min,
            )

        else:  # MODE_MPC
            # Opportunistic fire: if a low-risk asteroid happens to be
            # well-aligned with our current heading, fire without manoeuvring.
            # This doesn't compromise evasion since we're already facing that way.
            opp_fire = self._opportunistic_fire(
                asteroid_risks_ref, current_heading, can_fire
            )
            return self._emit(
                MODE_MPC,
                thrust    = mpc_thrust,
                turn_rate = mpc_turn_rate,
                fire      = opp_fire,
                drop_mine = False,
                r_global  = r_global,
                tau_min   = tau_min,
                note      = f'R={r_global:.2f} corridors={corridors_exist} opp_fire={opp_fire}',
            )

    # ------------------------------------------------------------------
    # Mode transition logic
    # ------------------------------------------------------------------

    def _update_mode(
        self,
        r_global:       float,
        corridors_exist: bool,
        mpc_feasible:   bool,
    ) -> None:
        """
        Apply hysteresis-based escalation / de-escalation between
        FUZZY and MPC_EVASION.
        """
        # --- Escalate to MPC ---
        should_escalate = (
            r_global > self.r_hi
            or not corridors_exist
        )
        if should_escalate and self._mode == MODE_FUZZY:
            self._set_mode(MODE_MPC)
            return

        # --- De-escalate to FUZZY ---
        # Requires sustained low risk for HOLD_FRAMES_MPC frames
        should_deescalate = (
            r_global < self.r_lo
            and corridors_exist
            and self._frames_in_mode >= self.hold_frames_mpc
        )
        if should_deescalate and self._mode == MODE_MPC:
            self._set_mode(MODE_FUZZY)

    # ------------------------------------------------------------------
    # Sacrifice control
    # ------------------------------------------------------------------

    def _sacrifice_control(
        self,
        r_global:          float,
        tau_min:           float,
        sacrifice_heading: float,
        lives_remaining:   int,
    ) -> ControlOutput:
        """
        Sacrifice mode:
        1. Drop mine immediately.
        2. Turn toward the asteroid cluster centroid (voluntary collision).
        3. Full thrust toward the cluster.

        After the death event, Kessler will trigger respawn, which the
        supervisor detects via respawn_timer > 0 in the next call.
        """
        self._set_mode(MODE_SACRIFICE)

        # Turn toward cluster at maximum rate
        turn_rate = math.copysign(180.0, sacrifice_heading)

        return self._emit(
            MODE_SACRIFICE,
            thrust    = 0,          # full thrust toward cluster
            turn_rate = turn_rate,
            fire      = True,           # concentrate fire during invulnerability
            drop_mine = True,           # trigger mine
            r_global  = r_global,
            tau_min   = tau_min,
            note      = f'sacrifice armed, lives={lives_remaining}',
        )

    # ------------------------------------------------------------------
    # Respawn control
    # ------------------------------------------------------------------

    def _respawn_control(
        self,
        mpc_thrust:   float,
        mpc_turn_rate: float,
        r_global:     float,
        tau_min:      float,
        respawn_timer: float,
    ) -> ControlOutput:
        """
        During the 3s invulnerability window:
        - Pure survival MPC (λ_d = 0, no fire)
        - Reset sacrifice counter and sacrifice state
        - Track remaining warmup frames for post-respawn
        """
        self._set_mode(MODE_RESPAWN)
        self._mpc_infeasible_streak = 0
        self._sacrifice_armed       = False

        # When respawn ends, plan warmup
        if respawn_timer < 0.1:   # last frames of respawn window
            self._post_respawn_frames = RESPAWN_WARMUP_FRAMES

        return self._emit(
            MODE_RESPAWN,
            thrust    = mpc_thrust,
            turn_rate = mpc_turn_rate,
            fire      = False,     # hard constraint: no firing during respawn
            drop_mine = False,
            r_global  = r_global,
            tau_min   = tau_min,
            note      = f'respawn {respawn_timer:.1f}s remaining',
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _opportunistic_fire(
        self,
        asteroid_risks: list,   # may be None
        current_heading: float,
        can_fire: bool,
    ) -> bool:
        """
        Fire in MPC mode if a safe, aligned asteroid is available.

        Conditions (all must hold):
        - can_fire is True (weapon cooldown)
        - Asteroid individual risk < MPC_FIRE_RISK_THRESH (not dangerous)
        - Asteroid TTC > MPC_FIRE_TAU_MIN (not imminent — we're not fleeing it)
        - Heading error to asteroid < MPC_FIRE_ANGLE_THRESH (already facing it)
        """
        if not can_fire or asteroid_risks is None:
            return False

        from toric_utils import angular_diff
        for ar in asteroid_risks:
            if ar.risk >= MPC_FIRE_RISK_THRESH:
                continue
            if ar.tau <= MPC_FIRE_TAU_MIN:
                continue
            heading_err = abs(angular_diff(current_heading, ar.bearing))
            if heading_err <= MPC_FIRE_ANGLE_THRESH:
                return True
        return False

    def _set_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode           = mode
            self._frames_in_mode = 0

    @staticmethod
    def _emit(
        mode: str,
        thrust: float,
        turn_rate: float,
        fire: bool,
        drop_mine: bool,
        r_global: float   = 0.0,
        tau_min:  float   = math.inf,
        note:     str     = '',
    ) -> ControlOutput:
        # Clamp to Kessler's legal ranges
        thrust    = max(-480.0, min(480.0, thrust))
        turn_rate = max(-180.0, min(180.0, turn_rate))
        return ControlOutput(
            mode      = mode,
            thrust    = thrust,
            turn_rate = turn_rate,
            fire      = fire,
            drop_mine = drop_mine,
            r_global  = r_global,
            tau_min   = tau_min,
            note      = note,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> str:
        return self._mode

    @property
    def mpc_infeasible_streak(self) -> int:
        return self._mpc_infeasible_streak

    @property
    def frames_in_mode(self) -> int:
        return self._frames_in_mode
