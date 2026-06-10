"""
params.py — tunable parameter vector for the fuzzy controller.

`ControllerParams` holds every knob the optimizer is allowed to change, with
defaults equal to the current hand-tuned values. `apply(p)` writes them back
into the controller modules.

IMPORTANT: apply() mutates module globals, and the controller's sub-objects read
those globals at *construction* time. So the optimizer MUST call apply(p) BEFORE
building a fresh Controller() (see optimize.fitness). Mutating globals does not
retro-actively change an already-built controller.

This is single-process safe. For parallel evaluation, run each individual in its
own process (globals are per-process).
"""
from __future__ import annotations
from dataclasses import dataclass, fields

import test_controller_fuzzy as tcf
import target_selector as tsel
import risk_field as rf
import angular_profile as ap


@dataclass
class ControllerParams:
    # --- target_selector ---
    FIRE_ANGLE_THRESHOLD: float = 12.0      # deg; fire cone (free ammo -> wider = more shots)
    PREDICT_DT: float = 1.0 / 3.0           # s; aim lead time
    COMMIT_FRAMES: float = 12.0             # frames a target is held (rounded to int)
    COMMIT_SWITCH_RATIO: float = 1.35       # new target must beat held by this factor
    AIM_FEASIBILITY_KAPPA: float = 35.0     # deg; aim-cost falloff in target score
    URGENCY_TAU_WEIGHT: float = 0.65        # urgency split (speed weight = 1 - this)

    # --- risk_field ---
    ALPHA_AGGREGATION: float = 0.60         # soft-max sharpness of global risk
    OUT_HIGH: float = 0.75                  # per-asteroid risk singleton (high)
    OUT_MEDIUM: float = 0.35                # (medium)
    OUT_LOW: float = 0.12                   # (low)

    # --- angular_profile ---
    RHO_THRESHOLD: float = 0.38             # below this a bearing counts as a free corridor
    SIGMA_BASE_DEG: float = 15.0            # min danger-lobe width
    SIGMA_TAU_SCALE: float = 20.0           # extra width for imminent asteroids
    W_FREEDOM: float = 0.40                 # corridor-pick weight: openness
    W_ALIGNMENT: float = 0.35               # corridor-pick weight: alignment to target

    # --- rule turn-gains (turn = gain * angular_error) ---
    K_A2: float = 2.0
    K_A3: float = 3.5
    K_A4: float = 4.0
    K_S1: float = 5.0
    K_S2: float = 4.5
    K_S3: float = 5.0
    K_S4: float = 4.0
    K_E1: float = 5.0

    # --- rule thrust setpoints (px/s^2) ---
    THR_B1: float = 40.0                    # medium-risk cruise (survival-sensitive)
    THR_B2: float = 60.0
    THR_S1: float = -220.0                  # imminent-collision reverse
    THR_S3: float = -260.0                  # saturated no-corridor escape
    THR_E1: float = 360.0                   # proximity-evade flee thrust

    # --- proximity-evade trigger window (px surface dist); trapezoid (0,0,C,D) ---
    PROX_C: float = 45.0
    PROX_D: float = 90.0

    # --- life / emergency ---
    LIFE_S_MULT: float = 1.25               # survival-rule weight boost on last life
    EMERGENCY_STRENGTH_TH: float = 0.15     # emergency latch threshold


# Min/max search bounds for each field (used by the optimizer).
BOUNDS = {
    "FIRE_ANGLE_THRESHOLD": (3.0, 30.0),
    "PREDICT_DT": (0.0, 0.8),
    "COMMIT_FRAMES": (1.0, 30.0),
    "COMMIT_SWITCH_RATIO": (1.0, 2.5),
    "AIM_FEASIBILITY_KAPPA": (10.0, 90.0),
    "URGENCY_TAU_WEIGHT": (0.0, 1.0),
    "ALPHA_AGGREGATION": (0.2, 0.95),
    "OUT_HIGH": (0.5, 1.0),
    "OUT_MEDIUM": (0.1, 0.7),
    "OUT_LOW": (0.0, 0.4),
    "RHO_THRESHOLD": (0.15, 0.70),
    "SIGMA_BASE_DEG": (5.0, 40.0),
    "SIGMA_TAU_SCALE": (0.0, 60.0),
    "W_FREEDOM": (0.0, 1.0),
    "W_ALIGNMENT": (0.0, 1.0),
    "K_A2": (0.5, 8.0),
    "K_A3": (0.5, 8.0),
    "K_A4": (0.5, 8.0),
    "K_S1": (1.0, 10.0),
    "K_S2": (1.0, 10.0),
    "K_S3": (1.0, 10.0),
    "K_S4": (1.0, 10.0),
    "K_E1": (1.0, 10.0),
    "THR_B1": (-200.0, 200.0),
    "THR_B2": (-200.0, 200.0),
    "THR_S1": (-480.0, 0.0),
    "THR_S3": (-480.0, 0.0),
    "THR_E1": (-480.0, 480.0),
    "PROX_C": (10.0, 150.0),
    "PROX_D": (20.0, 250.0),
    "LIFE_S_MULT": (1.0, 2.0),
    "EMERGENCY_STRENGTH_TH": (0.05, 0.50),
}

# Order of genes in the optimization vector. Trim this list to optimize fewer
# parameters, or keep all of them for a full search.
FREE = [f.name for f in fields(ControllerParams)]


def to_vector(p: ControllerParams) -> list:
    return [getattr(p, k) for k in FREE]


def from_vector(vec) -> ControllerParams:
    base = ControllerParams()
    for k, v in zip(FREE, vec):
        setattr(base, k, float(v))
    return base


def clamp_vector(vec) -> list:
    out = []
    for k, v in zip(FREE, vec):
        lo, hi = BOUNDS[k]
        out.append(min(hi, max(lo, float(v))))
    return out


def apply(p: ControllerParams) -> None:
    """Write parameters into the controller modules. Call BEFORE building Controller()."""
    # target_selector
    tsel.FIRE_ANGLE_THRESHOLD = p.FIRE_ANGLE_THRESHOLD
    tsel.PREDICT_DT = p.PREDICT_DT
    tsel.COMMIT_FRAMES = int(round(p.COMMIT_FRAMES))
    tsel.COMMIT_SWITCH_RATIO = p.COMMIT_SWITCH_RATIO
    tsel.AIM_FEASIBILITY_KAPPA = p.AIM_FEASIBILITY_KAPPA
    tsel.URGENCY_TAU_WEIGHT = p.URGENCY_TAU_WEIGHT
    tsel.URGENCY_SPEED_WEIGHT = 1.0 - p.URGENCY_TAU_WEIGHT
    # risk_field
    rf.ALPHA_AGGREGATION = p.ALPHA_AGGREGATION
    rf.OUT_HIGH = p.OUT_HIGH
    rf.OUT_MEDIUM = p.OUT_MEDIUM
    rf.OUT_LOW = p.OUT_LOW
    # angular_profile
    ap.RHO_THRESHOLD = p.RHO_THRESHOLD
    ap.SIGMA_BASE_DEG = p.SIGMA_BASE_DEG
    ap.SIGMA_TAU_SCALE = p.SIGMA_TAU_SCALE
    ap.W_FREEDOM = p.W_FREEDOM
    ap.W_ALIGNMENT = p.W_ALIGNMENT
    # test_controller_fuzzy gains / thrusts
    tcf.K_A2 = p.K_A2; tcf.K_A3 = p.K_A3; tcf.K_A4 = p.K_A4
    tcf.K_S1 = p.K_S1; tcf.K_S2 = p.K_S2; tcf.K_S3 = p.K_S3
    tcf.K_S4 = p.K_S4; tcf.K_E1 = p.K_E1
    tcf.THR_B1 = p.THR_B1; tcf.THR_B2 = p.THR_B2
    tcf.THR_S1 = p.THR_S1; tcf.THR_S3 = p.THR_S3; tcf.THR_E1 = p.THR_E1
    tcf.PROX_VERYCLOSE = (0.0, 0.0, p.PROX_C, p.PROX_D)
    tcf.LIFE_S_MULT = p.LIFE_S_MULT
    tcf.EMERGENCY_STRENGTH_TH = p.EMERGENCY_STRENGTH_TH
