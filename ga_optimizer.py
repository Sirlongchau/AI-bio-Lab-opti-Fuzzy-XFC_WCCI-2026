"""
ga_optimizer.py
===============
Genetic Algorithm optimizer for the Kessler multimodal controller system.

Architecture
------------
Optimises hyperparameters across three layers simultaneously:

  1. Supervisor      — mode-switching thresholds (R_lo, R_hi)
  2. RiskField       — FIS membership function breakpoints, aggregation alpha
  3. FuzzyController — TSK rule gains, thrust setpoints
  4. TargetSelector  — commitment, urgency, fire angle, acquisition FIS

The MPC (CasADi/IPOPT) layer is intentionally excluded: its parameters
have been validated independently and touching danger_scale, W_thrust etc.
produces ill-conditioned NLPs that inflate solve time unpredictably.

Each individual is a flat numpy array (genome) encoding all tuneable
parameters.  A Genome class wraps encode/decode logic so the GA never
needs to know about parameter semantics.

Fitness
-------
A single scenario (or a small set of rotating scenarios) is evaluated for
each individual.  Fitness = weighted combination of:
    - asteroids_hit      (maximise)
    - deaths             (minimise, penalised heavily)
    - accuracy           (maximise)
    - mean_eval_time     (minimise — stay under budget)

GA Features
-----------
- Tournament selection
- Uniform crossover + Gaussian mutation
- Elitism (top-k survive unchanged)
- Constraint repair: clamp genes back into legal bounds after mutation
- Parallel evaluation via multiprocessing.Pool
- Checkpoint / resume via JSON

Usage
-----
    python ga_optimizer.py                        # run with defaults
    python ga_optimizer.py --pop 40 --gen 30      # custom population/generations
    python ga_optimizer.py --resume checkpoint.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parameter schema — single source of truth for bounds and defaults
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """One tuneable parameter."""
    name:    str
    default: float
    lo:      float
    hi:      float
    group:   str        # 'supervisor' | 'risk_field' | 'fuzzy' | 'mpc'

    def clip(self, v: float) -> float:
        return float(np.clip(v, self.lo, self.hi))


# Build the full parameter list -----------------------------------------------

PARAM_SPECS: List[ParamSpec] = [

    # ---- Supervisor --------------------------------------------------------
    ParamSpec("R_lo",              0.80,  0.30, 0.85, "supervisor"),
    ParamSpec("R_hi",              0.90,  0.60, 0.99, "supervisor"),

    # ---- RiskField — aggregation -------------------------------------------
    ParamSpec("alpha_aggregation", 0.60,  0.20, 0.95, "risk_field"),

    # ---- RiskField — tau breakpoints (critical) ----------------------------
    ParamSpec("tau_crit_b",        0.25,  0.05, 0.60, "risk_field"),
    ParamSpec("tau_crit_d",        0.50,  0.20, 1.20, "risk_field"),

    # ---- RiskField — tau close ---------------------------------------------
    ParamSpec("tau_close_a",       0.25,  0.10, 0.80, "risk_field"),
    ParamSpec("tau_close_b",       0.50,  0.20, 1.20, "risk_field"),
    ParamSpec("tau_close_d",       1.50,  0.60, 3.00, "risk_field"),

    # ---- RiskField — tau medium --------------------------------------------
    ParamSpec("tau_med_a",         0.60,  0.20, 1.50, "risk_field"),
    ParamSpec("tau_med_c",         1.50,  0.60, 3.50, "risk_field"),
    ParamSpec("tau_med_d",         5.50,  2.00, 9.00, "risk_field"),

    # ---- RiskField — distance near -----------------------------------------
    ParamSpec("d_near_c",         50.0,  20.0, 120.0, "risk_field"),
    ParamSpec("d_near_d",         90.0,  50.0, 200.0, "risk_field"),

    # ---- RiskField — distance medium ---------------------------------------
    ParamSpec("d_med_a",          50.0,  20.0, 120.0, "risk_field"),
    ParamSpec("d_med_d",         220.0,  80.0, 400.0, "risk_field"),

    # ---- RiskField — output singletons -------------------------------------
    ParamSpec("out_high",          0.75,  0.40, 0.95, "risk_field"),
    ParamSpec("out_medium",        0.20,  0.05, 0.50, "risk_field"),
    ParamSpec("out_low",           0.15,  0.02, 0.35, "risk_field"),
    ParamSpec("out_negligible",    0.02,  0.001, 0.10,"risk_field"),

    # ---- FuzzyController — rule gains --------------------------------------
    ParamSpec("K_A1",   0.05, 0.01, 0.5,  "fuzzy"),
    ParamSpec("K_A2",   2.00, 0.5,  5.0,  "fuzzy"),
    ParamSpec("K_A3",   3.50, 1.0,  7.0,  "fuzzy"),
    ParamSpec("K_A4",   4.00, 1.0,  8.0,  "fuzzy"),
    ParamSpec("CA_B1_AIM",  1.65, 0.3, 4.0, "fuzzy"),
    ParamSpec("CA_B1_CORR", 1.35, 0.3, 4.0, "fuzzy"),
    ParamSpec("CA_B2_AIM",  0.90, 0.1, 3.0, "fuzzy"),
    ParamSpec("CA_B2_CORR", 2.45, 0.5, 5.0, "fuzzy"),
    ParamSpec("K_B3",   4.00, 0.5,  8.0,  "fuzzy"),
    ParamSpec("K_S1",   5.00, 1.0, 10.0,  "fuzzy"),
    ParamSpec("K_S2",   4.50, 1.0, 10.0,  "fuzzy"),
    ParamSpec("K_S3",   5.00, 1.0, 10.0,  "fuzzy"),
    ParamSpec("K_S4",   4.00, 1.0, 10.0,  "fuzzy"),
    ParamSpec("K_E1",   5.00, 1.0, 12.0,  "fuzzy"),
    ParamSpec("K_FALLBACK", 3.0, 0.5, 7.0,"fuzzy"),

    # ---- FuzzyController — thrust setpoints --------------------------------
    ParamSpec("THR_A1",   0.0,   -60.0,  60.0,  "fuzzy"),
    ParamSpec("THR_A2",   0.0,   -60.0,  60.0,  "fuzzy"),
    ParamSpec("THR_A3",   0.0,   -60.0,  60.0,  "fuzzy"),
    ParamSpec("THR_A4",   0.0,   -60.0,  60.0,  "fuzzy"),
    ParamSpec("THR_B1",  40.0,  -100.0, 120.0,  "fuzzy"),
    ParamSpec("THR_B2",  60.0,  -100.0, 150.0,  "fuzzy"),
    ParamSpec("THR_B3", -40.0,  -200.0,  60.0,  "fuzzy"),
    ParamSpec("THR_S1",-220.0,  -480.0,  -80.0,  "fuzzy"),
    ParamSpec("THR_S2",-120.0,  -480.0,  -40.0,  "fuzzy"),
    ParamSpec("THR_S3",-260.0,  -480.0,  -80.0,  "fuzzy"),
    ParamSpec("THR_S4",-300.0,  -480.0,  -80.0,  "fuzzy"),
    ParamSpec("THR_E1", 360.0,   120.0,  480.0,  "fuzzy"),

    # ---- FuzzyController — life multipliers --------------------------------
    ParamSpec("LIFE_S_MULT", 1.25, 0.8, 2.5, "fuzzy"),
    ParamSpec("LIFE_A_MULT", 1.10, 0.8, 2.0, "fuzzy"),

    # ---- TargetSelector — commitment & fire logic --------------------------
    ParamSpec("ts_commit_frames",       12,    4,   40,   "target_selector"),
    ParamSpec("ts_commit_switch_ratio", 1.35,  1.0,  2.5, "target_selector"),
    ParamSpec("ts_tau_urgent_override", 0.50,  0.10, 1.20, "target_selector"),
    ParamSpec("ts_fire_angle_threshold",10.0,  2.0,  18.0, "target_selector"),
    ParamSpec("ts_min_score",           0.001, 0.0,  0.05, "target_selector"),

    # ---- TargetSelector — urgency scoring ----------------------------------
    ParamSpec("ts_aim_feasibility_kappa", 35.0, 5.0, 100.0, "target_selector"),
    ParamSpec("ts_urgency_tau_cap",        2.0,  0.5,   5.0, "target_selector"),
    ParamSpec("ts_urgency_speed_scale",  180.0, 50.0, 400.0, "target_selector"),
    ParamSpec("ts_urgency_tau_weight",     0.65, 0.1,   0.9, "target_selector"),
    # urgency_speed_weight = 1 - tau_weight, kept consistent via repair()

    # ---- TargetSelector — predict lead -------------------------------------
    ParamSpec("ts_predict_dt",  0.333, 0.05, 0.50, "target_selector"),
    ParamSpec("ts_frag_penalty_large", 0.0, 0.0, 0.8, "target_selector"),

    # ---- TargetSelector — acquisition FIS: heading error breakpoints -------
    # HE_TINY  = (0, 0, b1, b2)
    ParamSpec("ts_he_tiny_b1",    2.0,  0.5,  5.0,  "target_selector"),
    ParamSpec("ts_he_tiny_b2",    6.0,  2.0, 15.0,  "target_selector"),
    # HE_SMALL = (a, b, c, d)
    ParamSpec("ts_he_small_a",    2.0,  0.5,  8.0,  "target_selector"),
    ParamSpec("ts_he_small_c",   12.0,  5.0, 30.0,  "target_selector"),
    ParamSpec("ts_he_small_d",   25.0, 10.0, 50.0,  "target_selector"),
    # HE_MEDIUM = (a, b, c, d)
    ParamSpec("ts_he_med_a",     15.0,  5.0, 40.0,  "target_selector"),
    ParamSpec("ts_he_med_b",     30.0, 10.0, 60.0,  "target_selector"),
    ParamSpec("ts_he_med_c",     60.0, 30.0, 120.0, "target_selector"),
    # HE_LARGE starts at ts_he_med_c by construction

    # ---- TargetSelector — acquisition FIS: distance breakpoints ------------
    # D_CLOSE = (0, 0, c, d)
    ParamSpec("ts_d_close_c",    90.0, 30.0, 200.0, "target_selector"),
    ParamSpec("ts_d_close_d",   180.0, 80.0, 350.0, "target_selector"),
    # D_MEDIUM = (a, b, c, d)
    ParamSpec("ts_d_med_a",     120.0, 50.0, 250.0, "target_selector"),
    ParamSpec("ts_d_med_b",     220.0, 80.0, 400.0, "target_selector"),
    ParamSpec("ts_d_med_c",     400.0,150.0, 700.0, "target_selector"),
    ParamSpec("ts_d_med_d",     600.0,250.0, 900.0, "target_selector"),

    # ---- TargetSelector — acquisition FIS: difficulty singletons -----------
    ParamSpec("ts_diff_easy",   0.05, 0.01, 0.30, "target_selector"),
    ParamSpec("ts_diff_medium", 0.40, 0.15, 0.70, "target_selector"),
    ParamSpec("ts_diff_hard",   0.85, 0.50, 1.00, "target_selector"),

]

# MPC (CasADi/IPOPT) parameters are intentionally excluded from GA optimisation.
# The MPC has been validated independently and its hyperparameters are stable.
# Letting the GA touch danger_scale, W_thrust etc. produces ill-conditioned NLPs
# that inflate solve time without improving gameplay.

N_PARAMS = len(PARAM_SPECS)
PARAM_NAMES = [p.name for p in PARAM_SPECS]


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

class Genome:
    """Wraps a numpy float64 array with encode/decode helpers."""

    def __init__(self, genes: Optional[np.ndarray] = None) -> None:
        if genes is None:
            self.genes = np.array([p.default for p in PARAM_SPECS], dtype=np.float64)
        else:
            self.genes = np.array(genes, dtype=np.float64)

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        genes = np.array([
            rng.uniform(p.lo, p.hi) for p in PARAM_SPECS
        ], dtype=np.float64)
        return cls(genes)

    def to_dict(self) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(PARAM_NAMES, self.genes)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Genome":
        genes = np.array([d.get(name, PARAM_SPECS[i].default)
                          for i, name in enumerate(PARAM_NAMES)], dtype=np.float64)
        return cls(genes)

    def repair(self) -> None:
        """Clamp all genes to their legal bounds in-place."""
        for i, spec in enumerate(PARAM_SPECS):
            self.genes[i] = spec.clip(self.genes[i])
        # Enforce R_lo < R_hi
        lo_idx = PARAM_NAMES.index("R_lo")
        hi_idx = PARAM_NAMES.index("R_hi")
        if self.genes[lo_idx] >= self.genes[hi_idx]:
            mid = (self.genes[lo_idx] + self.genes[hi_idx]) / 2.0
            self.genes[lo_idx] = max(PARAM_SPECS[lo_idx].lo, mid - 0.05)
            self.genes[hi_idx] = min(PARAM_SPECS[hi_idx].hi, mid + 0.05)

        # Enforce heading-error MF ordering: he_tiny_b1 < he_tiny_b2 < he_small_c < he_small_d
        def _order(a_name: str, b_name: str, gap: float = 0.5) -> None:
            ai = PARAM_NAMES.index(a_name)
            bi = PARAM_NAMES.index(b_name)
            if self.genes[ai] >= self.genes[bi]:
                mid = (self.genes[ai] + self.genes[bi]) / 2.0
                self.genes[ai] = max(PARAM_SPECS[ai].lo, mid - gap)
                self.genes[bi] = min(PARAM_SPECS[bi].hi, mid + gap)

        _order("ts_he_tiny_b1",  "ts_he_tiny_b2",  0.5)
        _order("ts_he_tiny_b2",  "ts_he_small_c",  1.0)
        _order("ts_he_small_c",  "ts_he_small_d",  1.0)
        _order("ts_he_small_d",  "ts_he_med_a",    1.0)
        _order("ts_he_med_a",    "ts_he_med_b",    2.0)
        _order("ts_he_med_b",    "ts_he_med_c",    5.0)

        # Enforce distance MF ordering
        _order("ts_d_close_c",  "ts_d_close_d",  5.0)
        _order("ts_d_close_d",  "ts_d_med_a",    1.0)
        _order("ts_d_med_a",    "ts_d_med_b",   10.0)
        _order("ts_d_med_b",    "ts_d_med_c",   10.0)
        _order("ts_d_med_c",    "ts_d_med_d",   10.0)

        # Enforce difficulty ordering: easy < medium < hard
        _order("ts_diff_easy",  "ts_diff_medium", 0.05)
        _order("ts_diff_medium","ts_diff_hard",   0.05)

    def copy(self) -> "Genome":
        return Genome(self.genes.copy())


# ---------------------------------------------------------------------------
# Parameter injection helpers
# ---------------------------------------------------------------------------

def apply_genome_to_modules(genome: Genome) -> None:
    """
    Patch module-level constants before instantiating controllers.
    Call this inside the worker process so the main process is unaffected.
    """
    d = genome.to_dict()

    # ---- risk_field --------------------------------------------------------
    try:
        import risk_field as rf

        rf.ALPHA_AGGREGATION = d["alpha_aggregation"]

        # Tau MFs (keep a=0 for critical/close; reuse existing shapes)
        cb = d["tau_crit_b"];  cd = d["tau_crit_d"]
        rf.TAU_CRITICAL = (0.0, 0.0, cb, cd)

        ca = d["tau_close_a"]; cbr = d["tau_close_b"]; cdr = d["tau_close_d"]
        rf.TAU_CLOSE = (ca, cbr, cbr, cdr)

        ma = d["tau_med_a"];   mc = d["tau_med_c"]; md = d["tau_med_d"]
        rf.TAU_MEDIUM = (ma, mc, mc, md)
        rf.TAU_FAR    = (mc, md, 99.0, 99.0)

        # Distance MFs
        dnc = d["d_near_c"];  dnd = d["d_near_d"]
        rf.D_NEAR = (0.0, 0.0, dnc, dnd)
        dma = d["d_med_a"]; dmd = d["d_med_d"]
        rf.D_MEDIUM = (dma, dnd, dnd, dmd)
        rf.D_FAR    = (dnd, dmd, 9999.0, 9999.0)

        # Output singletons
        rf.OUT_HIGH       = d["out_high"]
        rf.OUT_MEDIUM     = d["out_medium"]
        rf.OUT_LOW        = d["out_low"]
        rf.OUT_NEGLIGIBLE = d["out_negligible"]

    except ImportError:
        pass

    # ---- test_controller_fuzzy ---------------------------------------------
    try:
        import test_controller_fuzzy as fc

        fc.K_A1 = d["K_A1"];  fc.THR_A1 = d["THR_A1"]
        fc.K_A2 = d["K_A2"];  fc.THR_A2 = d["THR_A2"]
        fc.K_A3 = d["K_A3"];  fc.THR_A3 = d["THR_A3"]
        fc.K_A4 = d["K_A4"];  fc.THR_A4 = d["THR_A4"]
        fc.CA_B1_AIM  = d["CA_B1_AIM"];  fc.CA_B1_CORR = d["CA_B1_CORR"]
        fc.THR_B1     = d["THR_B1"]
        fc.CA_B2_AIM  = d["CA_B2_AIM"];  fc.CA_B2_CORR = d["CA_B2_CORR"]
        fc.THR_B2     = d["THR_B2"]
        fc.K_B3 = d["K_B3"];  fc.THR_B3 = d["THR_B3"]
        fc.K_S1 = d["K_S1"];  fc.THR_S1 = d["THR_S1"]
        fc.K_S2 = d["K_S2"];  fc.THR_S2 = d["THR_S2"]
        fc.K_S3 = d["K_S3"];  fc.THR_S3 = d["THR_S3"]
        fc.K_S4 = d["K_S4"];  fc.THR_S4 = d["THR_S4"]
        fc.K_E1 = d["K_E1"];  fc.THR_E1 = d["THR_E1"]
        fc.K_FALLBACK      = d["K_FALLBACK"]
        fc.LIFE_S_MULT     = d["LIFE_S_MULT"]
        fc.LIFE_A_MULT     = d["LIFE_A_MULT"]

    except ImportError:
        pass

    # ---- target_selector ---------------------------------------------------
    try:
        import target_selector as ts

        ts.COMMIT_FRAMES         = int(round(d["ts_commit_frames"]))
        ts.COMMIT_SWITCH_RATIO   = d["ts_commit_switch_ratio"]
        ts.TAU_URGENT_OVERRIDE   = d["ts_tau_urgent_override"]
        ts.FIRE_ANGLE_THRESHOLD  = d["ts_fire_angle_threshold"]
        ts.MIN_SCORE_TO_TARGET   = d["ts_min_score"]

        ts.AIM_FEASIBILITY_KAPPA = d["ts_aim_feasibility_kappa"]
        ts.URGENCY_TAU_CAP       = d["ts_urgency_tau_cap"]
        ts.URGENCY_SPEED_SCALE   = d["ts_urgency_speed_scale"]
        ts.URGENCY_TAU_WEIGHT    = d["ts_urgency_tau_weight"]
        ts.URGENCY_SPEED_WEIGHT  = 1.0 - d["ts_urgency_tau_weight"]

        ts.PREDICT_DT            = d["ts_predict_dt"]
        ts.FRAG_PENALTY_LARGE    = d["ts_frag_penalty_large"]

        # Acquisition FIS — heading error MFs
        b1 = d["ts_he_tiny_b1"];  b2 = d["ts_he_tiny_b2"]
        ts.HE_TINY   = (0.0, 0.0, b1, b2)
        sa = d["ts_he_small_a"]; sc = d["ts_he_small_c"]; sd = d["ts_he_small_d"]
        ts.HE_SMALL  = (sa, b2, sc, sd)      # shoulder starts where TINY ends
        ma = d["ts_he_med_a"]; mb = d["ts_he_med_b"]; mc = d["ts_he_med_c"]
        ts.HE_MEDIUM = (ma, mb, mc, 90.0)
        ts.HE_LARGE  = (mc, 90.0, 180.0, 180.0)

        # Acquisition FIS — distance MFs
        dcc = d["ts_d_close_c"]; dcd = d["ts_d_close_d"]
        ts.D_CLOSE  = (0.0, 0.0, dcc, dcd)
        dma = d["ts_d_med_a"]; dmb = d["ts_d_med_b"]
        dmc = d["ts_d_med_c"]; dmd = d["ts_d_med_d"]
        ts.D_MEDIUM = (dma, dmb, dmc, dmd)
        ts.D_FAR    = (dmc, dmd, 9999.0, 9999.0)

        # Acquisition FIS — difficulty singletons
        ts.DIFF_EASY   = d["ts_diff_easy"]
        ts.DIFF_MEDIUM = d["ts_diff_medium"]
        ts.DIFF_HARD   = d["ts_diff_hard"]

    except ImportError:
        pass

    # MPC (CasADi/IPOPT) constants are NOT patched — left at their validated defaults.


def apply_genome_to_supervisor(supervisor, genome: Genome) -> None:
    """Patch a live Supervisor instance (mode-switching thresholds)."""
    d = genome.to_dict()
    supervisor.R_lo = d["R_lo"]
    supervisor.R_hi = d["R_hi"]


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    fitness:       float
    asteroids_hit: int
    deaths:        int
    lives_remaining: int      # cumulated across all scenarios
    accuracy:      float
    eval_time_ms:  float
    n_scenarios:   int = 1
    error:         Optional[str] = None


# ---------------------------------------------------------------------------
# Fitness weights
# ---------------------------------------------------------------------------
# Design rationale:
#   - Survival is the hard constraint: dying costs far more than any asteroid
#     bonus can compensate.  W_DEATH is large and negative; W_SURVIVAL rewards
#     lives preserved across all scenarios cumulatively.
#   - Hits matter but must not make kamikaze profitable: even 30 extra hits
#     (~90 pts) should not offset 3 extra deaths (~150 pts penalty).
#   - Accuracy is a secondary objective: it signals shot quality, not raw
#     aggression.  Clamped to [0,1] before scaling so >100% is impossible.
#   - Eval-time penalty keeps controllers within the real-time budget.
# ---------------------------------------------------------------------------

W_HIT        =  3.0    # per asteroid hit (cumulated over scenarios)
W_DEATH      = -50.0   # per death — massive penalty to block kamikaze strategy
W_SURVIVAL   =  30.0   # per life preserved (max_deaths - actual_deaths)
W_ACCURACY   =  1.5    # × accuracy % — secondary, don't dominate
W_TIME       = -0.5    # per ms over TIME_BUDGET_MS
TIME_BUDGET_MS = 5.0   # ms — budget par frame contrôleur
EVAL_TIME_DISQUALIFY_MS = 30.0   # ms — disqualifié si trop lent


def compute_fitness(result: FitnessResult) -> float:
    # Note : eval_time_ms n'est PAS le temps contrôleur par frame —
    # c'est le temps de simulation Kessler total / nb frames (en ms).
    # Cette métrique est inutilisable pour pénaliser la lenteur du contrôleur,
    # donc on l'ignore complètement.
    acc_clamped = min(1.0, max(0.0, result.accuracy))
    f = (W_HIT      * result.asteroids_hit
       + W_DEATH    * result.deaths
       + W_SURVIVAL * result.lives_remaining
       + W_ACCURACY * acc_clamped * 100.0)
    return f


def _evaluate_worker(args: Tuple[np.ndarray, List[dict], dict]) -> FitnessResult:
    """
    Worker function for multiprocessing.  Must be top-level (picklable).

    args = (genes, scenario_configs, game_settings_override)
    """
    genes, scenario_configs, game_settings_override = args
    genome = Genome(genes)
    genome.repair()

    # Patch module constants before any import of controller classes
    apply_genome_to_modules(genome)

    try:
        from kesslergame import Scenario, KesslerGame, GraphicsType
        from Fuzzy_MPC_Controller import Controller

        n_scenarios     = len(scenario_configs)
        max_lives_total = n_scenarios * 3   # 3 lives per scenario

        total_hit            = 0
        total_death          = 0
        total_lives_remaining = 0
        acc_list             = []
        time_list            = []

        for cfg in scenario_configs:
            scenario = Scenario(**cfg)

            settings = {
                'perf_tracker':          True,
                'graphics_type':         GraphicsType.NoGraphics,
                'realtime_multiplier':   0,
                'graphics_obj':          None,
                'frequency':             30,
            }
            settings.update(game_settings_override)

            ctrl = Controller()
            apply_genome_to_supervisor(ctrl.supervisor, genome)

            game  = KesslerGame(settings=settings)
            score, perf = game.run(scenario=scenario, controllers=[ctrl])

            team = score.teams[0]
            total_hit            += team.asteroids_hit
            total_death          += team.deaths
            # Lives remaining this scenario (clamped — can't be negative)
            total_lives_remaining += max(0, 3 - team.deaths)

            # Use Kessler's native accuracy (already handles fragmentation correctly)
            acc_list.append(min(1.0, team.accuracy))
            # mean_eval_time est en secondes mais représente le temps de
            # simulation Kessler total / nb frames — pas le temps contrôleur.
            # On le stocke tel quel en ms pour la pénalité relative,
            # mais la disqualification est désactivée (seuil inatteignable).
            time_list.append(team.mean_eval_time * 1000.0)

        avg_acc  = float(np.mean(acc_list))  if acc_list  else 0.0
        avg_time = float(np.mean(time_list)) if time_list else 0.0

        result = FitnessResult(
            fitness          = 0.0,
            asteroids_hit    = total_hit,
            deaths           = total_death,
            lives_remaining  = total_lives_remaining,
            accuracy         = avg_acc,
            eval_time_ms     = avg_time,
            n_scenarios      = n_scenarios,
        )
        result.fitness = compute_fitness(result)
        return result

    except Exception as exc:
        return FitnessResult(
            fitness          = -9999.0,
            asteroids_hit    = 0,
            deaths           = 99,
            lives_remaining  = 0,
            accuracy         = 0.0,
            eval_time_ms     = 0.0,
            n_scenarios      = len(scenario_configs),
            error            = str(exc),
        )


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def tournament_select(
    population:  List[Genome],
    fitnesses:   List[float],
    k:           int,
    rng:         np.random.Generator,
) -> Genome:
    """Return the fittest genome among k randomly sampled candidates."""
    indices = rng.choice(len(population), size=k, replace=False)
    best    = max(indices, key=lambda i: fitnesses[i])
    return population[best].copy()


def uniform_crossover(
    parent_a: Genome,
    parent_b: Genome,
    p_swap:   float,
    rng:      np.random.Generator,
) -> Tuple[Genome, Genome]:
    """Swap each gene with probability p_swap."""
    mask  = rng.random(N_PARAMS) < p_swap
    alpha=rng.random()
    beta = rng.random()
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    child_a.genes=alpha*parent_a.genes+(1-alpha)*parent_b.genes
    child_b.genes=beta*parent_a.genes+(1-beta)*parent_b.genes
    # child_a = parent_a.copy()
    # child_b = parent_b.copy()
    # child_a.genes[mask] = parent_b.genes[mask]
    # child_b.genes[mask] = parent_a.genes[mask]
    return child_a, child_b


def gaussian_mutate(
    genome:     Genome,
    sigma_frac: float,          # mutation std as fraction of [lo, hi] range
    p_mutate:   float,          # per-gene mutation probability
    rng:        np.random.Generator,
) -> Genome:
    """Add Gaussian noise to each gene with probability p_mutate."""
    child = genome.copy()
    for i, spec in enumerate(PARAM_SPECS):
        if rng.random() < p_mutate:
            sigma = sigma_frac * (spec.hi - spec.lo)
            child.genes[i] += rng.normal(0.0, sigma)
    child.repair()
    return child


# ---------------------------------------------------------------------------
# GA Configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    pop_size:        int   = 30
    n_generations:   int   = 20
    elite_k:         int   = 3      # top-k carried to next gen unchanged
    tournament_k:    int   = 3      # tournament size
    p_crossover:     float = 0.7    # probability of crossover vs direct copy
    p_swap:          float = 0.5    # uniform crossover gene-swap rate
    sigma_frac:      float = 0.08   # Gaussian mutation std fraction
    p_mutate:        float = 0.15   # per-gene mutation probability
    n_workers:       int   = 4      # parallel evaluation workers
    seed:            int   = 42
    checkpoint_path: str   = "ga_checkpoint.json"
    scenario_configs: List[dict] = field(default_factory=lambda: [
        {
            "name":                  "GA_Scenario_Easy",
            "num_asteroids":         10,
            "ship_states":           [{'position': (400, 400), 'angle': 90,
                                       'lives': 3, 'team': 1,
                                       'mines_remaining': 3}],
            "map_size":              (1000, 800),
            "time_limit":            60,
            "ammo_limit_multiplier": 0,
            "stop_if_no_ammo":       False,
        },
        {
            "name":                  "GA_Scenario_Dense",
            "num_asteroids":         20,
            "ship_states":           [{'position': (400, 400), 'angle': 90,
                                       'lives': 3, 'team': 1,
                                       'mines_remaining': 3}],
            "map_size":              (1000, 800),
            "time_limit":            60,
            "ammo_limit_multiplier": 0,
            "stop_if_no_ammo":       False,
        },
    ])
    game_settings_override: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main GA class
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    generation:  int
    best_fitness: float
    mean_fitness: float
    best_genome:  Dict[str, float]
    best_result:  Optional[FitnessResult] = None


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for the Kessler multimodal controller.

    Example
    -------
        cfg = GAConfig(pop_size=40, n_generations=30)
        opt = GeneticOptimizer(cfg)
        best = opt.run()
        opt.apply_best()    # writes optimised params to module constants
    """

    def __init__(self, config: GAConfig) -> None:
        self.cfg  = config
        self.rng  = np.random.default_rng(config.seed)
        self.history: List[GenerationStats] = []
        self._best_genome: Optional[Genome] = None
        self._best_fitness: float = -math.inf

    # ------------------------------------------------------------------
    def _init_population(self) -> List[Genome]:
        pop = [Genome()]                               # always include defaults
        pop += [Genome.random(self.rng) for _ in range(self.cfg.pop_size - 1)]
        return pop

    # ------------------------------------------------------------------
    def _evaluate_population(
        self,
        population: List[Genome],
    ) -> List[FitnessResult]:
        """Evaluate all individuals, in parallel if n_workers > 1."""
        args_list = [
            (g.genes.copy(), self.cfg.scenario_configs, self.cfg.game_settings_override)
            for g in population
        ]

        if self.cfg.n_workers > 1:
            with mp.Pool(processes=self.cfg.n_workers) as pool:
                results = pool.map(_evaluate_worker, args_list)
        else:
            results = [_evaluate_worker(a) for a in args_list]

        return results

    # ------------------------------------------------------------------
    def _next_generation(
        self,
        population:  List[Genome],
        fitnesses:   List[float],
    ) -> List[Genome]:
        """Produce the next generation via elitism + tournament + crossover + mutation."""
        cfg = self.cfg

        # Sort by fitness descending
        order = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        sorted_pop = [population[i] for i in order]

        next_gen: List[Genome] = []

        # Elitism
        for i in range(min(cfg.elite_k, len(sorted_pop))):
            next_gen.append(sorted_pop[i].copy())

        # Fill remainder
        while len(next_gen) < cfg.pop_size:
            parent_a = tournament_select(population, fitnesses, cfg.tournament_k, self.rng)
            parent_b = tournament_select(population, fitnesses, cfg.tournament_k, self.rng)

            if self.rng.random() < cfg.p_crossover:
                child_a, child_b = uniform_crossover(parent_a, parent_b, cfg.p_swap, self.rng)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()

            child_a = gaussian_mutate(child_a, cfg.sigma_frac, cfg.p_mutate, self.rng)
            child_b = gaussian_mutate(child_b, cfg.sigma_frac, cfg.p_mutate, self.rng)

            next_gen.append(child_a)
            if len(next_gen) < cfg.pop_size:
                next_gen.append(child_b)

        return next_gen

    # ------------------------------------------------------------------
    def run(self) -> Genome:
        """
        Run the GA for cfg.n_generations.
        Returns the best Genome found.
        """
        print(f"\n{'='*60}")
        print(f"  Kessler GA Optimiser — {N_PARAMS} parameters")
        print(f"  Population: {self.cfg.pop_size}  |  Generations: {self.cfg.n_generations}")
        print(f"  Workers: {self.cfg.n_workers}")
        print(f"{'='*60}\n")

        population = self._init_population()

        # Resume from checkpoint if it exists
        ckpt = Path(self.cfg.checkpoint_path)
        start_gen = 0
        if ckpt.exists():
            population, start_gen = self._load_checkpoint(ckpt, population)

        for gen in range(start_gen, self.cfg.n_generations):
            t0 = time.perf_counter()
            results  = self._evaluate_population(population)
            fitnesses = [r.fitness for r in results]

            # Track global best
            best_idx = int(np.argmax(fitnesses))
            if fitnesses[best_idx] > self._best_fitness:
                self._best_fitness = fitnesses[best_idx]
                self._best_genome  = population[best_idx].copy()

            stats = GenerationStats(
                generation   = gen,
                best_fitness = float(np.max(fitnesses)),
                mean_fitness = float(np.mean(fitnesses)),
                best_genome  = self._best_genome.to_dict(),
                best_result  = results[best_idx],
            )
            self.history.append(stats)

            elapsed = time.perf_counter() - t0
            self._print_gen(stats, elapsed)
            self._save_checkpoint(ckpt, population, gen + 1)

            if gen < self.cfg.n_generations - 1:
                population = self._next_generation(population, fitnesses)

        print(f"\n✓ Optimisation complete.  Best fitness: {self._best_fitness:.3f}")
        self._print_best_params()
        return self._best_genome  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def apply_best(self) -> None:
        """Patch all module constants with the best genome found."""
        if self._best_genome is None:
            raise RuntimeError("No best genome — run() first.")
        apply_genome_to_modules(self._best_genome)
        print("✓ Best genome applied to module constants.")

    # ------------------------------------------------------------------
    def export_best(self, path: str = "best_params.json") -> None:
        """Write the best parameter dict to a JSON file."""
        if self._best_genome is None:
            raise RuntimeError("No best genome — run() first.")
        with open(path, "w") as f:
            json.dump(self._best_genome.to_dict(), f, indent=2)
        print(f"✓ Best parameters written to {path}")

    @staticmethod
    def load_and_apply(path: str) -> Genome:
        """Load a previously saved best_params.json and apply it."""
        with open(path) as f:
            d = json.load(f)
        genome = Genome.from_dict(d)
        apply_genome_to_modules(genome)
        print(f"✓ Loaded and applied parameters from {path}")
        return genome

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path, population: List[Genome], next_gen: int) -> None:
        data = {
            "next_generation": next_gen,
            "best_fitness":    self._best_fitness,
            "best_genome":     self._best_genome.to_dict() if self._best_genome else None,
            "population":      [g.to_dict() for g in population],
            "history":         [
                {
                    "generation":   s.generation,
                    "best_fitness": s.best_fitness,
                    "mean_fitness": s.mean_fitness,
                }
                for s in self.history
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_checkpoint(
        self, path: Path, default_pop: List[Genome]
    ) -> Tuple[List[Genome], int]:
        print(f"  ↩  Resuming from checkpoint: {path}")
        with open(path) as f:
            data = json.load(f)
        population = [Genome.from_dict(d) for d in data["population"]]
        start_gen  = data["next_generation"]
        if data.get("best_genome"):
            self._best_genome  = Genome.from_dict(data["best_genome"])
            self._best_fitness = data["best_fitness"]
        for h in data.get("history", []):
            self.history.append(GenerationStats(
                generation   = h["generation"],
                best_fitness = h["best_fitness"],
                mean_fitness = h["mean_fitness"],
                best_genome  = {},
            ))
        return population, start_gen

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_gen(self, stats: GenerationStats, elapsed: float) -> None:
        r = stats.best_result
        n_sc = r.n_scenarios if r else 1
        max_lives = n_sc * 3

        hit_str  = f"{r.asteroids_hit:3d}"               if r else "  ?"
        dead_str = f"{r.deaths:2d}/{max_lives}"          if r else " ?"
        live_str = f"{r.lives_remaining:2d}/{max_lives}" if r else " ?"
        acc_str  = f"{min(r.accuracy,1.0)*100:5.1f}%"   if r else "    ?"
        err_str  = f" [ERR: {r.error[:40]}]" if (r and r.error) else ""
        print(
            f"  Gen {stats.generation:03d}  "
            f"best={stats.best_fitness:8.2f}  "
            f"mean={stats.mean_fitness:8.2f}  "
            f"hit={hit_str}  dead={dead_str}  lives={live_str}  acc={acc_str}  "
            f"({elapsed:.1f}s){err_str}"
        )

    def _print_best_params(self) -> None:
        if self._best_genome is None:
            return
        d = self._best_genome.to_dict()
        groups: Dict[str, List[str]] = {}
        for spec in PARAM_SPECS:
            groups.setdefault(spec.group, []).append(spec.name)
        print("\n  Best parameters:")
        for grp, names in groups.items():
            print(f"\n    [{grp}]")
            for name in names:
                print(f"      {name:25s} = {d[name]:.6g}")


# ---------------------------------------------------------------------------
# Sensitivity analysis (one-at-a-time)
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    base_genome: Genome,
    scenario_configs: List[dict],
    n_steps: int = 5,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Vary each parameter independently over its full range (n_steps steps),
    evaluate fitness, return {param_name: [(value, fitness), ...]}.

    Useful for plotting sensitivity curves and identifying which parameters
    matter most.
    """
    results: Dict[str, List[Tuple[float, float]]] = {}
    game_settings = {}

    for i, spec in enumerate(PARAM_SPECS):
        print(f"  Sensitivity: {spec.name}  [{spec.lo:.4g}, {spec.hi:.4g}]")
        values  = np.linspace(spec.lo, spec.hi, n_steps)
        curve: List[Tuple[float, float]] = []

        for v in values:
            genome = base_genome.copy()
            genome.genes[i] = v
            genome.repair()
            r = _evaluate_worker((genome.genes, scenario_configs, game_settings))
            curve.append((float(v), r.fitness))
            print(f"    {spec.name}={v:.4g}  → fitness={r.fitness:.3f}")

        results[spec.name] = curve

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GA optimizer for Kessler multimodal controller")
    p.add_argument("--pop",      type=int,   default=40,   help="Population size")
    p.add_argument("--gen",      type=int,   default=20,   help="Number of generations")
    p.add_argument("--workers",  type=int,   default=12,    help="Parallel workers")
    p.add_argument("--elite",    type=int,   default=3,    help="Elite count")
    p.add_argument("--sigma",    type=float, default=0.08, help="Mutation sigma fraction")
    p.add_argument("--pmut",     type=float, default=0.25, help="Per-gene mutation probability")
    p.add_argument("--seed",     type=int,   default=42,   help="RNG seed")
    p.add_argument("--resume",   type=str,   default=None, help="Resume from checkpoint JSON")
    p.add_argument("--export",   type=str,   default="best_params.json",
                   help="Path to write best parameters")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run one-at-a-time sensitivity analysis after optimisation")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = GAConfig(
        pop_size       = args.pop,
        n_generations  = args.gen,
        n_workers      = args.workers,
        elite_k        = args.elite,
        sigma_frac     = args.sigma,
        p_mutate       = args.pmut,
        seed           = args.seed,
        checkpoint_path= args.resume or "ga_checkpoint.json",
    )

    opt  = GeneticOptimizer(cfg)
    best = opt.run()
    opt.export_best(args.export)

    if args.sensitivity:
        print("\nRunning sensitivity analysis on best genome…")
        sens = sensitivity_analysis(best, cfg.scenario_configs)
        with open("sensitivity.json", "w") as f:
            json.dump(sens, f, indent=2)
        print("✓ Sensitivity written to sensitivity.json")


if __name__ == "__main__":
    mp.freeze_support()
    main()