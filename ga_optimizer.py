"""
ga_optimizer.py
===============
Genetic Algorithm optimizer for the Kessler multimodal controller system.

Architecture
------------
Optimises hyperparameters across three layers simultaneously:

  1. Supervisor       — mode-switching thresholds (R_lo, R_hi)
  2. RiskField        — FIS membership function breakpoints, aggregation alpha
  3. TargetingSystem  — scoring weights, PID gains, repulsion, mine timing
                        (replaces the old FuzzyController / TargetSelector layer)

The MPC (CasADi/IPOPT) layer is intentionally excluded: its parameters have
been validated independently and touching them produces ill-conditioned NLPs.

Each individual is a flat numpy array (genome) encoding all tuneable
parameters.  A Genome class wraps encode/decode logic so the GA never
needs to know about parameter semantics.

Fitness
-------
A fixed set of rotating scenarios is evaluated per individual.
Fitness = weighted combination of:
    - asteroids_hit      (maximise)
    - deaths             (minimise, heavily penalised)
    - lives_remaining    (maximise)
    - accuracy           (maximise, secondary)

GA Features
-----------
- Tournament selection
- Blend crossover (BLX-alpha) + Gaussian mutation
- Elitism (top-k survive unchanged)
- Constraint repair: clamp + ordering enforcement after mutation
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
import json
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    group:   str

    def clip(self, v: float) -> float:
        return float(np.clip(v, self.lo, self.hi))


PARAM_SPECS: List[ParamSpec] = [

    # ── Supervisor ─────────────────────────────────────────────────────────
    # R_lo < R_hi enforced by repair(); both must be probabilities [0, 1]
    ParamSpec("R_lo",              0.50,  0.0,  1.0, "supervisor"),
    ParamSpec("R_hi",              0.60,  0.0,  1.0, "supervisor"),

    # ── RiskField — aggregation ────────────────────────────────────────────
    # alpha is a convex weight → [0, 1]
    ParamSpec("alpha_aggregation", 0.55,  0.0,  1.0, "risk_field"),

    # ── RiskField — tau breakpoints ────────────────────────────────────────
    # Must stay positive; ordering enforced by repair()
    ParamSpec("tau_crit_b",        0.25,  0.0, 10.0, "risk_field"),
    ParamSpec("tau_crit_d",        0.50,  0.0, 10.0, "risk_field"),
    ParamSpec("tau_close_a",       0.25,  0.0, 10.0, "risk_field"),
    ParamSpec("tau_close_b",       0.50,  0.0, 10.0, "risk_field"),
    ParamSpec("tau_close_d",       1.50,  0.0, 10.0, "risk_field"),
    ParamSpec("tau_med_a",         0.60,  0.0, 10.0, "risk_field"),
    ParamSpec("tau_med_c",         1.50,  0.0, 10.0, "risk_field"),
    ParamSpec("tau_med_d",         5.50,  0.0, 20.0, "risk_field"),

    # ── RiskField — distance breakpoints ──────────────────────────────────
    # Must stay positive; ordering enforced by repair()
    ParamSpec("d_near_c",          50.0,  0.0, 2000.0, "risk_field"),
    ParamSpec("d_near_d",          90.0,  0.0, 2000.0, "risk_field"),
    ParamSpec("d_med_a",           50.0,  0.0, 2000.0, "risk_field"),
    ParamSpec("d_med_d",          220.0,  0.0, 2000.0, "risk_field"),

    # ── RiskField — output singletons ──────────────────────────────────────
    # FIS outputs are risks → [0, 1]; ordering enforced by repair()
    ParamSpec("out_high",          0.75,  0.0,  1.0, "risk_field"),
    ParamSpec("out_medium",        0.20,  0.0,  1.0, "risk_field"),
    ParamSpec("out_low",           0.15,  0.0,  1.0, "risk_field"),
    ParamSpec("out_negligible",    0.02,  0.0,  1.0, "risk_field"),

    # ── TargetingSystem — scoring weights ──────────────────────────────────
    # Fully free: negative weight = reverse the signal (e.g. negative W_FRAG
    # would reward fragmentation, negative W_SMALL would prefer large targets)
    ParamSpec("W_RISK",      2.0,  -10.0, 10.0, "targeting"),
    ParamSpec("W_TTC",       1.5,  -10.0, 10.0, "targeting"),
    ParamSpec("W_KILL_EFF",  1.2,  -10.0, 10.0, "targeting"),
    ParamSpec("W_FRAG",      1.8,  -10.0, 10.0, "targeting"),
    ParamSpec("W_DIST",      0.4,  -10.0, 10.0, "targeting"),
    ParamSpec("W_SMALL",     0.8,  0.0, 10.0, "targeting"),

    # ── TargetingSystem — targeting geometry ───────────────────────────────
    # FIRE_CONE_DEG: (0, 180] — must be positive, capped at half-circle
    ParamSpec("FIRE_CONE_DEG",       4.0,   0.1, 180.0, "targeting"),
    # MAX_DIST_FIRE: must be positive (> MIN_DIST_FIRE = 10 px, fixed)
    ParamSpec("MAX_DIST_FIRE",     650.0,  11.0, 2000.0, "targeting"),
    # MIN_RISK_TO_TARGET: probability → [0, 1]
    ParamSpec("MIN_RISK_TO_TARGET",  0.05,  0.0,   1.0, "targeting"),
    # TTC_REF: must be > 0 (denominator in exp); enforced by repair()
    ParamSpec("TTC_REF",             3.0,   0.01, 30.0, "targeting"),
    # CHAIN_RADIUS: must be positive
    ParamSpec("CHAIN_RADIUS",       55.0,   0.0, 500.0, "targeting"),
    # FIRE_COOLDOWN_S: must be positive
    ParamSpec("FIRE_COOLDOWN_S",     0.10,  0.0,   2.0, "targeting"),

    # ── TargetingSystem — PID ─────────────────────────────────────────────
    # Fully free: negative Kp = reverse turn direction (valid if convention flipped)
    ParamSpec("Kp",  5.0, -30.0, 30.0, "targeting"),
    ParamSpec("Ki",  0.0, -10.0, 10.0, "targeting"),
    ParamSpec("Kd",  0.3, -10.0, 10.0, "targeting"),

    # ── TargetingSystem — repulsion ────────────────────────────────────────
    # REPULSION_THRUST: fully free (negative = attraction)
    ParamSpec("REPULSION_THRUST",   80.0, -480.0, 480.0, "targeting"),
    # TTC_FULL < TTC_MAX enforced by repair(); both must be positive
    ParamSpec("REPULSION_TTC_MAX",   2.0,   0.0,  30.0, "targeting"),
    ParamSpec("REPULSION_TTC_FULL",  0.5,   0.0,  30.0, "targeting"),

    # ── SacrificeController — mine timing ─────────────────────────────────
    # Must be positive (time before impact)
    ParamSpec("MINE_LEAD_S",  0.20,  0.0,  5.0, "targeting"),

    # ── TargetingController — target lock hysteresis ───────────────────────
    # SWITCH_RATIO = 1.0 → no hysteresis (switch freely)
    # SWITCH_RATIO > 1.0 → challenger must beat locked target by this factor
    # No upper bound: very large values = permanent lock until target destroyed
    ParamSpec("SWITCH_RATIO", 1.25,  1.0, 20.0, "targeting"),
]

N_PARAMS   = len(PARAM_SPECS)
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
        genes = np.array([rng.uniform(p.lo, p.hi) for p in PARAM_SPECS], dtype=np.float64)
        return cls(genes)

    def to_dict(self) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(PARAM_NAMES, self.genes)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Genome":
        genes = np.array(
            [d.get(name, PARAM_SPECS[i].default) for i, name in enumerate(PARAM_NAMES)],
            dtype=np.float64,
        )
        return cls(genes)

    def repair(self) -> None:
        """
        Enforce hard physical constraints only — no arbitrary bound clamping.
        The only things that must hold for the code not to crash or produce
        nonsense are:
          - probability values stay in [0, 1]
          - strictly-positive denominators (TTC_REF)
          - strictly-positive distances and times
          - monotone ordering of MF breakpoints and supervisor thresholds
        Everything else (weights, PID gains, repulsion) is left unconstrained.
        """
        def _idx(name: str) -> int:
            return PARAM_NAMES.index(name)

        def _get(name: str) -> float:
            return self.genes[_idx(name)]

        def _set(name: str, v: float) -> None:
            self.genes[_idx(name)] = v

        def _clip01(name: str) -> None:
            _set(name, float(np.clip(_get(name), 0.0, 1.0)))

        def _pos(name: str, minimum: float = 1e-6) -> None:
            """Clamp to a small positive value."""
            if _get(name) < minimum:
                _set(name, minimum)

        def _order(a: str, b: str, gap: float = 1e-4) -> None:
            ai, bi = _idx(a), _idx(b)
            if self.genes[ai] >= self.genes[bi]:
                mid = (self.genes[ai] + self.genes[bi]) / 2.0
                self.genes[ai] = mid - gap
                self.genes[bi] = mid + gap

        # ── Probabilities must stay in [0, 1] ─────────────────────────────
        for name in ("R_lo", "R_hi", "alpha_aggregation",
                     "out_high", "out_medium", "out_low", "out_negligible",
                     "MIN_RISK_TO_TARGET"):
            _clip01(name)

        # ── Supervisor: R_lo < R_hi ────────────────────────────────────────
        _order("R_lo", "R_hi", 0.01)

        # ── RiskField tau: all must be positive, monotone per MF ──────────
        for name in ("tau_crit_b", "tau_crit_d",
                     "tau_close_a", "tau_close_b", "tau_close_d",
                     "tau_med_a", "tau_med_c", "tau_med_d"):
            _pos(name)
        _order("tau_crit_b",  "tau_crit_d",  0.01)
        _order("tau_close_a", "tau_close_b", 0.01)
        _order("tau_close_b", "tau_close_d", 0.01)
        _order("tau_med_a",   "tau_med_c",   0.01)
        _order("tau_med_c",   "tau_med_d",   0.01)

        # ── RiskField distance: all must be positive, monotone per MF ─────
        for name in ("d_near_c", "d_near_d", "d_med_a", "d_med_d"):
            _pos(name)
        _order("d_near_c", "d_near_d", 0.1)
        _order("d_med_a",  "d_med_d",  0.1)

        # ── RiskField singletons: negligible < low < medium < high ────────
        _order("out_negligible", "out_low",    0.001)
        _order("out_low",        "out_medium", 0.001)
        _order("out_medium",     "out_high",   0.001)

        # ── TTC_REF: denominator — must be strictly positive ──────────────
        _pos("TTC_REF", 0.01)

        # ── Distances and times: must be positive ─────────────────────────
        _pos("MAX_DIST_FIRE",    11.0)   # > MIN_DIST_FIRE = 10 px
        _pos("CHAIN_RADIUS",      0.0)
        _pos("FIRE_COOLDOWN_S",   0.0)
        _pos("MINE_LEAD_S",       0.0)

        # ── Repulsion: TTC_FULL < TTC_MAX, both positive ──────────────────
        _pos("REPULSION_TTC_MAX",  0.0)
        _pos("REPULSION_TTC_FULL", 0.0)
        _order("REPULSION_TTC_FULL", "REPULSION_TTC_MAX", 0.01)

        # ── FIRE_CONE_DEG: (0, 180] ────────────────────────────────────────
        if _get("FIRE_CONE_DEG") <= 0.0:
            _set("FIRE_CONE_DEG", 0.1)
        if _get("FIRE_CONE_DEG") > 180.0:
            _set("FIRE_CONE_DEG", 180.0)

        # ── SWITCH_RATIO: must be ≥ 1.0 (below 1 the hysteresis inverts) ──
        if _get("SWITCH_RATIO") < 1.0:
            _set("SWITCH_RATIO", 1.0)

    def copy(self) -> "Genome":
        return Genome(self.genes.copy())


# ---------------------------------------------------------------------------
# Parameter injection
# ---------------------------------------------------------------------------

def apply_genome_to_modules(genome: Genome) -> None:
    """
    Patch module-level constants before instantiating controllers.
    Must be called inside the worker process (main process unaffected).
    """
    d = genome.to_dict()

    # ── risk_field ──────────────────────────────────────────────────────────
    try:
        import risk_field as rf

        rf.ALPHA_AGGREGATION = d["alpha_aggregation"]

        cb = d["tau_crit_b"];  cd = d["tau_crit_d"]
        rf.TAU_CRITICAL = (0.0, 0.0, cb, cd)

        ca = d["tau_close_a"]; cbr = d["tau_close_b"]; cdr = d["tau_close_d"]
        rf.TAU_CLOSE = (ca, cbr, cbr, cdr)

        ma = d["tau_med_a"]; mc = d["tau_med_c"]; md = d["tau_med_d"]
        rf.TAU_MEDIUM = (ma, mc, mc, md)
        rf.TAU_FAR    = (mc, md, 99.0, 99.0)

        dnc = d["d_near_c"]; dnd = d["d_near_d"]
        rf.D_NEAR   = (0.0, 0.0, dnc, dnd)
        dma = d["d_med_a"]; dmd = d["d_med_d"]
        rf.D_MEDIUM = (dma, dnd, dnd, dmd)
        rf.D_FAR    = (dnd, dmd, 9999.0, 9999.0)

        rf.OUT_HIGH       = d["out_high"]
        rf.OUT_MEDIUM     = d["out_medium"]
        rf.OUT_LOW        = d["out_low"]
        rf.OUT_NEGLIGIBLE = d["out_negligible"]

    except ImportError:
        pass

    # ── targeting_system ────────────────────────────────────────────────────
    try:
        import targeting_system as ts

        # Scoring weights
        ts.W_RISK       = d["W_RISK"]
        ts.W_TTC        = d["W_TTC"]
        ts.W_KILL_EFF   = d["W_KILL_EFF"]
        ts.W_FRAG       = d["W_FRAG"]
        ts.W_DIST       = d["W_DIST"]
        ts.W_SMALL      = d["W_SMALL"]

        # Targeting geometry
        ts.FIRE_CONE_DEG       = d["FIRE_CONE_DEG"]
        ts.MAX_DIST_FIRE       = d["MAX_DIST_FIRE"]
        ts.MIN_RISK_TO_TARGET  = d["MIN_RISK_TO_TARGET"]
        ts.TTC_REF             = d["TTC_REF"]
        ts.CHAIN_RADIUS        = d["CHAIN_RADIUS"]
        ts.FIRE_COOLDOWN_S     = d["FIRE_COOLDOWN_S"]

        # Repulsion
        ts.REPULSION_THRUST   = d["REPULSION_THRUST"]
        ts.REPULSION_TTC_MAX  = d["REPULSION_TTC_MAX"]
        ts.REPULSION_TTC_FULL = d["REPULSION_TTC_FULL"]

        # Mine timing
        ts.MINE_LEAD_S = d["MINE_LEAD_S"]

        # Target lock hysteresis
        ts.SWITCH_RATIO = d["SWITCH_RATIO"]

        # NOTE: Kp and Kd are constructor arguments on TargetingController.
        # They are read at __init__ time, so they cannot be patched on an
        # already-constructed instance.  The worker creates a fresh Controller()
        # AFTER calling apply_genome_to_modules(), so they are stored as module
        # globals here and picked up by TargetingController.__init__.
        ts._GA_Kp = d["Kp"]
        ts._GA_Ki = d["Ki"]
        ts._GA_Kd = d["Kd"]

    except ImportError:
        pass


def apply_genome_to_supervisor(supervisor, genome: Genome) -> None:
    """Patch a live Supervisor instance (mode-switching thresholds only)."""
    d = genome.to_dict()
    supervisor.R_lo = d["R_lo"]
    supervisor.R_hi = d["R_hi"]


# ---------------------------------------------------------------------------
# targeting_system monkey-patch for Kp/Kd
# ---------------------------------------------------------------------------
# TargetingController reads Kp/Kd at __init__ time (constructor args with
# defaults).  The GA stores _GA_Kp/_GA_Kd as module globals; we patch
# TargetingController.__init__ to read them when present.
#
# This patch is applied once at import time of the worker process,
# AFTER apply_genome_to_modules() has set ts._GA_Kp / ts._GA_Kd.
# ---------------------------------------------------------------------------

def _patch_targeting_controller_init() -> None:
    """
    Wrap TargetingController.__init__ so it reads ts._GA_Kp / ts._GA_Kd
    when available (set by apply_genome_to_modules).
    Called once per worker process, after module patching.
    """
    try:
        import targeting_system as ts
        _orig_init = ts.TargetingController.__init__

        def _patched_init(self, Kp=None, Ki=None, Kd=None):
            _Kp = getattr(ts, '_GA_Kp', 5.0) if Kp is None else Kp
            _Ki = getattr(ts, '_GA_Ki', 0.0) if Ki is None else Ki
            _Kd = getattr(ts, '_GA_Kd', 0.3) if Kd is None else Kd
            _orig_init(self, Kp=_Kp, Ki=_Ki, Kd=_Kd)

        ts.TargetingController.__init__ = _patched_init
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    fitness:          float
    asteroids_hit:    int
    deaths:           int
    lives_remaining:  int
    accuracy:         float
    eval_time_ms:     float
    n_scenarios:      int   = 1
    error:            Optional[str] = None


W_HIT      =  3.0
W_DEATH    = -80.0   # increased — suicidal spray must not be profitable
W_SURVIVAL =  40.0   # increased — living through a scenario is worth more than extra kills
W_ACCURACY =  2.0


def compute_fitness(r: FitnessResult) -> float:
    """
    Fitness computed as average per-scenario score, not raw sum.

    Each scenario contributes equally regardless of asteroid count —
    this prevents the high-density Stress scenario from dominating the
    gradient and rewarding spray-and-pray at the expense of survivability.

    Per-scenario score is normalised by max_asteroids_per_scenario so that
    hitting 10/10 in Sparse is worth the same as hitting 50/50 in Stress.
    """
    acc = min(1.0, max(0.0, r.accuracy))
    # Per-scenario averages (already averaged in the worker for acc)
    n  = max(r.n_scenarios, 1)
    avg_hit    = r.asteroids_hit   / n
    avg_death  = r.deaths          / n
    avg_lives  = r.lives_remaining / n

    # Survival gate: if the controller dies on average more than once per
    # scenario, apply a multiplier that suppresses the hit bonus.
    # This makes suicidal strategies strictly inferior even at high hit counts.
    survival_ratio = avg_lives / 3.0   # 1.0 = full survival, 0.0 = always 3 deaths
    hit_multiplier = 0.3 + 0.7 * survival_ratio   # floor at 0.3, full at 1.0

    return (W_HIT      * avg_hit * hit_multiplier
          + W_DEATH    * avg_death
          + W_SURVIVAL * avg_lives
          + W_ACCURACY * acc * 100.0)


def _evaluate_worker(args: Tuple[np.ndarray, List[dict], dict]) -> FitnessResult:
    genes, scenario_configs, game_settings_override = args
    genome = Genome(genes)
    genome.repair()

    apply_genome_to_modules(genome)
    _patch_targeting_controller_init()

    try:
        from kesslergame import Scenario, KesslerGame, GraphicsType
        from Fuzzy_MPC_Controller import Controller

        n_scenarios    = len(scenario_configs)
        fitness_list   = []
        total_hit      = 0
        total_death    = 0
        total_lives    = 0
        acc_list       = []

        for cfg in scenario_configs:
            scenario = Scenario(**cfg)
            settings = {
                'perf_tracker':        True,
                'graphics_type':       GraphicsType.NoGraphics,
                'realtime_multiplier': 0,
                'graphics_obj':        None,
                'frequency':           30,
            }
            settings.update(game_settings_override)

            ctrl = Controller()
            apply_genome_to_supervisor(ctrl.supervisor, genome)

            game     = KesslerGame(settings=settings)
            score, _ = game.run(scenario=scenario, controllers=[ctrl])
            team     = score.teams[0]

            s_hit    = team.asteroids_hit
            s_death  = team.deaths
            s_lives  = max(0, 3 - team.deaths)
            s_acc    = min(1.0, team.accuracy)

            # Per-scenario fitness — each scenario weighted equally
            s_result = FitnessResult(
                fitness=0.0, asteroids_hit=s_hit, deaths=s_death,
                lives_remaining=s_lives, accuracy=s_acc,
                eval_time_ms=0.0, n_scenarios=1,
            )
            s_result.fitness = compute_fitness(s_result)
            fitness_list.append(s_result.fitness)

            total_hit   += s_hit
            total_death += s_death
            total_lives += s_lives
            acc_list.append(s_acc)

        avg_acc = float(np.mean(acc_list)) if acc_list else 0.0
        result  = FitnessResult(
            fitness         = float(np.mean(fitness_list)),   # average, not sum
            asteroids_hit   = total_hit,
            deaths          = total_death,
            lives_remaining = total_lives,
            accuracy        = avg_acc,
            eval_time_ms    = 0.0,
            n_scenarios     = n_scenarios,
        )
        return result

    except Exception as exc:
        return FitnessResult(
            fitness=-9999.0, asteroids_hit=0, deaths=99,
            lives_remaining=0, accuracy=0.0, eval_time_ms=0.0,
            n_scenarios=len(scenario_configs), error=str(exc),
        )


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def tournament_select(
    population: List[Genome],
    fitnesses:  List[float],
    k:          int,
    rng:        np.random.Generator,
) -> Genome:
    indices = rng.choice(len(population), size=k, replace=False)
    best    = max(indices, key=lambda i: fitnesses[i])
    return population[best].copy()


def blend_crossover(
    parent_a: Genome,
    parent_b: Genome,
    alpha:    float,
    rng:      np.random.Generator,
) -> Tuple[Genome, Genome]:
    """BLX-alpha crossover: children sampled from [min-α·d, max+α·d]."""
    lo  = np.minimum(parent_a.genes, parent_b.genes)
    hi  = np.maximum(parent_a.genes, parent_b.genes)
    d   = hi - lo
    low = lo - alpha * d
    hig = hi + alpha * d
    child_a = Genome(rng.uniform(low, hig))
    child_b = Genome(rng.uniform(low, hig))
    child_a.repair()
    child_b.repair()
    return child_a, child_b


def gaussian_mutate(
    genome:     Genome,
    sigma_frac: float,   # kept in signature for API compatibility, used as scale factor
    p_mutate:   float,
    rng:        np.random.Generator,
) -> Genome:
    """
    Gaussian mutation with per-group absolute sigmas.

    Using sigma = sigma_frac × (hi - lo) explodes when bounds are wide
    (e.g. weights span [-10, 10] → sigma=1.6 per step, far too noisy).
    Instead we use semantically meaningful absolute sigmas per group,
    scaled by sigma_frac so the CLI --sigma flag still works as a global
    intensity knob.
    """
    # Absolute sigma per group — tuned to produce meaningful perturbations
    # without making the search chaotic on unbounded parameters.
    GROUP_SIGMA: Dict[str, float] = {
        "supervisor":  0.05,   # R_lo, R_hi are probabilities — small steps
        "risk_field":  0.15,   # tau/distance breakpoints, output singletons
        "targeting":   0.40,   # weights, gains, geometry — larger steps OK
    }

    child = genome.copy()
    for i, spec in enumerate(PARAM_SPECS):
        if rng.random() < p_mutate:
            base_sigma = GROUP_SIGMA.get(spec.group, 0.20)
            sigma = sigma_frac * base_sigma  # sigma_frac scales intensity globally
            child.genes[i] += rng.normal(0.0, sigma)
    child.repair()
    return child


# ---------------------------------------------------------------------------
# GA configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    pop_size:               int   = 30
    n_generations:          int   = 20
    elite_k:                int   = 3
    tournament_k:           int   = 3
    p_crossover:            float = 0.75
    blx_alpha:              float = 0.3
    sigma_frac:             float = 0.08
    p_mutate:               float = 0.20
    n_workers:              int   = 12
    seed:                   int   = 42
    checkpoint_path:        str   = "ga_checkpoint.json"
    scenario_configs:       List[dict] = field(default_factory=list)
    game_settings_override: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GenerationStats
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    generation:   int
    best_fitness: float
    mean_fitness: float
    best_genome:  Dict[str, float]
    best_result:  Optional[FitnessResult] = None


# ---------------------------------------------------------------------------
# GeneticOptimizer
# ---------------------------------------------------------------------------

class GeneticOptimizer:

    def __init__(self, config: GAConfig) -> None:
        self.cfg            = config
        self.rng            = np.random.default_rng(config.seed)
        self.history:       List[GenerationStats] = []
        self._best_genome:  Optional[Genome] = None
        self._best_fitness: float = -math.inf

        if not self.cfg.scenario_configs:
            self.cfg.scenario_configs = self._build_fixed_scenarios()

    # ------------------------------------------------------------------
    def _build_fixed_scenarios(self) -> List[dict]:
        rng = np.random.default_rng(self.cfg.seed)
        scenarios = []
        for name, n_ast in [("Sparse", 10), ("Medium", 20), ("Dense", 35), ("Stress", 50)]:
            asteroid_states = []
            for _ in range(n_ast):
                asteroid_states.append({
                    "position": (float(rng.uniform(0, 1000)), float(rng.uniform(0, 800))),
                    "angle":    float(rng.uniform(0, 360)),
                    "speed":    float(rng.uniform(20, 180)),
                    "size":     int(rng.integers(1, 5)),
                })
            scenarios.append({
                "name": f"GA_{name}",
                "asteroid_states": asteroid_states,
                "ship_states": [{
                    "position": (400, 400), "angle": 90,
                    "lives": 3, "team": 1, "mines_remaining": 3,
                }],
                "map_size":   (1000, 800),
                "time_limit": 60,
                "ammo_limit_multiplier": 0,
                "stop_if_no_ammo": False,
            })
        return scenarios

    # ------------------------------------------------------------------
    def _init_population(self) -> List[Genome]:
        pop  = [Genome()]   # defaults as anchor
        pop += [Genome.random(self.rng) for _ in range(self.cfg.pop_size - 1)]
        return pop

    # ------------------------------------------------------------------
    def _evaluate_population(self, population: List[Genome]) -> List[FitnessResult]:
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
        population: List[Genome],
        fitnesses:  List[float],
    ) -> List[Genome]:
        cfg = self.cfg
        order      = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        sorted_pop = [population[i] for i in order]

        next_gen: List[Genome] = [sorted_pop[i].copy() for i in range(min(cfg.elite_k, len(sorted_pop)))]

        while len(next_gen) < cfg.pop_size:
            pa = tournament_select(population, fitnesses, cfg.tournament_k, self.rng)
            pb = tournament_select(population, fitnesses, cfg.tournament_k, self.rng)

            if self.rng.random() < cfg.p_crossover:
                ca, cb = blend_crossover(pa, pb, cfg.blx_alpha, self.rng)
            else:
                ca, cb = pa.copy(), pb.copy()

            ca = gaussian_mutate(ca, cfg.sigma_frac, cfg.p_mutate, self.rng)
            cb = gaussian_mutate(cb, cfg.sigma_frac, cfg.p_mutate, self.rng)
            next_gen.append(ca)
            if len(next_gen) < cfg.pop_size:
                next_gen.append(cb)

        return next_gen

    # ------------------------------------------------------------------
    def run(self) -> Genome:
        print(f"\n{'='*60}")
        print(f"  Kessler GA Optimiser — {N_PARAMS} parameters")
        print(f"  Population: {self.cfg.pop_size}  |  Generations: {self.cfg.n_generations}")
        print(f"  Workers: {self.cfg.n_workers}")
        print(f"{'='*60}\n")

        population = self._init_population()

        ckpt      = Path(self.cfg.checkpoint_path)
        start_gen = 0
        if ckpt.exists():
            population, start_gen = self._load_checkpoint(ckpt, population)

        # Cache (genome, result) for the all-time best so elites are never
        # re-evaluated — avoids apparent regression when the game engine has
        # any internal non-determinism (RNG, collision ordering, etc.).
        _best_result: Optional[FitnessResult] = None

        for gen in range(start_gen, self.cfg.n_generations):
            t0       = time.perf_counter()
            results  = self._evaluate_population(population)
            fits     = [r.fitness for r in results]

            # ── Update global best ────────────────────────────────────────
            best_idx = int(np.argmax(fits))
            if fits[best_idx] > self._best_fitness:
                self._best_fitness = fits[best_idx]
                self._best_genome  = population[best_idx].copy()
                _best_result       = results[best_idx]

            # ── Stats always report the GLOBAL best, not generation best ──
            # This guarantees the printed best_fitness is non-decreasing.
            stats = GenerationStats(
                generation   = gen,
                best_fitness = self._best_fitness,        # global, not gen
                mean_fitness = float(np.mean(fits)),
                best_genome  = self._best_genome.to_dict(),
                best_result  = _best_result,              # global best result
            )
            self.history.append(stats)
            self._print_gen(stats, time.perf_counter() - t0, gen_best=float(np.max(fits)))
            self._save_checkpoint(ckpt, population, gen + 1)

            if gen < self.cfg.n_generations - 1:
                population = self._next_generation(population, fits)

        print(f"\n✓ Optimisation complete.  Best fitness: {self._best_fitness:.3f}")
        self._print_best_params()
        return self._best_genome  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def apply_best(self) -> None:
        if self._best_genome is None:
            raise RuntimeError("No best genome — run() first.")
        apply_genome_to_modules(self._best_genome)

    def export_best(self, path: str = "best_params.json") -> None:
        if self._best_genome is None:
            raise RuntimeError("No best genome — run() first.")
        with open(path, "w") as f:
            json.dump(self._best_genome.to_dict(), f, indent=2)
        print(f"✓ Best parameters written to {path}")

    @staticmethod
    def load_and_apply(path: str) -> Genome:
        with open(path) as f:
            d = json.load(f)
        genome = Genome.from_dict(d)
        apply_genome_to_modules(genome)
        _patch_targeting_controller_init()
        print(f"✓ Loaded and applied parameters from {path}")
        return genome

    # ------------------------------------------------------------------
    def _save_checkpoint(self, path: Path, population: List[Genome], next_gen: int) -> None:
        data = {
            "next_generation": next_gen,
            "best_fitness":    self._best_fitness,
            "best_genome":     self._best_genome.to_dict() if self._best_genome else None,
            "population":      [g.to_dict() for g in population],
            "history":         [
                {"generation": s.generation, "best_fitness": s.best_fitness,
                 "mean_fitness": s.mean_fitness}
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
                generation=h["generation"], best_fitness=h["best_fitness"],
                mean_fitness=h["mean_fitness"], best_genome={},
            ))
        return population, start_gen

    # ------------------------------------------------------------------
    def _print_gen(self, stats: GenerationStats, elapsed: float, gen_best: float = 0.0) -> None:
        r    = stats.best_result
        n_sc = r.n_scenarios if r else 1
        ml   = n_sc * 3
        hit  = f"{r.asteroids_hit:3d}"           if r else "  ?"
        dead = f"{r.deaths:2d}/{ml}"              if r else " ?"
        live = f"{r.lives_remaining:2d}/{ml}"     if r else " ?"
        acc  = f"{min(r.accuracy,1.0)*100:5.1f}%" if r else "    ?"
        err  = f" [ERR: {r.error[:40]}]" if (r and r.error) else ""
        print(
            f"  Gen {stats.generation:03d}  "
            f"best*={stats.best_fitness:8.2f}  gen={gen_best:8.2f}  "
            f"mean={stats.mean_fitness:8.2f}  "
            f"hit={hit}  dead={dead}  lives={live}  acc={acc}  "
            f"({elapsed:.1f}s){err}"
        )

    def _print_best_params(self) -> None:
        if not self._best_genome:
            return
        d      = self._best_genome.to_dict()
        groups: Dict[str, List[str]] = {}
        for spec in PARAM_SPECS:
            groups.setdefault(spec.group, []).append(spec.name)
        print("\n  Best parameters:")
        for grp, names in groups.items():
            print(f"\n    [{grp}]")
            for name in names:
                print(f"      {name:30s} = {d[name]:.6g}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GA optimizer for Kessler multimodal controller")
    p.add_argument("--pop",     type=int,   default=30,   help="Population size")
    p.add_argument("--gen",     type=int,   default=20,   help="Number of generations")
    p.add_argument("--workers", type=int,   default=12,    help="Parallel workers")
    p.add_argument("--elite",   type=int,   default=3,    help="Elite count")
    p.add_argument("--sigma",   type=float, default=0.08, help="Mutation sigma fraction")
    p.add_argument("--pmut",    type=float, default=0.20, help="Per-gene mutation probability")
    p.add_argument("--seed",    type=int,   default=42,   help="RNG seed")
    p.add_argument("--resume",  type=str,   default=None, help="Resume from checkpoint JSON")
    p.add_argument("--export",  type=str,   default="best_params.json")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg  = GAConfig(
        pop_size        = args.pop,
        n_generations   = args.gen,
        n_workers       = args.workers,
        elite_k         = args.elite,
        sigma_frac      = args.sigma,
        p_mutate        = args.pmut,
        seed            = args.seed,
        checkpoint_path = args.resume or "ga_checkpoint.json",
    )
    opt  = GeneticOptimizer(cfg)
    best = opt.run()
    opt.export_best(args.export)


if __name__ == "__main__":
    mp.freeze_support()
    main()