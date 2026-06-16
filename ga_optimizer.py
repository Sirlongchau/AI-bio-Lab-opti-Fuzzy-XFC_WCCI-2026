"""
ga_optimizer.py — v4
====================
GA pour le contrôleur multimodal Kessler, aligné sur les fichiers refactorés
(targeting_system.py « sticky-bearing » borné, risk_field.py, supervisor.py).

Ce que le GA optimise :
  - Supervisor : seuils d'hystérésis R_lo / R_hi.
  - RiskField  : MF tau / distance + singletons (corrige le retard d'évaluation).
  - Targeting  : W_SMALL, W_CLOSE, MAX_DIST_FIRE, HIT_FRACTION, STICK_DEG,
                 STICK_GAIN (mopping-de-cluster ⇄ réponse-à-l'imminent),
                 SWITCH_RATIO, TRACK_TOL_PX.
  - Sacrifice  : MINE_MIN_CATCH, MINE_MAX_HOLD.

Nouveautés v4 :
  - Parallélisme joblib/loky : exécuteur persistant réutilisé entre générations,
    init worker PARESSEUSE et idempotente (imports + NLP CasADi pré-construits +
    debug coupé), batch_size=1 → load-balancing optimal sur des parties de
    durées très inégales (un génome qui meurt vite = tâche courte).
  - Screening COMPARABLE : tout le monde est évalué sur un sous-ensemble des
    scénarios COMPLETS (pleine durée) ; les promus n'évaluent QUE les scénarios
    restants et on FUSIONNE avec les résultats de screening. La fitness des
    recalés (moyenne sur le sous-ensemble) et des promus (moyenne sur tout) est
    donc la moyenne de la même fonction par scénario → même échelle, et aucun
    scénario n'est rejoué deux fois.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Console output uses Unicode (✓, ⇄, ↩). On Windows the default console
# encoding is cp1252 and raises UnicodeEncodeError mid-print; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass  # older Python or already-wrapped stream — best effort only
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Parameter schema — aligné sur les modules refactorés
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    name:    str
    default: float
    lo:      float
    hi:      float
    group:   str
    is_int:  bool = False


PARAM_SPECS: List[ParamSpec] = [
    # ── Supervisor (hystérésis de mode) ───────────────────────────────────
    ParamSpec("R_lo", 0.60, 0.050, 0.92, "supervisor"),
    ParamSpec("R_hi", 0.80, 0.10, 0.98, "supervisor"),
    ParamSpec("MPC_TAU_TRIGGER",  1.25, 0.35, 3.00, "supervisor"),
    ParamSpec("MPC_NEAR_TRIGGER", 135.0, 40.0, 320.0, "supervisor"),
    ParamSpec("MPC_RISK_TRIGGER", 0.45, 0.05, 0.95, "supervisor"),
    ParamSpec("MPC_FAIL_LIMIT",   3.0,  1.0,  8.0, "supervisor", is_int=True),

    # ── RiskField (corrige le retard d'évaluation via les MF) ─────────────
    ParamSpec("alpha_aggregation", 0.60, 0.0, 1.0, "risk_field"),
    ParamSpec("tau_crit_b",   0.25, 0.05,  3.0, "risk_field"),
    ParamSpec("tau_crit_d",   0.50, 0.10,  5.0, "risk_field"),
    ParamSpec("tau_close_a",  0.25, 0.05,  5.0, "risk_field"),
    ParamSpec("tau_close_b",  0.50, 0.10,  6.0, "risk_field"),
    ParamSpec("tau_close_d",  1.50, 0.30,  8.0, "risk_field"),
    ParamSpec("tau_med_a",    0.60, 0.10,  8.0, "risk_field"),
    ParamSpec("tau_med_c",    1.50, 0.30, 10.0, "risk_field"),
    ParamSpec("tau_med_d",    5.50, 1.00, 20.0, "risk_field"),
    ParamSpec("d_near_c",    50.0,  10.0,  400.0, "risk_field"),
    ParamSpec("d_near_d",    90.0,  20.0,  600.0, "risk_field"),
    ParamSpec("d_med_a",     50.0,  10.0,  500.0, "risk_field"),
    ParamSpec("d_med_d",    220.0,  50.0, 1000.0, "risk_field"),
    ParamSpec("FFS", 0.02, 0.0, 1.0, "risk_field"),
    ParamSpec("FFM", 0.04, 0.0, 1.0, "risk_field"),
    ParamSpec("FFL", 0.08, 0.0, 1.0, "risk_field"),

    ParamSpec("FMS", 0.05, 0.0, 1.0, "risk_field"),
    ParamSpec("FMM", 0.10, 0.0, 1.0, "risk_field"),
    ParamSpec("FML", 0.18, 0.0, 1.0, "risk_field"),

    ParamSpec("FNS", 0.12, 0.0, 1.0, "risk_field"),
    ParamSpec("FNM", 0.22, 0.0, 1.0, "risk_field"),
    ParamSpec("FNL", 0.35, 0.0, 1.0, "risk_field"),

    ParamSpec("CFS", 0.10, 0.0, 1.0, "risk_field"),
    ParamSpec("CFM", 0.18, 0.0, 1.0, "risk_field"),
    ParamSpec("CFL", 0.30, 0.0, 1.0, "risk_field"),

    ParamSpec("CMS", 0.22, 0.0, 1.0, "risk_field"),
    ParamSpec("CMM", 0.35, 0.0, 1.0, "risk_field"),
    ParamSpec("CML", 0.50, 0.0, 1.0, "risk_field"),

    ParamSpec("CNS", 0.45, 0.0, 1.0, "risk_field"),
    ParamSpec("CNM", 0.65, 0.0, 1.0, "risk_field"),
    ParamSpec("CNL", 0.85, 0.0, 1.0, "risk_field"),

    ParamSpec("MFS", 0.18, 0.0, 1.0, "risk_field"),
    ParamSpec("MFM", 0.30, 0.0, 1.0, "risk_field"),
    ParamSpec("MFL", 0.45, 0.0, 1.0, "risk_field"),

    ParamSpec("MMS", 0.40, 0.0, 1.0, "risk_field"),
    ParamSpec("MMM", 0.60, 0.0, 1.0, "risk_field"),
    ParamSpec("MML", 0.80, 0.0, 1.0, "risk_field"),

    ParamSpec("MNS", 0.65, 0.0, 1.0, "risk_field"),
    ParamSpec("MNM", 0.85, 0.0, 1.0, "risk_field"),
    ParamSpec("MNL", 1.00, 0.0, 1.0, "risk_field"),

    # ── Targeting (turret sticky-bearing refactoré) ───────────────────────
    ParamSpec("W_SMALL",       0.30,    0.0,    1.0, "targeting"),
    ParamSpec("W_CLOSE",       0.25,    0.0,    1.0, "targeting"),
    ParamSpec("MAX_DIST_FIRE", 700.0, 200.0, 1200.0, "targeting"),
    ParamSpec("HIT_FRACTION",  0.85,    0.40,   1.0, "targeting"),
    ParamSpec("STICK_DEG",     50.0,   20.0,  120.0, "targeting"),
    ParamSpec("STICK_GAIN",    1.0,     0.0,    6.0, "targeting"),
    ParamSpec("SWITCH_RATIO",  1.30,    1.0,    3.0, "targeting"),
    ParamSpec("TRACK_TOL_PX",  28.0,   10.0,   80.0, "targeting"),

    # ── Sacrifice ─────────────────────────────────────────────────────────
    ParamSpec("MINE_MIN_CATCH", 3.0, 1.0,  8.0, "targeting", is_int=True),
    ParamSpec("MINE_MAX_HOLD", 15.0, 3.0, 40.0, "targeting", is_int=True),
    ParamSpec("MINE_TAU_MIN",      0.70, 0.20, 2.00, "targeting"),
    ParamSpec("MINE_TAU_MAX",      3.00, 1.00, 6.00, "targeting"),
    ParamSpec("MINE_RISK_TH",      0.55, 0.10, 0.95, "targeting"),
    ParamSpec("MINE_NEAR_SURFACE", 210.0, 60.0, 420.0, "targeting"),
]

N_PARAMS    = len(PARAM_SPECS)
PARAM_NAMES = [p.name for p in PARAM_SPECS]
_IDX        = {n: i for i, n in enumerate(PARAM_NAMES)}
_LO         = np.array([p.lo for p in PARAM_SPECS])
_HI         = np.array([p.hi for p in PARAM_SPECS])
_RANGE      = _HI - _LO
_IS_INT     = np.array([p.is_int for p in PARAM_SPECS])


# ---------------------------------------------------------------------------
# Genome — repair systématique
# ---------------------------------------------------------------------------

class Genome:

    __slots__ = ("genes",)

    def __init__(self, genes: Optional[np.ndarray] = None) -> None:
        if genes is None:
            self.genes = np.array([p.default for p in PARAM_SPECS], dtype=np.float64)
        else:
            self.genes = np.array(genes, dtype=np.float64)
        self.repair()                       # ← TOUJOURS valide, partout

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        return cls(rng.uniform(_LO, _HI))

    def to_dict(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(PARAM_NAMES, self.genes)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Genome":
        return cls(np.array(
            [d.get(n, PARAM_SPECS[i].default) for i, n in enumerate(PARAM_NAMES)],
        ))

    def key(self) -> bytes:
        """Clé de déduplication (quantifiée pour absorber le bruit float)."""
        return np.round(self.genes, 5).tobytes()

    def repair(self) -> None:
        np.clip(self.genes, _LO, _HI, out=self.genes)

        g = lambda n: float(self.genes[_IDX[n]])
        def s(n, v): self.genes[_IDX[n]] = float(v)
        def ordpair(a, b, gap):
            va, vb = g(a), g(b)
            if va >= vb - gap:
                mid = 0.5 * (va + vb)
                s(a, mid - gap / 2.0)
                s(b, mid + gap / 2.0)

        ordpair("R_lo", "R_hi", 0.05)                       # hystérésis correcte
        ordpair("tau_crit_b",  "tau_crit_d",  0.05)         # tau monotone
        ordpair("tau_close_a", "tau_close_b", 0.05)
        ordpair("tau_close_b", "tau_close_d", 0.05)
        ordpair("tau_med_a",   "tau_med_c",   0.05)
        ordpair("tau_med_c",   "tau_med_d",   0.10)
        ordpair("d_near_c", "d_near_d", 5.0)                # distance monotone
        ordpair("d_med_a",  "d_med_d", 10.0)
        ordpair("MINE_TAU_MIN", "MINE_TAU_MAX", 0.20)
        # ordpair("out_negligible", "out_low",    0.01)       # singletons ordonnés
        # ordpair("out_low",        "out_medium", 0.01)
        # ordpair("out_medium",     "out_high",   0.01)

        np.clip(self.genes, _LO, _HI, out=self.genes)
        self.genes[_IS_INT] = np.round(self.genes[_IS_INT])

    def copy(self) -> "Genome":
        return Genome(self.genes.copy())


# ---------------------------------------------------------------------------
# Injection des paramètres dans les modules (côté worker)
# ---------------------------------------------------------------------------

def apply_genome_to_modules(d: Dict[str, float]) -> None:
    """Patch les globals des modules. Idempotent, appelé à chaque tâche."""
    try:
        import supervisor as sp
        sp.MPC_TAU_TRIGGER  = d["MPC_TAU_TRIGGER"]
        sp.MPC_NEAR_TRIGGER = d["MPC_NEAR_TRIGGER"]
        sp.MPC_RISK_TRIGGER = d["MPC_RISK_TRIGGER"]
        sp.MPC_FAIL_LIMIT   = int(d["MPC_FAIL_LIMIT"])
    except ImportError:
        pass

    try:
        import risk_field as rf
        rf.ALPHA_AGGREGATION = d["alpha_aggregation"]
        rf.TAU_CRITICAL = (0.0, 0.0, d["tau_crit_b"], d["tau_crit_d"])
        cb = d["tau_close_b"]
        rf.TAU_CLOSE  = (d["tau_close_a"], cb, cb, d["tau_close_d"])
        mc = d["tau_med_c"]
        rf.TAU_MEDIUM = (d["tau_med_a"], mc, mc, d["tau_med_d"])
        rf.TAU_FAR    = (mc, d["tau_med_d"], 99.0, 99.0)
        dnd = d["d_near_d"]
        rf.D_NEAR   = (0.0, 0.0, d["d_near_c"], dnd)
        rf.D_MEDIUM = (d["d_med_a"], dnd, dnd, d["d_med_d"])
        rf.D_FAR    = (dnd, d["d_med_d"], 9999.0, 9999.0)
        for name in (
            "FFS","FFM","FFL",
            "FMS","FMM","FML",
            "FNS","FNM","FNL",

            "CFS","CFM","CFL",
            "CMS","CMM","CML",
            "CNS","CNM","CNL",

            "MFS","MFM","MFL",
            "MMS","MMM","MML",
            "MNS","MNM","MNL",
        ):
            setattr(rf, name, d[name])
    except ImportError:
        pass

    try:
        import targeting_system as ts
        ts.W_SMALL        = d["W_SMALL"]
        ts.W_CLOSE        = d["W_CLOSE"]
        ts.MAX_DIST_FIRE  = d["MAX_DIST_FIRE"]
        ts.HIT_FRACTION   = d["HIT_FRACTION"]
        ts.STICK_DEG      = d["STICK_DEG"]
        ts.STICK_GAIN     = d["STICK_GAIN"]
        ts.SWITCH_RATIO   = d["SWITCH_RATIO"]
        ts.TRACK_TOL_PX   = d["TRACK_TOL_PX"]
        ts.MINE_MIN_CATCH = int(d["MINE_MIN_CATCH"])
        ts.MINE_MAX_HOLD  = int(d["MINE_MAX_HOLD"])
        ts.MINE_TAU_MIN   = d["MINE_TAU_MIN"]
        ts.MINE_TAU_MAX   = d["MINE_TAU_MAX"]
        ts.MINE_RISK_TH   = d["MINE_RISK_TH"]
        ts.MINE_NEAR_SURFACE = d["MINE_NEAR_SURFACE"]
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Worker — init PARESSEUSE et idempotente (loky réutilise les process)
# ---------------------------------------------------------------------------

_W: dict = {}    # cache par process : modules + solveurs CasADi pré-construits


def _ensure_worker(map_sizes: Tuple[Tuple[float, float], ...]) -> None:
    """
    Payé UNE fois par process worker (loky garde les workers chauds entre les
    générations). Importe les modules, coupe le debug du contrôleur, et
    pré-construit le NLP CasADi par map_size (indépendant du génome/scénario).
    """
    global _W
    if _W.get("_ready"):
        return
    os.environ["KESSLER_DEBUG"] = "0"          # workers headless & rapides

    from kesslergame import Scenario, KesslerGame, GraphicsType
    from Fuzzy_MPC_Controller import Controller
    from risk_field import RiskField
    import Casadi_mpc as cm

    _W.update(
        Scenario=Scenario, KesslerGame=KesslerGame, GraphicsType=GraphicsType,
        Controller=Controller, RiskField=RiskField, cm=cm, solvers={},
    )
    for ms in map_sizes:
        try:
            slv = cm._CasADiSolver(tuple(ms))
            slv._build(cm.LAMBDA_D)
            _W["solvers"][tuple(ms)] = slv
        except Exception:
            pass                               # dégradé gracieux : build lazy
    _W["_ready"] = True


def _make_controller(map_size: Tuple[float, float]):
    """Controller frais branché sur le solveur CasADi pré-construit du worker."""
    ctrl = _W["Controller"]()
    try:
        key = tuple(map_size)
        slv = _W["solvers"].get(key)
        if slv is None:
            cm = _W["cm"]
            slv = cm._CasADiSolver(key); slv._build(cm.LAMBDA_D)
            _W["solvers"][key] = slv
        slv._u_prev = None                     # amorçage propre par partie

        mpc = ctrl.supervisor.controllers["mpc"]
        mpc._use_casadi = True
        mpc._map_size   = key
        mpc.risk_field  = _W["RiskField"](map_size=key)
        mpc._casadi_slv = slv
    except Exception:
        pass
    return ctrl


def _eval_task(gid: int, genes: np.ndarray, scenario_cfg: dict,
               settings_override: dict,
               map_sizes: Tuple[Tuple[float, float], ...]) -> Tuple[int, dict]:
    """UNE tâche = UN génome sur UN scénario."""
    _ensure_worker(map_sizes)
    genome = Genome(genes)
    d = genome.to_dict()
    apply_genome_to_modules(d)

    try:
        settings = {
            'perf_tracker':        False,
            'graphics_type':       _W["GraphicsType"].NoGraphics,
            'realtime_multiplier': 0,
            'graphics_obj':        None,
            'frequency':           30,
        }
        settings.update(settings_override)

        ctrl = _make_controller(scenario_cfg["map_size"])
        ctrl.supervisor.R_lo = d["R_lo"]
        ctrl.supervisor.R_hi = d["R_hi"]
        ctrl.supervisor.mpc_tau_trigger = d["MPC_TAU_TRIGGER"]
        ctrl.supervisor.mpc_near_trigger = d["MPC_NEAR_TRIGGER"]
        ctrl.supervisor.mpc_risk_trigger = d["MPC_RISK_TRIGGER"]
        ctrl.supervisor.mpc_fail_limit = int(d["MPC_FAIL_LIMIT"])

        score, _ = _W["KesslerGame"](settings=settings).run(
            scenario=_W["Scenario"](**scenario_cfg), controllers=[ctrl]
        )
        team = score.teams[0]
        stop_s = str(score.stop_reason)
        return gid, {
            "hit":    team.asteroids_hit,
            "deaths": team.deaths,
            "lives":  max(0, 3 - team.deaths),
            "acc":    min(1.0, team.accuracy),
            "stop_no_ships": ("no_ships" in stop_s),
            "n_asteroids": scenario_cfg.get("num_asteroids", 1),
            "error":  None,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return gid, {"hit": 0, "deaths": 3, "lives": 0, "acc": 0.0,
                     "stop_no_ships": True, "n_asteroids": scenario_cfg.get("num_asteroids", 1),
                     "error": str(exc)}


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

# v2 weights: the v1 death-gate (-200) was so steep that surviving a single hard
# scenario dominated the per-scenario mean, so the GA traded away accuracy
# everywhere (0.87 -> 0.42) to buy one survival — making it worse than default on
# easy/sparse scenarios. Softer death penalty + stronger accuracy reward keep
# survival valued (hit_mult already triples hits when surviving) without letting
# one survival swamp offense across all other scenarios.
# Safety-first v3:
#   - strong penalty for deaths / no_ships
#   - large bonus for zero-death runs
#   - still rewards hits and accuracy so the policy does not only flee
W_HIT            =  8.0
W_HIT_RATIO      =  250.0
W_DEATH          = -300.0
W_SURVIVAL       =  150.0
W_ZERO_DEATH     =  350.0
W_NO_SHIPS       = -500.0
W_ACCURACY       =  2.0


def compute_fitness(scenario_results: List[dict]) -> float:
    """Mean per-scenario fitness.

    Objective:
      1. Stay alive.
      2. Do not end with no_ships.
      3. Still destroy asteroids and maintain accuracy.
    """
    if not scenario_results:
        return -9999.0

    fits = []
    for r in scenario_results:
        if r.get("error"):
            fits.append(-9999.0)
            continue

        hit = float(r.get("hit", 0))
        deaths = float(r.get("deaths", 3))
        lives = float(r.get("lives", max(0, 3 - deaths)))
        acc = float(r.get("acc", 0.0))
        n_ast = max(float(r.get("n_asteroids", 1)), 1.0)
        hit_ratio = min(hit / n_ast, 2.0)

        score = (
            W_HIT       * hit
            + W_HIT_RATIO * hit_ratio
            + W_DEATH     * deaths
            + W_SURVIVAL  * lives
            + W_ACCURACY  * acc * 100.0
        )

        if deaths == 0:
            score += W_ZERO_DEATH

        if r.get("stop_no_ships", False):
            score += W_NO_SHIPS

        fits.append(score)

    return float(np.mean(fits))


# ---------------------------------------------------------------------------
# Opérateurs GA
# ---------------------------------------------------------------------------

def tournament_select(pop, fits, k, rng) -> Genome:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[max(idx, key=lambda i: fits[i])].copy()


def blend_crossover(pa, pb, alpha, rng) -> Tuple[Genome, Genome]:
    lo = np.minimum(pa.genes, pb.genes)
    hi = np.maximum(pa.genes, pb.genes)
    d  = hi - lo
    return (Genome(rng.uniform(lo - alpha*d, hi + alpha*d)),
            Genome(rng.uniform(lo - alpha*d, hi + alpha*d)))


def gaussian_mutate(genome, sigma_frac, p_mutate, rng) -> Genome:
    child = genome.copy()
    mask  = rng.random(N_PARAMS) < p_mutate
    noise = rng.normal(0.0, sigma_frac * _RANGE)   # σ ∝ plage de chaque paramètre
    child.genes = child.genes + mask * noise
    child.repair()
    return child


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    pop_size:        int   = 100
    n_generations:   int   = 25
    elite_k:         int   = 3
    tournament_k:    int   = 3
    p_crossover:     float = 0.75
    blx_alpha:       float = 0.3
    sigma_frac:      float = 0.10
    p_mutate:        float = 0.25
    n_workers:       int   = -1          # joblib n_jobs (-1 = tous les cœurs)
    seed:            int   = 42
    checkpoint_path: str   = "ga_checkpoint.json"
    time_limit:      float = 60.0
    # Screening comparable : tout le monde sur les `n_screen` premiers scénarios
    # (pleine durée), seuls les top `screen_frac` finissent les scénarios restants.
    use_screening:   bool  = True
    screen_frac:     float = 0.35
    n_screen:        int   = 2
    scenario_configs:       List[dict] = field(default_factory=list)
    game_settings_override: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GeneticOptimizer
# ---------------------------------------------------------------------------

class GeneticOptimizer:

    def __init__(self, config: GAConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._best_genome:  Optional[Genome] = None
        self._best_fitness: float = -math.inf
        self._cache: Dict[bytes, float] = {}
        self.history: List[dict] = []

        if not self.cfg.scenario_configs:
            self.cfg.scenario_configs = self._build_fixed_scenarios()

        # Exécuteur joblib persistant (loky) réutilisé à chaque génération.
        self._parallel = Parallel(
            n_jobs=self.cfg.n_workers, backend="loky",
            batch_size=1, pre_dispatch="all",
        )

    # ------------------------------------------------------------------

    def _map_sizes(self) -> Tuple[Tuple[float, float], ...]:
        seen = []
        for cfg in self.cfg.scenario_configs:
            ms = tuple(cfg["map_size"])
            if ms not in seen:
                seen.append(ms)
        return tuple(seen) or ((1000.0, 800.0),)

    def _build_fixed_scenarios(self) -> List[dict]:
        """
        Four hand-crafted scenarios spanning a difficulty gradient. Each one
        rewards a *different* sub-policy, so the average fitness pressures the
        GA into BOTH attacking well AND knowing when to flee.
 
          1. Skill_Sparse  (10 ast, slow)              — pure turret efficiency.
                                                         Punishes setups that
                                                         depress firing range or
                                                         broaden aim tolerance
                                                         beyond what's needed.
          2. Cluster_Cross (20 ast, medium, convergent)— a tight crossfire ring
                                                         around the ship. The
                                                         first big asteroid the
                                                         turret fragments rakes
                                                         the cluster; staying
                                                         put = swallowed by the
                                                         debris. The fleeing
                                                         policy WINS here.
          3. Wall_Pressure (28 ast, slow, aligned)     — a slow-moving wall the
                                                         turret can shave but
                                                         not stop in time.
                                                         Forces some MPC use.
          4. Storm         (40 ast, fast, random)      — chaos. Survival is the
                                                         dominant signal; the
                                                         fleeing policy loses
                                                         less than the standing
                                                         policy.
 
        Two centre screens (Cluster + Wall) act as the screening filter — they
        are the most discriminating between «attack-only» and «attack + flee».
        Sparse and Storm sit at the extremes to guard against overfitting.
        """
        rng = np.random.default_rng(self.cfg.seed)
        MAP_W, MAP_H = 1000.0, 800.0
        SHIP_X, SHIP_Y = 500.0, 400.0      # exact arena centre (was off-centre)
 
        def jitter(v, span):
            return float(v + rng.uniform(-span, span))
 
        # ---- 1. Skill_Sparse -------------------------------------------------
        # 10 slow asteroids spread out. Easy to clear with a good turret;
        # rewards long firing range and tight aim tolerance.
        skill = []
        for _ in range(10):
            skill.append({
                "position": (float(rng.uniform(50, MAP_W-50)),
                             float(rng.uniform(50, MAP_H-50))),
                "angle":    float(rng.uniform(0, 360)),
                "speed":    float(rng.uniform(30, 90)),
                "size":     int(rng.integers(1, 5)),
            })
 
        # ---- 2. Cluster_Cross ------------------------------------------------
        # A ring of 20 asteroids ~250 px from the ship, each one heading
        # roughly inward. Big asteroids placed deliberately so the inevitable
        # fragmentation creates a converging debris swarm. Standing still is
        # lethal; sliding 100–150 px sideways breaks the convergence.
        cluster = []
        for k in range(20):
            theta = 2 * math.pi * k / 20 + jitter(0, 0.08)
            r     = jitter(260.0, 25.0)
            x = SHIP_X + r * math.cos(theta)
            y = SHIP_Y + r * math.sin(theta)
            # Velocity aimed at ship centre, magnitude moderate so the GA can
            # actually outrun it with the MPC.
            inward_deg = (math.degrees(math.atan2(SHIP_Y - y, SHIP_X - x))
                          + jitter(0, 15.0)) % 360.0
            size = 3 if (k % 4 == 0) else int(rng.integers(1, 4))    # 25% large
            cluster.append({
                "position": (x, y),
                "angle":    inward_deg,
                "speed":    float(jitter(140.0, 25.0)),
                "size":     size,
            })
 
        # ---- 3. Wall_Pressure ------------------------------------------------
        # 28 asteroids stacked in a slow-moving wall on the right side, drifting
        # left. The turret can erode it but cannot annihilate it in time at any
        # fire rate — staying put eventually loses. Retreating left buys time.
        wall = []
        for row in range(4):
            for col in range(7):
                x = jitter(850.0 - 35.0 * col, 12.0)
                y = jitter(150.0 + 150.0 * row, 12.0)
                wall.append({
                    "position": (x, y),
                    "angle":    float(jitter(180.0, 12.0)),         # westward
                    "speed":    float(jitter(70.0, 15.0)),
                    "size":     int(rng.integers(2, 4)),            # mostly mid
                })
 
        # ---- 4. Storm --------------------------------------------------------
        # 40 random fast asteroids. Survivable only with active dodging;
        # rewards policies that limit deaths even when scoring is hard.
        storm = []
        for _ in range(40):
            # Bias spawns away from the ship centre to avoid instant deaths
            # that aren't the controller's fault.
            while True:
                px, py = rng.uniform(0, MAP_W), rng.uniform(0, MAP_H)
                if (px - SHIP_X)**2 + (py - SHIP_Y)**2 > 180.0**2:
                    break
            storm.append({
                "position": (float(px), float(py)),
                "angle":    float(rng.uniform(0, 360)),
                "speed":    float(rng.uniform(120, 280)),
                "size":     int(rng.integers(1, 5)),
            })
 
        # ---- Assemble --------------------------------------------------------
        # Screening order (first n_screen): Cluster + Wall — the two scenarios
        # where «attack + occasional flee» beats «attack only».
        specs = [
            ("Cluster_Cross", cluster),
            ("Wall_Pressure", wall),
            ("Skill_Sparse",  skill),
            ("Storm",         storm),
        ]
        full = []
        for name, asteroid_states in specs:
            full.append({
                "name":                  f"GA_{name}",
                "asteroid_states":       asteroid_states,
                "ship_states":           [{"position": (SHIP_X, SHIP_Y),
                                           "angle": 90, "lives": 3, "team": 1,
                                           "mines_remaining": 3}],
                # KesslerGame's Scenario type-checks map_size as tuple[int, int].
                "map_size":              (int(MAP_W), int(MAP_H)),
                "time_limit":            self.cfg.time_limit,
                "ammo_limit_multiplier": 0,
                "stop_if_no_ammo":       False,
            })
        return full
 

    # ------------------------------------------------------------------

    def _run_tasks(self, genomes: List[Tuple[int, Genome]],
                   scenarios: List[dict]) -> Dict[int, List[dict]]:
        """Évalue chaque (génome, scénario) en parallèle (joblib/loky)."""
        ms = self._map_sizes()
        ov = self.cfg.game_settings_override
        tasks = [(gid, g.genes.copy(), cfg, ov, ms)
                 for gid, g in genomes for cfg in scenarios]
        results: Dict[int, List[dict]] = {gid: [] for gid, _ in genomes}
        if not tasks:
            return results

        if self.cfg.n_workers == 1:
            _ensure_worker(ms)
            out = [_eval_task(*t) for t in tasks]
        else:
            out = self._parallel(delayed(_eval_task)(*t) for t in tasks)

        for gid, r in out:
            results[gid].append(r)
        return results

    # ------------------------------------------------------------------

    def _evaluate(self, population: List[Genome]) -> List[float]:
        """Dédup + screening COMPARABLE (sous-ensemble réutilisé) + complétion."""
        fits: List[Optional[float]] = [None] * len(population)

        # 1. Cache (élites, doublons, génomes revisités)
        fresh: List[Tuple[int, Genome]] = []
        for i, g in enumerate(population):
            cached = self._cache.get(g.key())
            if cached is not None:
                fits[i] = cached
            else:
                fresh.append((i, g))
        if not fresh:
            return fits  # type: ignore

        full = self.cfg.scenario_configs
        do_screen = (self.cfg.use_screening and len(fresh) > 4
                     and 0 < self.cfg.n_screen < len(full))

        if not do_screen:
            res = self._run_tasks(fresh, full)
            for gid, g in fresh:
                f = compute_fitness(res[gid])
                fits[gid] = f
                self._cache[g.key()] = f
            return fits  # type: ignore

        # 2. Screening : tout le monde sur le sous-ensemble (pleine durée)
        screen, rest = full[:self.cfg.n_screen], full[self.cfg.n_screen:]
        res_screen = self._run_tasks(fresh, screen)
        screen_fit = {gid: compute_fitness(res_screen[gid]) for gid, _ in fresh}

        order = sorted(fresh, key=lambda t: screen_fit[t[0]], reverse=True)
        n_full = max(2, int(math.ceil(len(fresh) * self.cfg.screen_frac)))
        promoted, dropped = order[:n_full], order[n_full:]

        # Recalés : fitness = moyenne sur le sous-ensemble (mêmes unités).
        for gid, g in dropped:
            fits[gid] = screen_fit[gid]
            self._cache[g.key()] = screen_fit[gid]

        # 3. Promus : on évalue UNIQUEMENT les scénarios restants, puis fusion.
        res_rest = self._run_tasks(promoted, rest) if rest else {gid: [] for gid, _ in promoted}
        for gid, g in promoted:
            f = compute_fitness(res_screen[gid] + res_rest[gid])
            fits[gid] = f
            self._cache[g.key()] = f

        return fits  # type: ignore

    # ------------------------------------------------------------------

    def run(self) -> Genome:
        cfg = self.cfg
        print(f"\n{'='*64}")
        print(f"  Kessler GA v4 — {N_PARAMS} params | pop {cfg.pop_size} "
              f"| gen {cfg.n_generations} | n_jobs {cfg.n_workers}")
        print(f"  screening={'ON' if cfg.use_screening else 'OFF'} "
              f"(n_screen={cfg.n_screen}/{len(cfg.scenario_configs)}) "
              f"| time_limit={cfg.time_limit}s")
        print(f"{'='*64}\n")

        population = [Genome()] + [Genome.random(self.rng)
                                   for _ in range(cfg.pop_size - 1)]

        ckpt = Path(cfg.checkpoint_path)
        start_gen = 0
        if ckpt.exists():
            population, start_gen = self._load_checkpoint(ckpt)

        try:
            for gen in range(start_gen, cfg.n_generations):
                t0   = time.perf_counter()
                fits = self._evaluate(population)

                gi = int(np.argmax(fits))
                if fits[gi] > self._best_fitness:
                    self._best_fitness = fits[gi]
                    self._best_genome  = population[gi].copy()

                self.history.append({
                    "generation":   gen,
                    "best_fitness": self._best_fitness,
                    "gen_best":     float(np.max(fits)),
                    "mean_fitness": float(np.mean(fits)),
                })
                print(f"  Gen {gen:03d}  best*={self._best_fitness:8.2f}  "
                      f"gen={np.max(fits):8.2f}  mean={np.mean(fits):8.2f}  "
                      f"cache={len(self._cache)}  "
                      f"({time.perf_counter() - t0:.1f}s)")

                self._save_checkpoint(ckpt, population, gen + 1)

                if gen < cfg.n_generations - 1:
                    population = self._next_generation(population, fits)
        finally:
            self._shutdown_workers()

        print(f"\n✓ Done. Best fitness: {self._best_fitness:.3f}")
        return self._best_genome  # type: ignore

    def _shutdown_workers(self) -> None:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except Exception:
            pass

    # ------------------------------------------------------------------

    def _next_generation(self, population, fits) -> List[Genome]:
        cfg   = self.cfg
        order = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
        next_gen = [population[order[i]].copy()
                    for i in range(min(cfg.elite_k, len(order)))]

        while len(next_gen) < cfg.pop_size:
            pa = tournament_select(population, fits, cfg.tournament_k, self.rng)
            pb = tournament_select(population, fits, cfg.tournament_k, self.rng)
            if self.rng.random() < cfg.p_crossover:
                ca, cb = blend_crossover(pa, pb, cfg.blx_alpha, self.rng)
            else:
                ca, cb = pa.copy(), pb.copy()
            next_gen.append(gaussian_mutate(ca, cfg.sigma_frac, cfg.p_mutate, self.rng))
            if len(next_gen) < cfg.pop_size:
                next_gen.append(gaussian_mutate(cb, cfg.sigma_frac, cfg.p_mutate, self.rng))
        return next_gen

    # ------------------------------------------------------------------

    def export_best(self, path: str = "best_params.json") -> None:
        if self._best_genome is None:
            raise RuntimeError("run() first.")
        self._best_genome.repair()
        with open(path, "w") as f:
            json.dump(self._best_genome.to_dict(), f, indent=2)
        print(f"✓ Written to {path}")

    def _save_checkpoint(self, path, population, next_gen) -> None:
        with open(path, "w") as f:
            json.dump({
                "next_generation": next_gen,
                "best_fitness":    self._best_fitness,
                "best_genome":     self._best_genome.to_dict()
                                   if self._best_genome else None,
                "population":      [g.to_dict() for g in population],
                "cache":           {k.hex(): v for k, v in self._cache.items()},
                "history":         self.history,
            }, f)

    def _load_checkpoint(self, path) -> Tuple[List[Genome], int]:
        print(f"  ↩  Resuming: {path}")
        with open(path) as f:
            data = json.load(f)
        pop = [Genome.from_dict(d) for d in data["population"]]
        if data.get("best_genome"):
            self._best_genome  = Genome.from_dict(data["best_genome"])
            self._best_fitness = data["best_fitness"]
        self._cache  = {bytes.fromhex(k): v
                        for k, v in data.get("cache", {}).items()}
        self.history = data.get("history", [])
        return pop, data["next_generation"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pop",       type=int,   default=100)
    p.add_argument("--gen",       type=int,   default=25)
    p.add_argument("--workers",   type=int,   default=-1, help="joblib n_jobs (-1 = all cores)")
    p.add_argument("--sigma",     type=float, default=0.10)
    p.add_argument("--pmut",      type=float, default=0.25)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--tl",        type=float, default=40.0)
    p.add_argument("--no-screen", action="store_true")
    p.add_argument("--export",    type=str,   default="best_params.json")
    args = p.parse_args()

    cfg = GAConfig(
        pop_size      = args.pop,
        n_generations = args.gen,
        n_workers     = args.workers,
        sigma_frac    = args.sigma,
        p_mutate      = args.pmut,
        seed          = args.seed,
        time_limit    = args.tl,
        use_screening = not args.no_screen,
    )
    opt = GeneticOptimizer(cfg)
    opt.run()
    opt.export_best(args.export)


if __name__ == "__main__":
    main()