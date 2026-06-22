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
import time
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
    # ── Supervisor (hystérésis de mode + horizon de sécurité dur) ─────────
    ParamSpec("R_lo", 0.60, 0.050, 0.92, "supervisor"),
    ParamSpec("R_hi", 0.80, 0.10, 0.98, "supervisor"),
    # Hard evade trigger: if the soonest surface-TTC drops below this, the
    # supervisor forces MPC regardless of the crowd aggregate. This is what
    # makes a SINGLE imminent collision evade (the fuzzy aggregate alone can
    # never saturate on one threat once alpha<R_hi).
    ParamSpec("TAU_EMERGENCY", 1.20, 0.50, 2.50, "supervisor"),

    # ── RiskField — partitions de l'unité par NŒUDS croissants ────────────
    # tau : tk0<tk1<tk2<tk3<tk4<tk5  (4 trapèzes : crit/close/med/far)
    # d   : dk0<dk1<dk2<dk3          (3 trapèzes : near/med/far)
    # L'ordre strict est garanti par Genome.repair() (chain()), donc le GA ne
    # peut JAMAIS produire de zone morte ni de trapèze invalide.
    ParamSpec("alpha_aggregation", 0.60, 0.0, 1.0, "risk_field"),
    ParamSpec("tau_k0", 0.40, 0.10,  2.0, "risk_field"),
    ParamSpec("tau_k1", 0.90, 0.20,  3.0, "risk_field"),
    ParamSpec("tau_k2", 1.60, 0.30,  5.0, "risk_field"),
    ParamSpec("tau_k3", 2.60, 0.50,  8.0, "risk_field"),
    ParamSpec("tau_k4", 4.00, 1.00, 12.0, "risk_field"),
    ParamSpec("tau_k5", 6.00, 2.00, 25.0, "risk_field"),
    ParamSpec("d_k0",  60.0, 10.0, 200.0, "risk_field"),
    ParamSpec("d_k1", 110.0, 20.0, 350.0, "risk_field"),
    ParamSpec("d_k2", 180.0, 40.0, 500.0, "risk_field"),
    ParamSpec("d_k3", 260.0, 60.0, 800.0, "risk_field"),
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
    ParamSpec("STICK_DEG",     40.0,   15.0,   60.0, "targeting"),
    ParamSpec("STICK_GAIN",    0.8,     0.0,    1.5, "targeting"),
    ParamSpec("SWITCH_RATIO",  1.30,    1.0,    3.0, "targeting"),
    ParamSpec("TRACK_TOL_PX",  28.0,   10.0,   80.0, "targeting"),

    # ── Sacrifice ─────────────────────────────────────────────────────────
    ParamSpec("MINE_MIN_CATCH", 3.0, 1.0,  8.0, "targeting", is_int=True),
    ParamSpec("MINE_MAX_HOLD", 15.0, 3.0, 40.0, "targeting", is_int=True),
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

        gax = self.genes
        g = lambda n: float(gax[_IDX[n]])
        def s(n, v): gax[_IDX[n]] = float(v)

        def ordpair(a, b, gap):
            va, vb = g(a), g(b)
            if va >= vb - gap:
                mid = 0.5 * (va + vb)
                s(a, mid - gap / 2.0)
                s(b, mid + gap / 2.0)

        def chain(names, gap):
            """Force names to be strictly increasing (push upward only)."""
            prev = g(names[0])
            for n in names[1:]:
                v = max(g(n), prev + gap)
                s(n, v)
                prev = v

        ordpair("R_lo", "R_hi", 0.05)                       # hystérésis correcte

        # Partition de l'unité : nœuds strictement croissants ⇒ pas de zone
        # morte ni de trapèze invalide possible, quel que soit le génome.
        chain(["tau_k0", "tau_k1", "tau_k2", "tau_k3", "tau_k4", "tau_k5"], 0.05)
        chain(["d_k0", "d_k1", "d_k2", "d_k3"], 8.0)

        # Consequents monotones sur le treillis (tau × dist × size).
        # risk décroît avec : urgence (C>M>F), proximité (N>M>F), taille (L>M>S).
        # ⇒ surface de risque physique, espace de recherche fortement réduit.
        def ci(t, di, sz): return _IDX[t + di + sz]
        for _ in range(3):                       # quelques passes ⇒ convergence
            for sz in "LMS":
                for di in "NMF":                 # urgence : C ≥ M ≥ F
                    for a, b in zip("CMF", "MF"):
                        gax[ci(b, di, sz)] = min(gax[ci(b, di, sz)], gax[ci(a, di, sz)])
                for t in "CMF":                  # proximité : N ≥ M ≥ F
                    for a, b in zip("NMF", "MF"):
                        gax[ci(t, b, sz)] = min(gax[ci(t, b, sz)], gax[ci(t, a, sz)])
            for di in "NMF":                     # taille : L ≥ M ≥ S
                for t in "CMF":
                    for a, b in zip("LMS", "MS"):
                        gax[ci(t, di, b)] = min(gax[ci(t, di, b)], gax[ci(t, di, a)])

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
        import risk_field as rf
        rf.ALPHA_AGGREGATION = d.get("alpha_aggregation", rf.ALPHA_AGGREGATION)

        # Rebuild the partitions of unity from SORTED knots (same scheme as the
        # risk_field defaults). sorted() is a safety net; Genome.repair() has
        # already ordered them. .get() keeps legacy param files runnable: a JSON
        # missing the knot keys simply falls back to the module defaults.
        tk = sorted(d.get(f"tau_k{i}", v)
                    for i, v in enumerate((0.4, 0.9, 1.6, 2.6, 4.0, 6.0)))
        rf.TAU_CRITICAL = (-1.0, -1.0, tk[0], tk[1])
        rf.TAU_CLOSE    = (tk[0], tk[1], tk[2], tk[3])
        rf.TAU_MEDIUM   = (tk[2], tk[3], tk[4], tk[5])
        rf.TAU_FAR      = (tk[4], tk[5], 99.0, 99.0)

        dk = sorted(d.get(f"d_k{i}", v)
                    for i, v in enumerate((60.0, 110.0, 180.0, 260.0)))
        rf.D_NEAR   = (-1.0, -1.0, dk[0], dk[1])
        rf.D_MEDIUM = (dk[0], dk[1], dk[2], dk[3])
        rf.D_FAR    = (dk[2], dk[3], 9999.0, 9999.0)
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
            setattr(rf, name, d.get(name, getattr(rf, name)))
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
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Worker — init PARESSEUSE et idempotente (loky réutilise les process)
# ---------------------------------------------------------------------------

_W: dict = {}    # cache par process : modules + solveurs CasADi pré-construits


def _ensure_worker(map_sizes: Tuple[Tuple[int, int], ...]) -> None:
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


def _make_controller(map_size: Tuple[int, int]):
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
               map_sizes: Tuple[Tuple[int, int], ...]) -> Tuple[int, dict]:
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
        ctrl.supervisor.TAU_EMERGENCY = d.get("TAU_EMERGENCY", ctrl.supervisor.TAU_EMERGENCY)

        score, _ = _W["KesslerGame"](settings=settings).run(
            scenario=_W["Scenario"](**scenario_cfg), controllers=[ctrl]
        )
        team = score.teams[0]
        return gid, {
            "hit":    team.asteroids_hit,
            "deaths": team.deaths,
            "lives":  max(0, 3 - team.deaths),
            "acc":    min(1.0, team.accuracy),
            "error":  None,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return gid, {"hit": 0, "deaths": 3, "lives": 0, "acc": 0.0,
                     "error": str(exc)}


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

W_HIT      =  3.0
W_DEATH    = -500.0
W_SURVIVAL =  40.0
W_ACCURACY =  2.0


def compute_fitness(scenario_results: List[dict]) -> float:
    """Moyenne PAR SCÉNARIO d'une fitness à gate de survie (échelle stable
    quel que soit le nombre de scénarios → screening comparable)."""
    if not scenario_results:
        return -9999.0
    fits = []
    for r in scenario_results:
        if r["error"]:
            fits.append(-9999.0)
            continue
        surv     = r["lives"] / 3.0
        hit_mult = 0.3 + 0.7 * surv
        fits.append(W_HIT      * r["hit"] * hit_mult
                  + W_DEATH    * r["deaths"]
                  + W_SURVIVAL * r["lives"]
                  + W_ACCURACY * r["acc"] * 100.0)
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

    def _map_sizes(self) -> Tuple[Tuple[int, int], ...]:
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
                "map_size":              (MAP_W, MAP_H),
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
    p.add_argument("--gen",       type=int,   default=50)
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