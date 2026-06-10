"""
optimize.py — parameter optimization for the fuzzy controller.

FITNESS = SCORE = mean asteroids destroyed over a set of scenarios.

`fitness(vector)` is framework-agnostic: it takes a list of floats (a genome)
ordered as params.FREE, and returns a single float to MAXIMIZE. Drop it into
EasyGA, DEAP, CMA-ES, Optuna, etc. A small built-in GA is included so you can
run end-to-end immediately:  `python optimize.py`

Seeds are split into train / validation / test so you don't overfit the search
to the exact training scenarios (the competition's scenarios are unknown).
Re-tune `*_SEEDS`, `POP`, `GENS` for your compute budget.
"""
from __future__ import annotations
import random
import statistics

from kesslergame import Scenario, TrainerEnvironment, GraphicsType
import params
from params import ControllerParams, apply, from_vector, to_vector, clamp_vector, FREE, BOUNDS

# --- scenario suites: (seed, num_asteroids). Disjoint sets. ---
TRAIN_SEEDS = [(1, 6), (2, 8), (3, 10), (4, 12), (5, 7)]
VAL_SEEDS   = [(6, 9), (7, 11)]
TEST_SEEDS  = [(8, 8), (9, 10), (10, 12)]

MAP_SIZE = (1000, 800)
TIME_LIMIT = 30.0
SHIP = [{"position": (500, 400), "angle": 90, "lives": 3, "team": 1, "mines_remaining": 3}]


def _mean_score(seeds) -> float:
    """Mean asteroids destroyed over `seeds`. Controller is built fresh per run
    AFTER params are applied, so the active parameter vector is in effect."""
    env = TrainerEnvironment(settings={"graphics_type": GraphicsType.NoGraphics, "frequency": 30})
    from Fuzzy_MPC_Controller import Controller
    hits = []
    for seed, n in seeds:
        sc = Scenario(name=f"s{seed}", num_asteroids=n, ship_states=SHIP, seed=seed,
                      map_size=MAP_SIZE, time_limit=TIME_LIMIT,
                      ammo_limit_multiplier=0, stop_if_no_ammo=False)
        score, _ = env.run(scenario=sc, controllers=[Controller()])
        hits.append(score.teams[0].asteroids_hit)
    return statistics.mean(hits)


def fitness(vector, seeds=TRAIN_SEEDS) -> float:
    """The objective to MAXIMIZE: mean asteroids destroyed."""
    apply(from_vector(clamp_vector(vector)))   # MUST apply before Controller() is built
    return _mean_score(seeds)


# ----------------------------- built-in GA --------------------------------
def _random_genome():
    return [random.uniform(*BOUNDS[k]) for k in FREE]


def _mutate(g, rate=0.25):
    out = []
    for k, v in zip(FREE, g):
        lo, hi = BOUNDS[k]
        if random.random() < rate:
            v = v + random.gauss(0.0, 0.15 * (hi - lo))
        out.append(min(hi, max(lo, v)))
    return out


def _crossover(a, b):
    return [random.choice(pair) for pair in zip(a, b)]


def evolve(pop=20, gens=15, elite=3, seed=0):
    """Simple elitist GA. Seeds the population with the current hand-tuned vector
    so the search starts from a known-good point and can only improve."""
    random.seed(seed)
    population = [to_vector(ControllerParams())] + [_random_genome() for _ in range(pop - 1)]
    best_overall, best_val = None, -1.0
    for gen in range(gens):
        scored = sorted(((fitness(g), g) for g in population), key=lambda t: -t[0])
        train_best, best_g = scored[0]
        val = fitness(best_g, VAL_SEEDS)                 # select on validation
        print(f"gen {gen:2d}  train_best={train_best:6.1f}  val={val:6.1f}")
        if val > best_val:
            best_val, best_overall = val, best_g
        elites = [g for _, g in scored[:elite]]
        population = elites + [
            _mutate(_crossover(random.choice(elites), random.choice(elites)))
            for _ in range(pop - elite)
        ]
    return best_overall


if __name__ == "__main__":
    default_vec = to_vector(ControllerParams())
    print("default  train=%.1f  val=%.1f  test=%.1f"
          % (fitness(default_vec), fitness(default_vec, VAL_SEEDS), fitness(default_vec, TEST_SEEDS)))
    # Small demo run. Scale POP/GENS up for a real search (and parallelize fitness).
    best = evolve(pop=8, gens=4)
    print("\nbest     train=%.1f  val=%.1f  test=%.1f"
          % (fitness(best), fitness(best, VAL_SEEDS), fitness(best, TEST_SEEDS)))
    print("\nbest params:")
    bp = from_vector(best)
    for k in FREE:
        print(f"  {k:24s} {getattr(bp, k):.4f}")
