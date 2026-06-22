# -*- coding: utf-8 -*-
"""
run_scenario.py
===============
Visualisation d'une partie avec les meilleurs paramètres GA.
Usage : python run_scenario.py [--params best_params.json]
"""

import os
import time
import json
import argparse
from pathlib import Path

# Interactive run: enable the controller's debug/heatmap pipeline. Must be set
# BEFORE importing the controller (the flag is read at import time).
os.environ.setdefault("KESSLER_DEBUG", "1")

# ── 1. Charger et appliquer les paramètres GA avant tout import contrôleur ──
from ga_optimizer import GeneticOptimizer, Genome, apply_genome_to_modules

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="best_params.json")
args = parser.parse_args()

BEST_PARAMS_PATH = args.params

_d = {}
if Path(BEST_PARAMS_PATH).exists():
    with open(BEST_PARAMS_PATH) as f:
        _d = json.load(f)

# ── 2. Import contrôleurs APRÈS le patch ────────────────────────────────────
from kesslergame import Scenario, KesslerGame, GraphicsType
from Fuzzy_MPC_Controller import Controller
#from Fuzzy_MPC_Controller_no_debug import Controller
from graphics_both import GraphicsBoth

# ── 3. Instancier le contrôleur et patcher l'instance ───────────────────────
ctrl = Controller()

if _d:
    apply_genome_to_modules(_d)
    # apply_genome_to_modules only patches risk_field / targeting_system module
    # globals — the supervisor thresholds are INSTANCE attributes and must be
    # set here, otherwise the interactive run silently uses the defaults
    # (R_lo=0.80 / R_hi=0.90) instead of the trained genome.
    sup = ctrl.supervisor
    if "R_lo" in _d:          sup.R_lo = _d["R_lo"]
    if "R_hi" in _d:          sup.R_hi = _d["R_hi"]
    if "TAU_EMERGENCY" in _d: sup.TAU_EMERGENCY = _d["TAU_EMERGENCY"]

# ── 4. Scénario ──────────────────────────────────────────────────────────────
my_test_scenario = Scenario(
    name='Test Scenario',
    num_asteroids=10,
    ship_states=[
        {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3},
    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False,
)

game_settings = {
    'perf_tracker':        True,
    'graphics_type':       GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj':        None,
    'frequency':           30,
}

# ── 5. Run ───────────────────────────────────────────────────────────────────
game = KesslerGame(settings=game_settings)

pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[ctrl])

print(f'\nScenario eval time: {round(time.perf_counter() - pre, 3)}s')
print(f'Stop reason:        {score.stop_reason}')
print(f'Asteroids hit:      {[team.asteroids_hit for team in score.teams]}')
print(f'Deaths:             {[team.deaths for team in score.teams]}')
print(f'Accuracy:           {[round(team.accuracy, 3) for team in score.teams]}')
print(f'Mean eval time:     {[round(team.mean_eval_time * 1000, 2) for team in score.teams]} ms')

# ── 6. Heatmap / explainability en fin de run ────────────────────────────────
# Called here (not via atexit) so the interactive display has a live event loop.
if hasattr(ctrl, "show_debug"):
    ctrl.show_debug(display_seconds=2.0)