# -*- coding: utf-8 -*-
"""
run_scenario.py
===============
Visualisation d'une partie avec les meilleurs paramètres GA.
Usage : python run_scenario.py [--params best_params.json]
"""

import time
import json
import argparse
from pathlib import Path

# ── 1. Charger et appliquer les paramètres GA avant tout import contrôleur ──
from ga_optimizer import GeneticOptimizer, Genome, apply_genome_to_modules

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="best_params.json")
args = parser.parse_args()

BEST_PARAMS_PATH = args.params

if Path(BEST_PARAMS_PATH).exists():
    with open(BEST_PARAMS_PATH) as f:
        _d = json.load(f)
#     print(f"✓ Best params loaded ({len(_d)} parameters)")
#     print(f"  Supervisor  R_lo={_d.get('R_lo', 0.50):.3f}  R_hi={_d.get('R_hi', 0.60):.3f}")
#     print(f"  Targeting   FIRE_CONE={_d.get('FIRE_CONE_DEG', 4.0):.1f}°  "
#           f"W_RISK={_d.get('W_RISK', 2.0):.2f}  W_SMALL={_d.get('W_SMALL', 0.8):.2f}  "
#           f"SWITCH_RATIO={_d.get('SWITCH_RATIO', 1.25):.2f}")
#     print(f"  Repulsion   thrust={_d.get('REPULSION_THRUST', 80.0):.1f}  "
#           f"TTC_max={_d.get('REPULSION_TTC_MAX', 2.0):.2f}s")
# else:
#     print("⚠ No best_params.json found — using default hyperparameters")
#     _d = {}

# ── 2. Import contrôleurs APRÈS le patch ────────────────────────────────────
from kesslergame import Scenario, KesslerGame, GraphicsType
from Fuzzy_MPC_Controller import Controller
from graphics_both import GraphicsBoth

# ── 3. Instancier le contrôleur et patcher l'instance ───────────────────────
ctrl = Controller()

if _d:
    apply_genome_to_modules(_d)

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