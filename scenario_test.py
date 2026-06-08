# -*- coding: utf-8 -*-
import time
import json
from pathlib import Path

# ── 1. Patch module constants BEFORE any controller import ──────────────────
from ga_optimizer import GeneticOptimizer, Genome, apply_genome_to_supervisor

BEST_PARAMS_PATH = "best_params.json"

if Path(BEST_PARAMS_PATH).exists():
    GeneticOptimizer.load_and_apply(BEST_PARAMS_PATH)
    print(f"✓ Best params loaded from {BEST_PARAMS_PATH}")
else:
    print("⚠ No best_params.json found — using default hyperparameters")

# ── 2. Import controllers AFTER patching ────────────────────────────────────
from kesslergame import Scenario, KesslerGame, GraphicsType
from Fuzzy_MPC_Controller import Controller
from graphics_both import GraphicsBoth

# ── 3. Instantiate controllers ──────────────────────────────────────────────
ctrl = Controller()

if Path(BEST_PARAMS_PATH).exists():
    with open(BEST_PARAMS_PATH) as f:
        genome = Genome.from_dict(json.load(f))
    apply_genome_to_supervisor(ctrl.supervisor, genome)
    print(f"✓ Supervisor thresholds: R_lo={ctrl.supervisor.R_lo:.3f}  R_hi={ctrl.supervisor.R_hi:.3f}")

# ── 4. Scenario & game settings ─────────────────────────────────────────────
my_test_scenario = Scenario(
    name='Test Scenario',
    num_asteroids=10,
    ship_states=[
        {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
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

game = KesslerGame(settings=game_settings)

# ── 5. Run ───────────────────────────────────────────────────────────────────
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[ctrl])

print('Scenario eval time: ' + str(time.perf_counter() - pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: '        + str([team.deaths        for team in score.teams]))
print('Accuracy: '      + str([team.accuracy      for team in score.teams]))
print('Mean eval time: '+ str([team.mean_eval_time for team in score.teams]))
