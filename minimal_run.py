# -*- coding: utf-8 -*-
"""
run_scenario.py
===============
Test scenario using hardcoded default parameters (no best_params.json).
"""

import time

from kesslergame import Scenario, KesslerGame, GraphicsType
from Fuzzy_MPC_Controller import Controller
from graphics_both import GraphicsBoth

# ── Controller with default hardcoded parameters ─────────────────────────────
ctrl = Controller()

# ── Scenario ─────────────────────────────────────────────────────────────────
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

# ── Run ───────────────────────────────────────────────────────────────────────
game = KesslerGame(settings=game_settings)

pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[ctrl])

print(f'\nScenario eval time: {round(time.perf_counter() - pre, 3)}s')
print(f'Stop reason:        {score.stop_reason}')
print(f'Asteroids hit:      {[team.asteroids_hit for team in score.teams]}')
print(f'Deaths:             {[team.deaths for team in score.teams]}')
print(f'Accuracy:           {[round(team.accuracy, 3) for team in score.teams]}')
print(f'Mean eval time:     {[round(team.mean_eval_time * 1000, 2) for team in score.teams]} ms')