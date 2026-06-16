# -*- coding: utf-8 -*-
"""
scenario_test_visual_heatmap.py
===============================
Visual test runner with post-run explainability heatmaps.

Usage:
    python .\scenario_test_visual_heatmap.py --params .\best_params.json
"""

import os

# Must be set BEFORE importing Fuzzy_MPC_Controller.
os.environ["KESSLER_DEBUG"] = "1"

import time
import json
import argparse
from pathlib import Path

from ga_optimizer import apply_genome_to_modules

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="best_params.json")
parser.add_argument("--display-seconds", type=float, default=2.5)
args = parser.parse_args()

params_path = Path(args.params)
_d = {}

if params_path.exists():
    with open(params_path, "r", encoding="utf-8") as f:
        _d = json.load(f)

    print(f"Loaded params: {params_path} ({len(_d)} keys)")
    apply_genome_to_modules(_d)
else:
    print(f"WARNING: params file not found: {params_path}")
    print("Running with hardcoded/default controller parameters.")

from kesslergame import Scenario, KesslerGame, GraphicsType
from Fuzzy_MPC_Controller import Controller

ctrl = Controller()

# Supervisor values are instance attributes, so patch them explicitly.
if _d:
    if "R_lo" in _d:
        ctrl.supervisor.R_lo = _d["R_lo"]
    if "R_hi" in _d:
        ctrl.supervisor.R_hi = _d["R_hi"]

    if "MPC_TAU_TRIGGER" in _d and hasattr(ctrl.supervisor, "mpc_tau_trigger"):
        ctrl.supervisor.mpc_tau_trigger = _d["MPC_TAU_TRIGGER"]
    if "MPC_NEAR_TRIGGER" in _d and hasattr(ctrl.supervisor, "mpc_near_trigger"):
        ctrl.supervisor.mpc_near_trigger = _d["MPC_NEAR_TRIGGER"]
    if "MPC_RISK_TRIGGER" in _d and hasattr(ctrl.supervisor, "mpc_risk_trigger"):
        ctrl.supervisor.mpc_risk_trigger = _d["MPC_RISK_TRIGGER"]
    if "MPC_FAIL_LIMIT" in _d and hasattr(ctrl.supervisor, "mpc_fail_limit"):
        ctrl.supervisor.mpc_fail_limit = int(_d["MPC_FAIL_LIMIT"])

scenario = Scenario(
    name="Test Scenario Visual Heatmap",
    num_asteroids=10,
    ship_states=[
        {
            "position": (400, 400),
            "angle": 90,
            "lives": 3,
            "team": 1,
            "mines_remaining": 3,
        }
    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False,
)

settings = {
    "perf_tracker": True,
    "graphics_type": GraphicsType.Tkinter,
    "realtime_multiplier": 1,
    "graphics_obj": None,
    "frequency": 30,
}

game = KesslerGame(settings=settings)

pre = time.perf_counter()
score, perf_data = game.run(scenario=scenario, controllers=[ctrl])

print("")
print(f"Scenario eval time: {round(time.perf_counter() - pre, 3)}s")
print(f"Stop reason:        {score.stop_reason}")
print(f"Asteroids hit:      {[team.asteroids_hit for team in score.teams]}")
print(f"Deaths:             {[team.deaths for team in score.teams]}")
print(f"Accuracy:           {[round(team.accuracy, 3) for team in score.teams]}")
print(f"Mean eval time:     {[round(team.mean_eval_time * 1000, 2) for team in score.teams]} ms")

# Heatmap / explainability display after the run.
if hasattr(ctrl, "show_debug"):
    ctrl.show_debug(display_seconds=args.display_seconds)
else:
    print("[debug_tools] Controller has no show_debug() method.")
