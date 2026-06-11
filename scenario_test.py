"""
scenario_test.py — visual run (Tkinter window).
One ship, one controller, 30 s, 3 lives, unlimited ammo.
Headless benchmarking: use `python benchmark.py` instead.
"""
import time
from kesslergame import Scenario, KesslerGame, GraphicsType
from Fuzzy_MPC_Controller import Controller

my_test_scenario = Scenario(
    name="Test Scenario",
    num_asteroids=10,
    ship_states=[{"position": (500, 400), "angle": 90, "lives": 3, "team": 1, "mines_remaining": 3}],
    map_size=(1000, 800),
    time_limit=30.0,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False,
)

game_settings = {
    "perf_tracker": True,
    "graphics_type": GraphicsType.Tkinter,   # set to GraphicsType.NoGraphics for headless
    "realtime_multiplier": 1,
    "frequency": 30,
}

game = KesslerGame(settings=game_settings)

pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[Controller()])

print("Scenario eval time: " + str(time.perf_counter() - pre))
print(score.stop_reason)
print("Asteroids hit: " + str([t.asteroids_hit for t in score.teams]))
print("Deaths: " + str([t.deaths for t in score.teams]))
print("Accuracy: " + str([t.accuracy for t in score.teams]))
print("Mean eval time (ms): " + str([t.mean_eval_time for t in score.teams]))
