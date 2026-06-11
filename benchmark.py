"""Headless benchmark: asteroids destroyed in 30s, 3 lives, unlimited ammo."""
import time, statistics
from kesslergame import Scenario, TrainerEnvironment, GraphicsType
from Fuzzy_MPC_Controller import Controller

def make_scenarios():
    common = dict(map_size=(1000, 800), time_limit=30.0,
                  ammo_limit_multiplier=0, stop_if_no_ammo=False)
    ships = [{'position': (500, 400), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3}]
    out = []
    for seed, n in [(1,6),(2,8),(3,10),(4,12),(5,7),(6,9),(7,11),(8,8)]:
        try:
            sc = Scenario(name=f"s{seed}", num_asteroids=n, ship_states=ships, seed=seed, **common)
        except TypeError:
            sc = Scenario(name=f"s{seed}", num_asteroids=n, ship_states=ships, **common)
        out.append(sc)
    return out

def run(label):
    env = TrainerEnvironment(settings={'graphics_type': GraphicsType.NoGraphics, 'frequency': 30})
    hits, deaths, accs, evals = [], [], [], []
    for sc in make_scenarios():
        score, perf = env.run(scenario=sc, controllers=[Controller()])
        t = score.teams[0]
        hits.append(t.asteroids_hit); deaths.append(t.deaths)
        accs.append(round(t.accuracy, 3)); evals.append(round(t.mean_eval_time*1000, 3))
    print(f"\n=== {label} ===")
    print("hits per scenario:  ", hits, " mean=%.1f" % statistics.mean(hits))
    print("deaths per scenario:", deaths, " mean=%.1f" % statistics.mean(deaths))
    print("accuracy:           ", accs)
    print("mean eval ms:       ", evals)
    return statistics.mean(hits)

if __name__ == "__main__":
    t0 = time.perf_counter()
    run("BASELINE (current code)")
    print("wall time: %.1fs" % (time.perf_counter()-t0))
