# XFC Kessler — Fuzzy/MPC Controller (tuned for asteroids-destroyed)

Objective assumed: **maximize asteroids destroyed** in a 30 s run, 3 lives, unlimited
ammo, difficulty ramping over time. No game-engine code was modified.

## Measured results (kesslergame 2.4.0, headless, 8 seeds, 30 s, 3 lives)

| | Mean hits | Mean deaths | Accuracy | Eval time / frame |
|---|---|---|---|---|
| Original code | 21.1 | 2.9 | 0.00–0.56 | ~19.0 ms |
| **This build** | **51.6** | **2.1** | 0.37–0.82 | **~5.9 ms** |

**+144% asteroids destroyed, fewer deaths, 3.2× faster per frame.**

## How to run

```bash
pip install -r requirements.txt          # casadi is optional (MPC fallback only)
# system tkinter is needed only for the visual window: e.g. apt-get install python3-tk

python scenario_test.py                   # visual Tkinter run (one ship, 30 s)
python benchmark.py                       # headless: 8 seeded scenarios, prints hits/deaths/accuracy
```

`Controller.explain()` returns the full per-frame decision trace
(`dominant_rule`, `rule_strengths`, target-score terms, corridor, emergency, fire) —
wire it into your XAI tuner / ghost-overlay.

## What changed (and why)

The biggest gains were bug fixes, not new features:

1. **Aim convention (the dominant bug).** Kessler fires bullets along `(cos θ, +sin θ)`
   and updates `heading += turn_rate·dt`. The code used `atan2(−dy, dx)` (a vertical
   mirror) *and* `turn = +k·(heading − aim)` (positive feedback → divergent). The ship
   was effectively spin-and-spraying. Fixed: `toric_bearing` uses `+dy`; the turn law
   is `+k·(aim − heading)` (`se_*` args flipped in `_build_inputs`).
2. **Stopped charging targets.** Low-danger rules and the rule-coverage fallback used to
   thrust 120–260 *forward* — straight into whatever the ship was aiming at. Now they
   hold position, rotate to aim, and fire.
3. **Fire while dodging + wider cone.** The fire decision no longer suppresses on
   `emergency`, and the alignment cone is 12° (was 6°). With free ammo, clearing
   incoming rocks is the best defense — this single change was the largest score jump
   and *also* lowered deaths.
4. **Proximity-evade** rule (`E1`) on raw nearest-distance, catching tangential/slow
   threats the radial-TTC model misses.
5. **Fragmentation neutralized** — splitting a big rock makes *more* targets, so for a
   kill objective destruction is always net-positive (no large-asteroid penalty).
6. **`lives_remaining` key fix** (was reading a nonexistent `lives` key, so life-aware
   weighting never engaged).
7. **Performance:** `angular_profile.build` / `repulse_direction` vectorized with numpy
   (19 ms → 5.9 ms/frame, behavior-identical).
8. **Removed dead modules** (`targeting.py`, `angular_field.py`, `weapons.py`,
   `vectorized_risk.py`, `viability_async.py` — all unimported).

## Tuning knobs (file : constant → direction)

- `target_selector.py : FIRE_ANGLE_THRESHOLD` (12°) — higher = more shots, free ammo.
- `target_selector.py : USE_INTERCEPT_LEAD` (False) — flip True for very fast / long-
  range fields. (Fixed ⅓-s lead `PREDICT_DT` benchmarked higher here; see Caveats.)
- `test_controller_fuzzy.py : PROX_VERYCLOSE` — proximity-evade trigger window (px).
- `test_controller_fuzzy.py : A1–A4 / B1 thrust` — forward speed when safe; kept low to
  avoid ramming. Raise cautiously to seek clusters.
- `risk_field.py : R_*/TAU_* breakpoints` and `supervisor.py : R_extreme/tau_extreme`
  (MPC escalation gate).

## Caveats — verify before the competition

- **Engine version.** Validated on **kesslergame 2.4.0**, where `ship_state` /
  `game_state` are objects (attribute access). `risk_field.py` and `target_selector.py`
  are already dict-or-object safe; if the competition uses a dict-based build, apply the
  same `_get_attr` helper to the core fields in `supervisor.py`, `test_controller_fuzzy.py`,
  and `Casadi_mpc.py`.
- **Aim sign** assumes facing = `(cos, +sin)`; confirmed on 2.4.0. Re-verify the ship
  converges onto a static target if you change engines.
- **~2 deaths/run are structural** — a station-keeping ship that thrusts only along its
  heading can't always out-run a fast sneak-up. Bigger survival gains need a real evasion
  redesign (perpendicular escape velocity) or the CasADi-MPC path (install `casadi`).
- **Experiments that regressed this objective and were reverted** (re-test if scoring
  ever weights survival): full intercept lead, real-px asteroid radius, urgency
  down-weighting, earlier/wider evasion, reverse-thrust evasion. For kill-max, a
  deliberately aggressive / under-sensitive risk model beats a "correct" cautious one.
- Benchmark numbers are scenario-dependent; the competition's scenarios will differ.

---

# Parameter optimization (`params.py`, `optimize.py`)

All tunable knobs are exposed as a single vector for your team to optimize.
**Fitness = score = mean asteroids destroyed.**

```bash
python optimize.py        # built-in GA demo: prints default vs evolved train/val/test score
```

- **`params.py`** — `ControllerParams` dataclass (33 knobs: rule gains, thrust
  setpoints, MF thresholds, fire cone, lead, urgency split, risk/corridor weights,
  proximity window), each with search `BOUNDS`. `apply(p)` writes them into the
  controller modules. Defaults equal the current tuned values, verified to
  reproduce the benchmark exactly (TRAIN=47.0, VAL=53.5).
- **`optimize.py`** — `fitness(vector) -> mean asteroids destroyed`, framework-
  agnostic (a list of floats ordered as `params.FREE`). Train/validation/test seed
  split so you don't overfit. Includes a self-contained elitist GA seeded with the
  current hand-tuned vector (so it can only improve).

**Plugging into EasyGA / DEAP / Optuna / CMA-ES:** point your optimizer at
`optimize.fitness` and `params.BOUNDS`. Example (EasyGA):

```python
import easyga, optimize
from params import FREE, BOUNDS
ga = easyga.GA()
ga.gene_impl   = lambda: None  # use chromosome_impl below instead
ga.chromosome_length = len(FREE)
ga.chromosome_impl = lambda: [__import__('random').uniform(*BOUNDS[k]) for k in FREE]
ga.fitness_function_impl = lambda chrom: optimize.fitness([g.value for g in chrom.gene_list])
ga.evolve()
```

**Two rules that matter:**
1. **Always call `apply(params)` before building a fresh `Controller()`** — the
   sub-objects read the module globals at construction time. `fitness()` already
   does this; preserve the order in any custom harness.
2. **Parallelize across processes, not threads** — `apply()` mutates module globals,
   which are per-process. Evaluate one individual per worker process.

**Tuning the search to your compute:** each fitness call runs `len(seeds)` 30 s games
(~1–2 s each headless). Pop 50 × 30 gens × 5 train seeds ≈ 7,500 games ≈ a couple of
hours single-process — so use process parallelism or shorter training scenarios, and
trim `params.FREE` to the knobs you care about most. MF breakpoint tuples (e.g.
`R_*`, `TAU_*`) are left fixed by default; add them to the search with ordering
constraints (a≤b≤c≤d) if you want deeper genetic-fuzzy tuning.
