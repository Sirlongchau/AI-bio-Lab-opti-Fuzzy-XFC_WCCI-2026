# XFC Safety-Mine Fuzzy-MPC Controller

Hybrid asteroid-control agent for the Kessler Game / XFC controller task.

This controller combines fuzzy risk reasoning, target selection, MPC fallback, and mine deployment. The current shipped configuration prioritizes survival, asteroid avoidance, useful firing, and mine use under pressure.

## Main Entry Point

Competition entry point:

    Fuzzy_MPC_Controller.py

Controller class:

    Controller

The controller returns the standard Kessler action tuple:

    (thrust, turn_rate, fire, drop_mine)

## Current Shipped Parameters

Active tuned parameter file:

    best_params.json

Copied from:

    best_params_safety_mine_v1.json

The safety-mine tuned version is the current best working candidate.

## Core Files

| File | Purpose |
|---|---|
| Fuzzy_MPC_Controller.py | Main competition controller entry point |
| supervisor.py | Selects between targeting, MPC, and sacrifice/mine logic |
| test_controller_fuzzy.py | Fuzzy avoidance and targeting controller |
| targeting_system.py | Aggressive targeting and sacrifice/mine behavior |
| target_selector.py | Predictive asteroid target selection |
| risk_field.py | Asteroid risk and time-to-collision reasoning |
| toric_utils.py | Toroidal geometry utilities |
| vectorized_risk.py | Faster risk calculations |
| angular_field.py | Angular danger representation |
| angular_profile.py | Safe corridor extraction |
| weapons.py | Weapon and mine arbitration helpers |
| viability_async.py | Viability checks |
| Casadi_mpc.py | MPC fallback controller |
| Parasitic_fire.py | MPC fire support |
| edge_fire_guard.py | Fire guard hook, currently permissive to preserve firing |
| debug_tools.py | Debug and visualization support |
| scenario_test.py | Standard local scenario runner |
| scenario_test_visual_heatmap.py | Visual run with heatmap/debug display |
| ga_optimizer.py | Genetic algorithm parameter tuning |
| requirements.txt | Python dependencies |

## Install

Install Python dependencies:

    pip install -r requirements.txt

If Kessler Game is not already installed in the environment, install it according to the course or competition setup.

## Run the Standard Test Scenario

Headless:

    python .\scenario_test.py --nogfx --params .\best_params.json

Visual:

    python .\scenario_test.py --params .\best_params.json

## Run the Visualizer With Heatmap Afterward

    python .\scenario_test_visual_heatmap.py --params .\best_params.json --display-seconds 3

## Expected Behavior

The controller should:

1. avoid obvious incoming asteroids,
2. keep moving instead of freezing,
3. shoot at useful moments,
4. drop mines under medium time-to-collision and high-risk pressure,
5. preserve behavior across all three lives,
6. avoid over-correcting around edge shots when firing is needed for survival.

## Training

The genetic optimizer is available through:

    python .\ga_optimizer.py

The currently shipped controller should not be retrained unless there is a repeated failure pattern.

Latest stable tuned file:

    best_params_safety_mine_v1.json

## Recommended Final Validation Before Submission

    python .\scenario_test.py --nogfx --params .\best_params.json
    python .\scenario_test.py --params .\best_params.json
    python .\scenario_test_visual_heatmap.py --params .\best_params.json --display-seconds 3

## Notes

This project went through several experimental patches. The current stable version intentionally restores normal firing behavior because overly strict edge-fire suppression caused the ship to miss key defensive shots and collide with asteroids.

Priority order for the shipped version:

    survive
    avoid asteroids
    shoot useful targets
    use mines under pressure
    avoid unnecessary overcorrection

## Author

Developed by Haidar Bin Hamid for the XFC / Kessler controller competition.
