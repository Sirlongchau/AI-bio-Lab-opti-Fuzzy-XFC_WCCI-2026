This repo is for the XFC and WCCI 2026.

# Overview of the project
We use a fuzzy risk assessment layer (Risk_field.py) that compute risk for each asteroid and then aggregate in a global risk. In order to do so, we base the math on toric distances and Time To Collision (TTC) in toric_utils.py.

The modal_supervisor.py uses the risk field and respawn status to select the control mode between fuzzy (nominal) where we are in a safe position and move little to none and shoot asteroids to get score, optimal control danger avoidance when danger is high, optimal path to safest reachable area and opportunistic fire, if optimal control fails to find a path, kamikaze mode, fire at everything and put a mine, finally respawn, no shooting and 3 seconde optimal rollout t find safest place.

The fuzzy mode is in the fuzzy_hybrid_controller.py and is under Haidar supervision.

The optimal control and modal_supervisor is under Hugo supervision.

The risk field is under Dang and Haidar collaboration.

The debug_tools is under Dang supervision.

The targeting and target selector is globally supervised.

# Running of the code
You need python 3.10 and pip install the requirements.
Then "python -m scenario_test"
