Welcome in this project aiming to create the best controller for the XFC/WCCI 2026 competitions based on the kessler game made by Thales Avionics North America.

This project cover a multimodal controller through an escalation logic based on a fuzzy risk assessment layer. 
As such we have an intelligent system switching between modes to oscillate between a Bang Bang control (optimal control scheme for linear systems under state constraints) for fast and precise targetting and destruction of targets, and a Full NLP (Non Linear Problem) optimal control solution with state and path constraints through the fuzzy risk as a weight (Dynamic programming).

In order to use this controller you will have to navigate the different files present here:
  - scenario_test.py
    This file cover in it's initialisation, the methods necessary to enforce the trained parameters found in the "best_params.json" data file.
    Here you can also select which controller you wish to use, namely the controller with or without debug, the no debug file is sensely named "Fuzzy_MPC_Controller_no_debug.py"
  - The debug files are three partites:
      - "debug_tools.py" is the main logger and visualisation method
      - "angular_profile.py" is a partially legacy visual that aims to detect feasible escape heading. This is mostly unusable in a dense environment.
  - The controller itself possess multiple necessary dependency:
      - "risk_field.py", Allow the system to evaluate a fuzzy risk for each individual asteroid and aggregate it as a global risk level at the ship position.
      - "ga_optimizer.py", is the genetic algorithm used for training and posses the method to apply the json parameters to the individual modules.
      - "supervisor.py", is the switching logic between the 3 modes, "aggressive", "active avoidance" and "sacrificial" mode based on the risk assessment and ship state.
      - "targeting_system.py" is the bang bang control for aggressive catch and destroy logic. It also embed the sacrificial mode as a variation of the targetting with a mine logic added.
      - "toric_utils.py", this is the actual core of the geometry for the controllers to understand warping as a toric clippic through the boundary layer.
      - "Casadi_mpc.py", is the NLP Dynamic programming avoidance protocal using the symbolic toolbox CASADI to allow a full second rollout path planning to reach a safe space.
      - "parasitic_fire.py", this is an add on to the CASADI modelling to shoot oppoportunistically in order to gain score even while actively fleeing and also to break a possible infinite loop of high risk -> avoid don't shoot -> don't reduce risk -> high risk still
  - The requirements for the environnement are defined in the "requirement.txt" file.
