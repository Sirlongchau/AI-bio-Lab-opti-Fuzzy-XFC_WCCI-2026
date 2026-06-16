from risk_field import RiskField
from targeting_system import TargetingController, SacrificeController
from Casadi_mpc import MPCController
import math

# GA-tunable module globals. ga_optimizer.apply_genome_to_modules patches these
# before Controller/Supervisor construction in each worker.
MPC_TAU_TRIGGER  = 1.25
MPC_NEAR_TRIGGER = 135.0
MPC_RISK_TRIGGER = 0.45
MPC_FAIL_LIMIT   = 3


class Supervisor:
    """
    Trainable safety/offense supervisor.

    Modes:
      target    = aggressive turret/braking controller
      mpc       = planned escape + parasitic firing
      sacrifice = emergency turret + mine fallback

    Key change:
      MPC can be triggered by physical danger gates, not only R_global >= R_hi.
      Mine deployment is checked in target and MPC modes when TTC is medium and
      risk is high.
    """

    def __init__(self):
        self.controllers = {
            "target": TargetingController(),
            "mpc": MPCController(),
            "sacrifice": SacrificeController(),
        }

        self.mode = "target"
        self.MPCfail = 0

        self.R_lo = 0.80
        self.R_hi = 0.90

        self.mpc_tau_trigger = MPC_TAU_TRIGGER
        self.mpc_near_trigger = MPC_NEAR_TRIGGER
        self.mpc_risk_trigger = MPC_RISK_TRIGGER
        self.mpc_fail_limit = int(MPC_FAIL_LIMIT)

    def _risk_snapshot(self, ship_state, game_state):
        ship_pos = tuple(ship_state.position)
        ship_vel = tuple(ship_state.velocity)

        self.risk_field = RiskField(map_size=game_state.map_size)
        risks = self.risk_field.compute_all(ship_pos, ship_vel, game_state.asteroids)
        R_global = self.risk_field.aggregate(risks)
        tau_min = self.risk_field.min_tau(risks)
        nearest_surface = min((ar.d_surface for ar in risks), default=9999.0)
        top_risk = max((ar.risk for ar in risks), default=0.0)

        return risks, R_global, tau_min, nearest_surface, top_risk

    def _mine_decision(self, ship_state, game_state, risks):
        if getattr(ship_state, "respawn_time_left", 0.0) > 0:
            return False
        try:
            return self.controllers["sacrifice"]._decide_mine(
                ship_state, game_state, risks
            )
        except Exception:
            return False

    def select_mode(self, ship_state, game_state, risks=None, R_global=None,
                    tau_min=None, nearest_surface=None, top_risk=None):

        if risks is None:
            risks, R_global, tau_min, nearest_surface, top_risk = self._risk_snapshot(
                ship_state, game_state
            )

        if getattr(ship_state, "respawn_time_left", 0.0) > 0:
            self.mode = "mpc"
            return

        # Physical danger gate: this is what the GA will tune to stop late reaction.
        imminent_by_tau = (
            math.isfinite(tau_min)
            and tau_min <= self.mpc_tau_trigger
            and top_risk >= self.mpc_risk_trigger
        )
        imminent_by_distance = (
            nearest_surface <= self.mpc_near_trigger
            and top_risk >= self.mpc_risk_trigger
        )

        if imminent_by_tau or imminent_by_distance:
            self.mode = "mpc"
            return

        # Existing fuzzy-risk hysteresis.
        if R_global <= self.R_lo:
            self.mode = "target"
            return

        if R_global >= self.R_hi:
            self.mode = "mpc"
            return

        # Otherwise preserve current mode.

    def compute(self, ship_state, game_state):
        asteroid_risks, R_global, tau_min, nearest_surface, top_risk = self._risk_snapshot(
            ship_state, game_state
        )

        self.select_mode(
            ship_state,
            game_state,
            risks=asteroid_risks,
            R_global=R_global,
            tau_min=tau_min,
            nearest_surface=nearest_surface,
            top_risk=top_risk,
        )

        controller = self.controllers[self.mode]

        if self.mode == "mpc":
            output = controller.compute(ship_state, game_state)
            drop_mine = self._mine_decision(ship_state, game_state, asteroid_risks)

            if output is not None and output.feasible:
                self.MPCfail = 0
                return (output.thrust, output.turn_rate, output.fire_decision, drop_mine)

            self.MPCfail += 1

            if self.MPCfail >= self.mpc_fail_limit:
                self.mode = "sacrifice"
                thrust, turn_rate, fire, drop_mine2 = self.controllers["sacrifice"].compute(
                    ship_state, game_state, asteroid_risks
                )
                return (thrust, turn_rate, fire, drop_mine or drop_mine2)

            # Fast fallback to target if MPC briefly fails.
            thrust, turn_rate, fire = self.controllers["target"].compute(
                ship_state, game_state, asteroid_risks
            )
            return (thrust, turn_rate, fire, drop_mine)

        if self.mode == "sacrifice":
            thrust, turn_rate, fire, drop_mine = controller.compute(
                ship_state, game_state, asteroid_risks
            )
            return (thrust, turn_rate, fire, drop_mine)

        # Normal aggressive target mode.
        thrust, turn_rate, fire = controller.compute(
            ship_state, game_state, asteroid_risks
        )
        drop_mine = self._mine_decision(ship_state, game_state, asteroid_risks)
        return (thrust, turn_rate, fire, drop_mine)
