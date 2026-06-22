from risk_field import RiskField
from targeting_system import TargetingController, SacrificeController
from Casadi_mpc import MPCController


class Supervisor:
    """
    Mode arbiter for the multimodal controller.

    Two independent escalation signals decide when to evade (MPC):
      1. Crowd danger  : the fuzzy aggregate R_global with hysteresis
                         (R_lo / R_hi) — "ambient" threat level, good for
                         deciding when to stop shooting and reposition.
      2. Safety horizon: the soonest surface-TTC. A SINGLE imminent collision
                         forces MPC even when R_global is low — because with
                         alpha < R_hi the aggregate can never saturate on one
                         threat, so without this gate a lone incoming rock is
                         never evaded (it just gets taken on the chin).
    """

    def __init__(self):
        self.controllers = {
            "target":    TargetingController(),
            "mpc":       MPCController(),
            "sacrifice": SacrificeController(),
        }

        self.mode = "target"
        self.MPCfail = 0

        # Overwritten by the GA (best_params) / scenario_test at runtime.
        self.R_lo = 0.80
        self.R_hi = 0.90
        self.TAU_EMERGENCY = 1.20   # s — hard evade trigger (see class docstring)

        # Built lazily per frame so a map-size change is always honoured.
        self.risk_field = None

    # ---------------------------------------------------------
    # Mode selection (operates on a pre-computed risk list)
    # ---------------------------------------------------------

    def select_mode(self, ship_state, risks):
        # During respawn invulnerability: stay in MPC to reposition safely.
        if ship_state.respawn_time_left > 0:
            self.mode = "mpc"
            return

        R_global = self.risk_field.aggregate(risks)
        tau_min  = self.risk_field.min_tau(risks)

        # (1) Hard safety horizon — single imminent collision overrides all.
        if tau_min <= self.TAU_EMERGENCY:
            self.mode = "mpc"
            return

        # (2) Crowd danger with hysteresis.
        if R_global <= self.R_lo:
            self.mode = "target"
            return
        if R_global >= self.R_hi:
            self.mode = "mpc"
            return
        # else: hysteresis band → preserve current mode.

    # ---------------------------------------------------------
    # Main routing function
    # ---------------------------------------------------------

    def compute(self, ship_state, game_state):
        # Risk field is computed ONCE per frame here and reused for mode
        # selection and the turret/sacrifice controllers. (The MPC keeps its
        # own internal risk_field; unifying that would require changing its
        # signature and is left as a separate clean-up.)
        self.risk_field = RiskField(map_size=game_state.map_size)
        asteroid_risks = self.risk_field.compute_all(
            ship_state.position, ship_state.velocity, game_state.asteroids
        )

        self.select_mode(ship_state, asteroid_risks)
        controller = self.controllers[self.mode]

        if self.mode == "mpc":
            output = controller.compute(ship_state, game_state)
        else:
            thrust, turn_rate, fire = controller.compute(
                ship_state, game_state, asteroid_risks
            )
            command = (thrust, turn_rate, fire, False)

        # -------------------------------------------------
        # MPC failure handling
        # -------------------------------------------------
        if ship_state.respawn_time_left > 0:
            self.MPCfail = 0

        if self.mode == "mpc" and output is not None:
            if not output.feasible:
                self.MPCfail += 1
                if self.MPCfail >= 10:
                    self.mode = "sacrifice"
                    controller = self.controllers[self.mode]
                    thrust, turn_rate, fire, drop_mine = controller.compute(
                        ship_state, game_state, asteroid_risks
                    )
                    command = (thrust, turn_rate, fire, drop_mine)
                else:
                    self.mode = "target"
                    controller = self.controllers[self.mode]
                    thrust, turn_rate, fire = controller.compute(
                        ship_state, game_state, asteroid_risks
                    )
                    command = (thrust, turn_rate, fire, False)
            else:
                self.MPCfail = 0
                self.controllers["sacrifice"].reset()
                command = (output.thrust, output.turn_rate,
                           output.fire_decision, False)

        elif self.mode == "mpc" and output is None:
            self.mode = "sacrifice"
            controller = self.controllers[self.mode]
            thrust, turn_rate, fire, drop_mine = controller.compute(
                ship_state, game_state, asteroid_risks
            )
            command = (thrust, turn_rate, fire, drop_mine)

        return command