from risk_field import RiskField
from test_controller_fuzzy import FuzzyController
from Casadi_mpc import MPCController

class Supervisor:

    def __init__(self, ):
        
        self.controllers = {
            "fuzzy": FuzzyController(),
            "mpc": MPCController(),
            "sacrifice": FuzzyController(),
        }

        self.mode = "fuzzy"

        self.MPCfail = 0

        self.R_lo = 0.80
        self.R_hi = 0.90

    # ---------------------------------------------------------
    # Mode selection only
    # ---------------------------------------------------------

    def select_mode(self, ship_state, game_state):
        ship_pos=ship_state.position
        ship_vel = ship_state.velocity
        asteroids=game_state.asteroids
        self.risk_field  = RiskField(map_size=game_state.map_size)

        risks = self.risk_field.compute_all(ship_pos,ship_vel, asteroids)
        R_global = self.risk_field.aggregate(risks)

        if ship_state.respawn_time_left > 0:
            self.mode = "mpc"
            return

        if R_global <= self.R_lo:
            self.mode = "fuzzy"

        elif R_global >= self.R_hi:
            self.mode = "mpc"

        # else:
        # hysteresis → preserve current mode

    # ---------------------------------------------------------
    # Main routing function
    # ---------------------------------------------------------

    def compute(self, ship_state, game_state):
        
        self.select_mode(ship_state, game_state)

        controller = self.controllers[self.mode]

        output = controller.compute(ship_state, game_state)
        #print("Supervisor selected mode: " + self.mode)
        #print(f"Output: {output}")

        # -------------------------------------------------
        # MPC failure handling
        # -------------------------------------------------

        if ship_state.respawn_time_left > 0:
            self.MPCfail = 0

        if self.mode == "fuzzy":
            command=output

        if self.mode == "mpc" and not output is None:

            if not output.feasible:

                self.MPCfail += 1

                if self.MPCfail >= 10:

                    self.mode = "sacrifice"

                else:

                    self.mode = "fuzzy"

                controller = self.controllers[self.mode]

                output = controller.compute(ship_state, game_state)
                command = output

            else:
                self.MPCfail = 0
                command = (output.thrust, output.turn_rate, output.fire_decision, False)
        elif self.mode == "mpc" and output is None:
            command= tuple([0.0, 0.0, False, False]) # default output for death validation

        return command