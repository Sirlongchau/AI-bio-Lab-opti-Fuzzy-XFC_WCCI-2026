
from Casadi_mpc import MPCController
from risk_field import RiskField
from test_controller_fuzzy import FuzzyController
#from SacrificeController import SacrificeController
from dataclasses import dataclass
from supervisor import Supervisor

class Controller:
    def __init__(self):
        self.fuzzy = FuzzyController()
        self.mpc = MPCController()
        self.sacrifice = FuzzyController() # placeholder for actual SacrificeController()
        self.supervisor = Supervisor()
        #self.debug=debug() # placeholder for actual debug tools like FrameDebugger, RiskHeatmap, etc.
    
    def actions(self, ship_state, game_state):
        #self.debug.update(ship_state, game_state) # update debug tools with current state
        output = self.supervisor.compute(ship_state, game_state) # get control output from supervisor
        __edge_output = output
        __edge_output = self._edge_guard_output(__edge_output, ship_state, game_state)
        return __edge_output
    

    def _edge_guard_output(self, output, ship_state, game_state):
        """
        Final fire guard. Movement, turning, and mines are unchanged.
        Only fire=True can be suppressed when the shot would leave the map
        before a direct non-wrapping intercept.
        """
        try:
            if output is None:
                return output

            if not isinstance(output, (tuple, list)):
                return output

            if len(output) < 3:
                return output

            if not bool(output[2]):
                return output

            from edge_fire_guard import edge_safe_fire

            if edge_safe_fire(ship_state, game_state):
                return output

            out = list(output)
            out[2] = False
            return tuple(out) if isinstance(output, tuple) else out

        except Exception:
            return output

    @property
    def name(self) -> str:
        return "fuzzy_mpc_hybrid_controller"
    
    # @property
    # def custom_sprite_path(self) -> str:
    #     return "A400m_kessler" 