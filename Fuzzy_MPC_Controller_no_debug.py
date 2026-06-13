
from Casadi_mpc import MPCController
from risk_field import RiskField
#from test_controller_fuzzy import FuzzyController
#from SacrificeController import SacrificeController
from dataclasses import dataclass
from supervisor import Supervisor

class Controller:
    def __init__(self):
        #self.fuzzy = FuzzyController()
        self.mpc = MPCController()
        #self.sacrifice = FuzzyController() # placeholder for actual SacrificeController()
        self.supervisor = Supervisor()
        #self.debug=debug() # placeholder for actual debug tools like FrameDebugger, RiskHeatmap, etc.
    
    def actions(self, ship_state, game_state):
        #self.debug.update(ship_state, game_state) # update debug tools with current state
        output = self.supervisor.compute(ship_state, game_state) # get control output from supervisor
        return output
    
    @property
    def name(self) -> str:
        return "fuzzy_mpc_hybrid_controller"
    
    # @property
    # def custom_sprite_path(self) -> str:
    #     return "A400m_kessler" 