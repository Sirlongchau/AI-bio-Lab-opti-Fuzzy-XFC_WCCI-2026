# runtime_controller.py

from supervisor import ModalSupervisor
from viability_async import AsyncViabilityWorker, ViabilityRequest, ViabilityKernel
from mpc import DirectionalMPC
from weapons import WeaponArbiter
from vectorized_risk import VectorizedRiskEngine

from toric_utils import toric_delta


class RuntimeController:
    def __init__(self, map_size):

        # Core subsystems
        self.risk = VectorizedRiskEngine(map_size)
        self.mpc = DirectionalMPC(map_size)
        self.supervisor = ModalSupervisor()
        self.weapons = WeaponArbiter()

        # Async safety layer (F8)
        self.viability_worker = AsyncViabilityWorker(
            ViabilityKernel()
        )

        self.map_size = map_size

        # persistent state
        self.last_interrupt = False

    # ------------------------------------------------------------
    # Main entry point (called every frame by game engine)
    # ------------------------------------------------------------

    def step(self, game_state):

        ship = game_state.ship
        asteroids = game_state.asteroids

        # --------------------------------------------------------
        # F1 + F9: perception + vector risk kernel
        # --------------------------------------------------------

        risk_out = self.risk.compute(
            ship.position,
            ship.velocity,
            asteroids
        )

        R_global = self.risk.aggregate(risk_out["risk"])
        tau_min = self.risk.min_ttc(risk_out["ttc"])

        corridor_exists = True  # placeholder (F3 integration assumed upstream)

        # --------------------------------------------------------
        # F6: interrupt-driven supervisor
        # --------------------------------------------------------

        mode, interrupt, flags = self.supervisor.step(
            tau_min=tau_min,
            corridor_exists=corridor_exists,
            r_global=R_global,
            r_dot=game_state.r_dot if hasattr(game_state, "r_dot") else 0.0
        )

        # --------------------------------------------------------
        # F8: viability worker (ONLY on interrupt edge)
        # --------------------------------------------------------

        if interrupt and not self.last_interrupt:

            req = ViabilityRequest(
                ship_pos=ship.position,
                ship_vel=ship.velocity,
                asteroids=asteroids,
                map_size=self.map_size,
                horizon=10,
                dt=0.25
            )

            # candidate headings from MPC discretization
            self.viability_worker.trigger(req, headings=list(range(0, 360, 30)))

        viability = self.viability_worker.get_result()

        self.last_interrupt = interrupt

        # --------------------------------------------------------
        # F5: MPC decision
        # --------------------------------------------------------

        theta = self.mpc.solve(
            ship.position,
            ship.velocity,
            ship.speed,
            corridors=game_state.corridors,
            asteroids=asteroids,
            risk_field=self.risk,
            interrupt=interrupt
        )

        # --------------------------------------------------------
        # F7: weapons
        # --------------------------------------------------------

        fire, mine = self.weapons.decide(
            heading=ship.heading,
            target_heading=theta,
            delta_risk=getattr(game_state, "delta_risk", 0.0),
            mode=mode,
            interrupt=interrupt,
            corridor_exists=corridor_exists,
            density_high=R_global > 0.6
        )

        # --------------------------------------------------------
        # low-level control
        # --------------------------------------------------------

        u_T = game_state.speed_controller(R_global, tau_min)
        omega = game_state.heading_controller(theta, ship.heading)

        return u_T, omega, fire, mine