from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModalSupervisor:
    def __init__(self):
        self.controllers = {
            "fuzzy": FuzzyController(),
            "mpc": MPCDirector(),
            "evade": EvadeController(),
        }

    def select_mode(self, ship_state, risks, target) -> str:
        """
        Replace this with your real logic.
        """
        min_risk = min(r.risk for r in risks) if risks else 0.0

        if min_risk > 0.8:
            return "evade"
        elif target is not None:
            return "mpc"
        else:
            return "fuzzy"

    def compute(self, ship_state, game_state, risks, target, context=None) -> ControlOutput:
        context = context or {}

        mode = self.select_mode(ship_state, risks, target)
        controller = self.controllers[mode]

        return controller.compute(ship_state, game_state, context)