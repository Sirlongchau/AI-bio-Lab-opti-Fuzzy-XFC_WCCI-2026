"""
f7_weapons.py
=============
Weapon arbitration layer.

Key rule:
- Weapons are NOT allowed to override interrupt safety state
- Supervisor has absolute priority
"""

from __future__ import annotations

from typing import Tuple


class WeaponArbiter:
    def __init__(self, fire_epsilon: float = 8.0):
        self.fire_eps = fire_epsilon

    def angle_diff(self, a: float, b: float) -> float:
        d = (a - b) % 360
        return d - 360 if d > 180 else d

    # ------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------

    def decide(
        self,
        heading: float,
        target_heading: float,
        delta_risk: float,
        mode: str,
        interrupt: bool,
        corridor_exists: bool,
        density_high: bool,
    ) -> Tuple[int, int]:

        fire = 0
        mine = 0

        # --------------------------------------------------------
        # HARD SAFETY LAYER (interrupt override)
        # --------------------------------------------------------
        if interrupt:
            return 0, 0

        # --------------------------------------------------------
        # FIRE LOGIC
        # --------------------------------------------------------
        aligned = abs(self.angle_diff(heading, target_heading)) < self.fire_eps

        if (
            aligned
            and delta_risk > 0
            and mode == "active"
        ):
            fire = 1

        # --------------------------------------------------------
        # MINE LOGIC
        # --------------------------------------------------------
        if (
            (not corridor_exists)
            and density_high
            and mode == "active"
        ):
            mine = 1

        return fire, mine