"""
f6_supervisor.py
=================
Event-driven modal supervisor.

Key design change:
- NO continuous viability computation
- ONLY edge-triggered interrupt events
- behaves like MCU interrupt controller

This is the main computational bottleneck fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SupervisorState:
    mode: str = "active"
    interrupt: bool = False
    escalation_level: int = 1
    last_interrupt: bool = False


class ModalSupervisor:
    def __init__(
        self,
        tau_interrupt: float = 1.2,
        rho_threshold: float = 0.25,
        gamma_r: float = 0.8,
        r_lo: float = 0.4,
        r_hi: float = 0.7,
    ):
        self.tau_interrupt = tau_interrupt
        self.rho_th = rho_threshold
        self.gamma_r = gamma_r
        self.r_lo = r_lo
        self.r_hi = r_hi

        self.state = SupervisorState()

    # ------------------------------------------------------------
    # Event detection (edge-triggered)
    # ------------------------------------------------------------

    def detect_interrupt(
        self,
        tau_min: float,
        corridor_exists: bool,
        r_global: float,
        r_dot: float,
    ) -> bool:

        return (
            tau_min < self.tau_interrupt
            or not corridor_exists
            or r_dot > self.gamma_r
        )

    # ------------------------------------------------------------
    # Escalation logic
    # ------------------------------------------------------------

    def update_escalation(self, r_global: float, corridor_exists: bool):
        if r_global < self.r_lo and corridor_exists:
            self.state.escalation_level = 1
        elif not corridor_exists or r_global >= self.r_lo:
            self.state.escalation_level = 2
        else:
            self.state.escalation_level = 3

    # ------------------------------------------------------------
    # Main step (interrupt-driven)
    # ------------------------------------------------------------

    def step(
        self,
        tau_min: float,
        corridor_exists: bool,
        r_global: float,
        r_dot: float,
    ) -> Tuple[str, bool, dict]:

        interrupt_now = self.detect_interrupt(
            tau_min, corridor_exists, r_global, r_dot
        )

        # edge-trigger behavior
        if interrupt_now and not self.state.last_interrupt:
            self.state.interrupt = True
        elif not interrupt_now:
            self.state.interrupt = False

        self.state.last_interrupt = interrupt_now

        # mode update
        self.state.mode = "respawn" if self.state.mode == "respawn" else "active"

        # escalation
        self.update_escalation(r_global, corridor_exists)

        flags = {
            "interrupt": self.state.interrupt,
            "level": self.state.escalation_level,
        }

        return self.state.mode, self.state.interrupt, flags