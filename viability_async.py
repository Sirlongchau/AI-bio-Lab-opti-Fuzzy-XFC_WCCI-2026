"""
f8_viability_async.py
=====================
Asynchronous viability kernel evaluation layer.

Key design shift:
- Viability is NOT computed per frame
- It is computed ONLY when triggered by F6 interrupt edges
- Runs in a background worker (threaded model)

This removes viability cost from the real-time control loop.

Design intent:
- decouple safety verification from MPC execution
- allow batched rollout simulation
- support future GPU/vector backend substitution
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

from toric_utils import toric_delta, toric_distance


Vec2 = Tuple[float, float]


# ============================================================
# Data structures
# ============================================================

@dataclass
class ViabilityResult:
    viable: bool
    worst_tau: float
    min_clearance: float
    trajectory_samples: int


@dataclass
class ViabilityRequest:
    ship_pos: Vec2
    ship_vel: Vec2
    asteroids: list
    map_size: Vec2
    horizon: int
    dt: float


# ============================================================
# Core viability evaluator (batch simulation engine)
# ============================================================

class ViabilityKernel:
    def __init__(
        self,
        n_samples: int = 16,
        safety_tau: float = 1.0,
        safety_distance: float = 25.0,
    ):
        self.n_samples = n_samples
        self.safety_tau = safety_tau
        self.safety_distance = safety_distance

    # ------------------------------------------------------------
    # Single rollout simulation
    # ------------------------------------------------------------

    def simulate_rollout(
        self,
        req: ViabilityRequest,
        heading: float,
    ) -> ViabilityResult:

        x, y = req.ship_pos
        vx, vy = req.ship_vel

        worst_tau = float("inf")
        min_clearance = float("inf")

        for _ in range(req.horizon):

            # simple inertial propagation (can be replaced by MPC model)
            vx += 0.0
            vy += 0.0

            x += vx * req.dt
            y += vy * req.dt

            W, H = req.map_size
            x %= W
            y %= H

            ship_pos = (x, y)

            # evaluate safety against asteroids
            for a in req.asteroids:

                dx, dy = toric_delta(ship_pos, a.position, req.map_size)
                dist = (dx * dx + dy * dy) ** 0.5

                min_clearance = min(min_clearance, dist)

                # pseudo TTC proxy (fast, conservative)
                rel_speed = (vx - a.velocity[0], vy - a.velocity[1])
                closing = dx * rel_speed[0] + dy * rel_speed[1]

                if closing > 0:
                    tau = dist / (closing + 1e-6)
                    worst_tau = min(worst_tau, tau)

        viable = (
            worst_tau > self.safety_tau
            and min_clearance > self.safety_distance
        )

        return ViabilityResult(
            viable=viable,
            worst_tau=worst_tau,
            min_clearance=min_clearance,
            trajectory_samples=req.horizon,
        )

    # ------------------------------------------------------------
    # Batch evaluation (vectorization hook)
    # ------------------------------------------------------------

    def batch_evaluate(
        self,
        req: ViabilityRequest,
        headings: List[float],
    ) -> List[ViabilityResult]:

        results = []

        for h in headings:
            results.append(self.simulate_rollout(req, h))

        return results


# ============================================================
# Async controller (interrupt-driven worker)
# ============================================================

class AsyncViabilityWorker:
    """
    Background worker triggered ONLY by F6 interrupt edges.
    """

    def __init__(self, kernel: ViabilityKernel):
        self.kernel = kernel

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self.latest_result: Optional[ViabilityResult] = None
        self.busy = False

    # ------------------------------------------------------------
    # Internal worker loop
    # ------------------------------------------------------------

    def _run(self, req: ViabilityRequest, headings: List[float]):
        results = self.kernel.batch_evaluate(req, headings)

        # collapse batch into worst-case viability
        worst = False
        worst_tau = float("inf")
        min_clearance = float("inf")

        for r in results:
            worst |= not r.viable
            worst_tau = min(worst_tau, r.worst_tau)
            min_clearance = min(min_clearance, r.min_clearance)

        final = ViabilityResult(
            viable=not worst,
            worst_tau=worst_tau,
            min_clearance=min_clearance,
            trajectory_samples=len(headings),
        )

        with self._lock:
            self.latest_result = final
            self.busy = False

    # ------------------------------------------------------------
    # Trigger (called ONLY by F6 interrupt edge)
    # ------------------------------------------------------------

    def trigger(
        self,
        req: ViabilityRequest,
        headings: List[float],
    ):
        if self.busy:
            return  # drop redundant trigger (MCU-style overwrite protection)

        self.busy = True

        self._thread = threading.Thread(
            target=self._run,
            args=(req, headings),
            daemon=True,
        )

        self._thread.start()

    # ------------------------------------------------------------
    # Non-blocking read
    # ------------------------------------------------------------

    def get_result(self) -> Optional[ViabilityResult]:
        with self._lock:
            return self.latest_result