"""
targeting.py
============
Tactical decision layer.

Selects asteroid targets based on:
    - risk reduction potential
    - fragmentation cost
    - angular accessibility
"""

from __future__ import annotations

from typing import List

from risk_field import AsteroidRisk


# ---------------------------------------------------------------------------
# Target scoring
# ---------------------------------------------------------------------------

def target_value(ar: AsteroidRisk) -> float:
    """
    Higher is better target.
    """
    risk_reduction = ar.risk

    fragmentation_penalty = max(0.0, ar.delta_risk_if_destroyed)

    # small TTC bonus (urgency)
    urgency = 1.0 / (1.0 + ar.tau)

    return (
        1.2 * risk_reduction
        + 0.8 * urgency
        - 1.5 * fragmentation_penalty
    )


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def select_target(
    asteroid_risks: List[AsteroidRisk],
) -> AsteroidRisk | None:
    """
    Choose best tactical asteroid.
    """
    if not asteroid_risks:
        return None

    return max(asteroid_risks, key=target_value)