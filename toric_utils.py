"""
toric_utils.py
==============
Geometric primitives for a toroidal arena.

All pairwise computations (distance, TTC, bearing) must account for
wrap-around boundaries.  This module centralises that logic so every
other module can call it without reimplementing it.

Coordinate convention
---------------------
- Origin at top-left corner.
- x increases rightward, y increases downward (Kessler default).
- Angles are measured in degrees, counter-clockwise from the +x axis,
  following Python's math.atan2 convention.  The caller is responsible
  for converting to Kessler's heading convention when needed.
"""

from __future__ import annotations

import math
from typing import Tuple

# ---------------------------------------------------------------------------
# Type aliases (plain tuples, no dataclass overhead at 60 fps)
# ---------------------------------------------------------------------------
Vec2 = Tuple[float, float]   # (x, y)  position or velocity


# ---------------------------------------------------------------------------
# Core toroidal arithmetic
# ---------------------------------------------------------------------------

def toric_delta(a: Vec2, b: Vec2, map_size: Vec2) -> Vec2:
    """
    Return the shortest signed displacement vector from b to a on a
    toroidal map of dimensions map_size = (W, H).

    Uses the "round to nearest" formula so the result always lies in
    (-W/2, W/2] x (-H/2, H/2].

    Parameters
    ----------
    a, b      : positions (x, y)
    map_size  : (width, height) of the arena

    Returns
    -------
    (dx, dy)  : displacement a - b on the torus
    """
    W, H = map_size
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    # Correct each axis independently
    dx -= W * round(dx / W)
    dy -= H * round(dy / H)
    return dx, dy


def toric_distance(a: Vec2, b: Vec2, map_size: Vec2) -> float:
    """
    Euclidean distance between a and b on the torus.
    """
    dx, dy = toric_delta(a, b, map_size)
    return math.hypot(dx, dy)


def toric_bearing(from_pos: Vec2, to_pos: Vec2, map_size: Vec2) -> float:
    """
    Bearing (in degrees, [0, 360)) from from_pos toward to_pos on the torus.

    Matches Kessler heading convention exactly:
        0   = right (+x), 90  = up (-y in screen coords),
        180 = left  (-x), 270 = down (+y in screen coords)

    Kessler uses math.atan2(-dy, dx) internally (negated y because screen
    y increases downward while heading treats up as positive).
    We mirror that so angular_diff(ship_heading, toric_bearing(...)) == 0
    when the ship is already facing the target.
    """
    dx, dy = toric_delta(to_pos, from_pos, map_size)
    return math.degrees(math.atan2(-dy, dx)) % 360


# ---------------------------------------------------------------------------
# Angular utilities
# ---------------------------------------------------------------------------

def angular_diff(a_deg: float, b_deg: float) -> float:
    """
    Signed angular difference a - b, wrapped to (-180, 180].

    Positive means a is counter-clockwise of b.
    """
    diff = (a_deg - b_deg) % 360
    if diff > 180:
        diff -= 360
    return diff


def shortest_turn(current_heading: float, target_heading: float) -> float:
    """
    Return the signed turn rate direction (in degrees) to reach
    target_heading from current_heading via the shortest arc.

    The caller multiplies this by a gain k_omega to get the actual turn
    rate command in degrees/s.
    """
    return angular_diff(target_heading, current_heading)


# ---------------------------------------------------------------------------
# Time-to-Collision
# ---------------------------------------------------------------------------

def time_to_collision(
    ship_pos: Vec2,
    ship_vel: Vec2,
    ast_pos: Vec2,
    ast_vel: Vec2,
    ast_radius: float,
    map_size: Vec2,
    margin: float = 0.0,
) -> float:
    """
    Estimate time (seconds) until the ship enters the collision radius of
    an asteroid, using a linearised closing-velocity model.

    The model is:
        TTC = (d_effective) / closing_speed

    where:
        d_effective   = toric_distance - ast_radius - margin  (surface distance)
        closing_speed = -d/dt(distance) = -(r_hat . v_rel)

    Returns
    -------
    float  : TTC in seconds, or math.inf if the asteroid is receding or
             already in collision range.

    Notes
    -----
    Caveat C1: the displacement vector is computed torically so that
    boundary-crossing asteroids are handled correctly.

    The model is first-order (ignores curvature of relative trajectory
    and ship inertia).  It is deliberately kept simple for real-time use
    inside the FIS and angular profile computation.
    """
    # Toroidal relative position: from ship to asteroid

    W, H = map_size

    rvx = ast_vel[0] - ship_vel[0]
    rvy = ast_vel[1] - ship_vel[1]

    best_ttc = math.inf

    for kx in (-1, 0, 1):
        for ky in (-1, 0, 1):

            dx = (ast_pos[0] + kx * W) - ship_pos[0]
            dy = (ast_pos[1] + ky * H) - ship_pos[1]

            dist = math.hypot(dx, dy)

            if dist < 1e-6:
                return 0.0

            d_eff = max(dist - ast_radius - margin, 0.0)

            ux = dx / dist
            uy = dy / dist

            closing = ux * rvx + uy * rvy

            if closing <= 0:
                continue

            ttc = d_eff / closing

            if ttc < best_ttc:
                best_ttc = ttc

    return best_ttc


# ---------------------------------------------------------------------------
# Heading conversion helpers
# ---------------------------------------------------------------------------

def kessler_heading_to_math(kessler_deg: float) -> float:
    """
    Convert Kessler's heading (0° = right, positive counter-clockwise,
    same as math convention) to [0, 360).

    In practice Kessler uses the same math.atan2 convention, so this
    is mostly a normalisation guard.
    """
    return kessler_deg % 360


def math_angle_to_turn_rate(
    current_heading: float,
    desired_heading: float,
    k_omega: float = 1.0,
    omega_max: float = 180.0,
) -> float:
    """
    Proportional heading controller.

    Returns a turn rate in degrees/s, clamped to [-omega_max, omega_max].

    Parameters
    ----------
    current_heading : ship's current heading in degrees
    desired_heading : target heading in degrees
    k_omega         : proportional gain (degrees of turn-rate per degree of error)
    omega_max       : maximum allowed turn rate (degrees/s)
    """
    error = shortest_turn(current_heading, desired_heading)
    omega = k_omega * error
    return max(-omega_max, min(omega_max, omega))