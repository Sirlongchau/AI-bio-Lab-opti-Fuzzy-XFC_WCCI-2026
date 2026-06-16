"""
edge_fire_guard.py
==================
Disabled final fire gate.

Reason:
    The edge-fire suppression was blocking useful shots and making the ship
    worse. This version restores normal firing behavior while leaving the
    controller structure intact.
"""

def edge_safe_fire(*args, **kwargs):
    return True
