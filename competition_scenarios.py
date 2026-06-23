# -*- coding: utf-8 -*-
"""
competition_scenarios.py
========================
Portfolio de scénarios *façon compétition* pour Kessler (XFC / FUZZ-IEEE, WCCI).

⚠️  AVERTISSEMENT D'HONNÊTETÉ
-----------------------------
Les scénarios EXACTS de la compétition 2026 ne sont PAS publics : ils sont
construits par les organisateurs et gardés privés (matchs en bracket contre des
agents baseline) précisément pour éviter le surapprentissage. Les vidéos YouTube
montrent les matchs mais ne contiennent ni coordonnées ni seeds. Ce fichier ne
prétend donc PAS reproduire les scénarios officiels — il fournit un portfolio
*représentatif* du format connu, calibré sur les conventions réelles du moteur
(ThalesGroup/kessler-game) :

    map_size = (1000, 800)          fréquence 30 Hz
    asteroïdes size ∈ {1..4}, radius = size*8
    vitesse aléatoire ∈ [0, 60*(2+(4-size)/4)]  → ~120 px/s (size4) … ~165 (size1)
    angle (deg) : vx = speed*cos(angle), vy = speed*sin(angle)   (y vers le bas)

Le portfolio couvre la même intention que les organisateurs :
    • des scénarios ALÉATOIRES seedés à densité croissante (= "random scenarios"),
    • des scénarios STRUCTURÉS déterministes (anneau, mur, couloirs croisés),
    • des DUELS 1v1 (deux équipes : on ne se tire pas dessus, mais mines + collisions).

Deux points d'entrée :
    competition_scenario_configs()  -> list[dict]      (prêts pour le GA)
    competition_scenarios()         -> list[Scenario]  (pour visualiser / évaluer)

Intégration GA (remplace _build_fixed_scenarios) :
    from competition_scenarios import competition_scenario_configs
    cfg = GAConfig(scenario_configs=competition_scenario_configs(time_limit=45))

NB : les duels (multi-ship) nécessitent DEUX contrôleurs au run ; ils sont donc
exclus du portfolio GA solo (qui tourne controllers=[ctrl]) et fournis à part
pour la validation contre un agent adverse.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

MAP_W, MAP_H = 1000, 800                     # ints: engine random placement uses randrange
CX, CY       = MAP_W / 2.0, MAP_H / 2.0      # centre arène (500.0, 400.0)

# Vitesse max moteur par taille : 60*(2+(4-size)/4). On reste dans ces bornes
# pour que les scénarios structurés aient la même "physique" que l'aléatoire.
def _engine_max_speed(size: int) -> float:
    return 60.0 * (2.0 + (4.0 - size) / 4.0)


def _ship(team: int = 1, pos=(CX, CY), angle: float = 90.0,
          lives: int = 3, mines: int = 3) -> Dict[str, Any]:
    return {"position": (float(pos[0]), float(pos[1])), "angle": float(angle),
            "lives": lives, "team": team, "mines_remaining": mines}


def _inward_angle(x: float, y: float, cx: float = CX, cy: float = CY) -> float:
    """Heading (deg, convention moteur) pointant de (x,y) vers (cx,cy)."""
    return math.degrees(math.atan2(cy - y, cx - x)) % 360.0


def _random_field(n: int, seed: int, vmin: float = 40.0, vmax: float = 150.0,
                  min_clear: float = 180.0) -> List[Dict[str, Any]]:
    """
    Champ aléatoire REPRODUCTIBLE à tailles mixtes (1..4), vitesses bornées et
    zone d'apparition dégagée autour du vaisseau (évite les morts instantanées
    qui ne sont pas la faute du contrôleur). Plus représentatif qu'un
    `num_asteroids` moteur (qui ne génère QUE des size-4 et peut spawn sur le
    vaisseau), et entièrement déterministe via random.Random(seed).
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    while len(out) < n:
        x = rng.uniform(20.0, MAP_W - 20.0)
        y = rng.uniform(20.0, MAP_H - 20.0)
        if (x - CX) ** 2 + (y - CY) ** 2 < min_clear ** 2:
            continue                               # garde le centre dégagé
        size = rng.randint(1, 4)
        out.append({"position": (x, y), "angle": rng.uniform(0.0, 360.0),
                    "speed": min(rng.uniform(vmin, vmax), _engine_max_speed(size)),
                    "size": size})
    return out


# ---------------------------------------------------------------------------
# Scénarios STRUCTURÉS (déterministes, façon "challenge organisateur")
# ---------------------------------------------------------------------------

def _ring_collapse(n: int = 22, radius: float = 300.0, speed: float = 110.0,
                   big_every: int = 4) -> List[Dict[str, Any]]:
    """Anneau d'astéroïdes convergeant vers le centre : esquive + ratissage de
    fragments. Rester immobile = avalé par le cône de débris."""
    out = []
    for k in range(n):
        th = 2.0 * math.pi * k / n
        x = CX + radius * math.cos(th)
        y = CY + radius * math.sin(th)
        size = 3 if (k % big_every == 0) else (2 if k % 2 else 1)
        out.append({"position": (x, y), "angle": _inward_angle(x, y),
                    "speed": min(speed, _engine_max_speed(size)), "size": size})
    return out


def _wall_sweep(rows: int = 5, cols: int = 7, x0: float = 880.0,
                dx: float = 26.0, speed: float = 80.0) -> List[Dict[str, Any]]:
    """Mur lent dérivant vers l'ouest : la tourelle l'érode mais ne l'annihile
    pas à temps → force un repli (mode MPC)."""
    out = []
    for r in range(rows):
        for c in range(cols):
            x = x0 - dx * c
            y = 120.0 + (MAP_H - 240.0) * r / max(rows - 1, 1)
            size = 2 + (c % 2)            # mix size 2/3
            out.append({"position": (x, y), "angle": 180.0,    # plein ouest
                        "speed": min(speed, _engine_max_speed(size)), "size": size})
    return out


def _crossfire_lanes(per_lane: int = 9, speed: float = 130.0) -> List[Dict[str, Any]]:
    """Deux flux antiparallèles décalés qui se croisent près du centre :
    teste le timing tir/esquive dans un couloir mouvant."""
    out = []
    for i in range(per_lane):
        x = 60.0 + (MAP_W - 120.0) * i / max(per_lane - 1, 1)
        size = 1 + (i % 3)
        # voie haute -> vers la droite ; voie basse -> vers la gauche
        out.append({"position": (x, CY - 110.0), "angle": 0.0,
                    "speed": min(speed, _engine_max_speed(size)), "size": size})
        out.append({"position": (MAP_W - x, CY + 110.0), "angle": 180.0,
                    "speed": min(speed, _engine_max_speed(size)), "size": size})
    return out


# ---------------------------------------------------------------------------
# Portfolio GA SOLO (un seul vaisseau, équipe 1) — drop-in _build_fixed_scenarios
# ---------------------------------------------------------------------------

def competition_scenario_configs(time_limit: float = 45.0,
                                 ammo_limit_multiplier: float = 0.0,
                                 stop_if_no_ammo: bool = False
                                 ) -> List[Dict[str, Any]]:
    """
    Renvoie des dicts directement consommables par Scenario(**cfg) et par le GA
    (mêmes clés que _build_fixed_scenarios). Ordre = gradient de difficulté ;
    les 2 premiers (Cluster + Wall) sont les plus discriminants → bons écrans.
    """
    base = dict(map_size=(MAP_W, MAP_H), time_limit=time_limit,
                ammo_limit_multiplier=ammo_limit_multiplier,
                stop_if_no_ammo=stop_if_no_ammo)
    ship = [_ship(team=1)]

    def rnd(name, n, seed, vmax=150.0):   # champ aléatoire seedé, tailles mixtes
        return {"name": name, "asteroid_states": _random_field(n, seed, vmax=vmax),
                "ship_states": ship, **base}

    def struct(name, states):     # scénario STRUCTURÉ déterministe
        return {"name": name, "asteroid_states": states,
                "ship_states": ship, **base}

    return [
        # — écrans discriminants (esquive obligatoire) —
        struct("Ring_Collapse",  _ring_collapse(n=22)),
        struct("Wall_Sweep",     _wall_sweep()),
        # — gradient aléatoire seedé (tailles mixtes, spawn dégagé) —
        rnd("Rand_Sparse",   8,  101),
        rnd("Rand_Light",    14, 202),
        rnd("Rand_Moderate", 22, 303),
        rnd("Rand_Dense",    34, 404),
        rnd("Rand_Storm",    50, 505, vmax=170.0),
        # — stress structuré supplémentaire —
        struct("Crossfire_Lanes", _crossfire_lanes()),
    ]


def competition_scenarios(time_limit: float = 45.0, **kw):
    """Mêmes scénarios solo, mais en objets Scenario (visualisation / éval directe)."""
    from kesslergame import Scenario
    return [Scenario(**cfg)
            for cfg in competition_scenario_configs(time_limit=time_limit, **kw)]


# ---------------------------------------------------------------------------
# DUELS 1v1 (multi-ship) — pour validation contre un agent adverse
# (nécessite DEUX contrôleurs : game.run(scenario, controllers=[mine, foe]))
# ---------------------------------------------------------------------------

def duel_scenarios(time_limit: float = 60.0):
    from kesslergame import Scenario
    ships = [_ship(team=1, pos=(CX - 250.0, CY), angle=0.0),
             _ship(team=2, pos=(CX + 250.0, CY), angle=180.0)]
    return [
        Scenario(name="Duel_Sparse", asteroid_states=_random_field(12, 11),
                 ship_states=ships, map_size=(MAP_W, MAP_H),
                 time_limit=time_limit, ammo_limit_multiplier=0.0),
        Scenario(name="Duel_Dense", asteroid_states=_random_field(30, 22),
                 ship_states=ships, map_size=(MAP_W, MAP_H),
                 time_limit=time_limit, ammo_limit_multiplier=0.0),
        Scenario(name="Duel_Ring", asteroid_states=_ring_collapse(n=20, radius=320),
                 ship_states=ships, map_size=(MAP_W, MAP_H),
                 time_limit=time_limit, ammo_limit_multiplier=0.0),
    ]


# ---------------------------------------------------------------------------
# Démo : visualiser un scénario du portfolio
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from kesslergame import KesslerGame, GraphicsType

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="index dans le portfolio solo")
    ap.add_argument("--tl", type=float, default=45.0)
    args = ap.parse_args()

    try:
        from Fuzzy_MPC_Controller import Controller
    except Exception:
        from Fuzzy_MPC_Controller_no_debug import Controller

    scen = competition_scenarios(time_limit=args.tl)[args.index]
    game = KesslerGame(settings={"graphics_type": GraphicsType.Tkinter,
                                 "realtime_multiplier": 1, "frequency": 30,
                                 "perf_tracker": True})
    score, _ = game.run(scenario=scen, controllers=[Controller()])
    print(f"\n[{scen.name}] stop={score.stop_reason}")
    print(f"  hits={[t.asteroids_hit for t in score.teams]}  "
          f"deaths={[t.deaths for t in score.teams]}  "
          f"acc={[round(t.accuracy,3) for t in score.teams]}")