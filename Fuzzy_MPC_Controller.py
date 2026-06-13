import atexit
import os
import matplotlib
matplotlib.use('Agg')  # file-only backend — must be set before any pyplot import

# Debug (per-frame snapshots + end-of-run heatmap) is ON by default for
# interactive runs, and OFF inside GA workers (which set KESSLER_DEBUG=0).
# When OFF, no snapshots are collected, no log is written and no atexit hook is
# registered — the controller runs at full speed with zero debug side effects.
_DEBUG = os.environ.get("KESSLER_DEBUG", "1") == "1"

from angular_profile import AngularProfile
from debug_tools import debug_snapshot, replay_heatmaps

from Casadi_mpc import MPCController
from risk_field import RiskField
#from test_controller_fuzzy import FuzzyController
#from SacrificeController import SacrificeController
from dataclasses import dataclass
from supervisor import Supervisor

_HEATMAP_DIR = "heatmaps"

class Controller:
    def __init__(self):
        #self.fuzzy = FuzzyController()
        self.mpc = MPCController()
        #self.sacrifice = FuzzyController() # placeholder for actual SacrificeController()
        self.supervisor = Supervisor()
        #self.debug=debug() # placeholder for actual debug tools like FrameDebugger, RiskHeatmap, etc.
        self._debug_ap   = AngularProfile()
        self._rf         = None
        self._map_size   = None

        self._frame          = 0
        self._last_snapshot  = None
        self._decision_log   = []   # lightweight per-sample explainability record
        self._frame_snapshots = []  # raw game state per 30-frame tick — rendered at game end

        # Fresh log file every game run (debug builds only)
        if _DEBUG:
            os.makedirs(_HEATMAP_DIR, exist_ok=True)
            self._log_path = os.path.join(_HEATMAP_DIR, "debug_log.txt")
            with open(self._log_path, "w", encoding="utf-8") as _f:
                _f.write("=== Controller Debug Log ===\n\n")
            atexit.register(self._finalize)
        else:
            self._log_path = None
    
    def actions(self, ship_state, game_state):
        self._frame += 1
        output = self.supervisor.compute(ship_state, game_state)  # control output

        if not _DEBUG:
            return output

        # Every 30 frames: record decision, collect snapshot data, append to log
        if self._frame % 30 == 0:
            _R_LO = self.supervisor.R_lo
            _R_HI = self.supervisor.R_hi
            self._record_decision(ship_state, game_state, output[0], output[1], output[2], _R_HI, _R_LO)
            self._collect_frame_data(ship_state, game_state)
            self._append_log_entry()

        # Keep latest frame for end-of-game heatmap
        self._last_snapshot = (
            tuple(ship_state.position),
            tuple(ship_state.velocity),
            ship_state.heading,
            [(tuple(a.position), tuple(a.velocity), a.size)
             for a in game_state.asteroids],
            game_state.map_size,
        )

        return output
    
        # ------------------------------------------------------------------
    # Per-tick data collection — no rendering on the hot path
    # ------------------------------------------------------------------

    def _collect_frame_data(self, ship_state, game_state):
        """Store raw game state for end-of-game heatmap replay — zero render overhead."""
        try:
            fc = getattr(self.supervisor, 'controllers', {}).get('fuzzy', None)
            explain_data = fc.explain() if fc is not None and hasattr(fc, 'explain') else {}
            self._frame_snapshots.append({
                "frame"        : self._frame,
                "ship_pos"     : tuple(ship_state.position),
                "ship_vel"     : tuple(ship_state.velocity),
                "ship_heading" : float(ship_state.heading),
                "ast_data"     : [(tuple(a.position), tuple(a.velocity), float(a.size))
                                  for a in game_state.asteroids],
                "map_size"     : game_state.map_size,
                "explain_data" : explain_data,
            })
        except Exception:
            pass  # debug must never crash the game

    # ------------------------------------------------------------------
    # Live text log — appended every 30 frames, replaced each game
    # ------------------------------------------------------------------

    def _append_log_entry(self):
        if not self._decision_log:
            return
        e = self._decision_log[-1]
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"[Frame {e['frame']}]\n")
                f.write(f"  Mode          : {e['mode']}\n")
                f.write(f"  Reason        : {e['mode_reason']}\n")
                f.write(f"  R_global      : {e['R_global']:.4f}\n")
                f.write(f"  Asteroids     : {e['n_asteroids']}\n")
                f.write(f"  Top threat    : {e['top_threat']}\n")
                f.write(f"  Fire decision : {e['fire_reason']}\n")
                f.write(f"  thrust={e['thrust']:+.0f}  turn_rate={e['turn_rate']:+.0f}\n")
                f.write("\n")
        except Exception:
            pass  # debug must never crash the game

    # ------------------------------------------------------------------
    # Explainability — collect one record per sample tick
    # ------------------------------------------------------------------

    def _record_decision(self, ship_state, game_state, thrust, turn_rate, fire,_R_HI,_R_LO):
        """Cheap snapshot of why the controller acted this frame."""
        try:
            if self._rf is None or self._map_size != game_state.map_size:
                self._map_size = game_state.map_size
                self._rf = RiskField(map_size=game_state.map_size)

            risks    = self._rf.compute_all(
                tuple(ship_state.position),
                tuple(ship_state.velocity),
                game_state.asteroids,
            )
            R_global = self._rf.aggregate(risks)

            # Infer mode from R_global vs supervisor thresholds
            if R_global >= _R_HI:
                mode   = "MPC"
                reason = f"R_global={R_global:.3f} >= {_R_HI} (danger threshold)"
            elif R_global <= _R_LO:
                mode   = "optimal Control"
                reason = f"R_global={R_global:.3f} <= {_R_LO} (safe — attack mode)"
            else:
                mode   = "hysteresis-previous mode"
                reason = f"R_global={R_global:.3f} in hysteresis band [{_R_LO}, {_R_HI}]"

            # Identify highest-risk asteroid
            top = max(risks, key=lambda r: r.risk, default=None)
            if top:
                threat_str = (
                    f"asteroid #{top.asteroid_id}  "
                    f"risk={top.risk:.3f}  tau={top.tau:.2f}s  "
                    f"d={top.d_surface:.0f}px  bearing={top.bearing:.0f}deg"
                )
            else:
                threat_str = "none"

            # Fire rationale
            if fire:
                fire_reason = "fired — heading aligned with target"
            else:
                fire_reason = "held fire — not aligned or no target"

            self._decision_log.append({
                "frame"      : self._frame,
                "mode"       : mode,
                "mode_reason": reason,
                "R_global"   : round(R_global, 4),
                "thrust"     : round(thrust, 1),
                "turn_rate"  : round(turn_rate, 1),
                "fire"       : fire,
                "fire_reason": fire_reason,
                "n_asteroids": len(risks),
                "top_threat" : threat_str,
                "top_tau"    : round(top.tau, 3) if top else None,
                "top_risk"   : round(top.risk, 3) if top else None,
            })
        except Exception:
            pass   # debug must never crash the game

    # ------------------------------------------------------------------
    # End-of-game: heatmap + explainability summary
    # ------------------------------------------------------------------

    def _finalize(self):
        """atexit hook: SAVE artifacts only. Interactive display happens in
        show_debug() while the interpreter is still alive (a GUI event loop is
        not available during interpreter shutdown)."""
        if not _DEBUG:
            return
        try:
            self._print_explainability_summary()
        except Exception:
            pass
        try:
            self._save_final_heatmap()
        except Exception:
            pass
        try:
            replay_heatmaps(self._frame_snapshots, interactive=False)  # -> PNGs
        except Exception:
            pass

    def show_debug(self, display_seconds: float = 2.0):
        """Render the end-of-run heatmaps to the screen.

        Call this AFTER game.run() returns (interpreter still alive) for a
        reliable interactive display; it switches to a GUI matplotlib backend
        on demand. Falls back to saving PNGs if no GUI backend is available.
        """
        if not _DEBUG:
            print("[debug_tools] KESSLER_DEBUG=0 — debug disabled, nothing to show.")
            return
        atexit.unregister(self._finalize)   # we render now; skip the shutdown save
        try:
            self._print_explainability_summary()
        except Exception:
            pass
        try:
            self._save_final_heatmap()
        except Exception as e:
            print(f"[debug_tools] Final heatmap failed: {e}")
        try:
            replay_heatmaps(self._frame_snapshots, interactive=True,
                            display_seconds=display_seconds)
        except Exception as e:
            print(f"[debug_tools] Replay failed: {e}")

    def _print_explainability_summary(self):
        log = self._decision_log
        if not log:
            print("[debug_tools] No decision log recorded.")
            return

        total       = len(log)
        n_fuzzy     = sum(1 for e in log if e["mode"] == "FUZZY")
        n_mpc       = sum(1 for e in log if e["mode"] == "MPC")
        n_hyst      = total - n_fuzzy - n_mpc
        n_fire      = sum(1 for e in log if e["fire"])
        avg_risk    = sum(e["R_global"] for e in log) / total
        max_risk    = max(e["R_global"] for e in log)
        peak_entry  = max(log, key=lambda e: e["R_global"])

        print()
        print("=" * 65)
        print("  CONTROLLER EXPLAINABILITY SUMMARY")
        print("=" * 65)
        print(f"  Frames sampled : {total}  (every 30 frames = every ~0.5 s)")
        print(f"  Asteroids avg  : {sum(e['n_asteroids'] for e in log)/total:.1f} on screen")
        print()
        print("  MODE DISTRIBUTION")
        print(f"    FUZZY (attack)   : {n_fuzzy:3d} samples  ({n_fuzzy/total*100:.0f}%)")
        print(f"    MPC   (evade)    : {n_mpc:3d} samples  ({n_mpc/total*100:.0f}%)")
        print(f"    Hysteresis band  : {n_hyst:3d} samples  ({n_hyst/total*100:.0f}%)")
        print()
        print("  GLOBAL RISK  (R_global ∈ [0, 1])")
        print(f"    Average : {avg_risk:.3f}")
        print(f"    Peak    : {max_risk:.3f}  at frame {peak_entry['frame']}")
        print(f"    Why MPC triggered: R_global >= {0.5309724169690697}  "
              f"(supervisor danger threshold)")
        print()
        print("  FIRE EVENTS")
        print(f"    Fired in {n_fire}/{total} sampled frames  ({n_fire/total*100:.0f}%)")
        if n_fire == 0:
            print("    Warning: ship never fired during sampled frames")
        print()

        # Show the 3 most dangerous moments
        top3 = sorted(log, key=lambda e: e["R_global"], reverse=True)[:3]
        print("  TOP 3 DANGEROUS MOMENTS")
        for i, e in enumerate(top3, 1):
            print(f"    [{i}] frame {e['frame']:4d}  R={e['R_global']:.3f}  "
                  f"mode={e['mode']}  fire={e['fire']}")
            print(f"         Why this mode : {e['mode_reason']}")
            print(f"         Top threat    : {e['top_threat']}")
            print(f"         Fire decision : {e['fire_reason']}")
            print(f"         thrust={e['thrust']:+.0f}  turn={e['turn_rate']:+.0f}")

        print("=" * 65)

    def _save_final_heatmap(self):
        if self._last_snapshot is None:
            return

        ship_pos, ship_vel, ship_heading, ast_data, map_size = self._last_snapshot

        class _Ast:
            __slots__ = ('position', 'velocity', 'size', 'radius')
            def __init__(self, p, v, s):
                self.position = p
                self.velocity = v
                self.size     = s
                self.radius   = s * 8.0      # engine: radius = size * 8

        try:
            asteroids = [_Ast(p, v, s) for p, v, s in ast_data]
            rf        = RiskField(map_size=map_size)
            risks     = rf.compute_all(ship_pos, ship_vel, asteroids)
            debug_snapshot(
                ship_pos        = ship_pos,
                ship_vel        = ship_vel,
                ship_heading    = ship_heading,
                asteroid_risks  = risks,
                map_size        = map_size,
                angular_profiler= self._debug_ap,
                save_prefix     = "final",
            )
            print("[debug_tools] Final heatmap saved → heatmaps/final_*.png")
        except Exception as e:
            print(f"[debug_tools] Final heatmap failed: {e}")
    
    @property
    def name(self) -> str:
        return "fuzzy_mpc_hybrid_controller"
    
    # @property
    # def custom_sprite_path(self) -> str:
    #     return "A400m_kessler"