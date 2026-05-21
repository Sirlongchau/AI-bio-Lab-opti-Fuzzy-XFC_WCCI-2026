"""
debug_tools.py
==============
Runtime debugging utilities for the FuzzyHybridController.

Two independent tools:

1. FrameDebugger  — conditional breakpoint system usable inside actions().
   Supports: break on frame N, break when condition(state) is True,
   break once, break every N frames.
   Uses Python's pdb so it works in any terminal without a GUI.

2. RiskHeatmap    — parallel Tkinter window showing the risk heatmap.
   Runs in a dedicated daemon thread so it never blocks the game loop.
   The controller pushes frame data via a queue; the Tk thread renders
   asynchronously.  No matplotlib, no external dependencies.

Threading model
---------------
                 game thread                    tk thread
                 ───────────                    ─────────
  actions() ──► queue.put(frame_data) ──────► _tk_loop() polls queue
                                               every ~50ms via after()
                                               renders to Canvas

The queue holds at most 1 item (maxsize=1).  If the Tk thread is slow,
the game thread drops the frame silently — the game is never blocked.
"""

from __future__ import annotations

import math
import pdb
import queue
import threading
import tkinter as tk
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np   # only for fast grid computation; stdlib otherwise

# ============================================================================
# 1. FrameDebugger
# ============================================================================

class FrameDebugger:
    """
    Conditional breakpoint manager for in-loop debugging.

    The debugger fires pdb.set_trace() when any registered condition is met.
    pdb gives a full interactive Python prompt in the terminal where you can
    inspect any variable, step through code, or continue.

    Useful pdb commands inside the breakpoint:
        n          — next line
        c          — continue to next breakpoint
        p expr     — print expression
        pp expr    — pretty-print
        l          — list source around current line
        q          — quit the game immediately
        u / d      — move up/down the call stack
        !x = 5     — assign variable x = 5 (useful to patch state live)

    Example
    -------
        dbg = FrameDebugger()
        dbg.break_at_frame(200)
        dbg.break_when(lambda s: s['tau_min'] < 0.5, label='imminent collision')
        dbg.break_every(300, label='periodic check')

        # In actions() each frame:
        dbg.tick({'frame': self._frame, 'r_global': r_global, ...})
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled    = enabled
        self._conditions: List[Dict[str, Any]] = []
        self._fired:      set = set()

    # ------------------------------------------------------------------
    def break_at_frame(self, frame_n: int, once: bool = True) -> None:
        """Stop at exactly frame number frame_n."""
        self._conditions.append({
            'label': f'frame={frame_n}',
            'fn':    lambda s: s.get('frame', -1) == frame_n,
            'once':  once,
        })

    def break_when(
        self,
        condition: Callable[[Dict], bool],
        label: str = 'user condition',
        once:  bool = False,
    ) -> None:
        """
        Stop when condition(state_dict) returns True.
        Set once=True to fire only the first time.
        """
        self._conditions.append({'label': label, 'fn': condition, 'once': once})

    def break_every(self, n_frames: int, label: str = '') -> None:
        """Stop every n_frames frames."""
        lbl = label or f'every {n_frames} frames'
        self._conditions.append({
            'label': lbl,
            'fn':    lambda s: s.get('frame', 0) % n_frames == 0,
            'once':  False,
        })

    def break_on_mode_change(self) -> None:
        """Stop whenever the supervisor mode changes."""
        self._last_mode = [None]
        def _check(s: Dict) -> bool:
            mode = s.get('mode')
            if mode != self._last_mode[0]:
                self._last_mode[0] = mode
                return True
            return False
        self._conditions.append({'label': 'mode change', 'fn': _check, 'once': False})

    def clear(self) -> None:
        self._conditions.clear()
        self._fired.clear()

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    # ------------------------------------------------------------------
    def tick(self, state: Dict[str, Any]) -> None:
        """
        Call once per frame.  Fires pdb.set_trace() if any condition met.
        'state' dict is in scope inside pdb so you can inspect everything.
        """
        if not self._enabled:
            return

        for i, cond in enumerate(self._conditions):
            if cond['once'] and i in self._fired:
                continue
            try:
                triggered = cond['fn'](state)
            except Exception:
                triggered = False

            if triggered:
                if cond['once']:
                    self._fired.add(i)
                print(
                    f"\n{'='*60}\n"
                    f"[FrameDebugger] Condition met: {cond['label']}\n"
                    f"  frame      = {state.get('frame', '?')}\n"
                    f"  mode       = {state.get('mode', '?')}\n"
                    f"  r_global   = {state.get('r_global', 0.0):.4f}\n"
                    f"  tau_min    = {state.get('tau_min', '?')}\n"
                    f"  ship_pos   = {state.get('ship_pos', '?')}\n"
                    f"  n_ast      = {len(state.get('asteroid_risks', []))}\n"
                    f"  corridors  = {len(state.get('corridors', []))}\n"
                    f"{'='*60}\n"
                    f"  Dropping into pdb. Type 'c' to continue, 'q' to quit.\n"
                    f"  Access 'state' dict for all frame variables.\n"
                    f"{'='*60}"
                )
                pdb.set_trace()   # 'state' is visible here
                break


# ============================================================================
# 2. RiskHeatmap  — pure Tkinter, runs in a daemon thread
# ============================================================================

# Colour gradient: safe(green) → watch(yellow) → danger(orange) → critical(red)
# Precomputed as a 256-entry RGB lookup table (no matplotlib needed)
def _build_lut() -> List[Tuple[int, int, int]]:
    lut = []
    stops = [
        (0.00, (13,  27,  42)),    # deep navy  — safe
        (0.30, (27, 100,  68)),    # dark green
        (0.55, (255, 214,  10)),   # yellow
        (0.75, (230, 100,  30)),   # orange
        (1.00, (220,  30,  30)),   # red        — critical
    ]
    for i in range(256):
        t = i / 255.0
        # Find segment
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0)
                r = int(c0[0] + alpha * (c1[0] - c0[0]))
                g = int(c0[1] + alpha * (c1[1] - c0[1]))
                b = int(c0[2] + alpha * (c1[2] - c0[2]))
                lut.append((r, g, b))
                break
    return lut

_LUT = _build_lut()


def _rgb_hex(r: int, g: int, b: int) -> str:
    return f'#{r:02x}{g:02x}{b:02x}'


class _HeatmapWindow:
    """
    Tkinter window that lives entirely in the Tk daemon thread.
    Never call any method on this object from the game thread —
    communicate only via the queue.
    """

    # Canvas logical dimensions (scaled from map_size on first frame)
    CANVAS_W = 600
    CANVAS_H = 480
    PANEL_H  = 90    # bottom info panel height

    def __init__(self, data_queue: 'queue.Queue') -> None:
        self._q    = data_queue
        self._root = tk.Tk()
        self._root.title('Risk Heatmap — Kessler Hybrid Controller')
        self._root.configure(bg='#1a1a2e')
        self._root.resizable(False, False)

        # Canvas for the heatmap
        self._canvas = tk.Canvas(
            self._root,
            width=self.CANVAS_W,
            height=self.CANVAS_H,
            bg='#0d1b2a',
            highlightthickness=0,
        )
        self._canvas.pack(side=tk.TOP)

        # Info panel below the canvas
        self._panel = tk.Frame(self._root, bg='#12122a', height=self.PANEL_H)
        self._panel.pack(side=tk.TOP, fill=tk.X)

        font_label = ('Consolas', 9)
        self._lbl_mode   = tk.Label(self._panel, text='mode: —',      fg='#aaaaff', bg='#12122a', font=font_label, anchor='w')
        self._lbl_risk   = tk.Label(self._panel, text='R_global: —',  fg='#ffdd88', bg='#12122a', font=font_label, anchor='w')
        self._lbl_tau    = tk.Label(self._panel, text='tau_min: —',   fg='#88ffaa', bg='#12122a', font=font_label, anchor='w')
        self._lbl_frame  = tk.Label(self._panel, text='frame: —',     fg='#aaaaaa', bg='#12122a', font=font_label, anchor='w')
        self._lbl_target = tk.Label(self._panel, text='target: —',    fg='#ffff44', bg='#12122a', font=font_label, anchor='w')
        self._lbl_solver = tk.Label(self._panel, text='solver: —',    fg='#44ffff', bg='#12122a', font=font_label, anchor='w')

        self._lbl_mode.grid  (row=0, column=0, sticky='w', padx=8, pady=2)
        self._lbl_risk.grid  (row=0, column=1, sticky='w', padx=8)
        self._lbl_tau.grid   (row=0, column=2, sticky='w', padx=8)
        self._lbl_frame.grid (row=0, column=3, sticky='w', padx=8)
        self._lbl_target.grid(row=1, column=0, sticky='w', padx=8, pady=2)
        self._lbl_solver.grid(row=1, column=1, sticky='w', padx=8, columnspan=2)

        # Photo image — reused every frame (avoids garbage-collection flicker)
        self._photo: Optional[tk.PhotoImage] = None

        # Map geometry (set on first frame)
        self._map_w: float = 1000.0
        self._map_h: float = 800.0
        self._res:   int   = 4       # px per grid cell

        # Start polling the queue
        self._root.after(50, self._poll)

    # ------------------------------------------------------------------
    def _poll(self) -> None:
        """Called by Tk's event loop every 50 ms to consume queue items."""
        try:
            data = self._q.get_nowait()
            self._render(data)
        except queue.Empty:
            pass
        self._root.after(50, self._poll)

    # ------------------------------------------------------------------
    def _render(self, data: Dict) -> None:
        """Render one frame of data onto the canvas."""
        map_size        = data['map_size']
        self._map_w     = float(map_size[0])
        self._map_h     = float(map_size[1])
        self._res       = data.get('resolution', 4)
        asteroid_risks  = data['asteroid_risks']
        ship_pos        = data['ship_pos']
        ship_heading    = data.get('ship_heading', 0.0)
        target          = data.get('target')
        profile         = data.get('angular_profile')   # List[float] | None
        mpc_traj        = data.get('mpc_trajectory')    # List[(x,y)] | None
        frame           = data.get('frame', 0)
        r_global        = data.get('r_global', 0.0)
        tau_min         = data.get('tau_min', float('inf'))
        mode            = data.get('mode', '?')
        solver          = data.get('solver_used', '?')

        W, H   = self._map_w, self._map_h
        CW, CH = self.CANVAS_W, self.CANVAS_H
        sx     = CW / W    # x scale factor map→canvas
        sy     = CH / H    # y scale factor

        # ---- Build heatmap grid ----
        cols = max(1, int(W / self._res))
        rows = max(1, int(H / self._res))
        grid = self._compute_grid(cols, rows, asteroid_risks)

        # ---- Convert grid to PhotoImage ----
        # Scale grid to canvas resolution using nearest-neighbour
        # (fast pure-numpy, no PIL required)
        canvas_grid = np.zeros((CH, CW), dtype=np.float32)
        for row in range(CH):
            gr = min(int(row / CH * rows), rows - 1)
            for col in range(CW):
                gc = min(int(col / CW * cols), cols - 1)
                canvas_grid[row, col] = grid[gr, gc]

        # Map [0,1] → LUT index [0,255]
        indices = np.clip((canvas_grid * 255).astype(np.int32), 0, 255)

        # Build PhotoImage row strings (Tk PPM format in memory)
        # Each row: "{#rrggbb #rrggbb ...}"
        photo = tk.PhotoImage(width=CW, height=CH)
        rows_data = []
        for row in range(CH):
            row_colors = [_rgb_hex(*_LUT[indices[row, col]]) for col in range(CW)]
            rows_data.append('{' + ' '.join(row_colors) + '}')
        photo.put(' '.join(rows_data))  # single put call — much faster

        self._canvas.delete('all')
        self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self._photo = photo   # hold reference to prevent GC
        self._last_grid = grid   # keep for iso-line drawing below

        # ---- Asteroid circles ----
        for ar in asteroid_risks:
            ax = ar.position[0] * sx
            ay = ar.position[1] * sy
            rr = max(3.0, ar.radius * min(sx, sy))
            # Colour by individual risk
            idx   = int(ar.risk * 255)
            color = _rgb_hex(*_LUT[idx])
            self._canvas.create_oval(
                ax - rr, ay - rr, ax + rr, ay + rr,
                outline=color, width=2, fill='',
            )
            self._canvas.create_text(
                ax, ay,
                text=f'R{ar.risk:.2f}\nτ{ar.tau:.1f}',
                fill='white', font=('Consolas', 7), justify=tk.CENTER,
            )

        # ---- MPC trajectory ----
        if mpc_traj and len(mpc_traj) > 1:
            pts = []
            for px_t, py_t in mpc_traj:
                pts += [px_t * sx, py_t * sy]
            self._canvas.create_line(pts, fill='#00ffff', width=2,
                                     smooth=True, dash=(4, 2))
            # Waypoint dots
            for px_t, py_t in mpc_traj:
                cx, cy = px_t * sx, py_t * sy
                self._canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill='#00ffff')

        # ---- Ship ----
        shx, shy = ship_pos[0] * sx, ship_pos[1] * sy
        # Draw heading arrow
        h_rad = math.radians(ship_heading)
        arrow_len = 22
        self._canvas.create_line(
            shx, shy,
            shx + arrow_len * math.cos(h_rad),
            shy - arrow_len * math.sin(h_rad),   # Kessler y-flip
            fill='white', width=2, arrow=tk.LAST,
        )
        # Ship cross
        s = 8
        self._canvas.create_line(shx-s, shy, shx+s, shy, fill='white', width=2)
        self._canvas.create_line(shx, shy-s, shx, shy+s, fill='white', width=2)

        # ---- Target ----
        if target is not None:
            t_rad = math.radians(target.bearing)
            dist  = 50
            tx = shx + dist * math.cos(t_rad)
            ty = shy - dist * math.sin(t_rad)
            self._canvas.create_text(tx, ty, text='★', fill='#ffff00',
                                     font=('Arial', 14))

        # ---- Angular danger profile — mini polar ring around ship ----
        if profile is not None:
            self._draw_polar(shx, shy, profile)

        # ---- Grid lines (light) ----
        for gx in range(0, CW, int(CW / 10)):
            self._canvas.create_line(gx, 0, gx, CH, fill='#1e1e3a', width=1)
        for gy in range(0, CH, int(CH / 8)):
            self._canvas.create_line(0, gy, CW, gy, fill='#1e1e3a', width=1)

        # ---- Decision boundary iso-lines (R_LO and R_HI) ----
        # These are the exact thresholds the supervisor uses to switch modes.
        # Drawn by marching through the grid and connecting threshold crossings.
        _g = getattr(self, '_last_grid', None)
        if _g is not None and _g.size > 0:
            self._draw_isoline(_g, CW, CH, level=0.45,
                               color='#44aaff', label='R_LO', dash=(6, 3))
            self._draw_isoline(_g, CW, CH, level=0.70,
                               color='#ff6622', label='R_HI', dash=(4, 2))

        # ---- Info panel ----
        tau_str = f'{tau_min:.2f}s' if tau_min < 999 else '∞'
        tgt_str = (f'#{target.asteroid_id} bear={target.bearing:.0f}°'
                   if target else 'none')
        mode_colors = {
            'fuzzy': '#aaaaff', 'mpc_evasion': '#ffbb44',
            'sacrifice': '#ff4444', 'respawn': '#44ff88',
        }
        self._lbl_mode  .config(text=f'mode: {mode}',
                                fg=mode_colors.get(mode, '#ffffff'))
        self._lbl_risk  .config(text=f'R_global: {r_global:.4f}')
        self._lbl_tau   .config(text=f'tau_min:  {tau_str}')
        self._lbl_frame .config(text=f'frame:    {frame}')
        self._lbl_target.config(text=f'target:   {tgt_str}')
        self._lbl_solver.config(text=f'solver:   {solver}')

    # ------------------------------------------------------------------
    def _compute_grid(
        self,
        cols: int,
        rows: int,
        asteroid_risks: list,
    ) -> np.ndarray:
        """
        Compute the risk grid using the EXACT same FIS (_fis_risk) that the
        supervisor calls on the real ship position.

        For each grid cell (cx, cy) and each asteroid i we compute:
            d_surface_i(cx,cy) — toroidal surface distance
            tau_i(cx,cy)       — linearised TTC from that cell
            R_i(cx,cy)         = _fis_risk(tau_i, d_surface_i, radius_i)

        Then aggregate exactly as RiskField.aggregate() does:
            grid(cx,cy) = alpha * max_i(R_i) + (1-alpha) * softsum_i(R_i)

        This means the heatmap contours ARE the supervisor's decision
        boundaries — the R_LO / R_HI iso-lines show exactly where mode
        switches happen, making the MPC safe path directly visible.

        Vectorisation strategy
        ----------------------
        _fis_risk uses trapezoid MFs (piecewise linear) — trivially
        vectorisable with numpy.  We replicate the MF arithmetic on full
        (rows×cols) arrays to avoid a Python loop over every cell.
        The total cost is O(n_ast × rows × cols) float operations, which
        for a 250×200 grid and 10 asteroids is ~500 k ops — about 3 ms.
        """
        grid = np.zeros((rows, cols), dtype=np.float32)
        if not asteroid_risks:
            return grid

        W, H = self._map_w, self._map_h
        r    = self._res

        # Grid cell centres
        cx_arr = np.linspace(r / 2, W - r / 2, cols, dtype=np.float64)
        cy_arr = np.linspace(r / 2, H - r / 2, rows, dtype=np.float64)
        CX, CY = np.meshgrid(cx_arr, cy_arr)   # (rows, cols)

        # Import FIS constants from risk_field (single source of truth)
        from risk_field import (
            TAU_CRITICAL, TAU_CLOSE, TAU_MEDIUM, TAU_FAR,
            D_NEAR, D_MEDIUM, D_FAR,
            S_SMALL, S_MEDIUM, S_LARGE,
            OUT_CRITICAL, OUT_HIGH, OUT_MEDIUM, OUT_LOW, OUT_NEGLIGIBLE,
        )
        ALPHA = 0.6   # must match RiskField.alpha

        def _trap_v(x: np.ndarray, a, b, c, d) -> np.ndarray:
            """Vectorised trapezoidal MF."""
            mu = np.zeros_like(x)
            # Rising slope
            mask = (x > a) & (x < b)
            mu[mask] = (x[mask] - a) / (b - a)
            # Plateau
            mu[(x >= b) & (x <= c)] = 1.0
            # Falling slope
            mask = (x > c) & (x < d)
            mu[mask] = (d - x[mask]) / (d - c)
            return mu

        # Per-asteroid FIS evaluation, accumulated into R_max and R_sum
        R_max = np.zeros((rows, cols), dtype=np.float64)
        R_sum = np.zeros((rows, cols), dtype=np.float64)

        for ar in asteroid_risks:
            ax = float(ar.position[0])
            ay = float(ar.position[1])
            vx = float(ar.velocity[0])
            vy = float(ar.velocity[1])
            radius = float(ar.radius)

            # --- Toroidal surface distance ---
            DX = CX - ax
            DY = CY - ay
            DX -= W * np.round(DX / W)
            DY -= H * np.round(DY / H)
            DIST   = np.sqrt(DX**2 + DY**2) + 1e-8
            D_SURF = np.maximum(DIST - radius, 0.0)

            # --- Toroidal TTC ---
            # Closing speed = -projection of asteroid velocity onto
            # approach vector (ship assumed stationary for the grid —
            # we show risk from each cell's perspective, not the ship's)
            UX = DX / DIST   # unit vector cell → asteroid
            UY = DY / DIST
            CLOSE = -(UX * vx + UY * vy)   # positive = approaching
            # Safe division: only divide where CLOSE > 0.1, else 99.0
            DENOM    = np.where(CLOSE > 0.1, CLOSE, 1.0)   # avoid /0
            TAU_GRID = np.where(CLOSE > 0.1, D_SURF / DENOM, 99.0)
            TAU_GRID = np.clip(TAU_GRID, 0.0, 99.0)

            # --- Fuzzify tau ---
            mu_t_crit   = _trap_v(TAU_GRID, *TAU_CRITICAL)
            mu_t_close  = _trap_v(TAU_GRID, *TAU_CLOSE)
            mu_t_medium = _trap_v(TAU_GRID, *TAU_MEDIUM)
            mu_t_far    = _trap_v(TAU_GRID, *TAU_FAR)

            # --- Fuzzify distance ---
            mu_d_near   = _trap_v(D_SURF, *D_NEAR)
            mu_d_medium = _trap_v(D_SURF, *D_MEDIUM)
            mu_d_far    = _trap_v(D_SURF, *D_FAR)

            # --- Fuzzify size (scalar, broadcast) ---
            mu_s_small  = float(_trap_v(np.array([radius]), *S_SMALL)[0])
            mu_s_medium = float(_trap_v(np.array([radius]), *S_MEDIUM)[0])
            mu_s_large  = float(_trap_v(np.array([radius]), *S_LARGE)[0])

            # --- Rule base (identical to _fis_risk) ---
            rules = [
                (np.minimum(mu_t_crit,   mu_d_near),                       OUT_CRITICAL),
                (np.minimum(mu_t_crit,   mu_d_medium),                     OUT_HIGH),
                (np.minimum(mu_t_crit,   np.full_like(mu_t_crit, mu_s_large)), OUT_HIGH),
                (np.minimum(mu_t_close,  mu_d_near),                       OUT_HIGH),
                (np.minimum(mu_t_close,  mu_d_medium),                     OUT_MEDIUM),
                (np.minimum(mu_t_close,  np.full_like(mu_t_close, mu_s_large)), OUT_MEDIUM),
                (np.minimum(mu_t_medium, mu_d_near),                       OUT_MEDIUM),
                (np.minimum(np.minimum(mu_t_medium, mu_d_medium),
                            np.full_like(mu_t_medium, mu_s_large)),        OUT_MEDIUM),
                (np.minimum(mu_t_medium, mu_d_medium),                     OUT_LOW),
                (np.minimum(mu_t_medium, mu_d_far),                        OUT_NEGLIGIBLE),
                (np.minimum(mu_t_far,    mu_d_near),                       OUT_LOW),
                (np.minimum(mu_t_far,    mu_d_medium),                     OUT_NEGLIGIBLE),
                (np.minimum(mu_t_far,    mu_d_far),                        OUT_NEGLIGIBLE),
            ]

            total_w = sum(w for w, _ in rules)
            W_sum   = sum(w for w, _ in rules)   # array sum
            # Vectorised weighted average
            num  = sum(w * o for w, o in rules)
            denom = sum(w for w, _ in rules)
            # denom is an array; avoid div-by-zero
            R_i = np.where(denom > 1e-9, num / denom, 0.0).astype(np.float32)

            # Accumulate for aggregation
            R_max = np.maximum(R_max, R_i)
            R_sum += R_i

        # --- Aggregate (same formula as RiskField.aggregate) ---
        R_sum_norm = R_sum / (1.0 + R_sum)   # soft normalise to [0,1)
        grid = (ALPHA * R_max + (1.0 - ALPHA) * R_sum_norm).astype(np.float32)

        return np.clip(grid, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _draw_isoline(
        self,
        grid:   'np.ndarray',
        CW:     int,
        CH:     int,
        level:  float,
        color:  str,
        label:  str,
        dash:   tuple = (4, 2),
    ) -> None:
        """
        Draw a contour line where grid == level using a simple marching-
        squares variant: scan horizontal pairs of cells and draw a segment
        where the value crosses the threshold.

        This directly shows where the FIS output equals R_LO / R_HI,
        i.e. the exact mode-switch boundaries of the supervisor.
        """
        rows, cols = grid.shape
        if rows < 2 or cols < 2:
            return

        cell_w = CW / cols
        cell_h = CH / rows

        # Horizontal crossings (between vertically adjacent cells)
        for r in range(rows - 1):
            for c in range(cols):
                v0 = float(grid[r,     c])
                v1 = float(grid[r + 1, c])
                if (v0 < level) != (v1 < level):
                    # Interpolate crossing row
                    t   = (level - v0) / (v1 - v0 + 1e-12)
                    y   = (r + t) * cell_h
                    x0  = c * cell_w
                    x1  = x0 + cell_w
                    self._canvas.create_line(x0, y, x1, y,
                                             fill=color, width=1,
                                             dash=dash)

        # Vertical crossings (between horizontally adjacent cells)
        for r in range(rows):
            for c in range(cols - 1):
                v0 = float(grid[r, c    ])
                v1 = float(grid[r, c + 1])
                if (v0 < level) != (v1 < level):
                    t   = (level - v0) / (v1 - v0 + 1e-12)
                    x   = (c + t) * cell_w
                    y0  = r * cell_h
                    y1  = y0 + cell_h
                    self._canvas.create_line(x, y0, x, y1,
                                             fill=color, width=1,
                                             dash=dash)

        # Label at top-left occurrence
        self._canvas.create_text(
            8, 8 + (0 if label == 'R_LO' else 18),
            text=f'── {label}={level}',
            fill=color, font=('Consolas', 8), anchor='nw',
        )

    # ------------------------------------------------------------------
    def _draw_polar(
        self,
        cx: float, cy: float,
        profile: List[float],
        r_inner: float = 36.0,
        r_outer: float = 60.0,
    ) -> None:
        """
        Draw the angular danger profile ρ(θ) as a filled polar ring
        around the ship position (cx, cy) on the canvas.
        """
        n = len(profile)
        for b, rho in enumerate(profile):
            angle_deg = b * 360.0 / n
            a0_rad    = math.radians(angle_deg - 0.5)
            a1_rad    = math.radians(angle_deg + 0.5)
            r_val     = r_inner + (r_outer - r_inner) * rho

            idx   = int(rho * 255)
            color = _rgb_hex(*_LUT[idx])

            # Draw a tiny trapezoid (two radial lines close enough to merge)
            x0i = cx + r_inner * math.cos(a0_rad)
            y0i = cy - r_inner * math.sin(a0_rad)
            x0o = cx + r_val   * math.cos(a0_rad)
            y0o = cy - r_val   * math.sin(a0_rad)
            x1o = cx + r_val   * math.cos(a1_rad)
            y1o = cy - r_val   * math.sin(a1_rad)
            x1i = cx + r_inner * math.cos(a1_rad)
            y1i = cy - r_inner * math.sin(a1_rad)

            self._canvas.create_polygon(
                x0i, y0i, x0o, y0o, x1o, y1o, x1i, y1i,
                fill=color, outline='', stipple='',
            )

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Blocking Tk main loop — called from the daemon thread."""
        self._root.mainloop()


# ============================================================================
# Public RiskHeatmap — controller-side interface
# ============================================================================

class RiskHeatmap:
    """
    Controller-side handle for the parallel Tk heatmap window.

    Instantiate once in FuzzyHybridController.__init__(), then call
    update() each frame (or every N frames for performance).

    The window opens in a daemon thread — it closes automatically when
    the game process exits.  No modification to the Kessler engine needed.

    Parameters
    ----------
    resolution   : px per grid cell (4 = good balance of detail vs speed)
    enabled      : set False to disable entirely (zero overhead)
    """

    def __init__(
        self,
        resolution: int  = 4,
        enabled:    bool = True,
    ) -> None:
        self.resolution = resolution
        self._enabled   = enabled
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._window: Optional[_HeatmapWindow] = None

        if enabled:
            self._start_thread()

    # ------------------------------------------------------------------
    def _start_thread(self) -> None:
        """Launch the Tk window in a daemon thread."""
        def _run() -> None:
            try:
                win = _HeatmapWindow(self._queue)
                self._window = win
                win.run()   # blocks until window closed
            except Exception as e:
                print(f'[RiskHeatmap] Tk thread error: {e}')

        t = threading.Thread(target=_run, name='HeatmapThread', daemon=True)
        t.start()

    # ------------------------------------------------------------------
    def update(
        self,
        ship_pos:        Tuple[float, float],
        ship_heading:    float,
        asteroid_risks:  list,
        map_size:        Tuple[float, float],
        r_global:        float,
        tau_min:         float,
        mode:            str,
        frame:           int,
        target          = None,
        angular_profile: Optional[List[float]] = None,
        mpc_trajectory:  Optional[List[Tuple[float, float]]] = None,
        solver_used:     str = '?',
    ) -> None:
        """
        Push one frame of data to the heatmap window.

        Called from the game thread — never blocks.  If the Tk thread
        hasn't consumed the previous frame yet, this frame is silently
        dropped (queue maxsize=1).
        """
        if not self._enabled:
            return

        data = dict(
            ship_pos        = ship_pos,
            ship_heading    = ship_heading,
            asteroid_risks  = asteroid_risks,
            map_size        = map_size,
            r_global        = r_global,
            tau_min         = tau_min,
            mode            = mode,
            frame           = frame,
            target          = target,
            angular_profile = angular_profile,
            mpc_trajectory  = mpc_trajectory,
            solver_used     = solver_used,
            resolution      = self.resolution,
        )
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            pass   # Tk thread is busy — drop frame, game never waits

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Signal the Tk window to close (optional — daemon exits anyway)."""
        if self._window is not None:
            try:
                self._window._root.quit()
            except Exception:
                pass