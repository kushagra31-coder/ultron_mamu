"""Ultron floating UI — animated geodesic icosphere (red & silver, like the reference).

Geometry: icosahedron subdivided 2× → 162 vertices, 480 edges, all normalized to unit sphere.
Rendering: perspective projection, painter's-algorithm depth sort, per-vertex glow simulation.

State colours (matching the reference image aesthetic):
  idle      — dark blood-red edges, dim silver nodes, slow rotation
  wake      — pulsing mid-red edges, silver nodes with subtle scale breath
  listening — bright vivid red, stark white glow nodes, fast spin
  speaking  — intense scarlet, pulsing white nodes synced to a beat
"""
from __future__ import annotations

import math
import os
import random
import requests
import threading
import tkinter as tk
from typing import Callable


# ---------------------------------------------------------------------------
# Icosphere geometry builder
# ---------------------------------------------------------------------------

def _normalize(v):
    x, y, z = v
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n)

def _midpoint(v1, v2, cache, verts):
    key = (min(v1, v2), max(v1, v2))
    if key in cache:
        return cache[key]
    p1 = verts[v1]
    p2 = verts[v2]
    mid = _normalize(((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2))
    idx = len(verts)
    verts.append(mid)
    cache[key] = idx
    return idx

def build_icosphere(subdivisions=2):
    """Return (vertices, edges) as lists. Vertices are unit-sphere (x,y,z) tuples."""
    t = (1 + math.sqrt(5)) / 2
    verts = [
        _normalize((-1,  t,  0)), _normalize(( 1,  t,  0)),
        _normalize((-1, -t,  0)), _normalize(( 1, -t,  0)),
        _normalize(( 0, -1,  t)), _normalize(( 0,  1,  t)),
        _normalize(( 0, -1, -t)), _normalize(( 0,  1, -t)),
        _normalize(( t,  0, -1)), _normalize(( t,  0,  1)),
        _normalize((-t,  0, -1)), _normalize((-t,  0,  1)),
    ]
    faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    for _ in range(subdivisions):
        new_faces = []
        cache = {}
        for f in faces:
            a,b,c = f
            ab = _midpoint(a, b, cache, verts)
            bc = _midpoint(b, c, cache, verts)
            ca = _midpoint(c, a, cache, verts)
            new_faces += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        faces = new_faces

    edge_set = set()
    for f in faces:
        a,b,c = f
        for e in ((min(a,b),max(a,b)), (min(b,c),max(b,c)), (min(a,c),max(a,c))):
            edge_set.add(e)

    return verts, list(edge_set)


# ---------------------------------------------------------------------------
# Main UI class
# ---------------------------------------------------------------------------

class UltronUI:
    SIZE = 220          # window / canvas px
    SPHERE_R = 88.0     # sphere radius in canvas px (before perspective)
    FOV_DIST = 350.0    # perspective camera distance

    # Colour themes per state  {edge_front, edge_back, node_bright, node_dim, glow}
    THEMES = {
        "idle":      ("#8B0000", "#2A0000", "#C0C0C0", "#555555", "#999999"),
        "wake":      ("#CC1111", "#440000", "#D8D8D8", "#888888", "#BBBBBB"),
        "listening": ("#FF3030", "#660000", "#FFFFFF", "#BBBBBB", "#FFFFFF"),
        "speaking":  ("#FF2020", "#880000", "#FFFFFF", "#CCCCCC", "#FFFFFF"),
    }

    def __init__(self,
                 toggle_callback: Callable,
                 get_recording_state_callback: Callable[[], bool],
                 get_wake_active_callback: Callable[[], bool] | None = None):

        self.toggle = toggle_callback
        self.get_recording = get_recording_state_callback
        self.get_wake = get_wake_active_callback or (lambda: True)

        self.verts, self.edges = build_icosphere(subdivisions=1)

        # ---- Tkinter boilerplate ----
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        TC = "#010101"
        self.root.config(bg=TC)
        self.root.attributes("-transparentcolor", TC)

        self.root.update_idletasks()
        ws = self.root.winfo_screenwidth() or 1920
        self.root.geometry(f"{self.SIZE}x{self.SIZE}+{ws - self.SIZE - 40}+40")

        self.canvas = tk.Canvas(self.root, width=self.SIZE, height=self.SIZE,
                                bg=TC, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",        self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_release)
        self._drag = {"x": 0, "y": 0, "moved": False}

        # Animation state
        self.state = "idle"
        self.ax = 0.0
        self.ay = 0.0
        self.pulse = 0.0

        self._poll_state()
        self._frame()

    # ---------------------------------------------------------------- drag
    def _on_press(self, e):
        self._drag = {"x": e.x, "y": e.y, "moved": False}

    def _on_drag(self, e):
        if abs(e.x - self._drag["x"]) > 2 or abs(e.y - self._drag["y"]) > 2:
            self._drag["moved"] = True
        nx = self.root.winfo_x() - self._drag["x"] + e.x
        ny = self.root.winfo_y() - self._drag["y"] + e.y
        self.root.geometry(f"+{nx}+{ny}")

    def _on_release(self, e):
        if not self._drag["moved"]:
            self.toggle()

    # --------------------------------------------------------- state poll
    def _poll_state(self):
        is_rec = self.get_recording()
        is_spk = False
        if not is_rec:
            try:
                r = requests.get("http://127.0.0.1:8000/status", timeout=0.15)
                is_spk = r.json().get("speaking", False)
            except Exception:
                pass

        if is_rec:
            self.state = "listening"
        elif is_spk:
            self.state = "speaking"
        elif self.get_wake():
            self.state = "wake"
        else:
            self.state = "idle"

        self.root.after(300, self._poll_state)

    # ----------------------------------------------------------- render
    def _frame(self):
        S = self.state

        # --- rotation speed / pulse rate per state ---
        if S == "listening":
            self.ax += 0.055;  self.ay += 0.080;  self.pulse += 0.25
        elif S == "speaking":
            self.ax += 0.035;  self.ay += 0.050;  self.pulse += 0.35
        elif S == "wake":
            self.pulse += 0.10
        else:  # idle
            self.pulse += 0.06

        pulse_amp = 0.06 if S in ("speaking", "wake") else 0.02
        scale = self.SPHERE_R * (1.0 + math.sin(self.pulse) * pulse_amp)

        # --- projection ---
        cx = cy = self.SIZE / 2
        ca_x, sa_x = math.cos(self.ax), math.sin(self.ax)
        ca_y, sa_y = math.cos(self.ay), math.sin(self.ay)

        projected = []
        for vx, vy, vz in self.verts:
            # Rotate X
            ry = vy * ca_x - vz * sa_x
            rz = vy * sa_x + vz * ca_x
            vy, vz = ry, rz
            # Rotate Y
            rx = vx * ca_y + vz * sa_y
            rz = -vx * sa_y + vz * ca_y
            vx, vz = rx, rz
            # Perspective divide
            d = self.FOV_DIST
            fac = d / (d + vz * scale)
            px = cx + vx * scale * fac
            py = cy + vy * scale * fac
            projected.append((px, py, vz, fac))   # fac: 1=front, <1=back

        # --- theme ---
        t = self.THEMES[S]
        ec_front, ec_back, nc_bright, nc_dim, glow_c = t

        # --- painter's: sort edges back→front by avg z ---
        edge_draw = []
        for i, j in self.edges:
            avg_z = projected[i][2] + projected[j][2]
            edge_draw.append((avg_z, i, j))
        edge_draw.sort(key=lambda x: x[0])  # lowest z = furthest = draw first
        node_z = sorted(range(len(projected)), key=lambda k: projected[k][2])

        self.canvas.delete("all")

        # Draw faint outer glow ring when active
        if S != "idle":
            r = scale + 8 + math.sin(self.pulse) * 5
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    outline=glow_c, width=1)
            r2 = r + 4
            self.canvas.create_oval(cx-r2, cy-r2, cx+r2, cy+r2,
                                    outline="#330000", width=1)

        def draw_edge(avg_z, i, j):
            p1, p2 = projected[i], projected[j]
            t_val = max(0.0, min(1.0, (avg_z + 1.0) / 2.0))
            color = self._lerp_hex(ec_back, ec_front, t_val)
            width = max(1, round(1 + t_val * 0.8))
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=width)

        def draw_node(k):
            px, py, pz, fac = projected[k]
            t_val = max(0.0, min(1.0, (pz + 1.0) / 2.0))
            color = self._lerp_hex(nc_dim, nc_bright, t_val)
            base_r = 1.5 + t_val * 2.5 + (math.sin(self.pulse * 1.3 + pz) * 0.6 if S in ("listening","speaking") else 0)
            r = base_r * fac
            if t_val > 0.4:
                for gr_mul, ga in [(2.8, "#220000"), (1.8, "#440000"), (1.0, color)]:
                    gr = r * gr_mul
                    self.canvas.create_oval(px-gr, py-gr, px+gr, py+gr, fill=ga, outline="")
            else:
                self.canvas.create_oval(px-r, py-r, px+r, py+r, fill=color, outline="")

        # 1. Back edges & nodes (Z < 0)
        for e in edge_draw:
            if e[0] < 0: draw_edge(*e)
        for k in node_z:
            if projected[k][2] < 0: draw_node(k)

        # 2. Dense Core (Z = 0)
        core_scale = scale * 0.40
        for r_mult, c_hex in [(1.0, ec_back), (0.7, ec_front), (0.3, nc_bright)]:
            cr = core_scale * r_mult + (math.sin(self.pulse * 2.0) * 2.0)
            self.canvas.create_oval(cx-cr, cy-cr, cx+cr, cy+cr, fill=c_hex, outline="")
        
        # Text embedded in the core (rendered before front wireframe)
        self.canvas.create_text(cx, cy, text="JARVIS", fill="#FFFFFF", font=("Consolas", 10, "bold"))

        # 3. Front edges & nodes (Z >= 0)
        for e in edge_draw:
            if e[0] >= 0: draw_edge(*e)
        for k in node_z:
            if projected[k][2] >= 0: draw_node(k)

        self.root.after(30, self._frame)

    # --------------------------------------------------------- helpers
    @staticmethod
    def _lerp_hex(c1: str, c2: str, t: float) -> str:
        """Linearly interpolate between two hex colours."""
        r1, g1, b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
        r2, g2, b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
        r = int(r1 + (r2-r1)*t)
        g = int(g1 + (g2-g1)*t)
        b = int(b1 + (b2-b1)*t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def run(self):
        self.root.mainloop()


def start_ui(toggle_callback, get_recording_state_callback, get_wake_active_callback=None):
    app = UltronUI(toggle_callback, get_recording_state_callback, get_wake_active_callback)
    app.run()
