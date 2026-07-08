#!/usr/bin/env python3
"""
Gallery world — minimal 3D room with affect-image billboards, raycast-rendered in numpy.

Replaces MiniWorld (PLAN.md) to avoid OpenGL/headless install risk: same function — an
egomotion pixel stream in which controlled affect content appears on walls — with fully
deterministic, dependency-free rendering. Wolfenstein-style column raycaster, 64x64 RGB.

Actions: 0 = turn left 15 deg, 1 = turn right 15 deg, 2 = forward 0.25.
Per-frame info: pixel-area fraction of every visible billboard (ground truth for probes).
"""
import numpy as np

WALL_H = 2.0          # world wall height; camera at mid-height
BB_V0, BB_V1 = 0.18, 0.82   # billboard vertical span on the wall (v in [0,1])
BB_W = 2.2            # billboard width in wall-u units


class Gallery:
    def __init__(self, textures, pool, size=64, room=8.0, n_billboards=5, seed=0, fov_deg=70.0):
        self.tex = textures            # dict img_idx -> (T,T,3) uint8
        self.pool = list(pool)         # image indices allowed on billboards
        self.S, self.room, self.n_bb = size, room, n_billboards
        self.fov = np.deg2rad(fov_deg)
        self.rng = np.random.RandomState(seed)
        self.focal = (size / 2) / np.tan(self.fov / 2)
        self.new_layout()
        self.reset_agent()

    # ---- layout ----
    def new_layout(self):
        self.bb = []  # (wall, u0, u1, img_idx)
        occupied = {w: [] for w in range(4)}
        idxs = self.rng.choice(self.pool, size=min(self.n_bb, len(self.pool)), replace=False)
        for img in idxs:
            for _ in range(40):
                w = self.rng.randint(4)
                u0 = self.rng.uniform(0.5, self.room - 0.5 - BB_W)
                if all(u0 + BB_W < a or u0 > b for a, b in occupied[w]):
                    occupied[w].append((u0, u0 + BB_W))
                    self.bb.append((w, u0, u0 + BB_W, int(img)))
                    break

    def reset_agent(self):
        self.x = self.rng.uniform(2, self.room - 2)
        self.y = self.rng.uniform(2, self.room - 2)
        self.th = self.rng.uniform(0, 2 * np.pi)

    # ---- geometry ----
    def _cast(self, ang):
        """Return (dist, wall, u) for a ray at absolute angle ang."""
        dx, dy = np.cos(ang), np.sin(ang)
        best = (np.inf, -1, 0.0)
        R = self.room
        cands = []
        if dy < -1e-9: cands.append(((0 - self.y) / dy, 0))
        if dx > 1e-9:  cands.append(((R - self.x) / dx, 1))
        if dy > 1e-9:  cands.append(((R - self.y) / dy, 2))
        if dx < -1e-9: cands.append(((0 - self.x) / dx, 3))
        for t, w in cands:
            if t <= 0 or t >= best[0]: continue
            hx, hy = self.x + t * dx, self.y + t * dy
            if w in (0, 2):
                if -1e-6 <= hx <= R + 1e-6:
                    u = hx if w == 0 else R - hx
                    best = (t, w, u)
            else:
                if -1e-6 <= hy <= R + 1e-6:
                    u = hy if w == 1 else R - hy
                    best = (t, w, u)
        return best

    def render(self):
        S = self.S
        img = np.empty((S, S, 3), np.uint8)
        img[: S // 2] = (205, 205, 210)      # ceiling
        img[S // 2:] = (70, 68, 66)          # floor
        counts = {}
        rows = np.arange(S)
        wall_base = [(150, 140, 130), (140, 150, 135), (135, 140, 150), (150, 148, 128)]
        for c in range(S):
            off = np.arctan((c - S / 2 + 0.5) / self.focal)
            dist, w, u = self._cast(self.th + off)
            if not np.isfinite(dist): continue
            d = max(dist * np.cos(off), 1e-3)
            half = (WALL_H / 2) * self.focal / d
            top, bot = S / 2 - half, S / 2 + half
            r0, r1 = max(int(np.ceil(top)), 0), min(int(np.floor(bot)), S - 1)
            if r1 < r0: continue
            v = (rows[r0:r1 + 1] - top) / max(bot - top, 1e-6)   # 0 top .. 1 bottom of wall
            shade = 1.0 / (1.0 + 0.12 * d)
            col = np.empty((r1 - r0 + 1, 3), np.float32)
            col[:] = np.array(wall_base[w], np.float32) * shade
            hit = None
            for (bw, u0, u1, ii) in self.bb:
                if bw == w and u0 <= u <= u1:
                    hit = (u0, u1, ii); break
            if hit is not None:
                u0, u1, ii = hit
                tex = self.tex[ii]; T = tex.shape[0]
                inb = (v >= BB_V0) & (v <= BB_V1)
                if inb.any():
                    tx = min(int((u - u0) / (u1 - u0) * T), T - 1)
                    ty = ((v[inb] - BB_V0) / (BB_V1 - BB_V0) * (T - 1)).astype(int)
                    col[inb] = tex[ty, tx].astype(np.float32) * (0.55 + 0.45 * shade)
                    counts[ii] = counts.get(ii, 0) + int(inb.sum())
            img[r0:r1 + 1, c] = np.clip(col, 0, 255).astype(np.uint8)
        total = S * S
        return img, {i: n / total for i, n in counts.items()}

    def step(self, a):
        if a == 0: self.th -= np.deg2rad(15)
        elif a == 1: self.th += np.deg2rad(15)
        elif a == 2:
            nx = self.x + 0.25 * np.cos(self.th)
            ny = self.y + 0.25 * np.sin(self.th)
            self.x = float(np.clip(nx, 0.4, self.room - 0.4))
            self.y = float(np.clip(ny, 0.4, self.room - 0.4))
        return self.render()

    # ---- scripted look-at-billboard behavior ----
    def bb_center(self, k):
        w, u0, u1, _ = self.bb[k]
        u = (u0 + u1) / 2; R = self.room
        return {0: (u, 0.0), 1: (R, u), 2: (R - u, R), 3: (0.0, R - u)}[w]

    def action_toward(self, k):
        cx, cy = self.bb_center(k)
        want = np.arctan2(cy - self.y, cx - self.x)
        dth = (want - self.th + np.pi) % (2 * np.pi) - np.pi
        if abs(dth) > np.deg2rad(12): return 1 if dth > 0 else 0
        if np.hypot(cx - self.x, cy - self.y) > 1.6: return 2
        return None  # arrived: stare
