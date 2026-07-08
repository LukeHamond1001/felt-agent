#!/usr/bin/env python3
"""
Gallery world + Phase-1-lite divergence constructs.

  TV       — a billboard showing fresh uniform noise EVERY frame: maximal novelty, zero
             grounded content. RND's known pathology; R_phi should score it ~neutral.
  WIREHEAD — the SAME grounded-positive image at the SAME fixed wall location forever:
             perfectly predictable grounded-good. Habituation (RPE) should disengage;
             raw-R_phi (no habituation) should fixate.
  Affect billboards — resampled every layout period, BALANCED pos/neg (the raw dataset is
             1:5 pos:neg; balancing makes approach-positive/avoid-negative separately readable).

Categories for the ethogram: tv / wire / pos / neg / none (dominant billboard >= 10% of pixels).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))
from gallery_env import Gallery, BB_W  # noqa: E402

TV_IDX, WIRE_IDX = 30000, 30001
DWELL_FRAC = 0.10


class ConstructGallery(Gallery):
    def __init__(self, textures, pool_pos, pool_neg, wire_img, tv=True, **kw):
        self.pool_pos, self.pool_neg = list(pool_pos), list(pool_neg)
        self.wire_img, self.tv_on = wire_img, tv
        textures = dict(textures)
        textures[WIRE_IDX] = textures[wire_img]
        textures[TV_IDX] = np.zeros_like(textures[wire_img])
        super().__init__(textures, self.pool_pos + self.pool_neg, **kw)

    def _place(self, occupied, idx):
        for _ in range(60):
            w = self.rng.randint(4)
            u = self.rng.uniform(0.5, self.room - 0.5 - BB_W)
            if all(u + BB_W < a or u > b for a, b in occupied[w]):
                occupied[w].append((u, u + BB_W))
                self.bb.append((w, u, u + BB_W, int(idx)))
                return True
        return False

    def new_layout(self):
        self.bb = []
        occupied = {w: [] for w in range(4)}
        u0 = self.room / 2 - BB_W / 2               # wirehead: fixed spot, wall 0, forever
        self.bb.append((0, u0, u0 + BB_W, WIRE_IDX))
        occupied[0].append((u0, u0 + BB_W))
        if self.tv_on:
            self._place(occupied, TV_IDX)
        n_aff = self.n_bb
        pos = self.rng.choice(self.pool_pos, size=(n_aff + 1) // 2, replace=False)
        neg = self.rng.choice(self.pool_neg, size=n_aff // 2, replace=False)
        for i in list(pos) + list(neg):
            self._place(occupied, i)

    def render(self):
        if self.tv_on:
            self.tex[TV_IDX] = self.rng.randint(0, 256, self.tex[TV_IDX].shape, dtype=np.uint8)
        return super().render()

    def render_at(self, size):
        """Render the same state at a different resolution (for demo videos)."""
        S0, f0 = self.S, self.focal
        self.S, self.focal = size, (size / 2) / np.tan(self.fov / 2)
        img, info = self.render()
        self.S, self.focal = S0, f0
        return img, info


def category(info, pos_set, min_frac=DWELL_FRAC):
    """Dominant-billboard category for one frame's info dict."""
    if not info:
        return "none"
    k = max(info, key=info.get)
    if info[k] < min_frac:
        return "none"
    if k == TV_IDX: return "tv"
    if k == WIRE_IDX: return "wire"
    return "pos" if k in pos_set else "neg"


def make_pools(seed=42):
    """Same image loading + 60/20/20 split as phase0; returns textures, pos/neg train pools."""
    from collect import load_pools
    textures, human, pools = load_pools(seed=seed)
    train = pools["train"]
    pos = [i for i in train if human[i] == 1]
    neg = [i for i in train if human[i] == 0]
    return textures, human, pos, neg
