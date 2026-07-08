#!/usr/bin/env python3
"""
Collect gallery rollouts.

Splits (seeded, image-level, fixed for the whole experiment):
  train-images   60%  — on billboards in TRAIN collection (world model + head training)
  heldout-weak   20%  — on billboards in TRAIN collection (WM sees them) but excluded from head training
  heldout-strong 20%  — NEVER rendered during training; appear only in EVAL collection

Policy: 70% random walk / 30% scripted approach-and-stare (so billboards get foveated).
Layout resamples every 800 steps. Saves frames.npy (uint8 N,S,S,3), actions.npy (int8),
dom_idx.npy (int16, dominant billboard image or -1), dom_frac.npy (float16), splits.json.
"""
import argparse, json, os
import numpy as np
from datasets import load_dataset
from PIL import Image
from gallery_env import Gallery

POS = {"happy"}; NEG = {"anger", "contempt", "disgust", "fear", "sad"}


def load_pools(tex_size=64, seed=42):
    ds = load_dataset("FastJobs/Visual_Emotional_Analysis", split="train")
    names = ds.features["label"].names
    textures, labels, human = {}, {}, {}
    kept = []
    for i, ex in enumerate(ds):
        nm = names[ex["label"]]
        if nm in POS: lab = 1
        elif nm in NEG: lab = 0
        else: continue
        im = ex["image"].convert("RGB").resize((tex_size, tex_size), Image.BICUBIC)
        textures[i] = np.asarray(im, np.uint8)
        human[i] = lab
        kept.append(i)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(kept)
    n = len(perm)
    pools = {"train": perm[: int(0.6 * n)].tolist(),
             "weak": perm[int(0.6 * n): int(0.8 * n)].tolist(),
             "strong": perm[int(0.8 * n):].tolist()}
    return textures, human, pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--frames", type=int, default=100_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or f"data_{a.mode}"
    os.makedirs(out, exist_ok=True)

    textures, human, pools = load_pools()
    pool = pools["train"] + pools["weak"] if a.mode == "train" else pools["strong"]
    rng = np.random.RandomState(a.seed + (0 if a.mode == "train" else 777))
    env = Gallery(textures, pool, seed=a.seed + (0 if a.mode == "train" else 777))

    S = env.S
    frames = np.empty((a.frames, S, S, 3), np.uint8)
    acts = np.empty(a.frames, np.int8)
    dom_idx = np.full(a.frames, -1, np.int16)
    dom_frac = np.zeros(a.frames, np.float16)

    img, info = env.render()
    target, stare, mode_scripted = None, 0, False
    for t in range(a.frames):
        if t % 800 == 0 and t > 0:
            env.new_layout(); env.reset_agent(); target = None
        if t % 60 == 0:  # new behavior segment
            mode_scripted = rng.rand() < 0.30
            target = rng.randint(len(env.bb)) if (mode_scripted and env.bb) else None
            stare = 0
        if mode_scripted and target is not None:
            act = env.action_toward(target)
            if act is None:                      # arrived: stare, wobbling gaze slightly
                stare += 1
                act = [0, 1][stare % 2]
                if stare > 14: mode_scripted = False
        else:
            act = int(rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6]))
        frames[t] = img; acts[t] = act
        if info:
            k = max(info, key=info.get)
            dom_idx[t] = k; dom_frac[t] = info[k]
        img, info = env.step(act)
        if t % 20000 == 0: print(f"{a.mode} {t}/{a.frames}")

    np.save(f"{out}/frames.npy", frames); np.save(f"{out}/actions.npy", acts)
    np.save(f"{out}/dom_idx.npy", dom_idx); np.save(f"{out}/dom_frac.npy", dom_frac)
    with open(f"{out}/splits.json", "w") as f:
        json.dump({"pools": pools, "human": {str(k): v for k, v in human.items()}}, f)
    n15 = int(((dom_frac >= 0.15)).sum())
    print(f"saved {a.frames} frames to {out}/ ; frames with billboard >=15% area: {n15}")


if __name__ == "__main__":
    main()
