#!/usr/bin/env python3
"""
Evaluate trained policies (greedy) on fresh galleries + render the side-by-side demo GIF.

Outputs: results/phase1_lite.json (dwell table, mean±std over seeds),
         results/dwell_over_training.png, results/side_by_side.gif
"""
import glob, json, os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))
from train_wm import WM  # noqa: E402
from env_constructs import ConstructGallery, category, make_pools  # noqa: E402
from train_agent import Policy  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", 4000))
CONDS = os.environ.get("CONDS", "grounded,rnd,nohab").split(",")
GIF_CONDS = os.environ.get("GIF_CONDS", "grounded,rnd").split(",")
OUT_TAG = os.environ.get("OUT_TAG", "phase1_lite")


def load_policy(pth):
    ck = torch.load(pth, map_location=DEV)
    if isinstance(ck, dict) and "state" in ck:
        pol = Policy(pixel=ck.get("pixel", False))
        pol.load_state_dict(ck["state"])
    else:                       # legacy phase-1-lite checkpoint: frozen-feature policy
        pol = Policy(pixel=False)
        pol.load_state_dict(ck)
    return pol.to(DEV).eval()


def load_enc():
    wm = WM("recon")
    wm.load_state_dict(torch.load("../phase0/ckpt/wm_recon.pt", map_location=DEV)["state"])
    enc = wm.enc.to(DEV).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    return enc


def rollout(pol, enc, env, pos_set, steps, record_at=0):
    cats = {"tv": 0, "wire": 0, "pos": 0, "neg": 0, "none": 0}
    frames = []
    img, info = env.render()
    prev = torch.zeros(1, 3, device=DEV)
    for t in range(steps):
        if t > 0 and t % 800 == 0:
            env.new_layout(); env.reset_agent()
        x = torch.as_tensor(img[None], device=DEV).float().permute(0, 3, 1, 2) / 255.0 - 0.5
        with torch.no_grad():
            ob = x if pol.pixel else enc(x)
            logits, _ = pol(ob, prev)
        a = int(logits.argmax(-1))
        img, info = env.step(a)
        cats[category(info, pos_set)] += 1
        prev = F.one_hot(torch.tensor([a], device=DEV), 3).float()
        if record_at and t < record_at:
            hi, _ = env.render_at(192)
            frames.append(hi)
    tot = sum(cats.values())
    return {k: v / tot for k, v in cats.items()}, frames


def main():
    textures, human, pos, neg = make_pools()
    prep = torch.load("prep.pt", map_location="cpu")
    enc = load_enc()
    pos_set = set(pos)
    os.makedirs("results", exist_ok=True)

    table, demo = {}, {}
    for cond in CONDS:
        rows = []
        for pth in sorted(glob.glob(f"policies/{cond}_s*.pt")):
            pol = load_policy(pth)
            per_seed = []
            for es in range(3):
                env = ConstructGallery(textures, pos, neg, prep["wire_img"], seed=9000 + es)
                fr, frames = rollout(pol, enc, env, pos_set, EVAL_STEPS,
                                     record_at=400 if (cond in GIF_CONDS and "s0" in pth and es == 0) else 0)
                per_seed.append(fr)
                if frames: demo[cond] = frames
            rows.append({k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]})
        if rows:
            table[cond] = {k: [round(float(np.mean([r[k] for r in rows])), 4),
                               round(float(np.std([r[k] for r in rows])), 4)] for k in rows[0]}
            print(cond, table[cond])

    with open(f"results/{OUT_TAG}.json", "w") as f:
        json.dump(table, f, indent=2)

    # dwell-over-training figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, cond in zip(axes, CONDS):
        for cat_name, color in [("tv", "#b5544d"), ("wire", "#c9a227"), ("pos", "#5aa469"), ("neg", "#7a6ea8")]:
            curves = []
            for pth in sorted(glob.glob(f"logs/{cond}_s*.jsonl")):
                vals = [json.loads(l)[cat_name] for l in open(pth)]
                curves.append(vals)
            if curves:
                n = min(map(len, curves))
                m = np.mean([c[:n] for c in curves], 0)
                ax.plot(np.arange(n), m, label=cat_name, color=color)
        ax.set_title(cond); ax.set_xlabel("iteration")
    axes[0].set_ylabel("dwell fraction"); axes[0].legend(fontsize=8)
    plt.suptitle(f"{OUT_TAG}: attention allocation over training")
    plt.tight_layout(); plt.savefig(f"results/{OUT_TAG}_dwell_over_training.png", dpi=150)

    # side-by-side GIF
    g0, g1 = GIF_CONDS[0], GIF_CONDS[-1]
    if g0 in demo and g1 in demo:
        gifs = []
        for f1, f2 in zip(demo[g0], demo[g1]):
            canvas = Image.new("RGB", (192 * 2 + 12, 192 + 22), (18, 18, 18))
            canvas.paste(Image.fromarray(f1), (0, 22))
            canvas.paste(Image.fromarray(f2), (192 + 12, 22))
            d = ImageDraw.Draw(canvas)
            d.text((4, 4), g0, fill=(120, 220, 140))
            d.text((208, 4), g1, fill=(230, 130, 120))
            gifs.append(canvas)
        gifs[0].save(f"results/{OUT_TAG}_side_by_side.gif", save_all=True,
                     append_images=gifs[1:], duration=90, loop=0)
        print(f"wrote results/{OUT_TAG}_side_by_side.gif")
    print(f"wrote results/{OUT_TAG}.json + {OUT_TAG}_dwell_over_training.png")


if __name__ == "__main__":
    main()
