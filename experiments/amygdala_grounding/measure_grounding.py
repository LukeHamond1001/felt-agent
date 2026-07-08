#!/usr/bin/env python3
"""
Measure the amygdala's PERCEPTUAL grounding.

Question: does a frozen CLIP representation carry human-labelled affect — can a **linear**
read of it separate approach from avoid on images it never trained on?

Scope, stated up front so the number is never oversold:
  - Measures the DESIGNED / KNOWN part — the grounded read-out generalizing.
  - Does NOT touch the developed value system ("the wall"), which needs scale.
  - A shipped datapoint, not validation of the thesis. And it is *facial* affect
    specifically (a prime amygdala stimulus), not general scenes — a subset.

Why a LINEAR probe (logistic regression), not a 10-50M head: a *linear* read separating the
classes is the stronger claim — it shows the affect is already present, linearly, in the
frozen features (nothing is "learned into" a big head). The shipped organ is a small net on
z_t; this probe is the minimal, hardest test of whether the grounding is really there.

Headline metric: **zero-shot** approach/avoid AUC (text-anchoring, NO labels at all) — the
prior being right *before* any experience. A trained linear probe is reported as a secondary
ceiling; note the frozen CLIP features do most of the work, so it is not the interesting number.

Dataset: FastJobs/Visual_Emotional_Analysis (800 images, 8 human-labelled emotions).
Approach = {happy}; avoid = {anger, contempt, disgust, fear, sad}. Neutral and *surprise*
are dropped: surprise is affectively ambiguous (a threat-orienting / salience signal, not a
clean approach), so it is excluded rather than forced onto either side.

Deps: torch, open_clip_torch, datasets, numpy, scikit-learn, matplotlib, pillow
  pip install torch open_clip_torch datasets numpy scikit-learn matplotlib pillow
Run:  python measure_grounding.py
"""
import os, numpy as np, torch, open_clip
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

POS_CLASSES = {"happy"}
NEG_CLASSES = {"anger", "contempt", "disgust", "fear", "sad"}   # neutral + surprise dropped
POS_ANCHORS = ["a joyful scene", "a happy smiling person", "something warm and pleasant", "something good"]
NEG_ANCHORS = ["a distressing scene", "a frightened or angry person", "something threatening or disgusting", "something bad"]


def main(out="results", seed=0):
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)

    ds = load_dataset("FastJobs/Visual_Emotional_Analysis", split="train")
    names = ds.features["label"].names
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(dev)
    tok = open_clip.get_tokenizer("ViT-B-32")

    imgs, y = [], []
    for ex in ds:
        nm = names[ex["label"]]
        if nm in POS_CLASSES: y.append(1)
        elif nm in NEG_CLASSES: y.append(0)
        else: continue
        imgs.append(pre(ex["image"].convert("RGB")))
    y = np.array(y)
    print(f"kept {len(y)} images ({y.sum()} approach / {(1-y).sum()} avoid)")

    feats = []
    for i in range(0, len(imgs), 64):
        x = torch.stack(imgs[i:i+64]).to(dev)
        with torch.no_grad():
            f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    feats = np.concatenate(feats)

    # HEADLINE — zero-shot grounded read, no labels at all
    with torch.no_grad():
        tp = model.encode_text(tok(POS_ANCHORS).to(dev)); tp = (tp / tp.norm(dim=-1, keepdim=True)).cpu().numpy()
        tn = model.encode_text(tok(NEG_ANCHORS).to(dev)); tn = (tn / tn.norm(dim=-1, keepdim=True)).cpu().numpy()
    zs = feats @ tp.mean(0) - feats @ tn.mean(0)
    zs_auc = roc_auc_score(y, zs)
    print(f"\n[headline] ZERO-SHOT grounded read (no labels):  approach/avoid AUC = {zs_auc:.3f}  (chance 0.50)")

    # secondary ceiling — a LINEAR probe, mean over 10 random 70/30 splits (features do the work)
    aucs, accs = [], []
    for s in range(10):
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y)); nt = int(0.3 * len(y)); te, tr = idx[:nt], idx[nt:]
        p = LogisticRegression(max_iter=1000).fit(feats[tr], y[tr]).predict_proba(feats[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); accs.append(accuracy_score(y[te], p > 0.5))
    base = max(y.mean(), 1 - y.mean())
    print(f"[secondary] linear probe, held-out AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
          f"acc = {np.mean(accs):.3f} (10 random 70/30 splits; base rate {base:.2f})")

    os.makedirs(out, exist_ok=True)
    import json
    with open(os.path.join(out, "grounding.json"), "w") as f:
        json.dump({
            "zero_shot_auc": round(float(zs_auc), 4),
            "probe_auc_mean": round(float(np.mean(aucs)), 4), "probe_auc_std": round(float(np.std(aucs)), 4),
            "probe_acc_mean": round(float(np.mean(accs)), 4), "base_rate": round(float(base), 4),
            "n_images": int(len(y)), "n_approach": int(y.sum()),
            "model": "CLIP ViT-B-32 laion2b_s34b_b79k", "dataset": "FastJobs/Visual_Emotional_Analysis",
        }, f, indent=2)
    print(f"metrics -> {out}/grounding.json")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist([zs[y == 1], zs[y == 0]], bins=25, alpha=0.8, label=["approach (happy)", "avoid (negative)"])
    ax.set_title(f"zero-shot grounded read, no labels  —  approach/avoid AUC = {zs_auc:.2f}")
    ax.set_xlabel("grounded valence score"); ax.set_ylabel("images"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "grounding.png"), dpi=140)
    print(f"\nfigure -> {out}/grounding.png")


if __name__ == "__main__":
    main()
