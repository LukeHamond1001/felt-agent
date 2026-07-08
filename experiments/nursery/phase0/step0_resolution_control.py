#!/usr/bin/env python3
"""
Phase 0, Step 0 — the resolution control (PLAN.md).

Question: does the frozen-CLIP affect read (AUC 0.95 @ native res) survive the pixel budgets
a world model actually consumes (64x64, 128x128)? This sets the FAIR CEILING (AUC_CLIP@res)
for every later gate: any AUC loss downstream decomposes into resolution loss (measured here)
vs latent-compression loss (measured after re-grounding). Without this number the encoder-swap
result is uninterpretable.

Protocol: identical to ../../amygdala_grounding/measure_grounding.py (same dataset, classes,
anchors, zero-shot read, linear-probe ceiling) except each image is first downsampled to
res x res (bicubic — approximates a renderer's output) before CLIP's own preprocessing
upsamples it back. res=224 reproduces the original 0.95 as a sanity check.

Gate G0-res: AUC_CLIP@res >= 0.85 at 64 or 128 → that resolution regime is viable.
"""
import json, os
import numpy as np, torch, open_clip
from PIL import Image
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

POS_CLASSES = {"happy"}
NEG_CLASSES = {"anger", "contempt", "disgust", "fear", "sad"}
POS_ANCHORS = ["a joyful scene", "a happy smiling person", "something warm and pleasant", "something good"]
NEG_ANCHORS = ["a distressing scene", "a frightened or angry person", "something threatening or disgusting", "something bad"]
RESOLUTIONS = [224, 128, 96, 64, 48, 32]


def encode_images(model, pre, pil_imgs, res, dev):
    feats = []
    batch = []
    for im in pil_imgs:
        if res < 224:
            im = im.resize((res, res), Image.BICUBIC)
        batch.append(pre(im))
        if len(batch) == 64:
            with torch.no_grad():
                f = model.encode_image(torch.stack(batch).to(dev))
                feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
            batch = []
    if batch:
        with torch.no_grad():
            f = model.encode_image(torch.stack(batch).to(dev))
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(feats)


def main(out="results"):
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)

    ds = load_dataset("FastJobs/Visual_Emotional_Analysis", split="train")
    names = ds.features["label"].names
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(dev)
    tok = open_clip.get_tokenizer("ViT-B-32")

    pil_imgs, y = [], []
    for ex in ds:
        nm = names[ex["label"]]
        if nm in POS_CLASSES: y.append(1)
        elif nm in NEG_CLASSES: y.append(0)
        else: continue
        pil_imgs.append(ex["image"].convert("RGB"))
    y = np.array(y)
    print(f"kept {len(y)} images ({y.sum()} approach / {(1 - y).sum()} avoid)")

    with torch.no_grad():
        tp = model.encode_text(tok(POS_ANCHORS).to(dev)); tp = (tp / tp.norm(dim=-1, keepdim=True)).cpu().numpy()
        tn = model.encode_text(tok(NEG_ANCHORS).to(dev)); tn = (tn / tn.norm(dim=-1, keepdim=True)).cpu().numpy()
    anchor = tp.mean(0) - tn.mean(0)

    results = {}
    rng = np.random.RandomState(0)
    for res in RESOLUTIONS:
        feats = encode_images(model, pre, pil_imgs, res, dev)
        zs_auc = roc_auc_score(y, feats @ anchor)
        # linear-probe ceiling, 10 random 70/30 splits (same as original protocol)
        probe_aucs = []
        for _ in range(10):
            idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
            tr, te = idx[:cut], idx[cut:]
            clf = LogisticRegression(max_iter=2000).fit(feats[tr], y[tr])
            probe_aucs.append(roc_auc_score(y[te], clf.decision_function(feats[te])))
        results[res] = {"zero_shot_auc": round(float(zs_auc), 4),
                        "probe_auc_mean": round(float(np.mean(probe_aucs)), 4),
                        "probe_auc_std": round(float(np.std(probe_aucs)), 4)}
        print(f"res {res:>3}px  zero-shot AUC {zs_auc:.3f}   probe AUC {np.mean(probe_aucs):.3f} ± {np.std(probe_aucs):.3f}")

    with open(os.path.join(out, "step0_resolution_control.json"), "w") as f:
        json.dump({"n_images": int(len(y)), "resolutions": results}, f, indent=2)

    g64 = results[64]["zero_shot_auc"] >= 0.85
    g128 = results[128]["zero_shot_auc"] >= 0.85
    print("\nGate G0-res (zero-shot >= 0.85):  64px", "PASS" if g64 else "FAIL",
          "  128px", "PASS" if g128 else "FAIL")
    print("Fair ceiling AUC_CLIP@res:", results[64 if g64 else 128]["zero_shot_auc"] if (g64 or g128) else "NOT MET — see PLAN.md escalation")


if __name__ == "__main__":
    main()
