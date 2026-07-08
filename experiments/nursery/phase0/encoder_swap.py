#!/usr/bin/env python3
"""
Phase 0 — Step 0 pre-flight: the cheap resolution + encoder-swap checks (no body, no Dreamer).

Two questions, both laptop-runnable, before any Habitat/Dreamer build:
  (1) RESOLUTION CONTROL (G0-res). Does the CLIP grounding survive being downsampled to
      Dreamer-scale pixels (64/128)? Faces may die in the downsample before a world model ever
      sees them. This sets the *fair* ceiling AUC_CLIP@res (0.95 was on 224px natural images).
  (2) ENCODER SWAP (pre-check for G0b). Does approach/avoid survive off text-aligned CLIP onto a
      language-free self-supervised latent (DINOv2)? An early read on whether the grounding is
      parasitic on CLIP's text alignment, before we spend days training a real z_t.
      NOTE: the 0.95 headline is zero-shot *text-anchored* on CLIP; a non-text encoder has no text
      tower, so DINOv2 is trained-probe only — that is exactly what re-grounding onto z_t will be.

Metric mirrors ../amygdala_grounding/measure_grounding.py: zero-shot text-anchored AUC (CLIP only)
plus a linear-probe AUC (mean +/- sd over 10 random 70/30 splits) against human labels.
Dataset: FastJobs/Visual_Emotional_Analysis. POS={happy}, NEG={anger,contempt,disgust,fear,sad}.
"""
import os, json, numpy as np, torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image

POS = {"happy"}
NEG = {"anger", "contempt", "disgust", "fear", "sad"}
POS_ANCHORS = ["a joyful scene", "a happy smiling person", "something warm and pleasant", "something good"]
NEG_ANCHORS = ["a distressing scene", "a frightened or angry person", "something threatening or disgusting", "something bad"]
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def probe(feats, y, n=10):
    aucs, accs = [], []
    for s in range(n):
        rng = np.random.default_rng(s); idx = rng.permutation(len(y)); nt = int(0.3 * len(y)); te, tr = idx[:nt], idx[nt:]
        p = LogisticRegression(max_iter=2000).fit(feats[tr], y[tr]).predict_proba(feats[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); accs.append(accuracy_score(y[te], p > 0.5))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs))


def load_imgs():
    ds = load_dataset("FastJobs/Visual_Emotional_Analysis", split="train")
    names = ds.features["label"].names
    imgs, y = [], []
    for ex in ds:
        nm = names[ex["label"]]
        if nm in POS: y.append(1)
        elif nm in NEG: y.append(0)
        else: continue
        imgs.append(ex["image"].convert("RGB"))
    return imgs, np.array(y)


def save(res):
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/encoder_swap.json", "w"), indent=2)


def main():
    print("device:", DEV)
    imgs, y = load_imgs()
    print(f"kept {len(y)} imgs ({int(y.sum())} approach / {int((1 - y).sum())} avoid)\n")
    res = {"n": int(len(y)), "pos": int(y.sum()), "device": DEV, "clip": {}, "dinov2": None}

    # ---- CLIP: zero-shot + linear probe at three resolutions ----
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(DEV); tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        tp = model.encode_text(tok(POS_ANCHORS).to(DEV)); tp = (tp / tp.norm(dim=-1, keepdim=True)).cpu().numpy()
        tn = model.encode_text(tok(NEG_ANCHORS).to(DEV)); tn = (tn / tn.norm(dim=-1, keepdim=True)).cpu().numpy()
    for R in [224, 128, 64]:
        batch = []
        for im in imgs:
            lo = im.resize((R, R), Image.BILINEAR).resize((224, 224), Image.BILINEAR) if R < 224 else im
            batch.append(pre(lo))
        feats = []
        for i in range(0, len(batch), 64):
            x = torch.stack(batch[i:i + 64]).to(DEV)
            with torch.no_grad():
                f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
        feats = np.concatenate(feats)
        zs = feats @ tp.mean(0) - feats @ tn.mean(0)
        zsauc = float(roc_auc_score(y, zs))
        pa, ps, pacc = probe(feats, y)
        res["clip"][f"res{R}"] = {"zeroshot_auc": zsauc, "probe_auc": pa, "probe_sd": ps, "probe_acc": pacc}
        print(f"[CLIP @ {R:3d}px]  zero-shot AUC {zsauc:.3f}  |  linear-probe AUC {pa:.3f} +/- {ps:.3f}")
        save(res)

    # ---- DINOv2 (no text tower): linear probe only = the encoder-swap read ----
    try:
        import timm
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).eval().to(DEV)
        cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, 224, 224); tf = timm.data.create_transform(**cfg, is_training=False)
        batch = [tf(im) for im in imgs]
        feats = []
        for i in range(0, len(batch), 64):
            x = torch.stack(batch[i:i + 64]).to(DEV)
            with torch.no_grad():
                f = m(x)
            feats.append(f.cpu().numpy())
        feats = np.concatenate(feats)
        pa, ps, pacc = probe(feats, y)
        res["dinov2"] = {"model": "vit_small_patch14_dinov2", "probe_auc": pa, "probe_sd": ps, "probe_acc": pacc}
        print(f"[DINOv2, no text] linear-probe AUC {pa:.3f} +/- {ps:.3f}   (encoder-swap survival read)")
    except Exception as e:
        res["dinov2"] = {"error": f"{type(e).__name__}: {e}"}
        print("DINOv2 failed:", type(e).__name__, e)
    save(res)
    print("\nsaved -> results/encoder_swap.json")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
