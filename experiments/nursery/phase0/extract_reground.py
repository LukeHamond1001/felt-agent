#!/usr/bin/env python3
"""
Re-ground R_phi onto the world-model latent and measure survival (PLAN.md steps 3-5).

For each checkpoint (recon / jepa):
  z_enc  = online encoder embedding (pre-RSSM)         } filtered over the rollout stream
  z_full = deter (+) posterior mean (what R_phi reads)  }
  Distilled head (shipped mechanism): MLP on z -> frozen-CLIP ZERO-SHOT valence of the frame
    (no human labels in training), trained on train-images frames only. 3 seeds.
  Linear probe (diagnostic): logistic regression z -> human label of the dominant billboard.

Evaluation (image-level: mean score over a billboard's >=15%-area frames -> one score/image):
  heldout-weak   : images the WM saw, head never saw   (tail slice of data_train)
  heldout-strong : images NOTHING ever saw             (data_eval)  <- primary
Reference ceilings: CLIP zero-shot on the same rendered 64px frames (render-domain ceiling).
"""
import json, os
import numpy as np, torch, open_clip
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from train_wm import WM, EMB, DETER, STOCH

POS_ANCHORS = ["a joyful scene", "a happy smiling person", "something warm and pleasant", "something good"]
NEG_ANCHORS = ["a distressing scene", "a frightened or angry person", "something threatening or disgusting", "something bad"]
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MIN_FRAC = 0.15


def clip_scores(frames_u8, batch=256):
    """Frozen-CLIP zero-shot valence for raw 64px frames (upsampled by CLIP preprocess)."""
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(DEV)
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        tp = model.encode_text(tok(POS_ANCHORS).to(DEV)); tp = F.normalize(tp, dim=-1).mean(0)
        tn = model.encode_text(tok(NEG_ANCHORS).to(DEV)); tn = F.normalize(tn, dim=-1).mean(0)
        anchor = (tp - tn)
        out = np.empty(len(frames_u8), np.float32)
        for i in range(0, len(frames_u8), batch):
            ims = [pre(Image.fromarray(f)) for f in frames_u8[i:i + batch]]
            f = model.encode_image(torch.stack(ims).to(DEV))
            out[i:i + batch] = (F.normalize(f, dim=-1) @ anchor).cpu().numpy()
    del model
    return out


def run_filter(wm, frames, actions, L=250, B=8):
    """Filter the whole stream in (B,L) chunks; returns z_enc, z_full float16 arrays."""
    N = (len(frames) // L) * L
    z_enc = np.empty((N, EMB), np.float16)
    z_full = np.empty((N, DETER + STOCH), np.float16)
    prev = np.concatenate([[0], actions[:-1]])
    starts = list(range(0, N, L))
    for i in range(0, len(starts), B):
        ss = starts[i:i + B]
        ob = np.stack([frames[s:s + L] for s in ss])
        ac = np.stack([prev[s:s + L] for s in ss])
        obs = torch.as_tensor(ob, device=DEV).float().permute(0, 1, 4, 2, 3) / 255.0 - 0.5
        act = F.one_hot(torch.as_tensor(ac, device=DEV).long(), 3).float()
        ze, zf = wm.filter(obs, act)
        for k, s in enumerate(ss):
            z_enc[s:s + L] = ze[k].cpu().numpy().astype(np.float16)
            z_full[s:s + L] = zf[k].cpu().numpy().astype(np.float16)
    return z_enc, z_full, N


def train_head(z, teacher, seeds=3, epochs=6):
    t = (teacher - teacher.mean()) / (teacher.std() + 1e-8)
    heads = []
    for sd in range(seeds):
        torch.manual_seed(sd)
        h = nn.Sequential(nn.Linear(z.shape[1], 256), nn.SiLU(), nn.Linear(256, 1)).to(DEV)
        opt = torch.optim.Adam(h.parameters(), 1e-3)
        zt = torch.as_tensor(z, device=DEV).float()
        tt = torch.as_tensor(t, device=DEV).float().unsqueeze(1)
        n = len(zt); idx = np.arange(n)
        for ep in range(epochs):
            np.random.RandomState(sd * 100 + ep).shuffle(idx)
            for j in range(0, n, 1024):
                b = idx[j:j + 1024]
                loss = F.mse_loss(h(zt[b]), tt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        heads.append(h.eval())
    return heads


def img_level_auc(scores, dom, human):
    """Aggregate frame scores by dominant image; AUC vs human labels."""
    agg = {}
    for s, d in zip(scores, dom):
        agg.setdefault(int(d), []).append(float(s))
    ids = [i for i in agg if str(i) in human]
    y = np.array([human[str(i)] for i in ids])
    x = np.array([np.mean(agg[i]) for i in ids])
    if len(np.unique(y)) < 2: return float("nan"), len(ids)
    return float(roc_auc_score(y, x)), len(ids)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-head-frac", type=float, default=0.0,
                    help="escalation 1: train head only on frames with billboard area >= this")
    ap.add_argument("--out", default="results/phase0_reground.json")
    a = ap.parse_args()
    with open("data_train/splits.json") as f:
        sp = json.load(f)
    human, pools = sp["human"], sp["pools"]
    weak, strong, trainp = set(pools["weak"]), set(pools["strong"]), set(pools["train"])

    ftr = np.load("data_train/frames.npy", mmap_mode="r")
    atr = np.load("data_train/actions.npy")
    dtr = np.load("data_train/dom_idx.npy"); rtr = np.load("data_train/dom_frac.npy").astype(np.float32)
    fev = np.load("data_eval/frames.npy", mmap_mode="r")
    aev = np.load("data_eval/actions.npy")
    dev_ = np.load("data_eval/dom_idx.npy"); rev = np.load("data_eval/dom_frac.npy").astype(np.float32)

    cut = int(0.8 * len(ftr))                      # head-train slice | weak-eval slice
    # --- teacher + render ceilings (world-model-independent; compute once) ---
    print("CLIP teacher on head-train subsample...")
    if a.min_head_frac > 0:                        # escalation 1: content-bearing frames only
        m = np.isin(dtr[:cut], list(trainp)) & (rtr[:cut] >= a.min_head_frac)
        head_idx = np.where(m)[0]
        if len(head_idx) > 30000:
            head_idx = head_idx[np.linspace(0, len(head_idx) - 1, 30000).astype(int)]
        print(f"escalation-1 head-train frames: {len(head_idx)}")
    else:
        sub = np.arange(0, cut, 3)                 # every 3rd frame, ~26k
        is_weak_dom = np.isin(dtr[sub], list(weak))
        head_idx = sub[~is_weak_dom]               # heldout-weak images never enter head training
    teacher = clip_scores(np.asarray(ftr[head_idx]))

    weak_mask = (np.arange(cut, len(ftr)))[np.isin(dtr[cut:], list(weak)) & (rtr[cut:] >= MIN_FRAC)]
    strong_mask = np.where(np.isin(dev_, list(strong)) & (rev >= MIN_FRAC))[0]
    print(f"eval frames: weak {len(weak_mask)}, strong {len(strong_mask)}")
    print("CLIP render-ceiling on eval frames...")
    clip_weak = clip_scores(np.asarray(ftr[weak_mask]))
    clip_strong = clip_scores(np.asarray(fev[strong_mask]))
    res = {"render_ceiling": {
        "weak": img_level_auc(clip_weak, dtr[weak_mask], human),
        "strong": img_level_auc(clip_strong, dev_[strong_mask], human)}}
    print("render ceiling:", res["render_ceiling"])

    for obj in ["recon", "jepa"]:
        ck = f"ckpt/wm_{obj}.pt"
        if not os.path.exists(ck): print("missing", ck); continue
        wm = WM(obj).to(DEV); wm.load_state_dict(torch.load(ck, map_location=DEV)["state"]); wm.eval()
        print(f"[{obj}] filtering train stream...")
        ze_tr, zf_tr, Ntr = run_filter(wm, ftr, atr)
        print(f"[{obj}] filtering eval stream...")
        ze_ev, zf_ev, Nev = run_filter(wm, fev, aev)
        hi = head_idx[head_idx < Ntr]
        te = teacher[: len(hi)] if len(hi) != len(head_idx) else teacher
        wk = weak_mask[weak_mask < Ntr]; st = strong_mask[strong_mask < Nev]
        out = {}
        for zname, ztr, zev in [("z_enc", ze_tr, ze_ev), ("z_full", zf_tr, zf_ev)]:
            heads = train_head(ztr[hi], te)
            aucs_w, aucs_s = [], []
            with torch.no_grad():
                for h in heads:
                    sw = h(torch.as_tensor(ztr[wk], device=DEV).float()).squeeze(1).cpu().numpy()
                    ss_ = h(torch.as_tensor(zev[st], device=DEV).float()).squeeze(1).cpu().numpy()
                    aucs_w.append(img_level_auc(sw, dtr[wk], human)[0])
                    aucs_s.append(img_level_auc(ss_, dev_[st], human)[0])
            # linear-probe diagnostic vs human labels (train-pool frames -> strong images)
            pr_idx = hi[np.isin(dtr[hi], list(trainp)) & (rtr[hi] >= MIN_FRAC)]
            ylab = np.array([human[str(int(d))] for d in dtr[pr_idx]])
            probe_auc = float("nan")
            if len(np.unique(ylab)) == 2:
                clf = LogisticRegression(max_iter=2000).fit(ztr[pr_idx].astype(np.float32), ylab)
                dsc = clf.decision_function(zev[st].astype(np.float32))
                probe_auc = img_level_auc(dsc, dev_[st], human)[0]
            out[zname] = {"head_auc_weak": [round(float(np.nanmean(aucs_w)), 4), round(float(np.nanstd(aucs_w)), 4)],
                          "head_auc_strong": [round(float(np.nanmean(aucs_s)), 4), round(float(np.nanstd(aucs_s)), 4)],
                          "probe_auc_strong": round(probe_auc, 4)}
            print(f"[{obj}][{zname}] weak {out[zname]['head_auc_weak']}  strong {out[zname]['head_auc_strong']}  probe {probe_auc:.3f}")
        res[obj] = out
        del wm, ze_tr, zf_tr, ze_ev, zf_ev

    os.makedirs("results", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
