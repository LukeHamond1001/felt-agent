"""Train the amygdala SHADOW head from a continuous affect annotation (see SPEC.md).

Fits  embedding(frozen perception) -> amygdala valence (signed pull: sign = toward/away, magnitude = intensity)
on one person's continuous affect trace. This is the perceptual shadow only -- NOT the full ineffable pull, NOT
the anticipatory value (the wall), and NOT dopamine/cortisol (those are downstream of the pull, not labeled here).
The trainable head is small (~10-50M) on FROZEN encoders -- one person's data can't train a big model. Honest by design.

Usage:
  python train_amygdala.py --media clip.mp4 --annotation annotation_clip_*.json --out amygdala_shadow.pt
  # sweep reaction lag (human affect trails the stimulus):
  python train_amygdala.py --media clip.mp4 --annotation a.json --reaction_lag_ms 300

Video path uses open_clip ViT-B/32 image embeddings (cv2 to grab frames). Audio path is a log-mel fallback
(swap in CLAP for real use). CPU-fine for short clips.

PROTOTYPE SCOPE (SPEC.md G1): the head here is fit on CLIP/log-mel features -- a METHODOLOGY PROTOTYPE, NOT the
agent's amygdala. Before agent use it must be RE-GROUNDED: re-embed the same annotations with the frozen real
trunk and refit this head on z_t -> valence. (The CLIP encoder can be upgraded to DINOv3/SigLIP2 features, but
the re-grounding requirement stands.)

CONSISTENCY CHECK (SPEC.md, protocol step 5): before trusting any head, run the self-consistency / test-retest
gate -- annotate the SAME clip twice (different days) and check the two traces reproduce:
  python train_amygdala.py --consistency a1.json a2.json
A low test-retest CCC (~0) means the signal is NOT learnable; do not trust any held-out fit on that grounder.
"""
import argparse, json, glob, numpy as np, torch, torch.nn as nn
torch.set_num_threads(4)

def load_annotation(path):
    p = sorted(glob.glob(path))[0] if any(c in path for c in "*?[") else path
    a = json.load(open(p)); s = a["samples"]
    t = np.array([x["t"] for x in s], np.float32)
    # amygdala valence: signed pull (up=+ drawn-to, down=- aversive); magnitude = intensity. Back-compat: derive from dopa/cort if needed.
    y = np.array([[x.get("valence", x.get("dopamine", 0.0) - x.get("cortisol", 0.0))] for x in s], np.float32)
    return a, t, y

def embed_video(media, times, embed_hz=8.0, device="cpu"):
    import cv2, open_clip
    from PIL import Image
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()
    cap = cv2.VideoCapture(media); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = float(times.max()) + 0.5
    grid = np.arange(0, dur, 1.0 / embed_hz, dtype=np.float32)          # embed on a coarse grid, assign nearest to samples
    embs = []
    for tg in grid:
        cap.set(cv2.CAP_PROP_POS_MSEC, tg * 1000.0); ok, frame = cap.read()
        if not ok: embs.append(embs[-1] if embs else np.zeros(512, np.float32)); continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            e = model.encode_image(pre(img)[None].to(device))[0]; e = e / e.norm()
        embs.append(e.cpu().numpy().astype(np.float32))
    cap.release()
    embs = np.array(embs, np.float32)
    idx = np.clip(np.searchsorted(grid, times), 0, len(grid) - 1)        # nearest grid frame per affect sample
    return embs[idx]

def embed_audio(media, times, win=0.5, device="cpu"):
    import librosa                                                        # fallback embedding; swap in CLAP for real use
    wav, sr = librosa.load(media, sr=16000)
    feats = []
    for t in times:
        a = int(max(0, (t - win / 2) * sr)); b = int(min(len(wav), (t + win / 2) * sr))
        seg = wav[a:b] if b > a else wav[:1]
        m = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=64)
        lm = librosa.power_to_db(m + 1e-9)
        feats.append(np.concatenate([lm.mean(1), lm.std(1)]).astype(np.float32))
    return np.array(feats, np.float32)

class Head(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))  # 1 output = amygdala valence (signed pull)
    def forward(s, x): return s.net(x)

def ccc(a, b):                                                            # Lin's concordance correlation
    a, b = a.ravel(), b.ravel()
    va, vb = a.var(), b.var(); ma, mb = a.mean(), b.mean()
    cov = ((a - ma) * (b - mb)).mean()
    return float(2 * cov / (va + vb + (ma - mb) ** 2 + 1e-9))

def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

def consistency(a1, a2):  # SPEC protocol step 5: test-retest CCC of two annotations of the SAME clip
    _, t1, y1 = load_annotation(a1); _, t2, y2 = load_annotation(a2)
    y2i = np.interp(t1, t2, y2[:, 0]).astype(np.float32)   # align the retest onto the first trace's timestamps
    c, r = ccc(y1[:, 0], y2i), pearson(y1[:, 0], y2i)
    print(f"self-consistency (test-retest, same clip annotated twice): CCC {c:+.2f}  pearson {r:+.2f}")
    print("  CCC >= ~0.5 = reproducible (a head can learn it).  CCC ~ 0 = NOT learnable -- do not trust any head fit on this grounder.")
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media"); ap.add_argument("--annotation")
    ap.add_argument("--consistency", nargs=2, default=None, metavar=("A1.json", "A2.json"),
                    help="test-retest CCC of two annotations of the SAME clip, then exit (no training) -- the mandatory gate from SPEC.md step 5")
    ap.add_argument("--out", default="amygdala_shadow.pt")
    ap.add_argument("--reaction_lag_ms", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--smooth", type=float, default=0.1); ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.consistency:                                   # run the self-consistency gate and exit (no training)
        consistency(args.consistency[0], args.consistency[1]); return
    if not args.media or not args.annotation:
        ap.error("--media and --annotation are required for training (or use --consistency A1.json A2.json)")

    meta, t, y = load_annotation(args.annotation)
    t_shift = t + args.reaction_lag_ms / 1000.0                           # shift affect later in stimulus time = align cause->effect
    kind = meta.get("media_kind", "video")
    print(f"embedding {kind}: {len(t)} samples ...")
    X = embed_video(args.media, t_shift, device=args.device) if kind == "video" else embed_audio(args.media, t_shift, device=args.device)

    n = len(X); cut = int(n * (1 - args.holdout))                         # TEMPORAL split (no leakage): train early, test late
    Xtr, Ytr, Xte, Yte = X[:cut], y[:cut], X[cut:], y[cut:]
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    head = Head(X.shape[1]).to(args.device); opt = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        p = head(Xtr_t)
        loss = ((p - Ytr_t) ** 2).mean() + args.smooth * ((p[1:] - p[:-1]) ** 2).mean()   # MSE + temporal smoothness
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pred = head(torch.tensor(Xte)).numpy()
    rep = {"reaction_lag_ms": args.reaction_lag_ms, "n_train": cut, "n_test": n - cut,
           "valence": {"pearson": round(pearson(pred[:, 0], Yte[:, 0]), 3), "ccc": round(ccc(pred[:, 0], Yte[:, 0]), 3)}}
    torch.save({"head": head.state_dict(), "in_dim": X.shape[1], "out_dim": 1, "meta": meta, "report": rep}, args.out)
    json.dump(rep, open("report.json", "w"), indent=2)
    print("held-out fit (higher=better; ~0 means the shadow isn't learnable from this run):")
    print(f"  amygdala valence  pearson {rep['valence']['pearson']:+.2f}  ccc {rep['valence']['ccc']:+.2f}")
    print(f"saved -> {args.out}.  NOTE: this is the perceptual SHADOW (signed pull), not the full ineffable pull (the wall).")

if __name__ == "__main__":
    main()
