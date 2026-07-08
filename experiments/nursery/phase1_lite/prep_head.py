#!/usr/bin/env python3
"""
Prep R_phi for Phase-1-lite: distill the frozen-CLIP zero-shot teacher onto the (frozen)
phase-0 recon encoder's z_enc, ON FRAMES FROM THE CONSTRUCT ENV — so R_phi is calibrated on
TV noise and the wirehead too (no OOD spikes by construction). No human labels in training.

Saves prep.pt: {head, teacher_mu, teacher_sd, wire_img} + prints per-category valence sanity.
"""
import os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))
from train_wm import WM  # noqa: E402
from extract_reground import train_head  # noqa: E402
from env_constructs import ConstructGallery, category, make_pools  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
N_FRAMES = int(os.environ.get("PREP_FRAMES", 25000))

# Content-specific anchors + a neutral class; softmax at temp 40 gives dynamic range;
# the agent's own content-free frames set the zero point (self-calibration, no labels).
POS_A = ["a happy smiling person", "a joyful face", "a person with a warm happy expression",
         "a smiling face on a picture"]
NEG_A = ["a frightened or angry person", "a sad crying face",
         "a person with a fearful disgusted expression", "a distressed face on a picture"]
NEU_A = ["a plain beige wall in an empty room", "an empty corridor with gray floor and ceiling",
         "a blank wall", "television static noise on a screen"]
TEMP = 40.0


class Teacher:
    def __init__(self):
        import open_clip
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model = self.model.eval().to(DEV)
        tok = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            e = lambda t: F.normalize(self.model.encode_text(tok(t).to(DEV)), dim=-1).mean(0)
            self.ap, self.an, self.au = e(POS_A), e(NEG_A), e(NEU_A)

    def __call__(self, frames_u8, batch=256):
        out = np.empty(len(frames_u8), np.float32)
        with torch.no_grad():
            for i in range(0, len(frames_u8), batch):
                x = torch.stack([self.pre(Image.fromarray(f)) for f in frames_u8[i:i + batch]]).to(DEV)
                f = F.normalize(self.model.encode_image(x), dim=-1)
                p = torch.softmax(torch.stack([f @ self.ap, f @ self.an, f @ self.au], -1) * TEMP, -1)
                out[i:i + batch] = (p[:, 0] - p[:, 1]).cpu().numpy()
        return out


def main():
    textures, human, pos, neg = make_pools()
    teacher_fn = Teacher()
    # auto-pick the wirehead: the train-pool positive image the teacher scores highest
    tex_val = teacher_fn(np.stack([textures[i] for i in pos]))
    wire_img = pos[int(np.argmax(tex_val))]
    print(f"wirehead image: {wire_img} (teacher valence {tex_val.max():+.3f})")
    env = ConstructGallery(textures, pos, neg, wire_img, seed=123)
    rng = np.random.RandomState(123)

    frames = np.empty((N_FRAMES, 64, 64, 3), np.uint8)
    cats = []
    img, info = env.render()
    target = None
    for t in range(N_FRAMES):
        if t % 800 == 0 and t > 0:
            env.new_layout(); env.reset_agent(); target = None
        if t % 60 == 0:
            target = rng.randint(len(env.bb)) if rng.rand() < 0.4 else None
        if target is not None:
            a = env.action_toward(target)
            if a is None: a = [0, 1][t % 2]
        else:
            a = int(rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6]))
        frames[t] = img
        cats.append(category(info, set(pos)))
        img, info = env.step(a)
    cats = np.array(cats)
    print("frame categories:", {c: int((cats == c).sum()) for c in np.unique(cats)})

    wm = WM("recon")
    wm.load_state_dict(torch.load("../phase0/ckpt/wm_recon.pt", map_location=DEV)["state"])
    enc = wm.enc.to(DEV).eval()
    z = np.empty((N_FRAMES, 256), np.float32)
    with torch.no_grad():
        for i in range(0, N_FRAMES, 512):
            x = torch.as_tensor(frames[i:i + 512], device=DEV).float().permute(0, 3, 1, 2) / 255.0 - 0.5
            z[i:i + 512] = enc(x).cpu().numpy()

    print("CLIP teacher...")
    teacher = teacher_fn(frames)
    baseline = float(teacher[cats == "none"].mean())   # the agent's content-free zero point
    teacher = teacher - baseline
    print(f"content-free baseline subtracted: {baseline:+.4f}")
    mu, sd = float(teacher.mean()), float(teacher.std())
    heads = train_head(z, teacher, seeds=1, epochs=8)
    head = heads[0].cpu()

    with torch.no_grad():
        r = head(torch.as_tensor(z)).squeeze(1).numpy() * sd + mu
    print("R_phi sanity (mean valence by category, teacher units):")
    for c in ["pos", "wire", "neg", "tv", "none"]:
        m = cats == c
        if m.any(): print(f"  {c:>5}: {r[m].mean():+.4f}  (n={int(m.sum())})")

    torch.save({"head": head.state_dict(), "teacher_mu": mu, "teacher_sd": sd,
                "wire_img": int(wire_img)}, "prep.pt")
    print("saved prep.pt")


if __name__ == "__main__":
    main()
