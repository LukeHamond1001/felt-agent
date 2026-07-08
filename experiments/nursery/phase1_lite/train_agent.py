#!/usr/bin/env python3
"""
Phase-1-lite agent: PPO over frozen phase-0 encoder features, three reward conditions.

  grounded : r = relu(R_phi) - P(z)  +  min(R_phi, 0)
             appetitive channel habituates via an online reward-predictor P (RPE); the
             aversive channel is a fixed floor, never predicted away. The architecture's
             scalarization, minus drives/damage (not present in the gallery).
  nohab    : r = R_phi raw (ablation: no habituation -> should fixate on the wirehead)
  rnd      : r = ||f_target(z) - f_pred(z)||^2, predictor trained online (tuned baseline)

Policy: MLP on [z_enc(256), prev_action(3)]. Identical across conditions.
Logs per-rollout dwell fractions {tv, wire, pos, neg, none} to logs/<cond>_s<seed>.jsonl.
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))
from train_wm import WM  # noqa: E402
from env_constructs import ConstructGallery, category, make_pools  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class RunningNorm:
    def __init__(self):
        self.m, self.v, self.n = 0.0, 1.0, 1e-4

    def __call__(self, x):
        self.n += len(x)
        d = x.mean() - self.m
        self.m += d * len(x) / self.n
        self.v = 0.99 * self.v + 0.01 * x.var()
        return x / (self.v ** 0.5 + 1e-6)


class Policy(nn.Module):
    """Frozen mode: input = frozen z_enc (256). Pixel mode: own plastic conv trunk, trained
    end-to-end by PPO — the phase-1.5 'unfreeze' arm."""

    def __init__(self, pixel=False):
        super().__init__()
        self.pixel = pixel
        if pixel:
            self.trunk = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
                nn.Flatten(), nn.Linear(256 * 4 * 4, 256), nn.LayerNorm(256))
        self.body = nn.Sequential(nn.Linear(259, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.pi = nn.Linear(256, 3)
        self.v = nn.Linear(256, 1)

    def forward(self, obs, prev):
        f = self.trunk(obs) if self.pixel else obs
        h = self.body(torch.cat([f, prev], -1))
        return self.pi(h), self.v(h).squeeze(-1)


def head_mlp():
    return nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))


def pix_net():
    return nn.Sequential(
        nn.Conv2d(3, 16, 4, 2, 1), nn.SiLU(),
        nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
        nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 128))


class Reward:
    """kind grounded/nohab: R_phi on the FROZEN phase-0 encoder (reward source constant across
    all conditions). kind rnd: novelty in feature space (phase-1-lite) or PIXEL space
    (phase-1.5 laundering check) per rnd_space."""

    def __init__(self, kind, prep, rnd_space="feat"):
        self.kind, self.rnd_space = kind, rnd_space
        if kind in ("grounded", "nohab"):
            self.head = head_mlp().to(DEV)
            self.head.load_state_dict(prep["head"])
            self.head.eval()
            self.mu, self.sd = prep["teacher_mu"], prep["teacher_sd"]
            if kind == "grounded":
                self.P = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1)).to(DEV)
                self.opt = torch.optim.Adam(self.P.parameters(), 1e-3)
        else:  # rnd
            net = pix_net if rnd_space == "pixel" else head_mlp
            self.tgt = net().to(DEV).eval()
            for p in self.tgt.parameters(): p.requires_grad_(False)
            self.prd = net().to(DEV)
            self.opt = torch.optim.Adam(self.prd.parameters(), 1e-3)

    def _rnd_in(self, z, fr):
        return fr if self.rnd_space == "pixel" else z

    @torch.no_grad()
    def __call__(self, z, fr=None):
        if self.kind == "rnd":
            x = self._rnd_in(z, fr)
            d = self.tgt(x) - self.prd(x)
            return (d ** 2).mean(-1)
        rphi = self.head(z).squeeze(-1) * self.sd + self.mu
        if self.kind == "nohab":
            return rphi
        app = F.relu(rphi)
        phasic = app - self.P(z).squeeze(-1)
        return phasic + torch.clamp(rphi, max=0.0)

    def update(self, z_batch, fr_batch=None):
        if self.kind == "nohab": return
        x_batch = self._rnd_in(z_batch, fr_batch) if self.kind == "rnd" else z_batch
        for _ in range(2):
            for i in range(0, len(x_batch), 1024):
                xb = x_batch[i:i + 1024]
                if self.kind == "rnd":
                    if self.rnd_space == "pixel": xb = xb.to(DEV).float().permute(0, 3, 1, 2) / 255.0 - 0.5
                    loss = ((self.tgt(xb).detach() - self.prd(xb)) ** 2).mean()
                else:
                    with torch.no_grad():
                        app = F.relu(self.head(xb).squeeze(-1) * self.sd + self.mu)
                    loss = F.mse_loss(self.P(xb).squeeze(-1), app)
                self.opt.zero_grad(); loss.backward(); self.opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["grounded", "rnd", "nohab"], required=True)
    ap.add_argument("--encoder", choices=["frozen", "pixel"], default="frozen")
    ap.add_argument("--rnd-space", choices=["feat", "pixel"], default="feat")
    ap.add_argument("--name", default=None, help="log/policy name (default: reward kind)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=256)
    a = ap.parse_args()
    name = a.name or a.reward
    pixel = a.encoder == "pixel"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    textures, human, pos, neg = make_pools()
    prep = torch.load("prep.pt", map_location=DEV)
    envs = [ConstructGallery(textures, pos, neg, prep["wire_img"], seed=a.seed * 100 + i)
            for i in range(a.envs)]
    pos_set = set(pos)

    wm = WM("recon")
    wm.load_state_dict(torch.load("../phase0/ckpt/wm_recon.pt", map_location=DEV)["state"])
    enc = wm.enc.to(DEV).eval()
    for p in enc.parameters(): p.requires_grad_(False)

    pol = Policy(pixel=pixel).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), 3e-4)
    rew = Reward(a.reward, prep, rnd_space=a.rnd_space)
    rnorm = RunningNorm()

    os.makedirs("logs", exist_ok=True)
    log = open(f"logs/{name}_s{a.seed}.jsonl", "w")

    def prep_frames(imgs):
        u8 = torch.as_tensor(np.stack(imgs))
        xf = u8.to(DEV).float().permute(0, 3, 1, 2) / 255.0 - 0.5
        return u8, xf

    obs = [e.render() for e in envs]
    u8, xf = prep_frames([o[0] for o in obs])
    z = enc(xf)
    prev = torch.zeros(a.envs, 3, device=DEV)
    step_count, t0 = 0, time.time()
    n_iters = a.steps // (a.envs * a.rollout)

    for it in range(n_iters):
        Z, FR, PA, A, LP, V, R = [], [], [], [], [], [], []
        cats = {"tv": 0, "wire": 0, "pos": 0, "neg": 0, "none": 0}
        for t in range(a.rollout):
            vs = it * a.rollout + t
            if vs > 0 and vs % 800 == 0:
                for e in envs:
                    e.new_layout(); e.reset_agent()
            with torch.no_grad():
                logits, v = pol(xf if pixel else z, prev)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            Z.append(z); FR.append(u8); PA.append(prev)
            A.append(act); LP.append(dist.log_prob(act)); V.append(v)
            imgs, infos = [], []
            for k, e in enumerate(envs):
                step_count += 1
                img, info = e.step(int(act[k]))
                imgs.append(img); infos.append(info)
                cats[category(info, pos_set)] += 1
            u8, xf = prep_frames(imgs)
            z = enc(xf)
            r = rew(z, xf)
            R.append(r)
            prev = F.one_hot(act, 3).float()
        with torch.no_grad():
            _, last_v = pol(xf if pixel else z, prev)
        Zt = torch.stack(Z); Rt = torch.stack(R); Vt = torch.stack(V)
        Rt = torch.as_tensor(rnorm(Rt.cpu().numpy()), device=DEV)
        adv = torch.zeros_like(Rt); gae = torch.zeros(a.envs, device=DEV)
        for t in reversed(range(a.rollout)):
            nxt = last_v if t == a.rollout - 1 else Vt[t + 1]
            delta = Rt[t] + 0.99 * nxt - Vt[t]
            gae = delta + 0.99 * 0.95 * gae
            adv[t] = gae
        ret = adv + Vt
        bz = Zt.reshape(-1, 256); bpa = torch.stack(PA).reshape(-1, 3)
        bfr = torch.stack(FR).reshape(-1, 64, 64, 3)          # uint8, CPU
        ba = torch.stack(A).reshape(-1); blp = torch.stack(LP).reshape(-1)
        badv = adv.reshape(-1); bret = ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-6)
        N = len(bz); idx = np.arange(N)
        for ep in range(4):
            np.random.shuffle(idx)
            for j in range(0, N, 1024):
                mb = idx[j:j + 1024]
                if pixel:
                    ob = bfr[mb].to(DEV).float().permute(0, 3, 1, 2) / 255.0 - 0.5
                else:
                    ob = bz[mb]
                logits, v = pol(ob, bpa[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(ba[mb]) - blp[mb]).exp()
                pl = -torch.min(ratio * badv[mb],
                                ratio.clamp(0.8, 1.2) * badv[mb]).mean()
                vl = F.mse_loss(v, bret[mb])
                loss = pl + 0.5 * vl - 0.02 * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step()
        rew.update(bz, bfr)
        tot = sum(cats.values())
        rec = {"iter": it, "steps": step_count,
               **{k: round(v / tot, 4) for k, v in cats.items()},
               "r_mean": round(float(Rt.mean()), 4)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % 20 == 0:
            print(f"[{name} s{a.seed}] it {it}/{n_iters} "
                  f"tv {rec['tv']:.3f} wire {rec['wire']:.3f} pos {rec['pos']:.3f} neg {rec['neg']:.3f} "
                  f"({step_count / (time.time() - t0):.0f} fps)", flush=True)

    os.makedirs("policies", exist_ok=True)
    torch.save({"state": pol.state_dict(), "pixel": pixel}, f"policies/{name}_s{a.seed}.pt")
    print("saved", f"policies/{name}_s{a.seed}.pt")


if __name__ == "__main__":
    main()
