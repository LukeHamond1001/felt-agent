#!/usr/bin/env python3
"""
Minimal RSSM world model, ONE backbone, TWO objectives (the confound PLAN.md flags):

  --objective recon : posterior state decodes the frame (DreamerV3-style pixel reconstruction).
                      Latent is PRESSURED to keep appearance. Optimistic floor.
  --objective jepa  : no decoder. From the prior state, predict an EMA target-encoder embedding
                      of the incoming frame (V-JEPA/BYOL-style latent prediction, stop-grad
                      target). Latent is FREE to discard appearance. The architecture's default.

Everything else — encoder, RSSM dims, data, steps, optimizer — is identical, so any grounding
gap between the two checkpoints is attributable to the objective alone.
"""
import argparse, os, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

DETER, STOCH, EMB = 512, 64, 256


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 4, EMB), nn.LayerNorm(EMB))

    def forward(self, x):  # x: (B,3,64,64) in [-0.5,0.5]
        return self.fc(self.c(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DETER + STOCH, 256 * 4 * 4)
        self.d = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, s):
        return self.d(self.fc(s).view(-1, 256, 4, 4))


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 512), nn.SiLU(), nn.Linear(512, o))


class WM(nn.Module):
    def __init__(self, objective):
        super().__init__()
        self.objective = objective
        self.enc = Encoder()
        self.gru = nn.GRUCell(STOCH + 3, DETER)
        self.prior = mlp(DETER, 2 * STOCH)
        self.post = mlp(DETER + EMB, 2 * STOCH)
        if objective == "recon":
            self.dec = Decoder()
        else:
            self.target_enc = Encoder()
            self.target_enc.load_state_dict(self.enc.state_dict())
            for p in self.target_enc.parameters(): p.requires_grad_(False)
            self.pred = mlp(DETER + STOCH, EMB)

    @staticmethod
    def dist(stats):
        mean, std = stats.chunk(2, -1)
        return mean, F.softplus(std) + 0.1

    def ema(self, m=0.996):
        with torch.no_grad():
            for p, tp in zip(self.enc.parameters(), self.target_enc.parameters()):
                tp.mul_(m).add_(p, alpha=1 - m)

    def loss(self, obs, act):
        """obs (B,L,3,64,64), act (B,L,3) one-hot of PREVIOUS action."""
        B, L = obs.shape[:2]
        emb = self.enc(obs.reshape(B * L, *obs.shape[2:])).view(B, L, EMB)
        deter = torch.zeros(B, DETER, device=obs.device)
        stoch = torch.zeros(B, STOCH, device=obs.device)
        kl_l, aux_l = 0.0, 0.0
        for t in range(L):
            deter = self.gru(torch.cat([stoch, act[:, t]], -1), deter)
            pm, ps = self.dist(self.prior(deter))
            if self.objective == "jepa":
                with torch.no_grad():
                    tgt = self.target_enc(obs[:, t])
                    tgt = F.normalize(tgt, dim=-1)
                pr = F.normalize(self.pred(torch.cat([deter, pm], -1)), dim=-1)
                aux_l = aux_l + (1 - (pr * tgt).sum(-1)).mean()
            qm, qs = self.dist(self.post(torch.cat([deter, emb[:, t]], -1)))
            stoch = qm + qs * torch.randn_like(qs)
            kl = (torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5).sum(-1)
            kl_l = kl_l + torch.clamp(kl, min=3.0).mean()
            if self.objective == "recon":
                rec = self.dec(torch.cat([deter, stoch], -1))
                aux_l = aux_l + F.mse_loss(rec, obs[:, t]) * 500.0
        return aux_l / L, kl_l / L

    @torch.no_grad()
    def filter(self, obs, act):
        """Run posterior filter; return z_enc (B,L,EMB), z_full (B,L,DETER+STOCH mean)."""
        B, L = obs.shape[:2]
        emb = self.enc(obs.reshape(B * L, *obs.shape[2:])).view(B, L, EMB)
        deter = torch.zeros(B, DETER, device=obs.device)
        stoch = torch.zeros(B, STOCH, device=obs.device)
        zf = torch.empty(B, L, DETER + STOCH, device=obs.device)
        for t in range(L):
            deter = self.gru(torch.cat([stoch, act[:, t]], -1), deter)
            qm, qs = self.dist(self.post(torch.cat([deter, emb[:, t]], -1)))
            stoch = qm
            zf[:, t] = torch.cat([deter, qm], -1)
        return emb, zf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["recon", "jepa"], required=True)
    ap.add_argument("--data", default="data_train")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=20)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    frames = np.load(f"{a.data}/frames.npy", mmap_mode="r")
    actions = np.load(f"{a.data}/actions.npy")
    actions = np.concatenate([[0], actions[:-1]])   # act[t] = action that led INTO frame t
    N = len(frames)
    wm = WM(a.objective).to(dev)
    opt = torch.optim.Adam(wm.parameters(), 3e-4)
    rng = np.random.RandomState(1)
    print(f"objective={a.objective} device={dev} frames={N} params={sum(p.numel() for p in wm.parameters())/1e6:.1f}M")

    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = rng.randint(0, N - a.seq - 1, size=a.batch)
        ob = np.stack([frames[i:i + a.seq] for i in idx])          # (B,L,64,64,3)
        ac = np.stack([actions[i:i + a.seq] for i in idx])          # prev-action alignment: a[t] led INTO frame t+1;
        obs = torch.as_tensor(ob, device=dev).float().permute(0, 1, 4, 2, 3) / 255.0 - 0.5
        act = F.one_hot(torch.as_tensor(ac, device=dev).long(), 3).float()
        aux, kl = wm.loss(obs, act)
        loss = aux + kl
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
        if a.objective == "jepa": wm.ema()
        if step % 500 == 0:
            print(f"step {step}/{a.steps}  aux {float(aux):.4f}  kl {float(kl):.3f}  {(time.time()-t0)/step:.2f}s/it", flush=True)
    os.makedirs("ckpt", exist_ok=True)
    torch.save({"state": wm.state_dict(), "objective": a.objective}, f"ckpt/wm_{a.objective}.pt")
    print("saved", f"ckpt/wm_{a.objective}.pt")


if __name__ == "__main__":
    main()
