# Phase 0 — re-grounding survival (no body, ~4 days, 1 GPU)

**Question:** does the amygdala's grounding survive the encoder swap — from frozen CLIP features
(where AUC = 0.95 lives, see [`../../amygdala_grounding/`](../../amygdala_grounding/README.md))
onto the learned latent `z_t` of a Dreamer world model trained for *prediction*, not affect?

**Threat model:** a world-model latent is optimized to predict dynamics and reconstruct
observations at low resolution. Facial/affect detail is exactly the kind of appearance
information a prediction objective is free to discard. If it does, `R_φ(z_t)` is blind no matter
how good the CLIP prior was, and the architecture needs an alignment term in the backbone loss —
better to learn that now for the price of a fine-tune.

**Design principle:** don't test this inside the full Nursery (weeks of env bring-up). Build the
minimal world that puts controlled affect content into a Dreamer's pixels: a **gallery world**.

---

## The gallery world

A procedurally generated 3D room-maze (Gymnasium **MiniWorld**, pip-installable, days-scale) whose
walls carry large image billboards textured with the 800 human-labelled affect images from
`FastJobs/Visual_Emotional_Analysis` (the exact dataset behind the 0.95). The agent (random policy
+ a scripted "look-at-billboard" behavior mixed in so billboards are actually foveated) walks the
gallery; every frame has ground-truth affect labels for free (we know which image is in view, and
its human label).

Why this is the *right* minimal test: it controls exactly what affect content exists in the world,
gives frame-level labels without annotation, and isolates the one question — does the signal
survive compression into `z_t` — from every Nursery confound (renderer realism, distress logic,
embodiment). Those belong to the full Phase 0 in [`../PROTOCOL.md`](../PROTOCOL.md); this is the
encoder-swap kill-check that gates everything.

**Image splits (fixed, seeded):**
- `train-images` (60%): appear on billboards during world-model training AND head training.
- `heldout-weak` (20%): appear in the world during WM training, but never in head training —
  tests the head generalizing over `z_t`.
- `heldout-strong` (20%): **never rendered anywhere** during any training — tests generalization
  through the whole stack (encoder + latent + head). **Primary split.**

---

## The confound to kill first: resolution

Dreamer consumes 64×64 (maybe 128×128) pixels; CLIP measured at 224. Faces may die in the
downsample *before* the world model ever sees them. So the fair ceiling is not CLIP@224 = 0.95,
it is **CLIP on resolution-matched frames**.

**Step 0 (hour one, no env needed):** rerun `measure_grounding.py` with images downsampled to
64×64 and 128×128 (then re-upsampled to CLIP's input size). This decomposes any later AUC loss
into *resolution loss* vs *latent-compression loss* — without it the result is uninterpretable.

- If CLIP@64 ≈ chance → run everything at 128×128 (DreamerV3 supports it; ~2–3× slower).
- If CLIP@128 also craters → the finding is already real: Dreamer-resolution pixel budgets cannot
  carry facial affect, and the architecture needs a higher-res or affect-dedicated perception
  stream. That's a Phase-0 result on its own; write it up.

All gates below are stated against the **resolution-matched ceiling** `AUC_CLIP@res`.

---

## The confound that matters most: reconstruction vs. prediction

Stock DreamerV3 carries a **pixel-reconstruction loss**, so its latent is *pressured to preserve
appearance* — faces included. But `build/ARCHITECTURE.md`'s default world model is
**reconstruction-free JEPA** (V-JEPA-2-AC), which predicts in latent space and is *free to discard*
appearance not needed for dynamics — **exactly the threat model this plan opens with.**
Reconstruction-DreamerV3 is insulated from the very failure we are testing for, so a pass on it does
**not** clear the JEPA default the architecture actually ships.

Run the world model in **two variants**, graded separately:

- **WM-recon (DreamerV3) — the optimistic floor.** If even the appearance-preserving model can't
  host the grounded read, the negative is decisive and cheap.
- **WM-jepa (reconstruction-free latent) — the real G0b.** A JEPA-style / latent-only-prediction
  model over the same rollouts; this is where the discard-appearance threat bites, and it is the
  world model the architecture ships. **G0-z is graded on WM-jepa**, with WM-recon as the reference
  upper bound.

WM-recon passing while WM-jepa fails is itself load-bearing: **the architecture cannot ground on its
reconstruction-free default and must add a reconstruction/alignment term to the backbone (or tap
pre-latent perception)** — a change to `build/ARCHITECTURE.md`, not a footnote.

**Interpreting a null (pessimism warning).** Gallery billboards are *static, predictable wallpaper* —
the easiest content for a prediction objective to compress away, doubly so for WM-jepa (predictable
*and* appearance-only). In the full Nursery, affect-laden content (avatars, distress) is **dynamic and
reward-relevant**, so the world model is pressured to encode it. So a gallery **null may be pessimistic**
relative to the real setting, while a gallery **pass is strong.** Mitigation: mix in ≥1
*behaviorally-relevant* affect cue (a billboard the scripted policy must approach to progress) and report
static-vs-relevant separately.

---

## Pipeline

1. **Env + data** (`gallery_env.py`, `collect.py`): MiniWorld gallery, seeded generation,
   billboard placement from `train-images ∪ heldout-weak`; policy = 70% random / 30% scripted
   approach-and-face-billboard; collect ~500k–1M frames at the chosen resolution. Log per frame:
   billboard-in-view id, view area fraction. Keep only frames with view fraction ≥ 15% for
   probe evaluation (pre-registered; a billboard 40 px tall tests nothing).
2. **World model** (`train_wm.py`): stock **DreamerV3** (small config, ~S), trained on the rollout
   buffer exactly as in normal Dreamer training (world model only; the policy can train too but is
   irrelevant here). ~12–24 h on one 4090/A100. No affect-related loss anywhere — that's the point.
3. **Latent extraction** (`extract_z.py`): run the trained WM over held-out rollouts (fresh seeds),
   record three latent variants per frame:
   - `z_enc` — encoder embedding (pre-RSSM),
   - `z_post` — posterior stochastic sample,
   - `z_full` — deter ⊕ stoch (**primary** — this is what `R_φ` reads in the architecture).
4. **Re-ground** (`reground.py`), two reads per latent variant:
   - **Distilled head (the shipped mechanism):** label every training frame with the frozen-CLIP
     *zero-shot* valence (same POS/NEG anchors as `measure_grounding.py` — no human labels enter
     training); train a small MLP head (2 hidden layers, ~1–4M params — days-scale stand-in for
     the 10–50M organ) on `z → teacher valence`. 3 seeds.
   - **Linear probe (the diagnostic):** logistic regression on `z` vs human labels, same protocol
     as the original experiment. This separates "the information isn't in `z_t`" (linear probe
     fails too) from "the distillation lost it" (probe fine, head bad).
5. **Measure:** AUC of each read against **human labels**, per split, image-level aggregation
   (mean score over a billboard's qualifying frames → one score per image, so frame counts don't
   inflate n). Report mean ± std over 3 head seeds and the frame-level number in an appendix.

---

## Gates (pre-registered)

| Gate | Criterion | Meaning |
|------|-----------|---------|
| **G0-res** | `AUC_CLIP@res ≥ 0.85` at 64 or 128 px | resolution regime is viable; sets the ceiling |
| **G0-z (primary)** | distilled `R_φ(z_full)` AUC on `heldout-strong` **(WM-jepa primary; WM-recon = reference)** ≥ 0.80 **and** ≥ `AUC_CLIP@res − 0.05` | re-grounding survives the encoder swap → Phase 1 is a go |
| **G0-diag** | linear probe on `z_full` ≥ 0.85 | affect is *in* the latent even if distillation is lossy |

**Escalation ladder if G0-z fails** (one step per half-day, in order):
1. If G0-diag passes → the head/distillation is the problem: more teacher-label frames, bigger
   head, train on `z_enc` too.
2. If G0-diag fails on `z_full` but passes on `z_enc` → the RSSM filters affect out: read `R_φ`
   from encoder features (architecture note: `R_φ` taps pre-RSSM perception — biologically fine,
   amygdala reads early sensory cortex).
3. If `z_enc` also fails → the encoder discards it under the prediction objective: add an
   **auxiliary CLIP-alignment loss** on the encoder (cosine between a projection of `z_enc` and
   frozen CLIP embedding, weight 0.1) and retrain the WM once.
4. If (3) fails → **blocked at re-grounding.** Write the negative up honestly: a
   reconstruction-trained world model at this scale cannot host the grounded read, and the
   architecture requires grounding-alignment pressure in the backbone loss as a load-bearing
   component, not an option. That changes `build/ARCHITECTURE.md` and is worth shipping.

---

## Schedule & budget

| Day | Work |
|-----|------|
| 1 AM | **Step 0 resolution control** (no env; existing script + downsample). Decide 64 vs 128. |
| 1 | Gallery env + collection pipeline; overnight: collect 500k–1M frames. |
| 2 | Train DreamerV3-S on the buffer (12–24 h, overnight). |
| 3 | Extract latents; train distilled heads (3 seeds) + linear probes; full AUC table. |
| 4 | Escalation ladder if needed; results README + figure (AUC vs {CLIP@224, CLIP@res, z_enc, z_post, z_full} × {weak, strong} — one bar chart tells the whole story). |

Compute: one 4090-class pod, ≤ 4 days. Storage: ~50–100 GB buffer.

## Deliverables

- `experiments/nursery/phase0/{gallery_env,collect,train_wm,extract_z,reground}.py`
- `results/` — AUC table (JSON) + the bar-chart figure
- `README.md` in the amygdala_grounding house style, with the scope box: *measures whether the
  designed grounding survives the encoder swap onto a learned world-model latent; still says
  nothing about the developed value (the wall); a second shipped datapoint, not validation.*

*(Post-execution note: the planned `extract_z.py` + `reground.py` shipped merged as `extract_reground.py` — one file, same two steps; the env is the raycast gallery described in the log below, not MiniWorld.)*

## Execution log (2026-07-06 — ran same-day, local M-series, ~4 h total)

- **Step 0 G0-res: PASS.** CLIP zero-shot AUC 0.950 @64px, 0.934 @32px (vs 0.952 @224) —
  resolution is a non-issue; the affect signal is low-frequency. `results/step0_resolution_control.json`.
- MiniWorld replaced by a numpy raycast gallery (`gallery_env.py`, ~3,400 fps) — zero install risk,
  and enabled a *cleaner* recon-vs-JEPA test: ONE RSSM backbone, single flipped objective bit.
- Pipeline ran: 100k train + 25k heldout-strong eval frames, both WMs (6M params, 12k steps),
  re-grounding + escalation 1. Full numbers in `README.md` + `results/phase0_reground*.json`.
- **Verdict: G0-z FAIL on `z_full`, G0-diag borderline-PASS on `z_enc`** — affect survives the
  encoder (probe 0.79–0.81 vs render ceiling 0.847) and dies in the RSSM state (0.56–0.59).
  Escalation ladder step 2 branch taken: **`R_φ` taps pre-RSSM perception.** Distillation loses a
  further ~0.1 to teacher noise on 64px renders → first pod follow-up = full-fidelity teacher.

## What this does not test

Renderer/domain realism (Habitat renders vs natural images), distress-as-multimodal-cue (audio),
online drift under a *plastic* backbone (H4 — the gallery WM is trained once, then frozen for
measurement). Those are the full Phase 0 / Phase 2 of [`../PROTOCOL.md`](../PROTOCOL.md). This
experiment answers exactly one question — and it's the one that blocks all the others.
