# Amygdala trainer — spec

The concrete program for **grounding the amygdala shadow** using a single emotionally-exceptional human (you). It implements the protocol in [../CONCEPTS.md §3](../CONCEPTS.md): one calm, uninterrupted, emotionally-attuned person maps their *intrinsic* felt pull **continuously** over rich audio/video, and that dense trace trains a thin head on top of frozen perception.

**What you label = the amygdala's felt pull.** Attention, innate dispositions, attraction — *not* good/bad. You report it as **signed valence**: the **sign** = toward/away, the **magnitude** = how strongly it pulls (a flicker of interest vs something gorgeous — the intensity *is* the pull itself). The *specific quality* ("this particular pull, not pull-in-general") is **not** in the label — it lives in the learned mapping `perception → your pull`. That binding is where "what makes us us" actually is.

**Not dopamine, not cortisol.** Those are **downstream** of the pull and are **not labeled**: dopamine = the reward-prediction-error the agent computes (pull − its own prediction); cortisol (G5) = **derived downstream as a slow low-pass of `relu(−valence)`** — a leaky accumulator over sustained *negative* pull, computed by the agent, never annotated. And neither can be measured directly in the cortex anyway (sub-second dopamine needs invasive electrodes; cortisol is a slow peripheral assay) — **the felt report is the only instrument**, which is the whole point.

**Honest scope.** This trains the *perceptual shadow* of the pull. It does **not** build the full ineffable pull or the anticipatory value — those are the wall, earned by a living agent. One person's taste: authentic, and unavoidably subjective.

---

## Pipeline

```
 stimulus (video/audio) ─► annotate.html ─► valence trajectory (.json)
                                                  │
 same stimulus ─► frozen perception encoder ─► embeddings (per timestamp)
                                                  │
                    align by timestamp ─► (embedding, valence) pairs
                                                  │
                    train_amygdala.py ─► amygdala_shadow head (.pt) + report
```

## Components

### `annotate.html` — timeline affect annotation (local, no install, nothing uploaded)
A video-editor-style tool: **video preview + a single vertical bar + a timeline**.
- The **bar** = the amygdala's felt pull. Drag **up = drawn-to**, **down = aversive**; **how far = how strongly** (signed intensity), center = neutral.
- The **timeline** shows the audio **waveform**, **playhead**, and your recorded **valence curve** (green above / red below), drawn live.
- **Mark as you go** (~30 Hz, timestamped). **Scrub to review / re-mark:** click/drag the timeline to jump back, see what you marked, and drag the bar while paused to tweak that moment; re-playing a region overwrites it.
- **No smoothing, no webcam, no countdown** — raw felt trace, by design.

### `train_amygdala.py` — fit the shadow head
- Embeds the stimulus with a **frozen** encoder at the annotation timestamps (video → `open_clip` ViT; audio → CLAP / log-mel fallback).
- Trains a **small MLP/transformer head** `embedding → valence` (MSE + a light temporal-smoothness term in the *loss*, not on your data).
- Optional `--reaction_lag_ms` *alignment* (shift the trace to line your response up with the causing frame) — alignment, not smoothing; off by default.
- Holdout eval: **Pearson r** + **CCC**; saves `amygdala_shadow.pt` + `report.json`.

## Sizing — the trainable part is SMALL
*Sizes are a starting point to tune.* The table lists the **target** frozen perception; the shipped prototype uses the lighter CLIP ViT-B/32 / log-mel encoders to validate the pipeline (see G1).
| piece | size | trained? |
|---|---|---|
| visual encoder (target: DINOv2 ViT-L; prototype: CLIP ViT-B/32) | ~0.3B / ~0.1B | frozen (borrowed perception) |
| audio encoder (target: CLAP; prototype: log-mel fallback) | ~0.2B | frozen |
| **amygdala head** (2–4 layer transformer / MLP → 1 valence output) | **~10–50M** | **yes — all your annotations train** |

One person's annotation is limited and temporally correlated; a big trainable model overfits instantly. Capacity lives in the **frozen** encoders. **You do not train a 200B amygdala** — you train a thin read-out. In the full humanoid, the same thin head reads the **frozen 200B trunk's `z_t`** instead of separate encoders.

### G1 — prototype scope: NOT transferable until re-grounded on `z_t`
The shipped `train_amygdala.py` head is a **methodology prototype** fit on **frozen CLIP ViT-B/32** image features (the table above lists DINOv2/CLAP as the *target* frozen perception — the prototype uses the lighter CLIP encoder to validate the *pipeline*, not to ship a usable head). **A head fit on CLIP features is not the agent's amygdala.** CLIP's feature space ≠ the agent trunk's `z_t`, so the learned `feature → valence` mapping does **not** transfer.

To make it usable on the real agent it must be **re-grounded**:
1. Re-run the *same* annotations through the **frozen real trunk** to get its latent `z_t` per timestamp (replacing the CLIP/CLAP embeddings).
2. **Refit the thin head on `z_t → valence`** (same loss, same holdout protocol).
3. Re-check held-out CCC on the real features before any agent use.

**Token read-out.** The trunk emits a set of tokens per timestamp, not a single vector. The head consumes them by either (a) **attention pooling** — a learned query that mean-pools a softmax-weighted sum over the `z_t` tokens — or (b) a small **cross-attention** read-out (1–2 layers, learned query attending over the token set) when spatial/temporal token structure matters. Either way the read-out is part of the **thin trainable head** (~10–50M, *starting point to tune*); the trunk stays frozen.

**Encoder upgrade (optional).** The prototype's CLIP encoder can be swapped for stronger frozen features — DINOv3 / SigLIP 2 or whatever is the **current SOTA generation** (*verify exact version/availability at build time — this moves fast*) — for better separability. This only improves the *prototype's* features; **it does not remove the re-grounding requirement.** The shipped agent head must still be fit on the real frozen trunk's `z_t`, because that is the space the agent actually perceives in.

## Data schema — `annotation_*.json`
```json
{ "media": "clip.mp4", "media_kind": "video", "duration_s": 184.2, "sample_hz": 30,
  "grounder": "luke", "reaction_lag_ms": 0,
  "channels": "amygdala valence: sign=toward/away, magnitude=pull intensity",
  "samples": [ {"t": 0.000, "valence": 0.02}, {"t": 0.033, "valence": 0.05} ],
  "notes": "" }
```

## Where it feeds (the 200B gameplan)
The shadow head output (the pull) → the **felt signal (reward)** → the **value system (BG)** learns to predict it → **dopamine = RPE** (pull − prediction) → the agent's continual learning. The 200B is **borrowed, frozen perception**; your annotations only ever train the thin affective read-out.

## Protocol checklist
1. One person, alone, **no interruptions**, full focus.
2. React **honestly and continuously** — track the moment-to-moment pull *and its intensity*, including rise and fall.
3. Prefer **rich, affect-laden, diverse** stimuli (film, music, real scenes); many short sessions across varied content > one long run.
4. Use the timeline to **review and re-mark** anything you got wrong.
5. **Self-consistency first (consistency check):** re-annotate the same clip on another day, then compute **test–retest CCC** between the two traces. Treat a **low retest CCC as "not learnable"** — if you can't reproduce your *own* trace, the head can't be expected to, and the held-out fit is untrustworthy. Run this *before* trusting any head:
   ```bash
   python train_amygdala.py --consistency a1.json a2.json
   ```
   (Prints test–retest CCC + Pearson and exits; no training. CCC ≈ 0 = not learnable.)
6. (Optional, external) record physiology separately and align by timestamp — kept outside the tool by design. **Paired physiology is a future validation aid, not part of the v1 grounding signal:** the felt report is the only instrument v1 trains on; physiology, if recorded, is for *later* cross-checking that the report tracks measured arousal, not an anchor the head is fit against.

## Honest caveats
- **Shadow only** — a signed-valence projection of the pull, not the full ineffable pull (the wall).
- **Single-source / subjective** — it is *your* affect; the amygdala inherits your taste.
- **Not a benchmark** — this grounds an organ; it is not a capability result.
