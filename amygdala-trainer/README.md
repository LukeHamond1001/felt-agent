# amygdala-trainer

Ground the **amygdala shadow** from one emotionally-attuned person's continuous affect, per the protocol in [../CONCEPTS.md §3](../CONCEPTS.md). Full design: [SPEC.md](SPEC.md).

**This trains the perceptual *shadow* only — not the ineffable pull or the anticipatory value (the wall). One person's affect: authentic, subjective. Not a capability benchmark.**

## Quickstart
1. **Annotate.** Open `annotate.html` in a browser (local, no install, nothing uploaded). Pick a local video/audio file, then track the **amygdala's felt pull** with a **single vertical bar**: **sign** = toward/away (up = drawn-to, down = aversive, center = neutral), **magnitude** = how strongly it pulls (a flicker of interest vs something gorgeous — the intensity *is* attention/beauty). This is the felt appraisal you can introspect — **not** dopamine, **not** cortisol (those are downstream signals the agent derives later, not labeled here).
   - **Hold & drag the bar to mark** as it plays (~30 Hz, timestamped). **Release to play it back** — the bar tracks your recorded curve (review, no overwrite). **Click/drag the timeline to seek**; **paused, drag the bar to tweak** that moment. Re-playing a region overwrites it. Space = play/pause.
   - The **timeline** shows the audio **waveform**, **playhead**, and your recorded **valence curve** (green above / red below), drawn live.
   - No 2D pad, no webcam, no countdown, no smoothing — a raw felt trace by design.
   - It exports `annotation_*.json` (a single signed-valence trace).
2. **Train.** `pip install -r requirements.txt` then:
   ```bash
   python train_amygdala.py --media clip.mp4 --annotation annotation_clip_*.json --out amygdala_shadow.pt
   ```
   Sweep `--reaction_lag_ms 0..600` and keep the lag with the best held-out CCC.
3. **Check yourself first.** Re-annotate the same clip on another day, compute test–retest CCC; if you can't reproduce your own trace, treat the head as **not learnable** before trusting it (self-consistency before generalization).

Output: `amygdala_shadow.pt` (the thin valence head) + `report.json` (held-out Pearson/CCC).

> **Prototype scope.** The trained head is a **methodology prototype** on frozen CLIP ViT-B/32 features. It is **not** transferable to the agent until **re-grounded** on the real frozen trunk's `z_t` (see [SPEC.md](SPEC.md) §G1).
