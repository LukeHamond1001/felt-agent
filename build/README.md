# The build — everything needed to build the humanoid

One unified predict-and-act cortex — warm-started from the best robot-foundation-model methods, then co-trained end-to-end — with the [felt system](../README.md) grown into it as it reads `z_t`. Read the **[root README](../README.md)** first for *why* and *what* — this folder is the *how*. It holds the humanoid spec (the parts + the phased plan), plus the amygdala demo and what to train it toward:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the parts: sensing, the unified predict-and-act cortex, the felt parts, actuation & latency, safety, and the authoritative **`z_t` workspace contract** every part reads. Opens with a parts table-of-contents.
- **[PERCEPTION-ACTION.md](PERCEPTION-ACTION.md)** — how it senses and acts: active foveated vision, auditory attention, gaze-as-action, and the action vector — all allocated by the amygdala's attention (perception and action are one loop).
- **[TRAINING.md](TRAINING.md)** — the phases, **Stage 0 → 6**: warm-start the cortex → embodiment priors → teleop imitation → sim-to-real whole-body control → **attach the felt system** → wire the deliberation tier → the feeling-driven life → **evaluation = large-scale falsification**. Opens with a stage-summary table.

## The amygdala, concretely — measured, and how to get its training data

The amygdala is a small net that reads perception (vision + audio) → an **approach/avoid pull**. It is the one part of the thesis that's **testable without a body** — here's the measured result and exactly what to train it toward.

**The measured result → [experiments/amygdala_grounding/](../experiments/amygdala_grounding/).** A **linear** read of frozen CLIP features separates human-labelled **approach/avoid** on *held-out* images ([`measure_grounding.py`](../experiments/amygdala_grounding/measure_grounding.py), reproducible from the HuggingFace Hub): **zero-shot AUC 0.95** — no labels, the grounded read alone, vs 0.50 chance; a trained linear probe barely improves it (0.96 ± 0.01, 10 splits), so the affect is already *in* the features. It measures the **designed perceptual grounding** — the known part generalizing to unseen affect — and explicitly **not** the developed value (the wall), which needs scale. A shipped datapoint, not validation. *(Facial affect specifically — a subset of what the amygdala reads; a linear probe of a general encoder, not a trained part.)*

![amygdala grounding — zero-shot approach/avoid separation](../experiments/amygdala_grounding/results/grounding.png)

**What it values (the labeling target):**
- **Approach (+):** other people and their positive affect (faces, smiling, laughter, warmth), nurturing / caregiving, safety and calm, resources. *Social cues dominate.*
- **Avoid (−):** danger (looming / fast-approaching things, heights, fire, aggression) — and above all **others' distress** (crying, screaming, fear, someone in pain).
- **Salience (attention):** sudden / looming / novel / moving things, faces and direct gaze, sharp or loud sounds.

**How to generate the data:** run agents (or curated video) through scenes / levels rich in those cues and label the affect — cheaply via CLIP/SigLIP text-anchoring or a VLM, or by hand for a personal layer — then fit a **thin head** (~10–50M → one signed valence) on the frozen perceptual features. Re-ground it on the trunk's `z_t` before agent use (Stage 4, [TRAINING.md](TRAINING.md)).

---

> **It can't be validated small.** At toy scale the felt system collapses to standard RL (see the [root README](../README.md#the-experiments-and-the-boundary-they-hit)); only a real body in a rich world can show the divergence the thesis predicts. The build therefore *ends* (Stage 6) at the honest test — at scale, does the felt agent keep developing with **no external reward** and **refuse to wirehead**? The parameter and rate numbers throughout are reasoned design choices, not measured optima. And **safety is not a footnote**: a body with self-generated desires is the maximal-autonomy configuration AI safety warns about — build it small, observable, bounded, and slowly (see [ARCHITECTURE.md](ARCHITECTURE.md) safety).
