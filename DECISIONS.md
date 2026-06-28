# Design decisions — what we chose, what we rejected, and why

The point of this document is that **every major fork was considered**, not stumbled into. If you wonder "why didn't they just do X?", the answer is here. Honesty is the project's discipline, so the rejected options and the open bets are stated as plainly as the choices.

---

## 0 · Where this sits on the frontier

There are **two** robotics frontiers, and they are not the same:

- **The deployed/shipped frontier** = the **LLM-VLA bolt-on** (Figure Helix, π0/π0.5, NVIDIA GR00T, Gemini Robotics). These control real robots *today*.
- **The research / future frontier** = the **world-model-centric agent** (DreamerV3, **V-JEPA 2-AC** [Meta 2025], **Cosmos** [NVIDIA 2025], TD-MPC2; LeCun's world-model blueprint). Predict-and-imagine, not react-and-imitate.

**We are on the world-model frontier** — current, well-backed, *rising*, and already demonstrated on real robots (V-JEPA 2-AC, 2025) — but **behind the VLA camp on shipped-on-hardware reliability.** So our backbone is *frontier on research and trajectory, underdog on deployment*. The felt system on top is **beyond** the current frontier — nobody builds it — which is the unproven part by design.

Honest one-liner: **we are betting the world-model camp is where robotics ends up (a real, serious bet), and adding the one thing neither camp has.** If world-models win (LeCun's thesis), we're early on the right substrate; if VLAs keep winning, our backbone is the underdog camp — but in *either* case the felt system is the differentiator, and it *requires* a predictive backbone the VLA camp can't provide.

---

## 1 · The major decisions (chose → rejected → why)

**Backbone: world-model-centric — *not* an LLM-VLA bolt-on.**
*Rejected:* Figure/π0/GR00T-style (a pretrained VLM with an action head + adapters bolted on).
*Why:* the VLA bolt-on is **reactive** — it maps perception→action with no model of the future, so it **cannot imagine**, cannot plan over imagined outcomes, cannot learn from imagined experience. Our entire mechanism (imagine-and-feel) needs prediction at the core. A predict-next world model has it; a reactive policy never can. (The VLA camp is *better for shipping a manipulator today* — we're optimizing for autonomy, which forces the harder substrate. See [BUILD.md](BUILD.md), [ARCHITECTURE.md](ARCHITECTURE.md).)

**One unified model — *not* "a frontier LLM with sensory/motor/world-model parts welded on."**
*Rejected:* the industry frankenstein (frozen text-LLM + adapters + separate world model + separate action head).
*Why:* a text-pretrained LLM was never a sensorimotor cortex; its representation is *descriptive*, not *dynamics-predictive*. We train **one** temporal multimodal model with a unified self-predictive + action objective, so perception/prediction/action share one latent `z_t`. Language is one modality it's trained on, not the skeleton.

**The felt system is a *separate* module reading `z_t` — not an output head of the cortex.**
*Rejected:* folding reward into the cortex as just another head.
*Why:* the cortex *represents and predicts*; the felt system *values*. That's a different functional type, and it mirrors real anatomy (cortex vs limbic). Keeping it separate is also what lets us **ground it from human affect** independently. The only other thing outside the cortex is the fast reflex controller — and that's not a bolt-on, it's *distilled out* of the one model (cortex → cerebellum/spinal arc) for latency.

**Borrow the backbone; spend all originality on the felt system.**
*Rejected:* inventing a novel backbone.
*Why:* the world-model backbone is a **known, de-risked paradigm** (Dreamer/JEPA/Cosmos). Inventing there wastes the originality budget. Concentrating novelty in one module (the felt reward) means the only unproven thing is *our* contribution, and "could the backbone work?" is already answered by others.

**Adopt each frontier lab's lead; keep ours where we lead.**
*Rejected:* copying any single stack wholesale (impossible — three are closed) or ignoring them.
*Why:* the leaders **don't overlap** — PI leads learning (**RECAP**, KI, RTC), NVIDIA leads infra (Cosmos, Isaac/Newton, SONIC, DreamGen, relative-EEF), Gemini leads reasoning (NL thinking-trace, success estimator), Figure leads the shipped WBC + tactile. We take the best *methods* from each (weights aren't borrowable; methods are) and keep our distinctive bets (foveation, auditory attention, felt/world-model/orchestrator). *Honest caveat:* a paper composite of best-of-breed parts is **not** a co-designed shipped system — integration is real risk, and ours is un-run while theirs ship.

**Vision: foveated active vision — *not* uniform high-res.**
*Rejected:* full-frame high-res everywhere.
*Why:* high-res is *possible* for one datacenter frame, but the binding case is multi-cam × real-time × **continuous world-model + replay for a lifetime** — where uniform high-res makes the always-on loops the most expensive *and* wastes world-model capacity on irrelevant periphery (less faithful imagination). Foveation points capacity where attention is. (Figure's palm-cameras are adopted as the cheap v1 manipulation fovea — [humanoid/PERCEPTION-ACTION.md](humanoid/PERCEPTION-ACTION.md).)

**Label the amygdala *pull* (signed valence) — *not* dopamine/cortisol.**
*Rejected:* annotating dopamine and cortisol channels.
*Why:* those are **downstream** — dopamine is the agent's *computed* RPE (pull − its own prediction); cortisol is a slow low-pass of sustained negative pull. Neither is introspectable *or* measurable in the cortex (sub-second dopamine needs invasive electrodes; cortisol is a slow peripheral assay). What a human *can* report is the felt pull — so that's what we label; the rest is derived. (Same category error as "CLIP is an amygdala.")

**Annotation: one signed-valence bar — *not* a 2-D pad or multi-channel.**
*Rejected:* a valence×arousal pad / separate wanting/liking/threat channels.
*Why:* a single bar's *magnitude = intensity* (attention/beauty) and *sign = toward/away* already captures the felt pull; the **specific quality lives in the perception-binding, not the label.** Adding a "wanting" channel would label the value system (the BG) — the thing that must be *earned by living*, not annotated. Usability also favors one continuous control. ([amygdala-trainer/SPEC.md](amygdala-trainer/SPEC.md).)

**A thin amygdala head on *frozen* perception — *not* a large trained amygdala.**
*Rejected:* training a big affect model on the annotations.
*Why:* one person's annotation is limited and temporally correlated; a big trainable model overfits instantly. Capacity lives in the frozen perception; the felt data only fits the ~10–50M read-out. (You do **not** train a 200B amygdala.)

**Sleep = continuous high-UTD replay — *not* a distinct sleep phase.**
*Rejected:* an offline "sleep" consolidation stage.
*Why:* biological sleep is a workaround for *not being able to learn while acting*; an artificial agent has a replay buffer and learns continuously — **experience replay is perpetual sleep.** A separate phase tied/lost to plain replay in our own tests. (Kept only for a real-time agent that genuinely can't learn online.)

**Dropped EWC for continual learning.**
*Rejected:* EWC (Fisher-penalty) for catastrophic forgetting.
*Why:* our own prior experiments found **EWC fails** (= finetune). We keep replay-CLEAR + plasticity-loss resets (ReDo/soft-reset) with an explicit arbitration trigger instead.

**No small-scale experiments / benchmarks in the repo.**
*Rejected:* shipping a toy demo as "validation."
*Why:* at small scale **every organ collapses into standard RL** — a toy positive is an artifact, a toy null is predicted; either way uninformative. Shipping toy "validation" would be dishonest. The thesis is **not a capability claim** (the frontier already wins on capability *without* a felt system), so a benchmark wouldn't even test it. (The one public empirical artifact is the separate, honest `curiosity-locomotion` repo.)

**The felt system is justified by *autonomy*, not capability.**
*Rejected:* claiming the felt system makes a better manipulator.
*Why:* all four frontier labs ship world-class robots with **no felt system and no world model** — so capability is *not* where the edge is. The felt/world-model layer earns its keep *only* in the open-ended, no-task-reward, lifelong regime: an agent that *wants* with no one instructing it. That sharpens the bet to exactly the thesis.

---

## 2 · What we are NOT claiming (the wall, stated plainly)

- **The ineffable amygdala pull cannot be fully built** — only the perceptual *shadow* is groundable; the rest is earned by living. We deliberately build **no placeholder module** for it.
- **The value/anticipatory system has no pretraining shortcut** — it's an unlived life; it develops only through embodied experience.
- **The thesis is hard to falsify** — "only works at scale" is the shape of an unfalsifiable claim; we own it and state the divergence prediction (no-reward open-ended development + no wireheading) that would falsify it, while admitting the "needs more life" escape hatch.
- **It is unbuilt** — a paper composite + a novel felt bet, not a shipped system. Every frontier lab ships; we have a spec and one honest locomotion result.

The honest summary: **borrowed (de-risked) world-model backbone + a novel, unproven felt system, on the rising research frontier — a real bet, cleanly isolated, stated without spin.**
