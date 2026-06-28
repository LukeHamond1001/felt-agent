# Architecture — the felt system

The [thesis](README.md): for super-complex agency in huge environments there is no reward we can write, and the only guiding principle we know of is human internal feeling. This document is what that principle looks like when you try to build it — a multimodal embodied agent that hears, touches, and sees, is moved by feeling, and acts through movement and voice.

Every part is tagged: `borrowed` (use what works), the **felt system** (the guiding principle made concrete), or `the wall` (the parts with no shortcut).

---

## Signal flow

```
   vision (cameras)  ─┐
   audio (mics)       ├─►  multimodal trunk   ─┬─►  amygdala  ──────────────┐
   touch / proprio   ─┘    (cortex + memory)   │    (the ineffable pull)     │
                                               │                            ▼
                            pain (nociception) ┤    hypothalamus (drives)  felt signal
                                               │                            │
                                               └─►  action heads:           ▼
                                                     motor (diffusion/flow)  basal ganglia
                                                     voice (audio out)       (value / wanting)
                                                          │                  │
                            world model (imagination) ◄───┘   dopamine = RPE ◄┘
                                   │
                            imagine-and-feel ──► the amygdala scores imagined futures
                                                          │
                                          actions move the body → new perception (closed loop)
```

*The boxes are **functional organs within one co-trained model**, not separate networks wired together — the trunk, world model, and action heads share the single latent `z_t` of the unified cortex (the felt system attaches as a separate module that reads `z_t`). The split is functional, not architectural.*

---

## `borrowed` — perception, trunk, action

Don't reinvent these; the field has them.

- **Perception.** Pretrained vision (ViT / DINOv2 / SigLIP), audio (CLAP / wav2vec), and touch/proprioception (joint angles, IMU, force/torque, tactile). A body is a POMDP; touch and proprioception are not optional.
- **Trunk.** A multimodal transformer fusing the senses over a time window (working memory — a few seconds of history).
- **Action.** A diffusion / flow-matching action expert for high-DoF continuous control, plus an optional voice head. These are **methods proven in GR00T / π0 / OpenVLA, assembled into our unified world-model cortex** — one co-trained model that predicts *and* acts over a shared latent `z_t`, rather than a reactive perception→action map.

This backbone gives the agent *representation* and *competence*. It does not give it *what to want* — that is the felt system's job.

> **Target vs. v1.** The target is the **single unified cortex** above (predict + act, one shared `z_t`). The **[humanoid/](humanoid/)** spec is the pragmatic **v1 path** toward it: it borrows a frontier VLM-VLA trunk (the deployed-frontier shortcut) with a **separate ~30B world model** as an interim measure, *because* today's borrowed trunks are not self-predictive. As trunks become self-predictive (the JEPA / Cosmos line), that separate world model folds into the cortex and v1 converges to the target. The borrowed VLA trunk is the v1 shortcut, not the end state.

---

## The felt system — the guiding principle, made concrete

These are separate modules on purpose. What feels like one thing — "good/bad" — is several systems that dissociate, and collapsing them into one scalar reward is the standard-RL move this whole project argues against. **The *guidance* comes from a trio — amygdala (the pull) + basal ganglia (value) + pain (the floor).** The hypothalamus below is demoted: it is bodily maintenance and the infant bootstrap, not a source of open-ended guidance.

### Hypothalamus — drives `maintenance, demoted`
Innate homeostatic setpoints: energy, damage load, temperature. Deficits that **pull the agent toward** resources. Tonic, slow, and **satiable** — eat and the hunger is gone. The oldest reward, and the most bodily — but **not** part of the guidance trio. For an imitation-pretrained agent that already has a seeded amygdala-pull and pain (an attractor and a repulsor), drives reduce to a minimal upkeep utility (recharge when low, don't cook the motors). Keep it small.

### Pain — nociception
A dedicated organ for **all physical pain**, and deliberately *not* folded into the hypothalamus. Pain is a different signal with a different job:

| | hypothalamus (drive) | pain (nociception) |
|---|---|---|
| signal | a deficit (something is low) | tissue damage (something is being harmed) |
| direction | **approach** a resource | **withdraw** from the source |
| dynamics | slow, tonic, satiable | fast, phasic, **does not habituate** |
| reflex | none — it motivates | a fast protective reflex that **bypasses the planner** |

Its defining property is that it **does not habituate** — pain that faded would stop protecting the body. It has a sensory side (where/how much) and an affective side (the unpleasantness) — and that unpleasantness is still *not* the amygdala; it is a hardwired, verbalizable alarm. Pain runs a fast withdrawal reflex *and* feeds a strong, non-fading aversive learning signal. Of all the felt organs, pain is the one that is cleanly buildable, because its target *is* specifiable: damage.

### Amygdala — the ineffable pull
This is the heart of the architecture and the hardest part, and it is **not a good/bad classifier.** The amygdala is the system of our **innate dispositions** — the pre-verbal pulls toward and away that you cannot put into words — and so it is what decides **attention**: what you notice, what you can't look away from, and how long it holds you.

That unverbalizability is the whole problem. You cannot label "the particular way this draws my attention" or "this specific draw," so you cannot build a dataset of it, so you cannot supervise it the way you train everything else. A vision-language model captures only the *verbalizable shadow* — semantic similarity to affect words — not the pull that casts it. **That is why the amygdala is so weird to train, and why a picture-labeling proxy was always going to be a category error.** It is `the wall`. The closest grounding obtainable is a single emotionally-exceptional human mapping their felt affect *sub-frame-by-sub-frame* over rich audio/video (the protocol in [CONCEPTS.md §3](CONCEPTS.md)) — the felt report is the grounding signal; paired physiology, if recorded, is a later validation aid, not part of it — and even that only seeds the shadow richly; the pull itself is earned by living.

### Habituation — anti-wireheading and curiosity
Felt reward (pleasure, not pain) **fades with repetition** and recovers slowly (sensory-specific satiety). This does two things: it **blocks single-stimulus wireheading** (you can't sit on the best stimulus — it stops being best the moment you sit on it) and it turns "seek good" into "seek *new* good." It is **not** a guarantee — an agent could still time-share satiated sources or learn to pace stimulation; habituation **biases toward** open-ended variety, a protective heuristic, not a proof. Curiosity that **emerges from the affect dynamics** rather than a bolted-on novelty bonus — promising, and untested at the scale where it would matter.

### Norepinephrine — uncertainty as a gain/precision modulator
NE is **not another reward term.** It is a scalar **gain** the rest of the felt system runs *through*, driven by the agent's own uncertainty: `NE ← ema(prediction_error + ensemble_disagreement + novelty(z_t))`. When the world is surprising or the world model disagrees with itself, NE rises and **turns up the gain** — scaling the felt signal and the learning rate so surprising experience writes harder and gets attended to; when the world is predictable it settles. It modulates *how much* the existing signals count, never adding a value of its own. This is the precision/learning-rate knob (attend and learn fast under uncertainty, coast when confident), kept deliberately separate from the appetitive/aversive channels it scales.

### Imagination — one multimodal faculty, feeding a shared workspace
A single learned world model — **not one imaginer per sense.** When it imagines a future it renders the *whole multimodal state at once* (sight, sound, touch, proprioception), because the world is all of them together; auditory and visual imagery are the same faculty in different channels (inner speech is the one near-exception, also borrowing the speech-motor system). Both the real senses and the world model write to **one shared perceptual workspace**, and the amygdala reads that workspace **blind to real-vs-imagined** — which is exactly how *imagine-and-feel* works: imagine a future, feel it as if it were real, act toward the future that pulls hardest. The only safeguard required is a **reality-monitoring tag** (real vs imagined) so the agent feels the imagined threat without acting on it as present — that tag failing is hallucination. The open problem is faithful, far-reaching, multi-leg search; the *feeling* is the easy part, the *search* is the bottleneck.

### Basal ganglia — the value system
What is worth pursuing over long horizons. This is the anticipatory half of affect made durable, and it is `the wall`: it **cannot be pretrained**, because it's about *this* agent's lived consequences, which no dataset contains. It develops through embodied experience over time (optionally accelerated by imagine-and-feel), and in a mature agent it also serves as a **cache** of the expensive imagine-and-feel computation — compiling slow deliberation (System 2) into fast reflex (System 1). Dopamine — the reward-prediction-error — is the signal that trains it.

### Sleep — not a phase
Biological sleep consolidates offline because the brain can't learn while busy perceiving and acting and its fast buffer is small. An artificial agent has a replay buffer and learns continuously — **experience replay is perpetual sleep.** So there is no distinct sleep phase here; consolidation is continuous (high update-to-data-ratio replay). A real offline pass is warranted only for an agent that genuinely cannot learn while acting (real-time, compute-limited deployment).

---

## How it would actually be built

You cannot RL a humanoid from scratch on felt reward — it's sample- and safety-prohibitive. The realistic recipe also matches biology (innate priors, then a lifetime):

1. **Imitation-pretrain** the perception→action backbone on demonstrations (teleoperation, human video) → competent base skills.
2. **Attach the felt system** — drives + pain (the buildable innate organs) + the amygdala + habituation + the world model and value cache.
3. **Feeling-driven open-ended learning** — the felt system becomes the autonomous drive that refines and extends the imitation-learned skills toward what the agent is drawn to, exploring via curiosity that emerges from habituation, after the demonstrations run out.

The felt system is not the *learner* of basic motor skill — imitation does that. It is the **guiding principle that decides what to pursue when there is no reward to hand the agent** — which is the entire point.

---

## Safety

An agent with self-generated, persistent, grounded desires it pursues and acts on is precisely the configuration AI safety treats as the core risk. The argument that *we must build the human guiding system because nothing else scales* is also an argument for building the most autonomy-laden kind of agent there is. Build it small, observable, bounded — and treat the autonomy that makes it interesting as the thing to be careful about.
