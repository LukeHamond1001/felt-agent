# Perception & action — active, foveated, attention-allocated

How the humanoid takes in vision and audio, and how it acts — designed with one principle:

> **Attention is the shared currency across perceive → think → act, and the amygdala's pull allocates it.** One orienting loop: salience aims the fovea/beam → high-res content enters the shared workspace → the amygdala feels it → it drives action, *including where to look/listen next*. Foveation and the felt system are the same attention system.

Tags: **`standard`** (buildable today), **`bet`** (novel integration). All numbers are starting points to tune. A few `standard` pieces are explicitly **adopted from Figure's Helix** where Figure already ships them in hardware (palm cameras, visual proprioception, a real talking voice path) — adopted to de-risk our distinctive bets (flow-matching+RTC, foveation, auditory attention, the felt/world-model/orchestrator stack), not to replace them. **The perception-action backbone is BORROWED** — a few load-bearing pieces are taken from current SOTA generation (state-relative EEF action space and native-aspect-ratio encoding from **NVIDIA**, train-time-conditioned RTC from **PI**), each named below. The distinctive bets stay ours.

---

## 1 · Vision — active foveation (not uniform high-res)

**Why not all-high-res.** For one image in a datacenter it is feasible (AnyRes VLMs do it), but the binding case is multi-camera × real-time × partly onboard × **continuous world-model + replay for a lifetime**. There, uniform high-res has real downsides beyond peak FLOPs: (i) the always-on loops (imagination, high-UTD consolidation) become the most expensive and the world model wastes capacity on irrelevant periphery → **less faithful** imagination; (ii) onboard power/thermal/latency; (iii) sample/representation inefficiency — foveation is an inductive bias that points capacity where it matters. So: foveate.

**Design (`standard` optics + `bet` gaze-from-affect).** Per camera, two streams:
- **Peripheral:** wide field, **low-res**, always on (~64–128 tokens). Situational awareness; motion/threat/salience detection.
- **Foveal:** a small **high-res crop** at the current gaze point (~224–336 px → full ViT). Task acuity.

**Aiming has two speeds (like eye saccade + head turn):**
- **Digital saccade** — shift the high-res crop within the wide frame. Fast, free, no motor. Runs a few Hz.
- **Head/neck motor** — large reorientation, at the WBC rate.
- The crop **rests at center by default** and *jumps* when pulled.

**What moves it (the orienting reflex).** A cheap **saliency map over the low-res periphery**, biased by the **amygdala's pull** (salient/threatening/novel → look there) + the task goal. Strong-salience orienting can be partly hardwired (a flinch/reflex, like the superior colliculus); the rest is learned. Gaze is an **action** (see §3).

**Build-to-a-tee:** periphery encoder = **DINOv3** ViT-S-class over a downsampled wide frame; foveal encoder = **DINOv3** ViT-L-class over the crop (a **SigLIP 2**-class encoder is the language-aligned alternative where vision-language grounding matters more than dense features; or a **C-RADIO**-style unified encoder distilling both — current SOTA generation, verify exact version/availability at build time, this moves fast); saliency = a small conv head over periphery features + a linear projection of the amygdala pull, softmax over locations → next crop center; saccade rate ~3–5 Hz; both streams tokenized into the workspace with a `fovea|periphery` channel tag and the gaze (x,y,scale) appended as tokens.

**Native-aspect-ratio / flexible-resolution encoding (`standard`, borrowed-from-NVIDIA).** Encode each stream at its **native aspect ratio with variable token counts** — no letterbox padding, no fixed-square crop. Both the wide peripheral frame (typically landscape) and the foveal crop feed the ViT at their real shape, the patch grid flexing to the resolution (the position-embedding scheme handling arbitrary H×W token grids — current SOTA generation, verify exact version/availability at build time, this moves fast). For a foveated front-end this is **strictly better than fixed-square crops**: no FLOPs wasted encoding pad pixels, no acuity thrown away by squashing a wide periphery into a square, and the foveal crop can be sized to the *content* (a thin tool, a wide face) rather than forced to a canonical box — capacity points exactly where the fovea aims it. Cheaper *and* sharper, which compounds in the always-on imagination/consolidation loops where the encoder runs constantly. Token budgets above (~64–128 periphery, ~224–336 px fovea) become **targets the variable-resolution grid tunes to**, not hard square sizes.

**Palm/wrist cameras — the v1 manipulation fovea (`standard`, adopted from Figure's Helix, which uses palm cameras instead of a learned fovea).** A cheap hardware route to high-acuity-where-the-hands-are: mount a small camera in each palm/wrist. For manipulation, the place that needs the most acuity *is* wherever the hands are, and a palm-cam delivers it for free — no motor, no saccade, no occlusion by the body. **Feed the palm-cam stream straight into the fovea token slot** (same `fovea` channel tag, same foveal encoder), so the rest of the stack is agnostic to whether the high-res crop came from a digital saccade or a wrist camera. Frame it this way: **palm-cam = the v1 manipulation fovea that de-risks the learned digital-saccade bet.** The learned head-mounted digital saccade stays the full version on top — palm-cams cover the manipulation workspace today while the general gaze-from-affect fovea (which also handles looking *away* from the hands, at faces, at distant salience) comes online. The two compose: head-fovea for orienting/scene, palm-fovea for the in-hand task.

**Visual proprioception (`standard`, adopted from Figure's Helix — the agent seeing its own hands in-frame).** Because both the head camera and the palm cameras routinely catch the hands in-frame, the agent **sees its own end-effectors** and can self-calibrate: visual hand pose corroborates (or corrects) joint-encoder proprioception, closing the loop on kinematic drift, slop, and miscalibration without an external tracker. Tag these tokens so the model can learn "that's *my* hand" — it grounds the body schema in vision, which the felt/world-model stack (§[ARCHITECTURE.md](ARCHITECTURE.md)) can then predict and use.

**MVP → full:** v1 = **palm/wrist cameras as the manipulation fovea** + head-camera fovea fixed at center, aim by head/neck only (no digital saccade). Full = add digital saccades + learned gaze on the head camera, with palm-cams retained as the standing manipulation fovea.

---

## 2 · Audio — auditory attention (the cocktail-party problem)

Foveation for sound. Same structure as vision:
- **Peripheral audio:** an always-on **coarse encoder over the full mixture** — awareness + the **auditory orienting reflex** (a sudden/loud sound grabs attention, amygdala-driven, exactly like peripheral motion).
- **Foveal audio = the attended stream:** **source-separate / beamform** one source (a voice, a sound of interest) and run a richer encoder on it.
- **"Where to listen" = which source/direction**, chosen by amygdala salience + task — the audio analog of gaze.
- **Aiming, two speeds:** **mic-array beamforming** (fast, electronic — the audio "saccade") + **head turn** (physical, slow).

**Build-to-a-tee:** mixture encoder = a streaming conformer or a frozen coarse audio encoder over the full array sum (coarse awareness only); attended stream = a separator (TF-GridNet / Conv-TasNet) **or** a delay-and-sum/MVDR beamformer steered by direction-of-arrival (GCC-PHAT over the array); attended-stream encoder = **AF-Whisper** (the Audio Flamingo 3 encoder — rich, speech + general-audio + language-aligned, current SOTA generation, verify exact version/availability at build time, this moves fast). DOA drives both the beamformer and the head-orienting target. Tokens enter the workspace with a `attended|ambient` tag. (Inner audio / auditory imagination already feeds the workspace via the world model — §[ARCHITECTURE.md](ARCHITECTURE.md).)

---

## 3 · Outputs — motor foveation, gaze-as-action, gated voice

The output mirror of foveation is **precision allocation**.

- **Foveal action** (high-rate, high-precision): the **attended effector** (the hands doing the task) — the flow-matching action expert at 50–200 Hz.
  - **Real-Time Chunking, conditioned at train time (`standard`, borrowed-from-PI).** Flow-matching emits action *chunks*, so the seam between consecutive chunks needs smoothing to avoid jerk when a new chunk overwrites the tail of the running one. Do this by **conditioning the action expert on the in-flight (already-committed) actions during training** — the model learns to generate a chunk that continues smoothly from what's still executing — rather than by inference-time inpainting that stitches chunks together after the fact. Train-time conditioning is cleaner and faster at inference (no extra denoising-with-constraints pass on the critical path), which matters for the 50–200 Hz foveal loop and even more for the always-on imagination rollouts that replay this same expert. Prefer it over inference-time inpainting — current SOTA generation, verify exact version/availability at build time, this moves fast.
- **Peripheral action** (automatic, cheap): **posture, balance, locomotion** — the low-level WBC reflexively, not deliberated each step. *This is the existing control hierarchy, reframed as attention allocation: deliberate precision goes where attention is; the rest runs automatic.*
- **Gaze/orientation is an ACTION.** Where to look and point the ears (head/neck + digital-fovea (x,y,scale) + beamform direction) are **outputs**. So attention is simultaneously input-selection *and* action — perception and action are one loop. The action vector therefore includes **gaze/orientation channels** alongside the joint targets.
- **Epistemic action (`bet`).** Because looking is an action and the world model predicts what you'd see, **imagine-and-feel can plan *where to look/listen*** — "is it worth turning to check?" Curiosity (habituation-driven) becomes a drive to *look*; threat becomes a drive to *check*. Information-seeking falls out of the felt system, no separate objective.
- **Voice output (`standard`):** a gated, intentional vocalization head — mostly silent (silence is the "periphery" of speech); externalized inner speech when the agent chooses. Gate = a learned scalar; content = CLAP-phoneme features or discrete audio tokens.
  - **Stand up a working speech-to-speech path EARLY (adopted from Figure's robots, which talk — ours is currently only a specced gate).** Don't leave voice as a paper feature behind the gate: wire an end-to-end speech-to-speech voice loop (mic → ASR/audio-LM → spoken reply) into the running system early, so the agent can actually converse from the start. **Keep our gated-voice + inner-speech framing on top:** the speech-to-speech path is the *transport*; the learned gate still decides *when* to speak (mostly silent), and what surfaces is externalized inner speech, not a chatbot that narrates constantly. Standing it up early de-risks the integration the same way palm-cams de-risk the fovea — a real voice channel today, the felt/intentional gating layered over it.
- **Felt/learning signal for gaze & voice (`bet`) — imagine-and-feel scores more than motor/pain futures.** These dims are in the action vector, so they must be *valued*, not just emitted. **Gaze:** the world model is conditioned on the gaze action and predicts the resulting `z_t` (what you'd see if you looked there), so imagine-and-feel scores a look by its **predicted info-gain / reduction in felt uncertainty (the NE channel)** — "worth turning to check?" gets a real predicted consequence the value head can feel, exactly like a motor action. **Voice:** in v1 the gate + content are **teleop-supervised** (imitation, like the rest of Stage 2) — there's no clean felt consequence for speaking yet; later the felt signal can shape it (social/communicative outcomes the amygdala values). **Inner speech** is the same voice content rendered to the *imagined* (`reality-tag = imagined`) workspace instead of the speaker — felt, not externalized — closing the auditory-imagination loop (§[ARCHITECTURE.md](ARCHITECTURE.md)).

**Action vector layout (build-to-a-tee).** Two interchangeable manipulation modes share one chunk:
- **Absolute joint targets** — `joint_targets(25–40 DoF)`, the WBC-native default; precise, robot-specific, what the low-level controller ultimately tracks.
- **Relative / state-relative end-effector mode (`standard`, borrowed-from-NVIDIA).** Express the hands' command as a **delta pose of each end-effector relative to its current state**, in a frame shared between human and robot, rather than absolute joint angles. This is **what makes human-video transfer work**: a hand moving 5 cm forward and rotating 10° is the *same action* whether it's a person's wrist in a YouTube clip or the robot's gripper, regardless of arm geometry or where the limb started — so demonstrations learned from human video map onto the robot's body without a per-embodiment retargeting step. A small IK/retarget layer turns the relative-EEF command back into joint targets the WBC tracks. Treat relative-EEF as the **transfer-friendly action representation** (train from human video here) and absolute joint targets as the **execution representation**; the action expert can emit either, selected by a mode flag — current SOTA generation, verify exact version/availability at build time, this moves fast.

`a = [ manip( joint_targets(25–40 DoF) | relative_EEF_delta(per-hand Δpose) ) + mode_flag | gaze(head/neck targets + fovea x,y,scale) | beamform_dir | voice_gate(1) + voice_content(D) ]`, emitted as flow-matching chunks `ℝ^{H×dim}`, H≈25–50.

---

## 4 · The loop, closed

```
 amygdala pull / task ──► saliency over low-res periphery (vision) + mixture (audio)
        │                                   │
        ▼                                   ▼
   orient: digital saccade + beamform (fast) ; head/neck turn (slow)
        │                                   │
        ▼                                   ▼
   high-res foveal crop  +  attended audio stream  ──► shared workspace (tagged fovea/attended, real/imagined)
                                   │
                                   ▼
                        amygdala feels it ; value + imagine-and-feel
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        ▼                                                        ▼
   motor (attended effector precise, periphery automatic)   next gaze/listen target (epistemic action)
```

Looking, listening, and acting are one orienting system driven by the felt pull. That is why the foveated front-end and the amygdala belong to the same architecture — the thing that decides *what is worth attending to* is the thing that decides *what is worth doing*.

**The legible intermediate.** This whole loop — what got attended to, what the amygdala felt about it, why gaze/listen/motor went where they did — is exposed in natural language as the **NL thinking-trace** (the system-2 deliberation layer, see §[ARCHITECTURE.md](ARCHITECTURE.md)). The trace is the human-readable seam between perceive → think → act: it narrates the orienting decision ("loud sound left, turning to check"), the felt pull, and the chosen action *before* the chunk is emitted, so the perception-action loop isn't a black box but a stream you can read, audit, and steer.
