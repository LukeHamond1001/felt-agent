# this humanoid spec

**A complete design for a ~200B-parameter humanoid: today's best robot-foundation-model backbone, with the [root thesis (../)](../README.md) felt system attached as a separate module reading `z_t` — the thing that decides what to *want*.**

> **This is the v1 de-risked path toward a unified world-model cortex.** The *target* is **one unified world-model-centric cortex** — a single co-trained model that predicts *and* acts, sharing one latent `z_t` (the thesis). This spec is the **pragmatic v1 bootstrap toward it**: borrow a frontier VLM-VLA trunk and run a **separate ~30B world model as an interim measure, precisely because today's borrowed trunks are not self-predictive**; as the trunk line goes self-predictive (JEPA/Cosmos), that world model folds into the cortex and the stack converges to the target. (The LLM is **rejected as the sensorimotor cortex** but **accepted as a deliberation tier above it** — the Tier-0 orchestrator in [ARCHITECTURE.md §2](ARCHITECTURE.md#2--the-backbone--a-tiered-vla-borrowed).)

The backbone is proven and borrowed. The felt system is the bet. The split is the whole point:

- **The backbone** (vision-language-action foundation model) gives the body *competence* — it perceives, understands instructions, and produces skilled whole-body motion. This is real, working technology (GR00T, π0, Helix-class systems). Use it.
- **The felt system** is designed to *enable* autonomy — to be what the body pursues when the instructions run out and there is no reward to hand it. Whether this design succeeds is the central bet. This is the [thesis](../README.md): for open-ended embodied life there is no reward anyone can write, and the only known guiding principle is human internal feeling. So we build that, and **attach it as a separate module reading the trunk's latent `z_t`** on top.

A humanoid is the *right* body for this, because the felt system can only be grounded in felt reality — and only an embodied agent living a real life has anywhere real to point.

---

## The two halves

```
                         ┌─────────────────────────  the felt system (the bet)  ─────────────────────────┐
                         │   hypothalamus (drives) · pain (nociception) · amygdala (the ineffable pull)   │
                         │   habituation · imagination (world model) · basal ganglia (value) · dopamine   │
                         └───────────────────────────────────▲───────────────┬───────────────────────────┘
                                          reads latent state  │               │  generates the reward / what-to-want
                         ┌────────────────────────────────────┴───────────────▼───────────────────────────┐
   cameras · mics ──────►│  System 2: multimodal VLM trunk + memory   (semantic perception + reasoning)     │
   tactile · proprio ───►│  System 1: flow-matching action expert     (high-frequency whole-body control)   │
   force/torque ────────►│  low-level: sim-to-real whole-body controller (balance, locomotion, contact)     │──► actuators · voice
                         └──────────────────────────  the backbone (borrowed, proven)  ────────────────────┘
```

The backbone is a **dual-system VLA**: a slow multimodal transformer that sees and reasons (System 2), driving a fast action expert that emits whole-body motion chunks (System 1), over a sim-to-real low-level controller for balance and contact. The felt system sits *above* it, reading the trunk's latent state and producing the agent's own objective — so that after demonstrations and instructions are exhausted, the humanoid still has something it is reaching for.

Full design in **[ARCHITECTURE.md](ARCHITECTURE.md)**. The build pipeline in **[TRAINING.md](TRAINING.md)**. How it senses and acts — active foveated vision, auditory attention, and outputs, all allocated by one attention system — in **[PERCEPTION-ACTION.md](PERCEPTION-ACTION.md)**.

---

## ~200B parameters, allocated

Illustrative, not empirically optimized — but in the proportions real VLAs use (the trunk dominates).

| component | role | ~params |
|---|---|---|
| Multimodal VLM trunk (System 2) | vision + audio + language + semantic reasoning | ~135 B |
| World model (imagination) | latent dynamics for imagine-and-feel | ~30 B |
| Felt system | amygdala, drives, pain, basal-ganglia value, habituation | ~15 B |
| Action expert (System 1) | flow-matching whole-body manipulation control | ~8 B |
| Memory / hippocampus | episodic store + continuous consolidation | ~7 B |
| Low-level WBC | sim-to-real balance / locomotion / contact | ~3 B |
| Voice head | vocalization | ~2 B |
| **total** | | **~200 B** |

> The world model (imagination) is architecturally part of the felt system but is budgeted separately above because of its size (~30B vs ~15B for the core felt organs).

## The control hierarchy (and where feeling lives in it)

| layer | function | rate |
|---|---|---|
| **pain reflex** | protective withdrawal — bypasses everything | ~1 kHz |
| low-level WBC | balance, locomotion, contact/force | ~0.5–1 kHz |
| action expert (System 1) | manipulation action chunks | ~50–200 Hz |
| world-model planning + value (imagine-and-feel, BG) | mid-level deliberation | ~10–50 Hz |
| **VLM + amygdala appraisal** (System 2 + the pull) | reasoning + what-to-want | ~1–10 Hz |

The felt system maps cleanly onto the hierarchy: **pain is the fastest loop** (a reflex, like a spinal arc), and **the amygdala is the slowest** (the pull that sets direction). Feeling brackets the stack at both ends.

---

## How the felt system attaches (a separate module reading `z_t`)

This is the only genuinely new engineering; everything else is integration of known parts.

1. **It reads, it doesn't replace.** The felt organs take the trunk's latent state as input — they don't re-encode the world. The backbone already built the representation; the amygdala reads a *pull* off it, the hypothalamus reads *drives*, pain reads *damage*.
2. **It generates the objective.** The felt signal (amygdala pull + drives − pain, modulated by habituation) *is* the reward for ongoing learning. No external reward function — the agent makes its own, which is the entire thesis.
3. **It plans by feeling futures.** The world model lets the agent imagine rollouts and let the amygdala feel them (*imagine-and-feel*); the basal ganglia caches that verdict into a fast value head (System 2 → System 1).
4. **It resists wireheading.** Habituation fades pleasure (not pain), so "seek good" becomes "seek new good" — open-ended curiosity that emerges from the affect dynamics, not a bolted-on bonus. It blocks single-stimulus fixation; it biases toward variety rather than guaranteeing it.

---

## Honest caveats

- **The backbone is proven; the felt system is not.** This design borrows a working stack and stakes everything on the felt module attached to it. The hard parts have no shortcut: the amygdala's target is *ineffable* (you can't label beauty, so you can't supervise it) and the value system is an *unlived life* (no dataset holds this body's future). Those are carried over from the root thesis unchanged — naming the bet, not hiding it.
- **It can't be validated small.** At toy scale the felt system collapses into standard RL; only a real body in a rich world can show the divergence the thesis predicts (open-ended development with no external reward; refusal to wirehead).
- **It is hard to falsify — the deepest caveat.** That divergence prediction is the honest falsification target, but the "it just needs more life/scale" escape hatch means almost any failure can be blamed on insufficient embodiment rather than a wrong idea. This design bets on a hypothesis that resists clean testing — one to be lived and observed more than proven. Naming that is part of the honesty, not a hole in it.
- **The numbers are illustrative.** Param allocations and control rates are reasoned design choices, not measured optima.
- **Safety is not a footnote.** A 200B humanoid with self-generated, persistent desires it pursues with a body is the maximal-autonomy configuration AI safety warns about. The argument *for* building it (nothing else scales) is also the argument for building it small, observable, bounded, and slowly. See [ARCHITECTURE.md](ARCHITECTURE.md#safety).

## License

MIT — see [LICENSE](../LICENSE).
