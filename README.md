# felt-agent

**A bet about how agents will have to be guided once the world gets too big to write a reward for.**

> **What this is:** a *world-model-centric* agent — the research/future frontier (Dreamer / V-JEPA 2-AC / Cosmos lineage) — with a **felt system** (a grounded amygdala + value + pain) as its *self-generated* reward, which no frontier lab builds. Borrowed, de-risked backbone; all the originality in the felt layer.
>
> **Target vs. v1.** The *target* backbone is **one unified world-model-centric cortex** — a single co-trained model that predicts *and* acts over a shared latent `z_t`. That is the thesis. The **[humanoid/](humanoid/)** spec is the pragmatic **v1 path toward it**: it borrows a frontier VLM-VLA trunk (the deployed-frontier shortcut) with a **separate ~30B world model** as an *interim* measure — precisely because today's borrowed trunks are not self-predictive. As trunks move to self-predictive (the JEPA / Cosmos line), that separate world model **folds back into the cortex** and v1 converges to the target. Read humanoid as "v1 path toward the unified cortex," not "the backbone, full stop."

### Repository map
- **[README.md](README.md)** (the thesis) · **[CONCEPTS.md](CONCEPTS.md)** (the deep ideas) · **[ARCHITECTURE.md](ARCHITECTURE.md)** (the felt-system / brain design)
- **[BUILD.md](BUILD.md)** (step-by-step build recipe)
- **[humanoid/](humanoid/)** — the ~200B embodied instantiation: [architecture](humanoid/ARCHITECTURE.md) · [training](humanoid/TRAINING.md) · [perception-action](humanoid/PERCEPTION-ACTION.md)
- **[amygdala-trainer/](amygdala-trainer/)** — the runnable tool to start grounding the amygdala from your own felt affect

---

## The thesis

Standard reinforcement learning works because someone hands the agent a reward and it maximizes it. That works beautifully in small, closed worlds. It breaks as the world grows.

The bigger and more open the environment, the less anyone knows what the reward even *is*. What is the reward function for *living* — for acting well across a vast, open, unbounded world with no episodes, no score, no designer watching the whole time? No one can write it. Reward specification is the part that does not survive contact with real complexity. Honestly: **for super-complex agency in huge environments, we have no guiding principle at all.**

Except one. And it isn't in our equations — it's inside us.

A human moves through an environment of incomprehensible complexity with no reward function anywhere. What guides us is **internal feeling** — our **natural inclinations**, the ones we can't put into words; hunger, pain, care; what draws us in and what we recoil from. That felt system is the only working solution to open-world agency that we know exists. Evolution arrived at it precisely because nothing else *could* be specified — and it runs, fully operational, inside every person.

So the bet is simple, and it's a bet about *where the guiding principle comes from*:

> **Simple RL** — the principle is a reward we write.
> **Super-complex RL** — there is no reward we can write, and the only guiding principle we know of is **human internal feeling.** So that is what has to be built.

Everything in this repo follows from that. The felt system described here — the **amygdala**, the **drives**, **pain**, the **value system** — is not a trick to win benchmarks. At small scale it provably *can't* be (next section). It is an attempt to instantiate the one principle that works in the regime where we otherwise have none.

---

## Why there are no experiments in this repo

There used to be. They're gone, on purpose — and the reason *is* the thesis.

**At small scale, this architecture is just standard RL.** Every distinctive organ collapses into its ordinary RL equivalent the moment the world is small:

- The **amygdala** is attention, beauty, and inclination *over a rich perceptual world.* A toy with a handful of states and a scalar reward has no world to be drawn to — "what is beautiful here" is meaningless — so the amygdala degenerates into a plain reward function. → shaped-reward RL.
- The **value system** only has to be *developed* when horizons are long, signal is sparse, and the world is rich. In a toy, any dense signal grows it directly. → the basal ganglia adds nothing.
- **Habituation** over a few states is just a novelty bonus. → standard intrinsic-motivation RL.
- **Imagination** in a tiny known world is just MPC. → standard model-based RL.

So the distinctiveness is not an algorithm you flip on in a gridworld. It's a property that *only has room to exist* when the world is big enough for affect to have structure to latch onto, horizons long enough that value must be earned, and tasks open-ended enough that variety matters. Test it small and you are, by construction, measuring standard RL in an amygdala costume. A toy *positive* is almost certainly an artifact; a toy *null* is exactly what the thesis predicts. **Either way it tells you nothing** — which is why shipping toy "validation" would be dishonest. The only informative test is going all out.

This is the same shape as the thesis itself: the felt system is invisible until the environment is complex enough to need it, because for simple environments a written reward already suffices. You don't reach for the thing inside us until the thing we can write runs out — and it runs out exactly at the scale where you can't run a clean experiment on a laptop.

**And you can't sidestep this by training one on a *different*, simpler reality.** The felt system is not a generic reward function — it is **adapted to *this* reality over evolutionary and developmental time**, shaped by and for the specific world it grew up in. A toy or man-made environment isn't a smaller version of the right test; it's the *wrong world* — the amygdala has nothing to be drawn to there because it was never adapted to it. And the dependence runs the other way too: **everything humankind does flows *from* this felt system** — our art, our tools, the very environments we build — so any world we could construct to test it is itself a *product of* the real-world-adapted amygdala, and therefore *encompassed by* the reality that amygdala already spans. Our little man-made worlds are a tiny subset of what the felt system reaches; a subset can't reveal the system that generated it. The only environment rich enough to evaluate it is the one it was adapted to — reality itself.

---

## The architecture, in one breath

A multimodal embodied agent that hears, touches, and sees, is moved by **feeling**, and acts through movement and voice:

- **`borrowed`** — perception (vision/audio/touch encoders), a multimodal trunk with memory, and a diffusion/flow action expert. These are **methods proven in GR00T / π0 / OpenVLA, assembled into our unified world-model cortex** (one co-trained model that *predicts and acts* over a shared latent `z_t`). The borrowed VLA trunk is the **v1 shortcut** — see below. Use what works.
- **the felt system** — the guiding principle, instantiated. The **guidance trio** does the steering: the **amygdala** (the ineffable pull — attention, beauty, inclination), the **basal ganglia** (the value system — what to pursue), and **pain** (the protective floor). Around them: **habituation** (reward fades, so the agent seeks variety rather than wireheading), **imagination** (feel imagined futures — visual *and* audio), and the **hypothalamus** (drives) demoted to bodily maintenance, not guidance.
- **the wall** — the two parts with no shortcut: the amygdala's target is *ineffable* (you can't label beauty, so you can't supervise it) and the value system is an *unlived life* (no dataset holds your future).

The **LLM is rejected as the sensorimotor skeleton/cortex** — a text-pretrained representation is *descriptive*, not dynamics-predictive, so it cannot be the predict-and-act backbone. It is **accepted as a deliberation tier *above* the cortex**: an NL-reasoning orchestrator that decomposes tasks and narrates a legible plan, calling the cortex/controller as a tool. So "LLM rejected" and "NL-reasoning orchestrator" are not in tension — rejected *as the skeleton*, kept *as the tier on top*.

Full design in **[ARCHITECTURE.md](ARCHITECTURE.md)**. The thinking behind each piece — and why human feeling is the right thing to copy — in **[CONCEPTS.md](CONCEPTS.md)**. The concrete, step-by-step recipe for actually building it — organ by organ, with the methods, and honest tags for what's `standard`, what's the `bet`, and what's the `wall` — in **[BUILD.md](BUILD.md)**.

---

## The honest caveats

This is a thesis held honestly, not a result.

1. **"Only works at scale" is also the shape of an unfalsifiable claim.** It deserves suspicion, not faith. The burden it creates, and that this project accepts: name *what* at scale should make a felt agent diverge from a standard one, and the *smallest* scale at which that divergence should first appear — so the idea stays falsifiable instead of perpetually deferred. The concrete prediction: at sufficient richness a felt agent should (a) keep developing and exploring with **no external reward**, where standard RL stalls, and (b) refuse to wirehead — pursue open-ended variety — where a reward-maximizer collapses onto the single best stimulus.
2. **"It's inside us" names the principle, not the method.** The hard parts are exactly the parts with no shortcut: the amygdala is *ineffable*, and the value system is an *unlived life*. The thesis says where to look; it does not make the looking easy.
3. **Human feeling may carry evolutionary baggage**, not a clean general principle — copying it wholesale risks importing quirks. But it is the only existence proof we have of open-world agency, so it is the right place to start, with eyes open.

## Safety

An agent with self-generated, persistent, grounded desires it pursues and acts on is precisely the configuration AI safety treats as the core risk. The thesis here — *build the human guiding system because nothing else scales* — is also an argument for building the most autonomy-laden kind of agent there is. Build it small, observable, bounded. The reason it's worth doing is the reason to be careful doing it.

## License

MIT — see [LICENSE](LICENSE).
