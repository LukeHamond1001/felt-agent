# Concepts — the thinking behind vision-brain

The long-form version of the ideas behind the [thesis](README.md). A research notebook, not a paper: it follows the questions that actually drove the project.

---

## 0 · The guiding principle

Start from the honest admission. For a small, closed task we know how to guide an agent: write a reward, maximize it. For a **super-complex agent in a huge open environment**, we do not. There is no reward function for *living*. Reward specification — the thing all of RL stands on — does not survive contact with real, open-ended complexity. We have no guiding principle for that regime.

But the regime is not hypothetical, and it is not unsolved. Every human is an existence proof: we navigate an environment of overwhelming complexity, with no episodes and no score, guided entirely by **internal feeling** — the pull of attention, the sense of beauty, inclinations we can't verbalize, hunger and pain and care. Evolution found this because it could not write us a reward either; feeling *is* the open-world guidance system, and it is sitting inside every person.

So the move is: when the world gets too complex to write a reward for, copy the only thing that is known to work — **the human felt system.** Not because it's elegant, but because it is the sole working example we have, and it's already in us. Everything below is an attempt to understand that system well enough to build it.

---

## 1 · Feeling is several systems, not one reward

The first mistake is to model "reward" as a single scalar. What feels like one thing — *good/bad* — is several systems, and you can feel them dissociate:

- **Hypothalamus — drives.** Homeostatic setpoints: hunger, thirst, fatigue, temperature, damage. Bodily, slow, satiable. Pulls you *toward* what you lack.
- **Pain — nociception.** Acute tissue-damage signalling. Fast, phasic, protective. Pushes you *away*.
- **Amygdala — the pull.** Attention, beauty, inclination; threat and salience. Pre-verbal, quick.
- **Basal ganglia — wanting.** The incentive to pursue. The value system.

The cleanest evidence they're separate is the **wanting ≠ liking** dissociation: you can *want* what you no longer *like* (addiction is exactly that — craving without pleasure). If they were one signal that couldn't happen. So an honest architecture keeps them as distinct modules — and most "intrinsic motivation" work quietly collapses them into one number, which is the very move that fails at scale.

But not all four do the *guiding*. The **guidance trio is amygdala (the pull) + basal ganglia (value) + pain (the floor)**: the amygdala says what's worth moving toward, the BG says what's worth pursuing over a horizon, pain says what would destroy you. The **hypothalamus is demoted** to bodily *maintenance* (recharge, don't overheat) plus the primordial *bootstrap* in infancy before curiosity and values exist. An imitation-pretrained agent with a seeded amygdala-pull and pain already has an attractor and a repulsor, so it doesn't need hunger to get going. Keep the hypothalamus minimal — it is not where open-ended guidance lives.

---

## 2 · The amygdala is not a good/bad classifier

This is the correction that reorganized the whole project. The amygdala is **not** a labeler that stamps things good or bad. It is the system that biases **attention** (what you notice, what you can't look away from), generates the sense of **beauty** (what draws you in), and produces **inclinations** — the pre-verbal pulls toward and away that you *cannot put into words*.

That ineffability is not a poetic flourish; it is the engineering problem. Everything we know how to train, we train against a target we can write down — a label, a reward, a number. The amygdala's output is exactly the thing you *can't* write down. There is no label for "the particular way this draws my eye" or "this specific beauty." You cannot build the dataset, so you cannot supervise it. **That is why it is so weird to train** — and why any attempt to stand in a good/bad picture classifier for it is a category error: it captures the *verbalizable shadow* (semantic similarity to affect words) and misses the pull that casts the shadow.

It also sharpens where the difficulty lives. The amygdala has a *perceptual* face (the pull you feel about what's in front of you) and an *anticipatory* face ("does this lead somewhere I'm drawn to"). The anticipatory face is literally a value function — expected future pull, discounted — which means it is about **your own lived consequences**, a life nothing has recorded. So neither face has a clean shortcut: the perceptual face is ineffable, the anticipatory face is unlived. The amygdala is the wall.

---

## 3 · How would you even train it?

If the target is unverbalizable, where can grounding come from at all? Only from felt reality, and there are exactly three taps:

1. **Your own body.** Wire feeling to physiology — pleasure/pain, approach/withdraw, autonomic state. Authentic, and the most data-hungry: it needs a body living a life.
2. **Others', recorded.** Affect-labelled physiology (EEG/EDA/HR + ratings). Someone else felt it and wrote down a coarse projection of it.
3. **Humanity's record, compressed.** Pretrained models — the internet is partly a vast log of what humans found beautiful, threatening, dear. But it's the *verbalizable shadow* again, not the pull.

What you cannot do is shape an amygdala on an **invented task** with a made-up reward — with no felt good/bad behind it you've just built a task's reward model and called it a feeling. This is the deep reason a *reality-shaped* felt system demands a *reality-embodied* agent: the grounding has to come from something real, and a toy has nowhere real to point. It is also why each person's amygdala differs and leads them down different lives — tap #1 is a different life for each of us.

### The protocol — insanely special training

Coarse labels (tap #3) only give the shadow. To seed the pull at the highest fidelity obtainable, you need a *person*, not a corpus:

- **One grounder, not a crowd.** A single human who is *exceptionally in touch with their own affect* — emotionally intelligent, self-aware enough to distinguish their innate guiding signals as they happen: **dopamine rising** (this is leading somewhere good — anticipation, lean in) and **falling** (disappointment, pull back); **cortisol rising** (mounting threat — act) and **falling** (safe, settle). Most people cannot report at this resolution; the source is rare by necessity, and the result is necessarily *one person's* taste — authentic, and unavoidably subjective.
- **Clean conditions.** No interruptions, full focus, a calm baseline-neutral mind, so the readings start from zero and the dynamics are legible.
- **Rich natural stimulus.** They watch video or listen to audio — real, affect-laden content, not toy stimuli.
- **Sub-frame-by-sub-frame mapping.** They annotate intrinsic emotion at fine temporal grain — not "this clip felt good" but the *moment-to-moment trajectory* of the pull and its rise and fall.
- **Paired physiology.** Record the body at the same time (EDA, HRV, pupil, EEG, cortisol proxies) so the felt report is anchored to *measured* neuromodulator dynamics, not words alone.

The output is a dense affect-*trajectory* — the closest thing to a real grounding signal there is. **Honest caveat:** even this only seeds the shadow *richly*; the truly ineffable pull is still the wall, earned by the agent living (§4 onward). And it inherits one person's affect — a feature (authentic) and a limit (subjective, single-source).

---

## 4 · Imagine-and-feel — and the bottleneck that's left

If anticipatory value needs lived consequences, the way to get it without living each one is to **imagine** the consequence and **feel** about the imagined result: roll the world model forward, let the amygdala feel the imagined future, act toward the future that pulls hardest. Feeling an *imagined* future, rather than only the present, is what lets an agent route around a near-term trap toward a distal good.

Two honest conditions and one honest limit: the imagination must **reach far enough** (you can't feel a future you can't imagine reaching), the model must be **faithful** (garbage rollouts produce confident wrong feelings), and on **multi-leg** problems the bottleneck is not the feeling but the **search** — finding the good trajectory in a vast action space. The frontier's planning machinery (MuZero/TD-MPC2-style search) is exactly about making that tractable; the feeling part was never the hard part.

**Imagination is one faculty, not one per sense.** There is no separate visual imaginer and audio imaginer — there is a single world model that, when it imagines a future, renders the *whole multimodal state at once* (sight, sound, touch, proprioception), because the real world is all of them together. Auditory and visual imagery are the same faculty in different channels. (Inner speech is the one near-exception — it also borrows the speech-*motor* system, covert articulation — but plain auditory imagery is just the one world model emitting sound.)

**The amygdala reads a shared workspace, blind to real-vs-imagined.** Both the real senses *and* the world model write to the same perceptual workspace; the amygdala reads it and feels whatever is there. It cannot tell perception from imagination and does not need to — which is exactly why you can dread an imagined future or be pulled toward an imagined good (the same loop that runs anxiety and hope in us). The only thing required on top is a **reality-monitoring tag** that marks each state real or imagined, so the agent *feels* the imagined threat but does not *act* on it as if present. When that tag fails in humans, that is hallucination.

This holds for **every channel, not just vision**: imagined *sights* and imagined *sounds* alike feed the felt system. An imagined melody can pull (beauty); an imagined scream can repel (threat) — exactly as a real one would, which is why a remembered tune can move you. Inner audio (a voice, a tune, inner speech) is not a separate reward path; it is the one imagination faculty rendering sound, felt by the one amygdala through the same workspace. So both visual imagination and audio thought close the same loop back into reward. *(Inner speech as a deliberate reasoning mode — a controllable inner monologue — is a plausible extension on top of this, but it is speculative, not established; the load-bearing claim is only that imagined multimodal content is felt.)*

---

## 5 · Habituation — anti-wireheading that *becomes* curiosity

A reward-maximizer handed a button presses it forever. That is the correct behavior for "maximize reward," which is why "maximize reward" is the wrong objective for an open-world agent. Feeling doesn't work that way: stare at one beautiful thing and the feeling fades (sensory-specific satiety).

Make that the mechanism — pleasure decays with repetition and recovers slowly — and two things happen at once: **single-stimulus wireheading is blocked** (you can't sit on the best stimulus; it stops being best the moment you sit on it) and **curiosity appears for free** ("seek good" becomes "seek *new* good"). It's a bias, not a guarantee — a determined agent could still time-share satiated sources or learn to pace stimulation; habituation pushes toward variety, it doesn't prove it. Crucially the curiosity *emerges from the affect dynamics* rather than being a separate novelty bonus bolted on. This is the one piece that might be genuinely novel as RL — and the honest status is "promising, untested at the scale where it would matter."

The asymmetry matters: **pleasure habituates, pain does not.** Pain that faded would stop protecting the body. So habituation applies to the appetitive side only — the system seeks new joys but never gets bored of an injury.

---

## 6 · The basal ganglia is a cache (System 2 → System 1)

Imagine-and-feel is deliberation, and you can't deliberate for every footstep. So the value system does what a cache does: it **learns to predict the planner's verdict**, compiling slow thinking into fast reflex. The first time through a problem you think it out; the thousandth time you just *know* the move. This is also the dopamine story — the reward-prediction-error that trains the cache is the dopamine signal — and it's the cleanest, most buildable of the value-side mechanisms. But what it caches (the anticipatory pull) is still the unlived thing; the cache is only as good as the life that filled it.

### Why affect is graded and dynamic, not "bad = mega-cortisol, good = dopamine flood"

The obvious-seeming design — slam a flood of dopamine for good and a blast of cortisol for bad — is wrong, and seeing *why* explains the real shape of feeling. The key fact: **dopamine is not "good = flood"; it is a prediction error** — better-than-expected (rise) or worse-than-expected (fall), not absolute reward. Everything follows:

- **A flood carries no learning signal.** Once the good is predicted, a constant flood teaches nothing — you can't improve on a saturated signal. The rising-and-falling *is* the learning gradient (RPE). Max-blast = no learning.
- **It auto-kills wireheading.** Once you predict the flood, the error goes to zero — you can't farm it by repeating. A literal "good = flood" agent would wirehead instantly.
- **You must rank, not just label.** Most options are "somewhat good"; binary blasts can't order good-vs-better-vs-best, graded affect can — and open-world action *is* ranking.
- **Adaptation preserves resolution.** If good always = max, you saturate and lose all discrimination (hedonic treadmill, Weber's law). Relative coding keeps a huge dynamic range usable.
- **The dynamics guide *anticipatory* action.** A *rise* says "good is coming — move toward it" *before* the outcome; a static blast only says "good happened," too late to steer.
- **Extremes are physically destructive.** Chronic mega-cortisol is tissue damage and allostatic load; evolution can't redline the system, so the working range *must* be graded, extremes emergency-only.
- **They're opponent processes you weigh.** Reward vs threat is a constant comparison; two saturating blasts can't be weighed against each other, graded signals can.

So feeling is graded, dynamic, prediction-error-based, and adapting because that is what makes it a system that *learns, ranks, anticipates, and survives*. The architecture already has the core right — **dopamine = TD error = RPE**; the lesson is to make the threat/cortisol channel graded-and-dynamic too, and keep the whole felt signal relative/adapting (habituation already does this for pleasure).

---

## 7 · Why we sleep — and why this agent doesn't, the way you do

A tempting hypothesis: we sleep because our learning rate is so low we have to keep re-hitting experiences to consolidate them. Partly right — that's the **Complementary Learning Systems** story (a fast hippocampus that grabs today, a deliberately slow cortex that integrates over a lifetime without overwriting it, and replay during sleep that bridges them). But it's not the whole story: sleep also **downscales synapses** to restore signal-to-noise (synaptic homeostasis), **flushes metabolic waste** (glymphatic clearance), and aids generalization and repair.

The computational punchline, and why this project deleted its old "sleep phase": **sleep-as-a-distinct-phase is a workaround for a constraint an artificial agent doesn't have.** Biology can't consolidate while awake — the same circuits are busy, the fast buffer is small — so consolidation is forced offline. An artificial agent has a replay buffer and learns anytime, so **experience replay *is* perpetual sleep** — the consolidation function, always on. A separate phase adds nothing. The one real exception: a real-time agent that genuinely can't learn while acting, for which an offline window is the only time to consolidate.

---

## 8 · One-shot learning falls out of this

Could a perfect felt system + imagination + reasoning learn from a *single* example? In humans it demonstrably does — but only because a lifetime of embodied, developmental, and evolutionary priors underpin it. The mechanism alone is not sufficient; one-shot is the *result* of those expensive foundations, not of the three parts by themselves. Granting the priors, the pieces map exactly: **the amygdala says what to attend to and remember** (a salient event burns in — flashbulb memory — because feeling tags it important); **imagination turns one example into many** (simulate variations and consequences — one real example becomes hundreds of imagined ones); **reasoning extracts the rule** (the principle that transfers, instead of the surface of the instance).

Two things already do this: humans are one-shot *as adults*, because a childhood and an evolution paid for the priors (a baby is not one-shot); and LLMs do a form of it (in-context learning *is* rich priors + reasoning generalizing from one example). The catch is the same wall: one-shot runs on rich priors and a *faithful* world model, both paid for slowly — and an unfaithful imagination confidently learns the *wrong* rule (that's how humans form one-shot superstitions). One-shot isn't free; it's the interest you earn on good priors. The poetic version, also true: *one-shot learning is imagination running fast on good priors, with feeling deciding what's worth imagining about.*

---

## 9 · Why small scale tells you nothing — and where the real claim lives

Everything above collapses to standard RL when the world is small, and this is not a disappointment — it's predicted by the thesis. The amygdala is attention/beauty/inclination *over a rich perceptual world*; give it a toy with three states and there is nothing to be drawn to, so it degenerates into a plain reward. The value system only has to be *developed* when horizons are long and signal sparse; in a toy any dense signal grows it directly. Habituation over a few states is a novelty bonus; imagination in a tiny known world is MPC. Each distinctive organ *is* its standard-RL equivalent at small scale — because at small scale a written reward still suffices, and the felt system is precisely the thing you only reach for when the written reward runs out.

So a toy positive is almost certainly an artifact and a toy null is exactly what's expected; either way, uninformative. **The real claim lives only at scale**, and to keep it honest it has to make a divergence prediction: at sufficient richness a felt agent should keep developing and exploring with **no external reward** (where standard RL stalls) and refuse to wirehead, pursuing open-ended variety (where a reward-maximizer collapses). That is the line that, if it failed at adequate scale, would falsify the thesis. Holding the claim honestly means owning that burden rather than hiding behind "it only works at scale."

---

## 10 · The honest shape of the bet

Strip it down. Mechanically, the learning core is regular RL; the two real twists are that the **reward is generated, not given** (feeling, over perception) and that it **habituates** (so the agent can't wirehead). Neither is a new algorithm. The contribution is not a method — it's a **claim about where guidance comes from in the regime we have no method for**: that the human felt system is the only known principle for open-world agency, that it's inside us, and that building it is therefore the path — with the hard parts named honestly (the ineffable amygdala, the unlived value system) rather than papered over. The discipline that matters here is refusing to dress up toy results as evidence for a claim only scale can test.
