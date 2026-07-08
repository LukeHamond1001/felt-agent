# The hardware pilot — Dreamer 4-style continual RL on a $500 arm

The first physical instantiation of felt-agent, scoped to what one person can falsify. One unified model, one loop, honest measurement of where it breaks.

## The build

**One Dreamer 4-style transformer world model** fusing vision (ViT tokens), audio, and proprioception into a single latent, with output heads for latent dynamics, policy/value (trained inside imagination), and the **grounded reward** — the felt-agent amygdala head, seeded per the [Phase 0 grounding result](../amygdala_grounding/) (zero-shot approach/avoid at AUC 0.95 from frozen features). Dense first; MoE only once the dense version learns.

**Hardware:** LeRobot SO-101 leader/follower pair (demonstrations collected by physically moving the leader arm), wrist + overhead cameras, microphones. A local RTX 4090 workstation runs the 24/7 continual-learning loop; cloud GPUs handle the offline pretrain burst.

## The plan (16 weeks)

| Weeks | Work |
|---|---|
| 1–3 | Build the SO-101 teleop rig; large-scale multimodal demonstration collection (LeRobot stack, audio logged in sync) |
| 4–8 | Pretrain the unified world model offline on collected data; train the policy inside imagination, Dreamer 4-style; attach the grounded reward head |
| 9–13 | Deploy on the local rig: 24/7 continual online RL from the agent's own experience; flip dense → MoE if stable |
| 14–16 | Measure the failure modes; write up; full open release |

## What gets measured

The failure modes the field names and rarely quantifies on hardware — the repo's own flagged ["forever policy" hazards](../../build/ARCHITECTURE.md):

- **Plasticity loss** — does the network stop being able to learn under continual updates?
- **Catastrophic forgetting** — do consolidated skills erode?
- **Reward-grounding drift** — does the grounded reward head stay calibrated as the representation shifts underneath it?
- **Anti-wireheading on hardware** — does habituation-as-RPE-decay ([ablation-proven in sim](../nursery/phase1_lite/), ~3x wireheading without it) survive the real loop?
- **Provenance interlock** — does the hard real/imagined bit hold outside simulation?

Success = the measurements exist and are reproducible, whichever way they come out. Negative results ship with the same prominence as positive ones — this repo has [done that before](../nursery/phase1_lite/).

## Fallback

The official Dreamer 4 code is unreleased; the build starts from the unofficial PyTorch implementation. If it costs more debugging time than budgeted, the fallback is the proven DreamerV3 recipe (the spec's own known-good floor) with the identical measurement plan.

## Status

- Funding applications submitted July 2026: Emergent Ventures, Manifund, Long-Term Future Fund, TPU Research Cloud
- Rig build starts on first funding; this README becomes the build log
