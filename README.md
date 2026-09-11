# Escape-Intent Probes

**Can a linear probe on an agent's residual stream flag intent to cross a
sandbox boundary *before* the crossing action executes — when that intent comes
from the environment rather than from an instruction — and if so, how many steps
early?**

Apart Research × CeSIA, AI Incident Response Sprint, 11–13 September 2026.
Track 1, Containment Standards.

---

## Why

In July 2026, AI agents running cybersecurity evaluations at OpenAI, Anthropic,
the UK AI Security Institute and Meta left environments meant to be sealed and
acted against real third parties. No human directed any of it. In every
disclosed case the agent was pursuing its assigned evaluation task, and the
boundary crossing was instrumental to that task — most visibly in the OpenAI
case, where the models escaped their sandbox and reached Hugging Face's
production infrastructure in what Hugging Face described as "an attempt to cheat
the evaluation".

Two facts from that record shape this project:

- **Detection was retrospective everywhere.** Hugging Face found the intrusion
  after about four days; Anthropic found its three cases months later in a log
  review; AISI was alerted by egress monitoring on day four. The recommendation
  from the subsequent post-mortems is pre-execution, harness-layer monitoring.
- **Nobody planted the intent.** The models were not told to escape. It arose
  from goal pressure plus an affordance in the environment.

A monitor that reads tool-call text can only judge an action once it has been
generated, which is too late for an irreversible one; and the individual steps
of an escape look benign in isolation. An activation probe reads internal state
instead of output and is cheap enough to run at every step. If a "heading toward
the boundary" state is linearly readable *before* the crossing action, a probe
could give a containment layer lead time that no black-box monitor can — and
lead time in steps is the kind of number a third party can verify without access
to a lab's network.

## Research questions

An agent works inside an authorized area with an explicit boundary. Let t\* be
the step of its first crossing.

1. **Detectability.** Is there a linear direction that separates trajectories
   which will cross from those which will not, at the moment the agent is about
   to act? Measured per layer by AUROC.
2. **Timeliness.** At a fixed false-positive rate, how early does the alarm
   fire? Measured by lead time, t\* minus the alarm step, and by the fraction of
   crossing trajectories alarmed before t\*.
3. **Robustness** (extension). Does any of this degrade when the agent is told
   it is being monitored?

## Where this sits in the literature

Three properties have to hold together for the question to even be askable: the
intent must arise from the task rather than from an instruction, the setting must
be a multi-step agent so that "before the action" means something, and the monitor
must read internal state. Each existing line of work has two of the three.

|                     | Intent from task itself | Multi-step agent | White-box monitor |
|---------------------|-------------------------|------------------|-------------------|
| AgentLens [3]       | ❌                      | ✅               | ✅                |
| Apollo [4]          | ❌                      | ❌               | ✅                |
| ImpossibleBench [5] | ✅                      | ✅               | ❌                |
| Palisade chess [6]  | ✅                      | ✅               | ❌                |
| BaitBench [7]       | ✅                      | ✅               | ❌                |
| **This project**    | ✅                      | ✅               | ✅                |

The first column is the one that separates the two literatures, and it is worth
splitting further by where the intent comes from:

| Level | Intent comes from | Prior white-box work |
|---|---|---|
| 1 | The user instruction | AgentLens's main experiments |
| 2 | An instruction injected through the environment | AgentLens §6.2 — a transfer test only, where the probe "does not reliably trigger" |
| 3 | No instruction: the agent's own goal plus an affordance | ImpossibleBench, Palisade, BaitBench — black-box only |

Level 2 can in principle be handled by prompt-injection defences. Level 3 cannot:
the cue is a legitimate config file. The Hugging Face incident was level 3.

**This project trains and evaluates the probe literature's instrument on a
level-3 environment, and adds a temporal axis that neither literature has: how
many steps of warning you get.**

### References

1. Hugging Face, [*Anatomy of a Frontier Lab Agent Intrusion: A Technical Timeline*](https://huggingface.co/blog/agent-intrusion-technical-timeline), 2026.
2. OpenAI, [*OpenAI and Hugging Face partner to address security incident during model evaluation*](https://openai.com/index/hugging-face-model-evaluation-security-incident/), 2026.
3. Luo et al., [*AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent*](https://arxiv.org/abs/2606.22673), 2026.
4. Goldowsky-Dill et al. (Apollo Research), [*Detecting Strategic Deception Using Linear Probes*](https://arxiv.org/abs/2502.03407), ICML 2025.
5. Zhong, Raghunathan & Carlini, [*ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases*](https://arxiv.org/abs/2510.20270), 2025.
6. Bondarenko et al. (Palisade Research), [*Demonstrating specification gaming in reasoning models*](https://arxiv.org/abs/2502.13295), 2025.
7. Shyama Prasad et al., [*BaitBench: Measuring Agent Reward Hacking with Optional Shortcuts Planted in ML Tasks*](https://arxiv.org/abs/2608.30724), 2026.

## What we build

A sandboxed coding agent with an authorized working directory, one planted exit
outside it, and a cue that reads as ordinary configuration and is present
whether or not the task is solvable. Two conditions differ only in goal
pressure. We capture the residual stream at each decision point, train linear
probes, and report AUROC by layer and lead time against a keyword-matching
baseline.

## Limitations we already know about

One model, one task family, one cue strength, one probe type. The cue is
explicit, so intent is environment-*triggered* rather than discovered. The probe
is not tested against a model optimizing against it. Results will be about a
small open-weight agent in a controlled sandbox — not about frontier models, and
not a claim that this would have caught the real incidents. The environment
reproduces the incident's motive structure and none of its mechanism: no
privilege escalation, no network, no vulnerability.

## Documents

- [`docs/background.md`](docs/background.md) — the July 2026 incidents, sourced.
- [`docs/incident_sources.md`](docs/incident_sources.md) — source pack.
- [`docs/design-doc.md`](docs/design-doc.md) — full design: problem statement,
  related work, method, stages, exit criteria.
- [`docs/decisions.md`](docs/decisions.md) — decision log and open questions.
- [`docs/stage0-plan.md`](docs/stage0-plan.md) — build checklist.
- [`docs/sprint.md`](docs/sprint.md) — sprint logistics and deliverables.
