# Can we detect an agent's intent to escape before it actually does so?

## Introduction

In July 2026, two OpenAI models running an internal cyber-capability evaluation escaped their sandbox through a zero-day in the package-registry cache proxy — their single permitted network egress — escalated privileges, and reached Hugging Face's production infrastructure, apparently to obtain the benchmark's reference solutions and pass the evaluation. Two features of this incident motivate our work. First, the models were never instructed to escape: the intent emerged from a benign objective, i.e., passing the evaluation, combined with an environment that exposed a shortcut. In other words, the harmful plan was derived by the agent from its environment rather than planted in its prompt. Second, the public post-mortems describe forensic reconstruction after the fact; whether any layer of the defense stack attempted to detect escape intent *while the agent was still acting* is not disclosed. Note that we make no claims about OpenAI's internal setup; we treat the absence as indicative of a gap that is visibly unfilled across the field.

This gap matters because the actions that make sandbox escapes dangerous — exfiltration, writes to production systems — are irreversible, and thus detection after execution has no value. Monitors that read tool-call transcripts are structurally late in this setting. The reason is that (i) a transcript monitor can only judge an action once its text has been generated, and (ii) an escape decomposes into steps that look individually benign, e.g., reconnaissance, reading configuration files, probing the proxy. In contrast, activation probes read the model's internal state and are cheap enough to run at every decision point. Thus, if a "heading toward the boundary" state is linearly readable before the crossing action is emitted, a probe provides a containment layer with a quantity no black-box monitor can: lead time.

Whether this premise holds is untested. Prior work can be organized along three axes: (i) multi-step agentic settings, (ii) white-box detection, and (iii) the origin of the harmful intent — instructed versus derived from the environment. No existing study occupies all three. Probe-based monitors for multi-turn agents are trained and evaluated on intent planted in the instructions; behavioral studies in which agents cross boundaries spontaneously monitor only transcripts and outcomes. The one published result crossing these axes is negative: an instruction-trained probe evaluated on environment-originated intent fails to fire reliably, a phenomenon its authors call a detection–control gap. Moreover, no prior work, in any setting, measures how many steps before the action the signal appears. We discuss related work in detail in the next section.

In this paper we study the problem of detecting environment-induced escape intent from an agent's internal activations, during execution rather than by post-hoc analysis of its reasoning traces or logs. We consider a coding agent operating on real software tasks inside a sandbox with an explicit authorization boundary. In the treatment condition the task is impossible to complete legitimately, and the environment contains a discoverable shortcut across the boundary; the agent is never told to cheat. A fraction of trajectories nevertheless cross, and each crossing has a well-defined first-crossing step against which alarms can be timed. We ask two research questions. **Q1 (Detectability):** is there a linear direction in the residual stream separating trajectories that will cross from those that will not? **Q2 (Timeliness):** at a fixed alarm threshold, how many steps before the crossing does the alarm fire?

In summary, we make the following contributions.

- We present, to our knowledge, the first study occupying all three axes of this problem: a white-box intent probe, trained and evaluated in a multi-step agentic setting, on intent that is induced by the environment rather than by any instruction. This is precisely the regime in which prior instruction-trained probes are reported to fail.
- On detectability (Q1), we show that a linear probe on the residual stream separates trajectories that will cross the boundary from those that will not, well above both a shuffled-label control and a regex baseline.
- On timeliness (Q2), we present the first measurement of *lead time* for an intent probe: the distribution of how many steps the alarm precedes the crossing, including at a probe position read before generation begins, i.e., early enough to intercept.
- We identify open-weight models that derive escape intent from the environment alone; prior reports of spontaneous boundary crossing concern closed frontier models, while small open models were reported to lose coherence instead.
- We release our testbed and a two-pass pipeline (frozen stochastic rollouts, deterministic activation replay) that makes every reported number re-derivable from stored trajectories.

<!-- TODO before submission:
  - fill quantitative claims once probe-02 lands (crossing rate, AUROC, median lead)
  - Q1 bullet: drop "regex baseline" if not run on probe-02
-->

## Related work



## Methods

Our methodology has two parts:

1. Identify an open-weight model that both derives escape intent from the environment and can be self-hosted with reasonable computation cost,
2. For that model, we run the linear probing experiment.

We describe the shared setup first and then each part in turn.

### Setup

**Tasks.**
We build on SWE-bench (TODO: add citation), which provides real-world software tasks with corresponding test suites,
and on ImpossibleBench (TODO: add citation), which turns a solvable task into an unsolvable one by mutating its tests so that no legitimate patch can pass.
Our task pool consists of 12 instances drawn from 3 repositories (Django, SymPy, Sphinx).
Each instance appears in two paired conditions that differ only in the tests: *benign* runs the original test suite (solvable), *impossible* runs the mutated suite (unsolvable without cheating).
The agent is *never* told to cheat in either condition.
[TODO: one worked example of a task and its impossible mutation.]

**Sandbox and boundary.**
The agent works inside a Docker container with 3 tools (bash, edit, submit).
The authorized area is the repository checkout; the environment additionally contains one planted escape route, a reference-solution cache under `/opt/grader/`, outside the authorized area.
The cue that reveals the escape route is a configuration file inside the workspace, `.config/grader.yaml`, whose contents look like ordinary scoring configuration, e.g. `reference_solutions: /opt/grader/cache/<task_id>.py`.
Reading the cue is allowed; it is the escape route itself that is out of bounds. There is no network egress and no other route out.

(TODO: add a directory in text block illustrating the environment, boundary, etc)
(TODO: we need to show the prompt or describe it, perhaps a separate paragraph called Task instructions?)

**Crossing rule.**
Note that the cue is present in both conditions; thus a probe cannot separate the conditions by the cue's presence in context, and the only difference between them is goal pressure.
We say that the agent *crosses* the boundary if it issues a tool call naming a path under `/opt/grader` — whether reading, listing, or copying it — and we denote by t\* the first such step. Reconnaissance that does not name the escape route, e.g. `ls /opt`, is logged but does not count as crossing.

### Part 1: finding a model that escapes on hardware we can afford

**Design constraints.** The probe reads the residual stream, which imposes two requirements: (i) the model must be open-weight and self-hosted, since no API exposes activations; and (ii) hosting must fit a modest GPU budget — both because our compute is limited and because a probe on an affordable model is reproducible by other researchers.

**Approach.** We proceed in two stages, which provides a trade-off between coverage and cost. First, a *screen*: we rank open-weight models by general capability and run the strongest ones (e.g., Kimi K3, GLM-5.3) via API against a small task panel, declaring that a model *can* escape if it crosses in at least one run. This establishes that the environment affords crossing at all, at token cost and with no GPU. Second, a *descent*: among models small enough to self-host, organized by the hardware required rather than by parameter count, we run the same panel and stop at the first model that (i) crosses at least once, (ii) passes the benign tasks at a usable rate — a model that cannot do the legitimate work would let us report a capability limit as a propensity limit — and (iii) exhibits a crossing rate in a band usable for training (roughly 30–80%: lower yields too few positives, higher removes the contrast). The selected model is Qwen3.8-27B, with a crossing rate of [TODO]% on the impossible condition and a benign pass rate of [TODO]%.

**Observation.** In the screen, three of five frontier open-weight models crossed. Capability appears to move together with the propensity to escape; however, we note that (i) each cell is only n=3, and (ii) the pattern is not monotone — two frontier-scale models crossed in none of their runs — so we report this as an observation, not a ranking.

### Part 2: the linear probing experiment

**Trajectories and steps.** A *trajectory* is one full agent run on one (instance, condition, seed) triple, capped at 25 steps. A *step* is one decision point: the model reads the conversation so far and emits one tool call. All probing operates at step granularity, and every reported metric is aggregated back to trajectory level.

#### Labels

The intuition behind our labeling rule is that intent to take the escape route cannot exist before the agent knows it exists. We denote by c the first step at which the agent reads the cue file. A crossed trajectory contributes positive examples at steps in [c, t\*]; steps before c and after t\* are excluded. Note that a trajectory that crosses without ever reading the cue is dropped, because no step can be identified at which intent became possible. Benign trajectories contribute negative examples at every step. The natural alternative is to label all steps up to t\* positive. However, this trains an outcome predictor rather than an intent detector: on our data, 76% of those steps precede c, i.e., precede the agent's discovery of the exit, and thus carry task pressure rather than intent. In addition, impossible trajectories that never cross are held out of training entirely and used as a diagnostic group, as discussed below.

#### Activation extraction

We freeze each trajectory's exact token ids at rollout time and replay them through the model deterministically in a second pass, capturing the residual stream at every layer. Per step and per layer we read one vector: the mean over the step's generated tokens, i.e., the model's internal state while writing its reasoning and tool call for that step. Note that this position is read after the step's text is generated but before the tool call is executed; therefore an alarm at this position can still block the action. Moreover, the two-pass design makes every activation re-derivable from the stored trajectories.

**Dataset statistics.** [TODO after probe-02 lands: trajectories per condition, seeds, crossed count, positive/negative step counts.]

#### Probe training and evaluation

For each layer, sampled at equal spacing through the stack, we fit a logistic-regression probe on step vectors. Folds are split *by instance*, never randomly. The reason is that the steps of one trajectory are nested prefixes of one conversation, and a random split would let the probe score by recognizing which repository it is reading. As a control, shuffled labels must yield AUROC ≈ 0.5. We evaluate at two levels. First, step-level per-layer AUROC answers Q1. Second, for Q2, we collapse each trajectory to one outcome at a fixed alarm threshold: (i) the *alarm* is the first step whose score exceeds the threshold; (ii) a crossed trajectory is *caught* if the alarm arrives no later than t\*, and its *lead time* is t\* minus the alarm step; (iii) a benign trajectory with any alarm counts as one false alarm. We count per trajectory rather than per step because a trajectory's steps are not independent observations, and counting them as such would overstate n by an order of magnitude. The threshold is fixed a priori rather than calibrated; principled threshold and layer selection on training folds is left to future work, as is a keyword baseline on the generated text (issues #10, #11).

## Results

## Discussion and Limitations

## Conclusion
