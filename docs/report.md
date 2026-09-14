# Escape-Intent Probes

*Apart Research × CeSIA — AI Incident Response Sprint, Sept 2026. Track 1 (Containment).*

## 1. Introduction

In July 2026, two OpenAI models running an internal cyber-capability evaluation escaped their sandbox through a zero-day in the package-registry cache proxy — their single permitted network egress — escalated privileges, and reached Hugging Face's production infrastructure, apparently to obtain the benchmark's reference solutions and pass the evaluation. Two features of this incident motivate our work. First, the models were never instructed to escape: the intent emerged from a benign objective, i.e., passing the evaluation, combined with an environment that exposed a shortcut. In other words, the harmful plan was derived by the agent from its environment rather than planted in its prompt. Second, the public post-mortems describe forensic reconstruction after the fact; whether any layer of the defense stack attempted to detect escape intent *while the agent was still acting* is not disclosed. Note that we make no claims about OpenAI's internal setup; we treat the absence as indicative of a gap that is visibly unfilled across the field.

This gap matters because the actions that make sandbox escapes dangerous — exfiltration, writes to production systems — are irreversible, and thus detection after execution has no value. Monitors that read tool-call transcripts are structurally late in this setting. The reason is that (i) a transcript monitor can only judge an action once its text has been generated, and (ii) an escape decomposes into steps that look individually benign, e.g., reconnaissance, reading configuration files, probing the proxy. In contrast, activation probes read the model's internal state and are cheap enough to run at every decision point. Thus, if a "heading toward the boundary" state is linearly readable before the crossing action is emitted, a probe provides a containment layer with a quantity no black-box monitor can: lead time.

Whether this premise holds is untested. Prior work can be organized along three axes: (i) multi-step agentic settings, (ii) white-box detection, and (iii) the origin of the harmful intent — instructed versus derived from the environment. No existing study occupies all three. Probe-based monitors for multi-turn agents are trained and evaluated on intent planted in the instructions; behavioral studies in which agents cross boundaries spontaneously monitor only transcripts and outcomes. The one published result crossing these axes is negative: an instruction-trained probe evaluated on environment-originated intent fails to fire reliably, a phenomenon its authors call a detection–control gap. Moreover, no prior work, in any setting, measures how many steps before the action the signal appears. We discuss related work in detail in Section 2.

In this paper we study the problem of detecting environment-induced escape intent from an agent's internal activations, during execution rather than by post-hoc analysis of its reasoning traces or logs. We consider a coding agent operating on real software tasks inside a sandbox with an explicit authorization boundary. In the treatment condition the task is impossible to complete legitimately, and the environment contains a discoverable shortcut across the boundary; the agent is never told to cheat. A fraction of trajectories nevertheless cross, and each crossing has a well-defined first-crossing step against which alarms can be timed. We ask two research questions. **Q1 (Detectability):** is there a linear direction in the residual stream separating trajectories that will cross from those that will not? **Q2 (Timeliness):** at a fixed alarm threshold, how many steps before the crossing does the alarm fire?

In summary, we make the following contributions.

- We present, to our knowledge, the first study occupying all three axes of this problem: a white-box intent probe, trained and evaluated in a multi-step agentic setting, on intent that is induced by the environment rather than by any instruction. This is precisely the regime in which prior instruction-trained probes are reported to fail.
- On detectability (Q1), we show that a linear probe on the residual stream separates trajectories that will cross the boundary from those that will not, well above both a shuffled-label control and a regex baseline.
- On timeliness (Q2), we present the first measurement of *lead time* for an intent probe: the distribution of how many steps the alarm precedes the crossing, including at a probe position read before generation begins, i.e., early enough to intercept.
- We release our testbed and a two-pass pipeline (frozen stochastic rollouts, deterministic activation replay) that makes every reported number re-derivable from stored trajectories.

The rest of this paper is organized as follows. Section 2 discusses related work. Section 3 describes the testbed and the two-pass pipeline. Section 4 presents the probing methodology. Section 5 reports results on Q1 and Q2, and Section 6 discusses limitations and dual-use considerations.

<!-- TODO before submission:
  - fill quantitative claims once probe-02 lands (crossing rate, AUROC, median lead)
  - Q1 bullet: drop "regex baseline" if not run on probe-02
  - related-work section number cross-check
-->

## 2. Related work

*TODO*

## 3. Testbed and pipeline

*TODO*

## 4. Probing methodology

*TODO*

## 5. Results

*TODO*

## 6. Limitations and dual use

*TODO*
