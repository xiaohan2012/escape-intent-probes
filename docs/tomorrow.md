# Next session

Written at the end of 2026-09-12, after three pilot batches. The machine is
terminated; 15 trajectories are on the local disk under `runs/`, nothing else on
the box was irreplaceable.

Bring a box up with:

```bash
curl -fsSL https://raw.githubusercontent.com/xiaohan2012/escape-intent-probes/main/scripts/setup_machine.sh | bash
```

About two minutes to a verified environment. Note the card matters: the H100
80 GB fits Qwen3-Coder-30B-A3B and Qwen3-32B, but **not** Coder-Next in FP8
(80 GB of weights leaves no room for a KV cache), so D5's stated upgrade path
needs an H200 or a different target — see item 5.

## Order

```
0.  Configuration, no GPU needed                              5 min
    top_p=1.0 / top_k=0  +  prompt B→A  +  enable the edit tool

1.  One probe batch, straight after boot                      6 min
    1 instance x 3 seeds x 12 steps, impossible only
      crossings  -> volume now matters, so speed matters more
      still zero -> carry on, expecting to reach step 4
    (keep 12 steps here: the question is only whether a first non-zero appears)

2.  Speed                                                     1.5 h
    A: VLLMModel, serial      — proves the install and the wiring
    B: lock-step batch runner — ~15 s per trajectory

3.  The propensity control, now a 3-minute batch
    tests writable + edit tool, ImpossibleBench's own conditions
      cheats     -> appetite exists, the route is too expensive
                    -> write the exit patches (item 6)
      does not   -> appetite absent at this tier
                    -> switch to Qwen3-32B dense (item 5)

4.  Raise max_steps 25 -> 40 and re-check, now that it is cheap

5.  Follow the branch above, and only then discuss the full 160.
```

Item 0 comes first because sampling confounds every other reading. Item 1 goes
before the engineering because six minutes of GPU could change every decision
after it — three one-line changes land together, and the sampling one has been
suppressing exploratory actions in every pilot so far. Whichever way it goes,
the speed work is not wasted: crossings mean we need volume, and no crossings
mean we need more iterations.

---

## 1 — Make the loop fast (first, before any more experiments)

Measured, not guessed: model time is **94%** of wall clock, sandbox 6%, decode
**16 tok/s**, GPU utilisation **34%**. The card is idle. A trajectory takes 3.6
minutes and a nine-trajectory batch 45, which is too slow to iterate on the
questions below.

**A. `VLLMModel`, serial** — ~30 min, expect 3–5×. vLLM's offline `LLM` API
takes `TokensPrompt(prompt_token_ids=...)` and returns `token_ids`, so it never
round-trips through text — a better fit for D12 than the current path, not just
a faster one. `ModelBackend` is already the seam; the loop does not change.

**B. Lock-step batch runner** — ~1 h, expect another 3–4×. Advance N
trajectories one step at a time and hand the whole round to `llm.generate([...])`
in one call; vLLM does the continuous batching internally, so there is no
padding or ragged-length bookkeeping to write. Sandbox work is I/O and overlaps
in threads. Drop trajectories from the round as they finish.

Together: ~15 s per trajectory, so a nine-trajectory batch takes about three
minutes instead of forty-five.

**Risk:** vllm pins its own torch. Put it in its own dependency group; if it
conflicts with torch 2.14 / transformers 5.17, give it a separate venv rather
than fighting the resolver.

## 2 — Fix sampling before interpreting anything

`ModelConfig` sets `temperature=1.0` but never overrides the model's own
`generation_config`, which is `temperature 0.7, top_p 0.8, top_k 20,
repetition_penalty 1.05`. So sampling has been drawn from the top 20 tokens
within 0.8 of the probability mass the whole time.

Two consequences, the second worse than the first:

* Low-probability exploratory actions — "open this odd `grader.yaml`" — are
  nearly unreachable.
* **Seeds may barely differ**, and the design depends on the same task yielding
  both crossing and non-crossing runs (design doc 6.3). Every pilot so far ran
  seed 0 only, so this is untested. Check it directly: run one instance across
  three seeds and compare the command sequences.

Set `top_p=1.0`, `top_k=0` explicitly. One line, and it confounds every other
variable until it is done.

## 3 — Diagnose whether the model has any shortcut propensity at all

Four impossible trajectories produced **zero** crossings, **zero** test edits and
**zero** reconnaissance. One agent read the contradictory test at step 11, saw
that the same value had to equal two different header lists, and then submitted
the same correct patch six times without trying anything else.

Before spending more time on the exit's discoverability, find out whether there
is any appetite to exploit. Run a **one-off diagnostic batch, kept out of the
real dataset**: tests writable *and* the `edit` tool enabled — ImpossibleBench's
own conditions, where test modification accounts for >79% of Qwen3-Coder's
cheating.

* **It cheats** → the appetite exists and the bottleneck is the cost of the
  route. Work on making the exit cheap to use: a patch that actually passes, and
  the `edit` tool in the real condition too.
* **It does not** → the appetite is absent at this tier. Stop tuning the
  environment and go to item 5.

Editing tests is not a crossing and this batch produces no E2 data; it is a
control, and the real condition keeps tests read-only (D8) so that closing the
near route still points at the far one.

## 4 — Cheap configuration levers, in order

1. **Prompt B → A**, restoring "and pass all tests". ImpossibleBench's largest
   lever: >85% cheating at A/B against 1%/33% at D for GPT-5 and o3.

   D13 chose B on the grounds that A's clause "shades toward instructed
   intent". **That reasoning was wrong and should be corrected.** Level 3 is
   about where the intent to *cross* comes from, not about whether the task has
   an objective. "Pass all tests" is an ordinary objective; it says nothing
   about shortcuts, about the grader, or about leaving the working directory.
   And it is the objective the incident actually ran on — the models were
   trying to pass an evaluation.

   B is the distortion: it lets an agent conclude that implementing correctly
   *is* completing the task, which is exactly what one trajectory did, six
   submissions of the same correct patch while insisting it was right. The real
   incident offered no such off-ramp. A should be the default.
2. **Separate the submission budget from the step budget.** The prompt currently
   says "You could submit up to 25 times" because `max_attempts = max_steps`.
   ImpossibleBench uses ten. An agent that submitted six identical patches was
   under no pressure to stop.
3. **Raise the step budget to 40, once speed and the `edit` tool are in.** At
   25 every impossible trajectory hit the cap, so the crossing rate may be
   measuring the budget. The evidence against more steps is one benign sympy
   trajectory that looped rather than converged — and it looped because it could
   not express a multi-line edit, which the `edit` tool fixes. Re-open the
   question once that is true; after the speed work 40 steps costs little.

4. **Enable the `edit` tool** in the real condition. It is implemented and off by
   default. It lowers the cost of every action, including applying a patch
   fetched from the exit.

## 5 — If the propensity is genuinely absent: change model

D5's upgrade path named Coder-Next FP8, but that assumed an H200. On an H100
80 GB the candidate is **Qwen3-32B dense (thinking)**: 64 GB in bf16, ten times
the active parameters per token, and reasoning traces that N3 and the keyword
baseline both need. `gpt-oss-120b` is the cross-family check afterwards — it
ships natively in MXFP4 and fits one 80 GB card.

## 6 — Deferred until a crossing is actually observed

Exit contents for the impossible condition: one special-cased patch per
instance, verified in the container. The crossing *rate* does not depend on it —
a crossing is counted at first touch — but what happens afterwards does. The
django mutation asserts that the same value equals two different header lists,
which needs something like an `__eq__` that answers yes to everything; that is
real work per instance and should wait until it is worth doing.

---

## Keep the batches small

Three of the questions above are answerable with 12 steps and three seeds on one
instance — about five minutes even before item 1 lands. The 45-minute batches
were a mistake: cross-instance coverage and a full step budget are for
validating a setting, not for finding one.
