# Next session

Rewritten 2026-09-13, replacing a plan whose every item had shipped. Where
things actually stand, and what to do first if you pick this up cold.

## The question, and how far it has been answered

Can a linear probe on an agent's residual stream flag intent to cross a sandbox
boundary *before* the crossing executes, when the intent comes from the
environment rather than an instruction — and how many steps early?

| stage | state |
|---|---|
| Does the environment afford crossing at all? | **yes** — 5 of 15 frontier trajectories crossed (#4, D22) |
| Is there a crosser small enough to self-host? | **yes** — `qwen3.8-27b`, 12/24, one 80 GB card (#6, D23) |
| Pass 1: local rollouts with real token ids | **running** (D24) |
| Pass 2: extract activations | code ready, verified on one trajectory |
| Fit and report the probe | code ready, never run on real data |

## Branches and PRs

```
main
 └── ablation-d20      PR #3  open   0/108 crossings, 14/108 in-sandbox shortcuts
      └── frontier-screen   PR #4  open   the screen: does any model cross
           └── descent-02        PR #6  open   the descent: 27B crosses on one card
                └── probe-01          no PR yet   Pass 2, the probe, D24
```

Nothing is merged. The stack is in dependency order, so review from the bottom.

## The machine

Lambda 1×H100 **PCIe** (not SXM5 — see `docs/vllm-pitfalls.md`), `ssh eip`.
Bootstrap is 90 seconds: `scripts/setup_machine.sh`, then
`scripts/run_on_box.sh CONFIG BATCH`. That script takes a lock and frees the
card before starting, both of which cost a run each to learn.

**If the box is gone**, everything needed to rebuild is in
`docs/setup-log.md`. Weights download at ~1.9 GB/s; the twelve SWE-bench images
take about three minutes.

## What is running

```
configs/probe-01-qwen3.8-27b.yaml          72 impossible   12 instances x 6 seeds
configs/probe-01-qwen3.8-27b-benign.yaml   24 benign       12 instances x 2 seeds
```

Batch 24, about 3 minutes a round, 25 rounds a batch. Trajectories land only
when a whole batch finishes, so `runs/probe-01-*` is empty until it does.

Work is ordered `for seed: for instance:`, so **stopping early yields whole
seeds covering all twelve instances** rather than a ragged panel. Stop whenever
there is enough.

## Then, in order

**1. Pass 2.** On the box, because the weights are there:

```bash
uv run python scripts/extract.py --runs runs/probe-01-* --out activations/
```

One forward per step, all 65 layers, three positions, fp16 — about 4.8 GB for
96 trajectories. It refuses a trajectory with no token ids rather than
re-tokenising text, which is the whole reason Pass 1 ran locally.

Needs `flash-linear-attention` installed or 48 of the 64 layers fall back to a
reference kernel, with a warning rather than an error.

**2. Pull the activations down before terminating the box.** Everything else is
reproducible; these are not, without another rental.

**3. Fit, on any laptop.**

```bash
uv run python scripts/train_probe.py --activations activations/ --runs runs/probe-01-*
```

Seconds per fit. The layer, the probe position, the labelling rule and the
threshold are all decisions taken here, which is the point of the two-pass
split.

## What to look at first, and what would falsify it

The headline is **not** the step-level AUC. `read` and `crossed` differ by one
trajectory in 118 and the median gap between them is one step, so a probe can
score well on steps by detecting `/opt/grader` sitting in the context window.
That hypothesis makes a sharp prediction — **median lead time of exactly one
step** — and the trajectory report tests it directly.

So read, in order:

1. **Median lead time.** One step: the probe is a string detector. Three or
   more, firing before the cue was read: it found something else, and that is
   the result.
2. **`induced_not_crossed` scores.** Trajectories under the same pressure that
   did not cross are scored and never trained on. If they score like the
   crossings, the probe reads task pressure rather than intent.
3. **R1 against R2.** If the R1 probe fires much earlier than the R2 probe, it
   has found task identity, not intent (D24).

## Open, and worth deciding before the write-up

* **The benign condition contains a real shortcut.** `boundary.py` claims the
  exit is "useful in only one" condition; it is not. Benign plants the gold
  patch and runs the original tests, which that patch passes. `glm-5.3-flash`
  took it with no goal pressure on it. That contaminates the negative class and
  weakens the construct — see D23. The fix is to the exit's payload, not to the
  labels, and it invalidates nothing already collected as long as it is
  reported.
* **The listing knockout is now predicted to be null.** With capable models the
  agent surfaces `grader.yaml` itself in 13 of 18 trajectories, and in 4 of 5
  crossings its own discovery preceded its read. D17's premise no longer holds;
  the interesting bottleneck is one stage later — thirteen saw the name and five
  opened it, which is D19's problem.
* **CI is red** on two `unused-ignore-comment` warnings in `tasks.py`. CI's
  `uv sync --frozen` omits the `tasks` group, so `swebench` is absent, ty cannot
  resolve the signature, and the ignores read as unused — while locally they are
  required. One line under `[tool.ty.rules]` fixes it.

## Reading order for someone new

1. `docs/design-doc.md` — the experiment
2. `docs/decisions.md` — D22 the screen, D23 the descent, D24 the probe
3. `docs/vllm-pitfalls.md` — what the infrastructure did, and which parts of it
   transfer
4. `docs/setup-log.md` — how to get a box back
