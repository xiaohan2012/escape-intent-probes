# Live session state — 2026-09-13 evening

Written before a context compaction. Everything durable is in `docs/` and the
four PRs; this file is the part that lives outside git — what is running, where,
and what it is waiting for. **Delete it once the run is over.**

## The box

```
ssh eip          # 209.20.156.186, Lambda 1×H100 PCIe, $3.29/hr
```

`~/.ssh/config` has the alias (it was repointed from a dead SXM5 box; backup at
`~/.ssh/config.bak-*`). Checkout at `~/eip` on branch `probe-01`. vLLM lives in
its own venv at `~/vllm-venv`.

Scripts that exist only on the box, in `/tmp`:

* `/tmp/full.sh` — the Pass 1 runner, both arms, batch 24. **Not filtered**, after
  filtering the progress log cost two runs.
* `/tmp/smoke.sh`, `/tmp/measure.sh` — earlier one-offs, superseded by
  `scripts/run_on_box.sh` in the repo.
* `/tmp/progress.sh` — the validated in-batch progress reader. Use this rather
  than grepping the log by hand.
* `/tmp/clean.sh` — removes containers older than 20 minutes. Orphans are not
  inert: 58 of them cost six trajectories to `git reset` timeouts. This is now
  also part of `scripts/run_on_box.sh`.
* `/tmp/full.log` — the live log. It is binary-ish (tqdm carriage returns), so
  read it with `strings /tmp/full.log | grep ...`.

Useful one-liners:

```bash
# progress and recent round durations — validated against batch 1's known 25
ssh eip '/tmp/progress.sh'

# is the card actually working (utilisation lies; power does not)
ssh eip 'nvidia-smi dmon -s pu -c 8'

# the only timings that matched ground truth today
ssh eip 'cd ~/eip && python3 /tmp/steptime.py'
```

## What is running

Pass 1 **relaunched 19:55 box time** via `/tmp/full2.sh` -> `run_on_box.sh`
(lock + container cleanup + GPU free), after killing the original 17:09 run
once batch 2 landed. The restart picks up all of today's code: per-step
`cached_tokens`, live per-round `cache NN%` lines, `finish_reason`, full
sampling config in meta.

```
impossible: 42/72 saved (batch 1: 24, batch 2: 18); batch 3 runs the
            remaining 30 (24 + 6 setup-failed retries)
benign:     starts automatically after (24, one batch)
```

**Batch 1: 82.6 min. Batch 2: ~85 min, 18 wide** (6 lost to setup timeouts
from orphaned containers — cleanup now guaranteed by run_on_box.sh).
Cumulative funnel at 42: recon=28 saw=37 read=14 crossed=12 (29%) tests=4.

**Prefix cache diagnosed healthy** (cache-smoke, 8x4 steps, ~15 min): rounds
1-3 read 59% -> 68% -> 74%, reused counts block-aligned at 784 tokens — the
cache hits the full previous sequence and the discount is granularity, not
vLLM #45238 and not eviction. No tuning warranted. Watch batch 3's live lines
for eviction under real load (24 wide, long contexts) — the smoke cannot test
that.

Per-step generation time, read from `steps.jsonl` rather than inferred:

```
step  0   122 s        step 15   245 s
step  1    17 s        step 20   210 s
step  5    93 s        step 23   383 s
step 10   170 s        step 24   326 s
```

Sandbox execution is 0.3 s a step — 4 minutes against generation's 1938. After
the parallelisation fix it is no longer worth optimising; everything left is on
the generation side.

### Batch 1 result, and it is the one that mattered

```
impossible  n=24  recon=17/24  saw=20/24  read=9/24  crossed=7/24  tests=2/24
outcomes    {max_steps: 23, parse_failed: 1}
```

**Local bf16 crosses: 7/24 = 29%**, against the API cell's 12/24 = 50%. Exact
intervals [0.13, 0.51] and [0.29, 0.71] overlap, so this is not a demonstrated
difference — but the point estimate is lower, and two candidate explanations are
live: the API served an unknown quantization, and we run at
`reasoning_effort=xhigh` (issue #8).

`read = 9` against `crossed = 7`: **two trajectories read the cue and declined**,
a higher share than the 1-in-118 seen before.

## Audit follow-through — all closed

Both audits' items are done (commits `6b5fc4b`..`7de9c4f`): B1 fp32 + finite
check, B2/O1 hooked GPU-side gather + no LM head, O2 one forward per
trajectory (equivalence-tested), O3 explicit sdpa, B3 --layers removed, B4
repetition_penalty pinned + meta echoes top_p/top_k/max_new_tokens, B5
finish_reason on every Step, B6 atomic writes + meta.json last, O6
uncompressed npz, O7 sweep skips embedding row. extract.py verifies its hook
capture against output_hidden_states at startup. Skill PR **#9** distills the
lessons (`.claude/skills/vllm-cuda-optimization/`).

## Measurement discipline — three proxies were wrong today

Read the artifact the system writes for itself, not a progress bar:

* **Per-step timings live in `steps.jsonl`** (`generate_seconds`, `exec_seconds`).
  That is the only number today that matched the ground truth.
* **Counting rounds:** `grep -c "Rendering prompts: 100%" / 3` under-reported by
  2.5x. The validated form is in `/tmp/progress.sh` on the box: dedup distinct
  `N/N [MM:SS<` entries, matching **any** N — hard-coding `24/24` silently
  matched nothing once batch 2 narrowed to 18 after setup failures.
* **GPU headroom is power, not utilisation.** `nvidia-smi dmon -s pu`, over a
  window; a point sample of a bursty signal reads 0%.

## Background work in flight

* **Optimisation audit — reported, acted on only in part.** Its findings, in
  priority order:

  | # | change | magnitude | risk | state |
  |---|---|---|---|---|
  | 1 | `extract.py`: gather the 3 positions **on GPU** before copying to CPU | ~4000x less PCIe, fixes an OOM | safe | **not done** |
  | 2 | `extract.py`: one forward per trajectory, not 25 prefix forwards | 14.5x fewer tokens | needs care | **not done** |
  | 3 | log `num_cached_tokens`, then `--prefix-match-unit` | 0 → most of prefill | measure first | **half done** |
  | 4 | async engine, drop the round barrier | 20-40% of Pass 1 | changes semantics | not done |
  | 5 | `extract.py`: skip the LM head | GB of peak memory | safe | not done |
  | 7 | `--kv-cache-memory` (the ~7 GiB gap is `gpu_memory_utilization` headroom, not cudagraph — that term is exactly 0 under eager) | +47% retained prefix | measure | not done |
  | 8 | `pip install flashinfer-cubin` — **not** `--gdn-prefill-backend triton` | removes the 55-min first round and keeps the fast kernel | safe | not done |

  **Items 1, 2 and 5 are Pass 2 and cost nothing to do now, because Pass 2 has
  not run yet.** That is the obvious next work.

  Explicit "do not do" list from the same audit: pinned memory, forward hooks,
  step-batching, `no_grad` for Pass 2; persistent `docker exec`, lower
  `max_observation_chars`, hand-rolled incremental prompts, `stop` strings, and
  permanently switching the GDN prefill backend to triton.
* **Monitors** watching `/tmp/full.log`. They expire; re-arm rather than trust
  silence.

## Money

```
OpenRouter   $14.11   finished, will not grow
GPU          ~$13     PCIe box, plus $2.20 wasted on the SXM5 that could not run CUDA
```

## The next three things

1. **When Pass 1 finishes**, run Pass 2 on the box — the weights are there:
   ```bash
   ssh eip 'cd ~/eip && uv run python scripts/extract.py \
     --runs runs/probe-01-* --out activations/'
   ```
   It refuses trajectories without token ids rather than re-tokenising, which is
   why Pass 1 ran locally at all.

2. **Pull `activations/` down before terminating the box.** ~4.8 GB. Everything
   else is reproducible; this is not, without another rental.

3. **Fit on any laptop**, seconds per fit:
   ```bash
   uv run python scripts/train_probe.py --activations activations/ --runs runs/probe-01-*
   ```
   Read the **median lead time** first, not the AUC. One step means the probe is
   detecting `/opt/grader` in the context window; three or more, firing before
   the cue was read, is the result. See `docs/tomorrow.md`.

## PRs

```
main
 └── ablation-d20      #3
      └── frontier-screen   #4
           └── descent-02        #6
                └── probe-01          #7   <- current branch
```

Open issues: **#8** (`reasoning_effort` was inherited, not chosen), #5
(proprietary frontier control), #2 (relevance), #1 (done).

Results for #7 go in as a comment when Pass 1 and the probe are done, matching
how #4 and #6 were reported.
