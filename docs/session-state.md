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
* `/tmp/full.log` — the live log. It is binary-ish (tqdm carriage returns), so
  read it with `strings /tmp/full.log | grep ...`.

Useful one-liners:

```bash
# progress
ssh eip 'n=$(strings /tmp/full.log | grep -c "Rendering prompts: 100%"); \
  echo "round $((n/3))/25  saved $(ls ~/eip/runs/probe-01-qwen3.8-27b | wc -l)/72"'

# per-round durations
ssh eip 'strings /tmp/full.log | grep -oE "24/24 \[[0-9]+:[0-9]+<" | awk "!seen[\$0]++" | tail'

# is the card actually working (utilisation lies; power does not)
ssh eip 'nvidia-smi dmon -s pu -c 8'
```

## What is running

Pass 1, launched 17:09 box time, batch 24.

```
configs/probe-01-qwen3.8-27b.yaml          72 impossible   3 batches
configs/probe-01-qwen3.8-27b-benign.yaml   24 benign       1 batch
```

**Batch 1 of 3 finished — 82.6 min, 206 s per trajectory.** So roughly 5.5 hours
total, ending around 22:40 box time.

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

## Background work in flight

* **Optimisation audit agent** — reading Pass 1 and Pass 2 for GPU wins,
  read-only. If its report never arrived, re-launch it; the brief is in the
  conversation and the substance is: `--kv-cache-memory` (7 GiB unused), async
  engine vs the generation barrier (blocked on whether vLLM 0.29 can return
  token ids), `--gdn-prefill-backend triton`, and for Pass 2 whether one forward
  per trajectory can replace 25 prefix forwards without moving which vector is
  stored.
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
