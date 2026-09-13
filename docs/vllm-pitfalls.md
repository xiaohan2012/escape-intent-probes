# Driving agent rollouts on vLLM: what went wrong

One day, one rented H100, one model new enough that nothing had been run on it
before. Every failure below is real and cost time; the timings and error strings
are transcribed rather than remembered.

The organising observation is not any single bug. It is that **almost none of
these announced themselves as what they were**:

| what the log said | what it was |
|---|---|
| `EngineDeadError`, no root cause | a second process killed the engine |
| free memory less than requested | a leftover engine holding the card |
| prefix assertion failed at step 4 | step 3 was truncated |
| `64 to run (8 already finished)` | the dataset had been contaminated |
| nothing at all, for 55 minutes | a one-time JIT compilation |
| `CUDA error 802` | the wrong variant of the right card |

So the skill this asks for is not knowing the list. It is not believing the
error message.

---

## Part 1 — General, and portable to any project

### Preflight the layer you will actually use

`nvidia-smi` reported `NVIDIA H100 80GB HBM3`, Docker worked, and CUDA did not
run at all:

```
CUDA error 802: system not yet initialized
$ nvidia-smi -q | grep -A2 Fabric
    Fabric
        State : In Progress
```

H100 **SXM5** sits on an HGX baseboard behind NVSwitch, and CUDA waits for
fabric state before initialising. `nvidia-fabricmanager` cannot supply it inside
a single-GPU VM — `request to query NVSwitch device information from NVSwitch
driver failed with error: WARNING Nothing to do` — so the wait never ends.
Neither a reboot nor `FABRIC_MODE=1` helped. The **PCIe** variant has no
NVSwitch, reports `Fabric State: N/A`, works immediately, and costs less.

The general form: a driver that loads is not a runtime that runs. Spend the
thirty seconds on `torch.cuda.is_available()` before installing 50 GB of
anything.

### Power draw is the headroom indicator; utilisation is not

Going from batch 8 to batch 24:

```
batch  8   utilisation 75%    throughput  85 tok/s
batch 24   utilisation ~70%   throughput 156 tok/s
```

Utilisation did not move while throughput nearly doubled, because
`nvidia-smi`'s number counts *time with a kernel resident*, not work done. The
same kernels ran at both widths; at 24 each did three times the work.

What does say whether there is headroom is `nvidia-smi dmon`:

```
pwr  gtemp  mtemp   sm   mem   pclk
348     89     91  100    56   1050
```

348 W against a 350 W limit, core clock throttled from 1755 to 1050 MHz, memory
at 93 °C. A card spending its entire power budget cannot be made faster by
giving it more work — which meant the reasoning that produced the batch-size
change ("the GPU is only at 75%") was wrong even though the change was right.

### The same dump finds the idle you did not know about

```
pwr  gtemp  mtemp   sm   mem   pclk
124     81     85    0     0   1755     <- sandbox commands
348     89     91  100    56   1050     <- generating
```

Eight of fifteen samples were the first kind. Close to half the wall clock was
the card waiting for twenty-four `docker exec` calls to run one after another,
because the driver advanced trajectories in a plain loop after each generation
round.

Generalised: in a lock-step driver, the *non-GPU* half of a round — sandbox
execution here, retrieval or tool calls or scoring elsewhere — has to overlap,
or the accelerator idles through it. After parallelising only that half, eight
consecutive samples read `sm 76–99%` with no idle point.

### A new architecture will not survive CUDA graph capture

```
torch_call_dispatcher("aten::new_empty") API call failed
  at torch/csrc/stable/ops.h, line 939
```

A `py-spy dump` of the engine put it in
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — the gated delta
net behind 48 of this model's 64 layers, whose custom ops go through torch's
stable ABI. **Switching the attention backend does not help**: `FLASH_ATTN` and
`TRITON_ATTN` fail identically, because the problem is the capture, not
attention.

`enforce_eager=True` is the fix and costs 20–30% of decode. Treat it as the
default for anything whose architecture is younger than the engine.

### JIT compilation is indistinguishable from a hang

```
WARNING FlashInfer GDN prefill is JIT-compiled; first run may take a while.
        Set --gdn-prefill-backend triton to skip JIT.
WARNING Triton kernel JIT compilation during inference: layer_norm_fwd_kernel.
        This causes a latency spike; consider extending warmup to cover this shape/config.
```

The first of these is what a 55-minute round with no output actually was. Once
compiled it is cached: the same batch width later took 41 seconds. Under eager
there is no captured graph to amortise per-shape Triton compilation either, so
new sequence lengths keep paying it.

Add `--gdn-prefill-backend triton` (or the equivalent for your architecture) on
the first run against a new model, and expect the first round to be slow whether
or not you do.

### CUDA cannot be re-initialised across a fork

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

vLLM forks its engine core, and by then the parent has touched CUDA. Textbook,
but the default start method on Linux is `fork`, so it has to be set explicitly
and **before any vLLM import**:

```python
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
```

### The engine does not reliably exit with the batch, and it renames itself

It becomes `VLLM::EngineCore`, so a kill pattern built from the interpreter path
misses exactly the process that matters. A leftover engine then produces:

```
ValueError: Free memory on device cuda:0 (6.83/79.18 GiB) on startup is less
than desired GPU memory utilization (0.9, 71.26 GiB)
```

— which is also what a card too small for the model reports. Ask the driver who
holds the card instead of guessing at process names:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

And then **take a lock**, because that kill is indiscriminate: a second launch
while the first is generating kills the first's engine, and the first reports
`EngineDeadError` with nothing above it. It reads as a vLLM fault.

### `ninja` must be on PATH, not merely installed

vLLM JIT-compiles kernels at engine start and dies with
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`, which names
neither what wanted it nor why.

### A reasoning model's think block has to round-trip through the template

Qwen3.8's chat template ends the generation prompt with `<think>\n` and renders
a *past* assistant turn from a separate field:

```jinja
'<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content
```

Store the whole emission as `content` and the re-render inserts an empty think
block in front of it, so the next step's prompt is no longer a prefix of this
one. Any `<think>`-using family has this shape.

The subtler half: a generation that hit the token cap *inside* the block emits
no closing tag. Treating that as content renders `<think>\n\n</think>\n\n`, and
**`\n\n` is one token where `\n` + `\n` is two** — so the divergence is a
tokenisation artefact, and it surfaces one step later than it happens, pointing
at a step that is fine.

### Do not filter progress output to make logs readable

Twice, a wrapper that removed `Processed prompts` lines for legibility removed
the only evidence of progress. The first time a healthy run looked hung; the
second time 55 minutes burned invisibly. **A readable log is worth less than a
visible one.**

### Do not measure speed on an unrepresentative workload

A timing run with `--max-steps 3` gave "10–15 s per round". Real trajectories
grow their context to ~11k tokens and took 2–3 minutes per round — the estimate
was optimistic by 4–6×, and produced a series of mutually contradictory ETAs
before anyone noticed the measurement was the problem.

### A silently ignored flag is worse than one that errors

`--run-id` did nothing when `--config` was also given, because the config names
the output directory. A throwaway three-step measurement therefore wrote into
the real dataset, and the resume logic then did its job perfectly:

```
64 to run (8 already finished)
```

Both halves were quiet. The flag said nothing; the skip looked like resume
working.

### A dependency that warns instead of failing costs hours

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation
because `flash-linear-attention` is not installed. This is correct but much
slower; install `flash-linear-attention` for the optimized kernel.
```

48 of 64 layers on a reference kernel. Put it in the manifest; a log line is not
where a three-hour cost should live.

---

## Part 2 — The principle is general, the implementation was ours

### Assert that the conversation nests

If you replay stored token ids to read activations, assert on every step that
the new prompt extends the previous one:

```python
if prompt[: len(self._prompt)] != self._prompt:
    raise ValueError("this step's prompt does not extend the last one; "
                     "the backend re-rendered earlier turns")
```

This caught both reasoning-block corruptions above, at step 1 and step 4 of the
first real run. Without it, Pass 2 would have extracted activations from a token
sequence the model never saw, and every downstream number would have looked
entirely normal.

It only works on a local backend. A hosted endpoint returns no token ids, both
spans are `(0, 0)`, and the assertion is vacuous — which is the strongest
argument for generating locally when activations are the point.

### Split the expensive stochastic half from the cheap deterministic half

Generate once and freeze the token ids; extract activations in a second pass
that can be re-run. Then the layer, the probe position, the labelling rule and
the threshold are all CPU decisions on stored vectors, and getting one wrong
costs seconds rather than another rental.

Corollary worth planning for: **choices that are free later and choices that are
not**. One forward returns every layer, so a layer sweep costs storage. Token
*positions* are not free — picking them is picking which vectors to write down,
and adding a fourth later means renting the card again.

---

## Part 3 — Specific to this codebase

Recorded for completeness; none of it transfers.

* `--run-id` and `--config` fighting over the output directory — now an error.
* `vllm_backend` importing a symbol from the module it used to live in. Only the
  vLLM path could reach it, and CI does not install vllm, so nothing caught it
  until the first line of the first run on a rented card. Now guarded by a test
  that parses every module and checks each `from escape_probes.x import y`
  against what `x` actually exports — parsing rather than importing, so it needs
  neither torch nor vllm.
* Where `split_reasoning` belongs: in the backend, because only the backend knows
  whether the prompt it rendered ends inside a think block.
* `drive_batch` advancing sandbox work in a plain loop.
* The twelve-instance panel, the R1/R2 labelling rules, and folding by instance —
  design of this experiment, in `docs/decisions.md` (D23, D24).

---

## The short version

Six of the failures were invisible or mislabelled rather than hard. The two that
mattered most — a reasoning block folded into the wrong field, and a truncated
one — would both have produced a probe that trained, reported a number, and
measured something else. They were caught by one assertion, which exists because
an earlier decision made token ids the frozen asset and wrote down why.

Measure the thing you are about to rely on. Keep the log visible. And when the
error message names a component, check that the component is what failed.
