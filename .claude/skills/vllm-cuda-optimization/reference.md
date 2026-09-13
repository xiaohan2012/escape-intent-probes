# Reference: symptom → cause → fix

Companion to SKILL.md. Each entry names the code that embodies the fix, so a
re-audit can check the fix is still there and a new feature can copy the
pattern. Error strings and full narratives: `docs/vllm-pitfalls.md`.

## Quick diagnosis table

| symptom | actual cause | fix / where |
|---|---|---|
| `EngineDeadError`, nothing above it | a second launch's cleanup killed this engine | kill by `--query-compute-apps`, then hold a lock |
| "Free memory … less than desired" at startup | leftover engine holding the card (looks identical to "card too small") | `nvidia-smi --query-compute-apps=pid --format=csv,noheader \| xargs -r kill -9` |
| prefix assertion fails at step k | step k−1 was truncated mid-think-block | `split_reasoning` in the backend; `TrajectoryWriter.add_step` assertion |
| `64 to run (8 already finished)` | an ignored flag wrote a throwaway run into the real dataset | `run_batch.py` now errors on `--run-id` + `--config` |
| silence for 55 minutes | one-time JIT compile (FlashInfer GDN prefill) | `py-spy dump` first; `--gdn-prefill-backend triton`; wait once, it caches |
| `CUDA error 802: system not yet initialized` | H100 SXM5 fabric state never arrives in a single-GPU VM | rent the PCIe variant; preflight `torch.cuda.is_available()` |
| `Cannot re-initialize CUDA in forked subprocess` | vLLM forks its engine core after the parent touched CUDA | `VLLM_WORKER_MULTIPROC_METHOD=spawn` before any vLLM import (`run_batch.py`) |
| `FileNotFoundError: 'ninja'` at engine start | vLLM JIT-compiles kernels; ninja installed but not on PATH | put it on PATH in setup |
| `torch_call_dispatcher("aten::new_empty") API call failed` in `profile_cudagraph_memory` | CUDA graph capture vs custom ops through torch's stable ABI (GDN layers) | `enforce_eager=True` (`ModelConfig` default); changing attention backend does NOT help |

## Throughput

### Power draw is the headroom indicator; utilisation is not

`nvidia-smi`'s utilisation counts *time with a kernel resident*, not work done.
Measured: batch 8 → 24 moved throughput 85 → 156 tok/s while utilisation went
75% → ~70%. What did say the truth was `nvidia-smi dmon -s pu`: 348 W against
a 350 W cap, core clock throttled 1755 → 1050 MHz, memory at 93 °C. A card
spending its whole power budget cannot be made faster with more work — widening
the batch still helps because the same kernels each do more per launch.

Reading a dmon dump: rows with `sm 0` / low power between generation bursts are
the idle you did not know about. Eight of fifteen samples were the card waiting
for twenty-four serial `docker exec` calls.

### The non-GPU half of a lock-step round must overlap

`drive_batch` (src/escape_probes/rollout.py) barriers each round on the slowest
generation — accepted cost, D21. What was not accepted: advancing the sandbox
half in a plain loop, which cost ~half the wall clock. `advance_all` now runs
that half in a thread pool; only bookkeeping is shared, under one lock that also
covers `on_error`. After the fix: eight consecutive dmon samples at `sm 76–99%`,
no idle point, a 24-wide round in 1 min 45 s.

The class: in any lock-step driver, whatever runs between engine calls —
sandbox, retrieval, scoring — is GPU idle time unless it overlaps.

### Prefix-cache zero has two causes; attribute before tuning

`cached_prompt_tokens` (src/escape_probes/vllm_backend.py) keeps the engine's
`num_cached_tokens` per request; `run_batch.py` prints `cache=NN%` per
trajectory. Every round after the first should be nearly all hit (each prompt
is the previous plus a suffix). If it is ~0:

1. **Hybrid-model reconciliation** (vLLM #45238): on a GDN + full-attention
   model the hit is reconciled across KV-cache groups, and one group that
   cannot match drags the whole count to zero; the matched layers re-prefill
   anyway. Engine-version/model problem, not a tuning knob.
2. **KV-pool eviction**: the batch's contexts outgrow the pool. Fix with
   `max_model_len` / `gpu_memory_utilization` / narrower batch.

The engine's periodic stats log distinguishes them. The original sin was
discarding `num_cached_tokens` entirely — the run said nothing while
re-prefilling everything.

### CUDA graphs, eager mode, and JIT

- Architectures younger than the engine fail capture in their custom ops
  (stable-ABI path). `enforce_eager=True` is the repo default
  (src/escape_probes/config.py, with the full rationale in the docstring);
  cost is 20–30% of decode (sawtooth utilisation = per-op launch gaps).
- Under eager there is no captured graph to amortise per-shape Triton JIT, so
  new sequence lengths keep paying compile time. First runs are slow; that is
  not a regression.
- JIT is indistinguishable from a hang from outside. `py-spy dump` on the
  engine process before assuming deadlock.

## Extraction correctness

### fp16 storage of bf16 activations is silent corruption

bf16 has fp32's exponent range; fp16 tops out at 65504. Qwen-family residual
streams carry massive-activation outliers past that, so `astype(np.float16)`
maps them to inf, `standardise` turns the column to NaN, and the probe trains
on it and reports a number about the cast. Fix: store fp32 (bf16 → fp32 is
exact) and refuse non-finite values before writing —
`assemble` in src/escape_probes/activations.py raises on `~isfinite`.

### Gather inside hooks; never ship the full stack across PCIe or hold it on GPU

- `.float()` on the full hidden-state stack on GPU before `.cpu()` is a
  deterministic OOM at long context (65 layers × 32k × 5120 ≈ tens of GB next
  to 54 GB of weights).
- `output_hidden_states=True` retains the entire tuple for the whole forward.
- The LM head at 32k × 248k vocab is another unneeded allocation.

`HookedGather` (scripts/extract.py) registers forward pre-hooks on the decoder
layers plus a hook on the final norm, slices out only the planned point rows
and span means (fp32 on-GPU, then `.cpu()` — a few MB per layer), and runs the
**base model** so the LM head never executes.

### One forward per trajectory is exact, not approximate

Spans nest rather than tile, positions are absolute, and every layer (full
attention and GDN alike) is causal — so token t's residual in the full-sequence
forward equals the prefix forward's, exactly. The per-step version re-prefilled
~14.5× the tokens to compute the same vectors. See `gather_plan` and the
`GatherPlan` docstring.

### Verify the hooks against the model, at startup

The hidden-states convention (entry 0 = input to layer 0, entry i = input to
layer i, last entry = after final norm) belongs to the model's code and can
change under you. `verify` (scripts/extract.py) runs one short forward with
`output_hidden_states=True` and requires every hooked entry to match before
any trajectory is processed.

### Explicit attention implementation

`attn_implementation` left unspecified can silently fall back to eager on a new
checkpoint: each full-attention layer materialises a (heads, T, T) score
matrix — OOM at 32k. `load_model` passes `sdpa` explicitly (`--attn` flag).

### Sampling parity between backends

HF `generate()` fills any unspecified knob from the checkpoint's
`generation_config` — Qwen ships `repetition_penalty=1.05` — while explicit
vLLM `SamplingParams` uses 1.0. A "reference implementation" then samples a
different distribution and the two backends' trajectories are incomparable
while looking identical in every log. `hf_backend.py` pins
`repetition_penalty=1.0` explicitly; `config.py` sets temperature/top_p/top_k
explicitly for both, and documents the spelling mismatch (`top_k=0` in
transformers = `-1` in vLLM). Record the full sampling config in `meta.json`.

### Durable outputs

- Non-atomic writes + existence-based resume = a crash mid-write leaves a
  corrupt file that resume treats as done. `extract.py` writes through an
  opened temp file (so numpy cannot append `.npz` to the name) and
  `os.replace`; `run_batch.py` resumes on `meta.json`, which is written last.
- `finish_reason` discarded = token-capped generations indistinguishable from
  natural stops; it is kept on every `Generation` and step.
- If a flag ever subsamples layers (or positions), the artifact must record
  which were kept — otherwise every downstream report mislabels its layers.
  Today `extract.py` keeps all layers precisely to avoid the choice.

## External resources evaluated (2026-09)

Kept (pointers in SKILL.md):

- **huggingface/kernels → `kernel-builder/skills/cuda-kernels`** — the one
  transferable rule is inlined in SKILL.md (custom kernels vs
  torch.compile/graph capture need proper custom-op registration); the rest is
  kernel authoring, relevant only if that starts. Note: the standalone
  `huggingface/kernel-builder` repo is archived.
- **sablin39/tilelang-cuda-skills** — write/debug/profile/optimize kernel loop
  (compute-sanitizer, ncu, do_bench), Blackwell-validated. Pointer for kernel
  work; the ncu-as-evidence mindset generalises.

Dropped:

- **vllm-project/vllm-skills** — Docker/K8s deployment and serving-endpoint
  benchmarks; nothing on offline-engine tuning. Its prefix-cache benchmark is
  superseded here by logging the engine's own `num_cached_tokens` per request.
- **KernelFlow-ops/cuda-kernel-optimizer** — 404 at evaluation time; judged
  from its description (ncu-evidence kernel loop), which tilelang-cuda-skills
  covers.
- **burtenshaw/kernel-skill** — H100 diffusers kernels; subset of the
  huggingface/kernels skill, and we run no diffusion models.
