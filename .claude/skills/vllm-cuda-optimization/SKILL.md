---
name: vllm-cuda-optimization
description: Use when working on vLLM serving or CUDA/PyTorch efficiency in this repo — the rollout batch (run_batch.py, drive_batch, vllm_backend.py), Pass 2 extraction (extract.py, activations.py), a new model or card, or a re-audit of any of it. Loads the checklist two GPU-rental audits paid for.
---

# vLLM / CUDA optimization and extraction correctness

Two audits of the GPU pipeline — one on Pass 1 throughput, one on Pass 2
correctness — produced everything below. Every item is already fixed in the
code; what recurs is the *class* of mistake. Before trusting a change, a new
model, or a rented card, walk the relevant checklist.

Detail lives in two places, in this order:

- `reference.md` (this directory) — symptom → cause → fix for each item, with
  the code that embodies the fix.
- `docs/vllm-pitfalls.md` — the full war stories with transcribed error
  strings. Do not duplicate it; link it.
- `docs/decisions.md` — D21 (why vLLM + a lock-step driver), D23 (the hardware
  ladder), D24 (probe positions, and the throughput section with the dmon
  numbers).

## Three principles (apply before any checklist)

1. **Measure the artifact the system writes for itself, not a proxy.** Six
   contradictory ETAs came from proxies (`--max-steps 3` timing runs, GPU
   "utilisation"). The engine's own numbers — `num_cached_tokens`, per-step
   `generate_seconds`/`exec_seconds`, `nvidia-smi dmon` power draw — are the
   measurements. Keep the progress log visible; a wrapper that filters
   `Processed prompts` lines for legibility has twice hidden the only evidence
   a run was alive.
2. **Error messages name a component, not the cause.** `EngineDeadError` = a
   second process killed the engine. "Free memory less than requested" = a
   leftover engine holding the card. A prefix assertion at step 4 = truncation
   at step 3. When a message names a component, verify the component is what
   failed before touching it.
3. **Silent fallbacks cost more than crashes.** A dependency that warns and
   falls back (flash-linear-attention → reference PyTorch, 48/64 layers), a
   flag that is quietly ignored (`--run-id` with `--config`), an unspecified
   `attn_implementation`, a checkpoint's `generation_config` filling sampling
   knobs — each ran for hours before being noticed. Make fallbacks fail, or
   assert the fast path is active.

## Throughput checklist (Pass 1 / vLLM engine)

- [ ] **Headroom = power draw, not utilisation %.** `nvidia-smi dmon -s pu`.
      Utilisation counts kernel-resident time, not work: it sat at ~70–75%
      while batch 8→24 nearly doubled throughput (85→156 tok/s). 348 W against
      a 350 W cap and a clock throttled 1755→1050 MHz means no headroom.
- [ ] **Find the idle half of the round.** `sm 0%` samples between generation
      bursts = the non-GPU half (docker exec, scoring, tool calls) running
      serially. It must overlap or the card idles ~half the wall clock. Fixed
      in `drive_batch`'s `advance_all` (thread pool over the sandbox half).
- [ ] **Widen the batch on a power-capped card.** Throughput scales with width
      while utilisation does not move; measure tok/s, not %.
- [ ] **Log the engine's prefix-cache hits.** `num_cached_tokens` per request
      (`cached_prompt_tokens` in `vllm_backend.py`, `cache=NN%` in run_batch
      output). Zero from round 2 onward means every conversation is re-prefilled
      every round. Two causes, different fixes — attribute before tuning
      (reference.md).
- [ ] **New architecture ⇒ expect CUDA graph capture to fail.**
      `enforce_eager=True` is the repo default for this reason; switching
      attention backends does not help (the capture fails, not attention).
      Costs 20–30% of decode.
- [ ] **A silent first round is probably JIT, not a hang.** FlashInfer GDN
      prefill compiled for 55 minutes with no output, then cached (41 s later).
      Check with `py-spy dump` before killing anything.

## Correctness checklist (Pass 2 / extraction, HF forwards)

- [ ] **Never store bf16 activations as fp16.** Qwen residual streams exceed
      fp16's 65504; the cast maps outliers to inf, `standardise` NaNs the
      column, and the probe trains anyway. fp32 (exact from bf16) + assert
      finite before writing (`activations.assemble`).
- [ ] **Never materialise the full hidden-state stack on GPU.**
      65 × 32k × 5120 is tens of GB. Gather the few planned rows inside
      forward hooks; a few MB crosses PCIe (`extract.HookedGather`). Call the
      base model, not the CausalLM wrapper — skip the 32k × 248k logits.
- [ ] **One forward per trajectory, not per step-prefix.** Nested spans +
      absolute positions + causal layers make it exact, at 14.5× fewer tokens.
- [ ] **Pass `attn_implementation="sdpa"` explicitly.** An eager fallback
      materialises (heads, T, T) per full-attention layer — OOM at 32k.
- [ ] **Verify hooks against `output_hidden_states=True` at startup.** The
      hidden-states convention belongs to the model's code (`extract.verify`).
- [ ] **Sampling parity with the reference implementation.** HF `generate()`
      fills unset knobs from the checkpoint's `generation_config` (Qwen ships
      `repetition_penalty=1.05`); explicit vLLM SamplingParams uses 1.0. Set
      every knob explicitly in both backends, and mind spelling differences
      (`top_k`: 0 in transformers = -1 in vLLM).
- [ ] **Atomic writes + a last-written marker.** Existence-based resume treats
      a corrupt partial file as done. Temp file + `os.replace`; write
      `meta.json` last.
- [ ] **Record what you'd otherwise have to guess:** `finish_reason` (capped
      vs natural stop), which layers were kept if you ever subsample, backend
      and sampling config in `meta.json`.

## New model / new card preflight

1. `torch.cuda.is_available()` before installing anything — a loaded driver is
   not a running runtime (H100 SXM5 fabric-state trap; PCIe variant works).
2. `nvidia-smi --query-compute-apps=pid` — who holds the card; kill by query,
   not by process name (the engine renames itself `VLLM::EngineCore`), then
   take a lock so a second launch cannot kill the first.
3. `VLLM_WORKER_MULTIPROC_METHOD=spawn` before any vLLM import (set in
   `run_batch.py`); `ninja` on PATH.
4. Smoke run that **generates**, not one that only loads — capture failures and
   dense-vs-MoE bandwidth costs are invisible before decode runs.
5. Expect the first round to be slow (JIT); budget from a representative
   context length, never from a 3-step run.

## If the work turns into writing custom kernels

We write none today; the transferable rule already bit us from the other side:
**custom CUDA kernels and `torch.compile`/graph capture are mutually exclusive
unless registered as proper PyTorch custom ops** — our capture failure was
GDN custom ops going through torch's stable ABI. When kernel authoring starts:

- huggingface/kernels — `kernel-builder/skills/cuda-kernels` skill: H100/A100/T4
  guides, custom-op registration, benchmark scripts (kernel-builder repo is
  archived; this is its new home).
- sablin39/tilelang-cuda-skills — write → debug (compute-sanitizer) → profile
  (ncu) → optimize loop; the ncu-evidence-driven mindset applies to any
  before/after perf claim even without kernels.
