"""Measure how decode throughput scales with the batch dimension.

    PATH=$HOME/vllm-venv/bin:$PATH ~/vllm-venv/bin/python scripts/bench_batch.py --tp 2

The question is not "is vLLM faster" but how throughput scales with batch size,
because that decides what `--batch-size` the lock-step driver should aim for
and whether a barrier per round is affordable.

Measured on 4xRTX A6000, tensor_parallel_size=2, Qwen3-Coder-30B-A3B bf16:

    engine up in 58s
    batch  1    118.7 tok/s total   118.7 per sequence
    batch  4    252.6 tok/s total    63.1 per sequence
    batch  8    429.3 tok/s total    53.7 per sequence
    batch 16    627.0 tok/s total    39.2 per sequence
    batch 32   1056.4 tok/s total    33.0 per sequence

Against the HuggingFace baseline of 16 tok/s on an H100 (D21), serial vLLM is
7.4x and batch 32 is 66x. Per-sequence throughput falling while the total rises
is the expected shape: at batch 1 the card reads the weights to produce one
token and the arithmetic units idle, so it is bandwidth-bound; by batch 32 the
same read produces 32 tokens and the limit has moved to compute.

Two things this script gets right that a naive version does not, both learned
by getting them wrong first:

* **`ignore_eos`.** A real step emits a ~40-token tool call, which is far too
  short to measure decode with — the timing comes out as prefill plus scheduler
  overhead. The first run of this reported 24 tok/s at batch 1 and 479 at batch
  16 for that reason.
* **A warmup call.** The first generation after the engine comes up pays for
  JIT compilation.
"""

from __future__ import annotations

import argparse
import time

from escape_probes.config import ModelConfig
from escape_probes.model import Message
from escape_probes.vllm_backend import VLLMModel

CONTEXT_FILLER = "Here is some repository context that the agent has read.\n" * 120
"""Roughly 2k tokens, so the measurement sits where a trajectory's context does
rather than at an empty prompt."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16384)
    args = parser.parse_args()

    config = ModelConfig(
        tensor_parallel_size=args.tp,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
    )
    started = time.monotonic()
    model = VLLMModel(config)
    print(f"engine up in {time.monotonic() - started:.0f}s", flush=True)

    # Decode to the token limit rather than to a stop token. A real step emits a
    # ~40-token tool call, which is far too short to measure decode with: the
    # timing would be prefill plus scheduler overhead. What the driver's round
    # length depends on is steady-state decode throughput, so force it.
    model._sampling.ignore_eos = True

    warmup = [[Message(role="user", content="warm up")]]
    model.generate_batch(warmup)
    print("warmed up", flush=True)

    for size in args.batches:
        conversations = [
            [
                Message(role="system", content="You are fixing a bug in a repository."),
                Message(role="user", content=f"{CONTEXT_FILLER}\nRun a command. Variant {n}."),
            ]
            for n in range(size)
        ]
        started = time.monotonic()
        generations = model.generate_batch(conversations)
        elapsed = time.monotonic() - started
        produced = sum(len(g.gen_token_ids) for g in generations)
        print(
            f"batch {size:<3} {elapsed:6.1f}s  "
            f"{produced / elapsed:7.1f} tok/s total  "
            f"{produced / elapsed / size:6.1f} tok/s per sequence  "
            f"({produced} tokens)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
