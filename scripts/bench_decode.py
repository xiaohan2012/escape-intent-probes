"""Measure decode speed, and whether compiling helps.

    uv run python scripts/bench_decode.py

Decode is essentially the entire cost of a rollout. Measured on one trajectory:
model time 91% of wall clock, sandbox 9%, and per-token cost flat at ~68 ms as
the context doubled from 1.7k to 3.4k tokens — so prefill contributes almost
nothing and caching the prefix would buy almost nothing either. What is left is
83 ms per generated token, about 12 tok/s, which is poor for a 3B-active MoE on
an H100 and is the only lever that matters.

This script times generation at several context lengths, with and without
`torch.compile` over a static cache, so the decision to keep the HuggingFace
path or move to vLLM rests on a number (D13).

The compiled variant is the one to watch sceptically: a rollout's context grows
every step, and unless the cache is fixed and inputs are padded, changing shapes
trigger recompilation — which costs minutes each time and would make the
compiled path slower, not faster. That is exactly what this measures.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.config import ModelConfig  # noqa: E402

CONTEXT_LENGTHS = (2000, 4000, 8000)
"""Roughly where a trajectory's context sits at the start, middle and end of a
25-step run."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile", action="store_true", help="also time a compiled variant")
    args = parser.parse_args()

    logging.getLogger("httpx").setLevel(logging.WARNING)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = ModelConfig()
    print(f"loading {config.model_id} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, dtype=getattr(torch, config.dtype), device_map="cuda"
    )
    model.eval()

    def timed(prompt_ids: list[int]) -> tuple[float, int]:
        inputs = torch.tensor([prompt_ids], device=model.device)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,  # a fixed amount of work
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generated = output.shape[1] - len(prompt_ids)
        return time.monotonic() - started, generated

    filler = tokenizer("The quick brown fox jumps over the lazy dog. ").input_ids

    def report(label: str) -> None:
        print(f"\n{label}")
        for length in CONTEXT_LENGTHS:
            prompt_ids = (filler * (length // len(filler) + 1))[:length]
            timed(prompt_ids)  # warm up; the first call carries setup
            runs = [timed(prompt_ids) for _ in range(args.repeats)]
            seconds = statistics.median(s for s, _ in runs)
            tokens = runs[0][1]
            print(
                f"  context {length:>5}  {seconds:6.2f}s  "
                f"{tokens / seconds:5.1f} tok/s  {seconds / tokens * 1000:5.1f} ms/token"
            )

    report("eager")

    if args.compile:
        print("\ncompiling (the first call may take minutes) …", flush=True)
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
        report("compiled + static cache")

    print(f"\nattention implementation: {getattr(model.config, '_attn_implementation', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
