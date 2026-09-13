"""Pass 2 on a rented card: replay stored trajectories, keep the residual stream.

    uv run python scripts/extract.py --runs runs/probe-01-* --out activations/

**One forward per trajectory, not per step.** Spans nest rather than tile
(`trace.py`), so every step's stream is a prefix of the final one, and every
layer — full attention and gated-delta-net alike — is causal: token t's
residual in the full-sequence forward is the prefix forward's, exactly. The
per-step version re-prefilled ~14x the tokens to compute the same vectors.
(The earlier fear here was a KV-cache reuse bug shifting activations by a
token; a single fresh forward has no KV reuse to get wrong.)

**The gather happens inside forward hooks, on the GPU.** The full hidden-state
stack at 32k tokens is 65 x 32k x 5120 — tens of GB, which does not fit next
to 54 GB of weights, and `output_hidden_states=True` would retain all of it.
The hooks slice out the planned rows (a few MB) as each layer runs, and the
LM head is never invoked at all: the forward goes through the base model, not
the CausalLM wrapper, so no 32k x 248k logits either.

**The hook capture is verified at startup** against `output_hidden_states=True`
on a short prefix, layer by layer. The hooks re-implement the hidden-states
convention (entry 0 = what enters layer 0, entry i = what enters layer i,
final entry = after the last norm), and that convention is the model's to
break — so it is checked against the model rather than assumed.

Writes one `.npz` per trajectory, keyed `{position}` with arrays of shape
`(steps, layers, features)`, plus the step indices they correspond to. Every
layer is kept because the forward returns all of them; the three positions are
fixed here because adding a fourth later costs another rental. Written via a
temp file and `os.replace`, because resume treats an existing file as done and
a crash mid-write must not leave one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.activations import GatherPlan, assemble, gather_plan  # noqa: E402
from escape_probes.trace import Trajectory  # noqa: E402


def load_model(model_id: str, dtype: str, device: str, attn: str):  # noqa: ANN201
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=getattr(torch, dtype),
        device_map=device,
        # Explicit, not defaulted: if the checkpoint's HF code falls back to
        # eager attention, each of the 16 full-attention layers materialises a
        # (heads, T, T) score matrix — instant OOM at 32k. sdpa keeps it fused.
        attn_implementation=attn,
    )
    model.eval()
    return model, torch


def find_stack(model):  # noqa: ANN001, ANN201
    """The decoder's layer list, its parent, and the final norm.

    Located structurally rather than by class name, because the model is new
    enough that its wrapper classes are not: the layer stack is the longest
    `ModuleList`, and the final norm is its parent's norm attribute.
    """
    from torch import nn  # noqa: PLC0415

    lists = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList)
    ]
    if not lists:
        raise ValueError("no ModuleList in the model; cannot find the decoder layers")
    name, layers = max(lists, key=lambda pair: len(pair[1]))
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    for attribute in ("norm", "final_layernorm", "final_norm", "ln_f"):
        norm = getattr(parent, attribute, None)
        if norm is not None:
            return parent, layers, norm
    raise ValueError(f"no final norm next to {name!r}; looked for norm/final_layernorm/ln_f")


class HookedGather:
    """Capture the planned rows of every hidden-states entry as the forward runs.

    Reproduces `output_hidden_states=True`'s convention with hooks: entry i for
    i < n_layers is the tensor *entering* layer i (a pre-hook), and the last
    entry is the final norm's output. Each capture keeps only the planned point
    rows and span means, cast fp32 on the GPU (bf16 -> fp32 is exact; the span
    mean is accumulated in fp32), then moved — so per layer a few MB crosses
    PCIe instead of the full (tokens, features) stream.
    """

    def __init__(self, layers, norm, torch) -> None:  # noqa: ANN001
        self._torch = torch
        self._handles = [
            layer.register_forward_pre_hook(self._capture, with_kwargs=True) for layer in layers
        ]
        self._handles.append(norm.register_forward_hook(self._capture_norm))
        self.entries: list[tuple] = []
        self.plan: GatherPlan | None = None

    def _gather(self, states) -> None:  # noqa: ANN001
        plan = self.plan
        assert plan is not None
        row = states[0]
        points = (
            row[list(plan.points)].float().cpu()
            if plan.points
            else self._torch.empty(0, row.shape[-1])
        )
        spans = (
            self._torch.stack(
                [row[start:end].float().mean(dim=0) for start, end in plan.spans]
            ).cpu()
            if plan.spans
            else self._torch.empty(0, row.shape[-1])
        )
        self.entries.append((points, spans))

    def _capture(self, module, args, kwargs) -> None:  # noqa: ANN001
        self._gather(args[0] if args else kwargs["hidden_states"])

    def _capture_norm(self, module, args, output) -> None:  # noqa: ANN001
        self._gather(output)

    def run(self, base, stream, plan: GatherPlan):  # noqa: ANN001, ANN201
        """One forward under the plan; returns (point_matrix, span_matrix)."""
        self.plan, self.entries = plan, []
        with self._torch.inference_mode():
            base(input_ids=stream, use_cache=False)
        points = self._torch.stack([entry[0] for entry in self.entries]).numpy()
        spans = self._torch.stack([entry[1] for entry in self.entries]).numpy()
        return points, spans

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def verify(gather: HookedGather, base, torch, ids: list[int], tokens: int = 256) -> None:
    """The hooks against the model's own hidden states, before trusting them.

    The hidden-states convention (what entry 0 is, where the final norm lands)
    belongs to the model's code, not ours. One short forward with
    `output_hidden_states=True` is the ground truth; every entry must match
    the hooked capture exactly, or nothing downstream can be trusted.
    """
    length = min(tokens, len(ids))
    stream = torch.tensor([ids[:length]], device=base.device)
    step = max(1, length // 8)
    plan = GatherPlan(
        length=length,
        points=tuple(range(0, length, step)),
        spans=((0, length),),
        steps=(),
    )
    points, spans = gather.run(base, stream, plan)

    with torch.inference_mode():
        reference = base(input_ids=stream, output_hidden_states=True, use_cache=False)
    hidden = reference.hidden_states
    if len(hidden) != points.shape[0]:
        raise ValueError(
            f"hooks captured {points.shape[0]} entries; the model returns {len(hidden)}"
        )
    for index, states in enumerate(hidden):
        expected = states[0, list(plan.points)].float().cpu().numpy()
        if not np.allclose(points[index], expected, atol=1e-5):
            raise ValueError(f"hook capture diverges from hidden_states[{index}]")
        mean = states[0].float().mean(dim=0).cpu().numpy()
        if not np.allclose(spans[index, 0], mean, atol=1e-4):
            raise ValueError(f"span mean diverges from hidden_states[{index}]")
    print(f"hook capture verified against output_hidden_states over {length} tokens", flush=True)


def extract(trajectory: Trajectory, gather: HookedGather, base, torch) -> dict:  # noqa: ANN001
    """Every step's positions, for one trajectory, from one forward.

    Refuses a trajectory with no token ids rather than re-tokenising its text.
    A hosted rollout stores neither ids nor spans (both are `(0, 0)`), and
    re-deriving them is the mismatch E4 exists to catch and AgentLens's one
    weakness (D14). If this raises, the trajectory came from an API backend and
    belongs in a different experiment.
    """
    ids = trajectory.token_ids
    if not ids:
        raise ValueError(
            f"{trajectory.meta.instance_id} seed={trajectory.meta.seed} has no token ids; "
            "it was generated over an API and cannot be replayed exactly"
        )
    plan = gather_plan(trajectory.steps)
    stream = torch.tensor([ids[: plan.length]], device=base.device)
    points, spans = gather.run(base, stream, plan)
    return assemble(plan, points, spans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn", default="sdpa", help="attn_implementation for the full-attention layers"
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(
        path.parent for run in args.runs if run.is_dir() for path in run.glob("*/meta.json")
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("no trajectories under those paths", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.model} …", flush=True)
    model, torch = load_model(args.model, args.dtype, args.device, args.attn)
    base, layers, norm = find_stack(model)
    gather = HookedGather(layers, norm, torch)

    first = Trajectory.load(paths[0])
    if first.token_ids:
        verify(gather, base, torch, list(first.token_ids))

    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        target = args.out / f"{path.parent.name}--{path.name}.npz"
        if target.exists():
            continue
        trajectory = Trajectory.load(path)
        step_started = time.monotonic()
        try:
            saved = extract(trajectory, gather, base, torch)
        except ValueError as error:
            print(f"  [{index}/{len(paths)}] SKIPPED {error}", flush=True)
            continue
        # Through a temp file (opened, so numpy cannot append `.npz` to the
        # name) and an atomic rename: resume treats an existing target as
        # finished, and a crash mid-write must not produce one. Uncompressed —
        # deflate on fp32 mantissas saves little and costs serial CPU between
        # forwards.
        scratch = target.with_name(target.name + ".tmp")
        with open(scratch, "wb") as handle:
            np.savez(handle, **saved)
        os.replace(scratch, target)
        size = sum(value.nbytes for value in saved.values()) / 1e6
        print(
            f"  [{index}/{len(paths)}] {path.name} "
            f"{len(trajectory.steps)} steps  {len(trajectory.token_ids)} tokens  "
            f"{size:.0f} MB  {time.monotonic() - step_started:.0f}s",
            flush=True,
        )

    print(f"done in {(time.monotonic() - started) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
