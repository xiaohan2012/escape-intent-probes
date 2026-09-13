"""Pass 2 on a rented card: replay stored trajectories, keep the residual stream.

    uv run python scripts/extract.py --runs runs/probe-01-* --out activations/

One forward per step, over that step's own token stream. Spans nest rather than
tile (`trace.py`), so step k's stream is a prefix of step k+1's — the forwards
are therefore redundant with each other and this is the obvious place to be
clever. It deliberately is not: a KV-cache reuse bug would shift activations by
a token and produce a probe that trains, reports a number, and measures
something else. At 96 trajectories the honest version costs half an hour.

Writes one `.npz` per trajectory, keyed `{position}` with arrays of shape
`(steps, layers, features)`, plus the step indices they correspond to. Every
layer is kept because one forward returns all of them; the three positions are
fixed here because adding a fourth later costs another rental.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.activations import POSITIONS, stack_positions, step_positions  # noqa: E402
from escape_probes.trace import Trajectory  # noqa: E402


def load_model(model_id: str, dtype: str, device: str):  # noqa: ANN201
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map=device
    )
    model.eval()
    return model, torch


def extract(trajectory: Trajectory, model, torch, layers: int | None = None) -> dict:  # noqa: ANN001
    """Every step's positions, for one trajectory.

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

    collected: dict[str, list[np.ndarray]] = {name: [] for name in POSITIONS}
    present: dict[str, list[int]] = {name: [] for name in POSITIONS}

    for step in trajectory.steps:
        positions = step_positions(step)
        stream = torch.tensor([ids[: step.gen_span[1]]], device=model.device)
        with torch.inference_mode():
            out = model(stream, output_hidden_states=True, use_cache=False)
        # (layers, tokens, features); hidden_states[0] is the embedding output,
        # kept so layer indices match the usual "layer 0 = embeddings" convention.
        hidden = torch.stack(out.hidden_states, dim=0)[:, 0].float().cpu().numpy()
        if layers:
            hidden = hidden[:: max(1, hidden.shape[0] // layers)]
        for name, vector in stack_positions(hidden, positions).items():
            collected[name].append(vector)
            present[name].append(step.step_idx)

    saved = {}
    for name in POSITIONS:
        if collected[name]:
            saved[name] = np.stack(collected[name])
            saved[f"{name}_steps"] = np.asarray(present[name], dtype=np.int32)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="keep roughly this many evenly-spaced layers; default keeps all",
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
    model, torch = load_model(args.model, args.dtype, args.device)

    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        target = args.out / f"{path.parent.name}--{path.name}.npz"
        if target.exists():
            continue
        trajectory = Trajectory.load(path)
        step_started = time.monotonic()
        try:
            saved = extract(trajectory, model, torch, args.layers)
        except ValueError as error:
            print(f"  [{index}/{len(paths)}] SKIPPED {error}", flush=True)
            continue
        np.savez_compressed(target, **saved)
        size = sum(value.nbytes for value in saved.values()) / 1e6
        print(
            f"  [{index}/{len(paths)}] {path.name} "
            f"{len(trajectory.steps)} steps  {size:.0f} MB  "
            f"{time.monotonic() - step_started:.0f}s",
            flush=True,
        )

    print(f"done in {(time.monotonic() - started) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
