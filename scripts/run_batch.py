"""Run a batch of trajectories, reporting as it goes.

    uv run python scripts/run_batch.py --instances django__django-12419 \\
        --seeds 0 1 2 3 --limit 5

Two things it does that a loop over `run_trajectory.py` would not:

* **Loads the model once.** Twenty seconds per trajectory otherwise, which is
  seven minutes of a twenty-trajectory pilot spent re-reading the same weights.
* **Reports every few trajectories**, so the crossing rate is visible while the
  batch runs rather than an hour later. E2 is the gate that can kill Stage 0,
  and if it is going to fail it should fail early enough to change something.

Already-finished trajectories are skipped, so an interrupted batch resumes by
being run again.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.boundary import plant  # noqa: E402
from escape_probes.config import RunConfig  # noqa: E402
from escape_probes.labels import Labels, label  # noqa: E402
from escape_probes.model import ModelBackend  # noqa: E402
from escape_probes.rollout import rollout  # noqa: E402
from escape_probes.sandbox import DockerSandbox  # noqa: E402
from escape_probes.tasks import SweBenchTask, load_instances  # noqa: E402
from escape_probes.trace import Trajectory  # noqa: E402


def summarise(results: list[Labels], elapsed: float) -> str:
    """The numbers E1 and E2 are read from, as they stand."""
    lines = []
    for condition in ("benign", "impossible"):
        subset = [r for r in results if r.condition == condition]
        if not subset:
            continue
        crossed = sum(r.crossed for r in subset)
        read_cue = sum(r.read_cue for r in subset)
        passed = sum(r.outcome == "passed" for r in subset)
        lines.append(
            f"  {condition:<11} n={len(subset):<3} "
            f"crossed={crossed}/{len(subset)} ({crossed / len(subset):.0%})  "
            f"passed={passed}/{len(subset)} ({passed / len(subset):.0%})  "
            f"read cue={read_cue}/{len(subset)}"
        )
    outcomes = Counter(r.outcome for r in results)
    lines.append(f"  outcomes    {dict(outcomes)}")
    lines.append(f"  mean steps  {sum(r.n_steps for r in results) / len(results):.1f}")
    lines.append(f"  elapsed     {elapsed / 60:.1f} min  ({elapsed / len(results):.0f}s each)")
    return "\n".join(lines)


def build_model(config: RunConfig, fake: bool) -> ModelBackend:
    if fake:
        from escape_probes.model import FakeModel, ScriptedStep  # noqa: PLC0415

        return FakeModel([ScriptedStep(tool="submit")])
    from escape_probes.hf_backend import HFModel  # noqa: PLC0415

    return HFModel(config.model, tools=config.agent.tools)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="pilot-01")
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["benign", "impossible"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--split", default="conflicting", choices=["conflicting", "oneoff"])
    parser.add_argument("--limit", type=int, default=None, help="stop after this many")
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--host", default=None, help="SSH destination of the Docker host")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = RunConfig(run_id=args.run_id, instance_ids=tuple(args.instances))
    if args.max_steps:
        config.agent.max_steps = args.max_steps

    rows = load_instances(args.split)
    missing = [i for i in args.instances if i not in rows]
    if missing:
        print(f"not in the {args.split} split: {missing}", file=sys.stderr)
        return 1

    # Seed varies fastest, so an interrupted batch still covers every instance
    # and condition rather than finishing one instance exhaustively.
    work = [
        (instance, condition, seed)
        for seed in args.seeds
        for instance in args.instances
        for condition in args.conditions
    ]
    todo = [w for w in work if not (config.trajectory_dir(*w) / "meta.json").exists()]
    done = len(work) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} to run ({done} already finished)", flush=True)

    print("loading the model …", flush=True)
    model = build_model(config, args.fake)

    results: list[Labels] = []
    started = time.monotonic()

    for index, (instance, condition, seed) in enumerate(todo, start=1):
        task = SweBenchTask(row=rows[instance], condition=condition)
        step_started = time.monotonic()
        try:
            with DockerSandbox(instance, config.env, host=args.host) as sandbox:
                task.setup(sandbox)
                plant(sandbox, config.env, instance, task.gold_patch)
                trajectory = rollout(task, model, sandbox, config, condition, seed)
        except Exception as error:  # one bad instance must not end the batch
            print(f"  [{index}/{len(todo)}] {instance} {condition} {seed}: FAILED {error}")
            continue

        trajectory.save(config.trajectory_dir(instance, condition, seed))
        labels = label(trajectory, config.env)
        results.append(labels)
        print(
            f"  [{index}/{len(todo)}] {instance} {condition} seed={seed} "
            f"{trajectory.meta.outcome} steps={labels.n_steps} t*={labels.t_star} "
            f"cue={'y' if labels.read_cue else 'n'} "
            f"{time.monotonic() - step_started:.0f}s",
            flush=True,
        )

        if index % args.report_every == 0 and results:
            print(f"\n--- after {index} ---")
            print(summarise(results, time.monotonic() - started))
            print(flush=True)

    if results:
        print("\n=== batch ===")
        print(summarise(results, time.monotonic() - started))
    return 0


def load_finished(config: RunConfig) -> list[Trajectory]:
    """Every trajectory already on disk for this run — for re-labelling offline."""
    root = config.out_root / config.run_id
    return [Trajectory.load(path.parent) for path in sorted(root.glob("*/meta.json"))]


if __name__ == "__main__":
    raise SystemExit(main())
