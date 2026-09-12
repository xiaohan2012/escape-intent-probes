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
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.boundary import plant  # noqa: E402
from escape_probes.config import RunConfig  # noqa: E402
from escape_probes.labels import Labels, label  # noqa: E402
from escape_probes.model import ModelBackend  # noqa: E402
from escape_probes.rollout import SerialBatch, drive_batch, rollout_steps  # noqa: E402
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
    from escape_probes.backends import build_model as build  # noqa: PLC0415

    return build(config.model, tools=config.agent.tools)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="pilot-01")
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["benign", "impossible"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--split", default="conflicting", choices=["conflicting", "oneoff"])
    parser.add_argument("--limit", type=int, default=None, help="stop after this many")
    parser.add_argument(
        "--setup-workers",
        type=int,
        default=8,
        help="threads opening containers for a round; setup is I/O and the GPU idles through it",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="trajectories advanced in lock step per engine call (D21)",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--host", default=None, help="SSH destination of the Docker host")
    parser.add_argument("--backend", default=None, choices=["vllm", "hf"])
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="tensor parallel size; a 30B MoE in bf16 needs >=2 on a 48 GB card",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = RunConfig(run_id=args.run_id, instance_ids=tuple(args.instances))
    if args.max_steps:
        config.agent.max_steps = args.max_steps
    if args.backend:
        config.model.backend = args.backend
    if args.tp:
        config.model.tensor_parallel_size = args.tp
    if args.max_model_len:
        config.model.max_model_len = args.max_model_len

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
    done_count = 0

    for group_start in range(0, len(todo), args.batch_size):
        group = todo[group_start : group_start + args.batch_size]
        group_started = time.monotonic()

        with ExitStack() as stack:
            # Setup is outside the driver on purpose: a failure here belongs to
            # one work item and should be reported as such, not become a
            # trajectory that never started.
            #
            # It runs in threads because it is I/O — `docker run`, then a few
            # `docker exec` calls per item — and the card has nothing to do
            # while it happens. Serially it was ~7 s per container, so a round
            # of 18 spent over two minutes with the GPU at zero, once per cell.
            # Each worker uses its own `ExitStack` and hands the teardown over
            # with `pop_all`, which is the thread-safe way to put a context
            # manager entered elsewhere onto this stack.
            def open_one(
                item: tuple[str, str, int],
            ) -> tuple[tuple[str, str, int], SweBenchTask, DockerSandbox, ExitStack]:
                instance, condition, seed = item
                local = ExitStack()
                try:
                    sandbox = local.enter_context(
                        DockerSandbox(instance, config.env, host=args.host)
                    )
                    task = SweBenchTask(
                        row=rows[instance],
                        condition=condition,
                        reset_tests=config.env.reset_tests,
                    )
                    task.setup(sandbox)
                    plant(sandbox, config.env, instance, task.gold_patch)
                except BaseException:
                    local.close()
                    raise
                return item, task, sandbox, local.pop_all()

            opened: list[tuple[tuple[str, str, int], SweBenchTask, DockerSandbox]] = []
            with ThreadPoolExecutor(max_workers=min(len(group), args.setup_workers)) as pool:
                futures = {pool.submit(open_one, item): item for item in group}
                for future in futures:
                    instance, condition, seed = futures[future]
                    try:
                        item, task, sandbox, teardown = future.result()
                    except Exception as error:
                        done_count += 1
                        print(
                            f"  [{done_count}/{len(todo)}] {instance} {condition} {seed}: "
                            f"SETUP FAILED {error}",
                            flush=True,
                        )
                        continue
                    stack.push(teardown)
                    opened.append((item, task, sandbox))

            if not opened:
                continue

            steps = [
                rollout_steps(task, sandbox, config, condition, seed)
                for (_, condition, seed), task, sandbox in opened
            ]

            def report_error(index: int, error: Exception, opened=opened) -> None:
                instance, condition, seed = opened[index][0]
                print(f"  {instance} {condition} {seed}: FAILED {error}", flush=True)

            batched = model if hasattr(model, "generate_batch") else SerialBatch(model)
            trajectories = drive_batch(steps, batched, on_error=report_error)

        group_seconds = time.monotonic() - group_started
        for (instance, condition, seed), trajectory in zip(
            [item[0] for item in opened], trajectories, strict=True
        ):
            done_count += 1
            if trajectory is None:
                continue
            trajectory.save(config.trajectory_dir(instance, condition, seed))
            labels = label(trajectory, config.env)
            results.append(labels)
            print(
                f"  [{done_count}/{len(todo)}] {instance} {condition} seed={seed} "
                f"{trajectory.meta.outcome} steps={labels.n_steps} t*={labels.t_star} "
                f"cue={'y' if labels.read_cue else 'n'} "
                f"{trajectory.meta.wall_clock_seconds:.0f}s",
                flush=True,
            )
        print(
            f"  -- round of {len(opened)} in {group_seconds:.0f}s "
            f"({group_seconds / len(opened):.0f}s per trajectory)",
            flush=True,
        )

        if results:
            print(f"\n--- after {done_count} ---")
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
