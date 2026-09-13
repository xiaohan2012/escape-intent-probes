"""Fit the probe on stored activations and report it two ways.

    uv run python scripts/train_probe.py --activations activations/ --runs runs/probe-01-*
    uv run python scripts/train_probe.py --activations activations/ --runs runs/probe-01-* \\
        --layers 8 --rule R1 R2 --position last_prompt command generated

CPU only, seconds per fit. Everything expensive happened in Pass 1 and Pass 2,
which is the point of the split: the layer, the labelling rule, the probe
position and the threshold are all decisions taken here, on stored vectors,
and re-taking them costs nothing.

**Two reports, because they answer different questions.** The step-level AUC is
what the probing literature reports and what makes this comparable to AgentLens
(D14). The trajectory-level report is what this study actually asks — did the
alarm arrive before the act, and how many steps before — and it is the one that
can tell a probe from a string matcher. With `read` and `crossed` almost
perfectly correlated in our 118 trajectories, a step-level AUC can be high
because the probe detects `/opt/grader` sitting in the context window. That
shows up immediately as a median lead time of one step.

**The threshold is chosen on the training folds.** Picking it on the held-out
fold is how a lead time gets manufactured; at 60 trajectories it would move the
headline by several steps.

**`induced_not_crossed` is scored but never trained on.** It is the diagnostic
that separates the two things a probe could have learned: if trajectories under
pressure that did *not* cross score like the crossings, the probe reads task
pressure rather than intent.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.config import EnvConfig  # noqa: E402
from escape_probes.fit import LinearProbe, roc_auc, standardise  # noqa: E402
from escape_probes.labels import Labels, label  # noqa: E402
from escape_probes.probe import (  # noqa: E402
    LABEL_RULES,
    Example,
    LabelRule,
    fold_by_instance,
    label_steps,
    trajectory_report,
)
from escape_probes.trace import Trajectory  # noqa: E402


def load_labels(runs: list[Path]) -> dict[tuple[str, str, int], Labels]:
    """`(run, instance, seed) -> Labels` for every finished trajectory."""
    env = EnvConfig()
    found = {}
    for run in runs:
        if not run.is_dir():
            continue
        for meta in sorted(run.glob("*/meta.json")):
            labels = label(Trajectory.load(meta.parent), env)
            found[(run.name, labels.instance_id, labels.seed)] = labels
    return found


def load_activations(directory: Path, position: str) -> dict[tuple[str, str, int], dict[int, Any]]:
    """`(run, instance, seed) -> {step: vector over layers}` for one position."""
    loaded: dict = {}
    for path in sorted(directory.glob("*.npz")):
        run, trajectory = path.stem.split("--", 1)
        instance, condition, seed = trajectory.rsplit("--", 2)
        del condition
        with np.load(path) as data:
            if position not in data:
                continue
            vectors, steps = data[position], data[f"{position}_steps"]
        loaded[(run, instance, int(seed))] = dict(zip(steps.tolist(), vectors, strict=True))
    return loaded


def build(
    labels: dict, activations: dict, rule: LabelRule, layer: int
) -> tuple[list[Example], np.ndarray]:
    """Labelled examples and their feature matrix, for one rule and one layer."""
    examples, features = [], []
    for key, one in labels.items():
        steps = label_steps(one, rule)
        vectors = activations.get(key)
        if not steps or vectors is None:
            continue
        for step, y in sorted(steps.items()):
            if step not in vectors:
                continue
            examples.append(
                Example(
                    instance_id=one.instance_id,
                    seed=one.seed,
                    step=step,
                    y=y,
                    key=(one.instance_id, one.seed),
                )
            )
            features.append(vectors[step][layer])
    return examples, np.asarray(features, dtype=np.float64)


def evaluate(
    examples: list[Example],
    features: np.ndarray,
    labels: dict,
    activations: dict,
    layer: int,
    folds: int,
    l2: float,
) -> tuple[float, object]:
    """Cross-validated AUC, and the trajectory report built from the same folds."""
    index = {id(example): position for position, example in enumerate(examples)}
    held_scores: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    auc_pieces: list[tuple[np.ndarray, np.ndarray]] = []

    for train, test in fold_by_instance(examples, folds):
        rows = np.array([index[id(e)] for e in train])
        y = np.array([e.y for e in train])
        if len(np.unique(y)) < 2:
            continue
        centre, scale = standardise(features[rows])
        probe = LinearProbe(l2=l2).fit((features[rows] - centre) / scale, y)

        test_rows = np.array([index[id(e)] for e in test])
        if len(test_rows):
            scores = probe.score((features[test_rows] - centre) / scale)
            auc_pieces.append((np.array([e.y for e in test]), scores))

        # Every step of every held-out trajectory, including ones no rule
        # labelled: the trajectory report needs a score before t*, and R2 labels
        # nothing there.
        held = {e.instance_id for e in test}
        for key, one in labels.items():
            if one.instance_id not in held:
                continue
            vectors = activations.get(key)
            if vectors is None:
                continue
            steps = sorted(vectors)
            matrix = np.asarray([vectors[s][layer] for s in steps], dtype=np.float64)
            for step, score in zip(steps, probe.score((matrix - centre) / scale), strict=True):
                held_scores[(one.instance_id, one.seed)][int(step)] = float(score)

    y_all = np.concatenate([piece[0] for piece in auc_pieces]) if auc_pieces else np.zeros(0)
    s_all = np.concatenate([piece[1] for piece in auc_pieces]) if auc_pieces else np.zeros(0)
    auc = roc_auc(y_all, s_all) if len(np.unique(y_all)) == 2 else float("nan")

    scored = [
        (one, held_scores[(one.instance_id, one.seed)])
        for one in labels.values()
        if held_scores.get((one.instance_id, one.seed))
    ]
    return auc, scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--position", nargs="+", default=["last_prompt", "command", "generated"])
    parser.add_argument("--rule", nargs="+", default=list(LABEL_RULES))
    parser.add_argument("--layers", type=int, default=8, help="how many layers to sweep")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    labels = load_labels(args.runs)
    if not labels:
        print("no trajectories under those paths", file=sys.stderr)
        return 1
    print(f"{len(labels)} trajectories")

    print(f"\n{'position':<12} {'rule':<5} {'layer':>6} {'n':>6} {'pos':>5} {'AUC':>6}  trajectory")
    for position in args.position:
        activations = load_activations(args.activations, position)
        if not activations:
            print(f"{position:<12} (no activations stored)")
            continue
        depth = next(iter(next(iter(activations.values())).values())).shape[0]
        layers = np.linspace(0, depth - 1, args.layers, dtype=int)

        for rule in args.rule:
            for layer in layers:
                examples, features = build(labels, activations, rule, int(layer))
                if len(examples) < 2 * args.folds:
                    continue
                auc, scored = evaluate(
                    examples, features, labels, activations, int(layer), args.folds, args.l2
                )
                report = trajectory_report(scored, args.threshold)
                positives = sum(e.y for e in examples)
                print(
                    f"{position:<12} {rule:<5} {layer:>6} {len(examples):>6} "
                    f"{positives:>5} {auc:>6.3f}  {report.summary()}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
