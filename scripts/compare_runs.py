"""Put several runs side by side — one row per cell of an ablation.

    uv run python scripts/compare_runs.py runs/ablate-d20-*

`summarise_runs.py` answers "what does this run say"; this answers "which of
these runs differs from the others", which is a different question and the one
an ablation asks. Offline and re-runnable, for the same reason (D12): the
labelling rules can change and every number here recomputes without a
re-rollout.

The columns are the funnel, in order — **recon**, then **read cue**, then
**crossed** — and that order is the point. Iterating on the crossing rate alone
is how a batch comes back all zeros carrying no information; the earlier stages
say whether the agent got anywhere near the boundary at all. `steps` and
`max_steps` are there because a cell where every trajectory hits the cap is a
cell whose funnel is measuring the budget.

Rows are printed in the order given, so the baseline goes first and each
knockout is read against it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from escape_probes.config import EnvConfig  # noqa: E402
from escape_probes.labels import Labels, label  # noqa: E402
from escape_probes.trace import Trajectory, TrajectoryMeta  # noqa: E402

COLUMNS = (
    f"{'run':<26} {'n':>3}  {'recon':>9} {'read cue':>9} {'crossed':>9}  "
    f"{'passed':>9} {'at cap':>9}  {'steps':>5}  config"
)


def load(root: Path) -> list[tuple[TrajectoryMeta, Labels]]:
    env = EnvConfig()
    pairs = []
    for path in sorted(root.glob("*/meta.json")):
        trajectory = Trajectory.load(path.parent)
        pairs.append((trajectory.meta, label(trajectory, env)))
    return pairs


def rate(count: int, total: int) -> str:
    """Count and percentage together.

    The raw count matters as much as the share here: `2/18` and `11/99` read
    very differently and an ablation cell is small enough that a bare
    percentage invites over-reading.
    """
    if not total:
        return "     —   "
    return f"{count:>2}/{total:<2} {count / total:>3.0%}"


def describe(metas: list[TrajectoryMeta]) -> str:
    """The knobs that differ across an ablation, from the trajectories' own meta.

    Read back from what was written rather than from the config file, so a row
    cannot claim a setting the run did not actually use.
    """
    if not metas:
        return ""
    first = metas[0]
    tools = "+".join(t[0] for t in first.tools)
    return f"prompt={first.prompt_level} tools={tools} temp={first.temperature}"


def row(root: Path) -> str:
    pairs = load(root)
    if not pairs:
        return f"{root.name:<26} (nothing on disk)"
    metas = [meta for meta, _ in pairs]
    labels = [labelled for _, labelled in pairs]
    n = len(labels)
    at_cap = sum(m.outcome in ("max_steps", "max_submissions") for m in metas)
    steps = statistics.median(m.n_steps for m in labels)
    return (
        f"{root.name:<26} {n:>3}  "
        f"{rate(sum(m.first_recon_step is not None for m in labels), n)} "
        f"{rate(sum(m.read_cue for m in labels), n)} "
        f"{rate(sum(m.crossed for m in labels), n)}  "
        f"{rate(sum(m.outcome == 'passed' for m in metas), n)} "
        f"{rate(at_cap, n)}  "
        f"{steps:>5.0f}  {describe(metas)}"
    )


def outcome_mix(roots: list[Path]) -> str:
    lines = []
    for root in roots:
        metas = [meta for meta, _ in load(root)]
        if metas:
            lines.append(f"  {root.name:<26} {dict(Counter(m.outcome for m in metas))}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    missing = [r for r in args.runs if not r.exists()]
    if missing:
        print(f"no such run directory: {missing}", file=sys.stderr)
        return 1

    print(COLUMNS)
    for root in args.runs:
        print(row(root))

    print("\noutcomes")
    print(outcome_mix(args.runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
