"""Pass 2: replay a stored trajectory and keep the residual stream.

Pass 1 is stochastic, needs a sandbox, and costs a rented card; it happens once
and freezes the token ids (D12). This is the deterministic half — feed those ids
back through the model with `output_hidden_states=True` and keep a small number
of vectors per step. Everything after it (which layer, which labelling rule,
which threshold, which analysis unit) is a CPU decision on the stored result,
which is the whole reason for the split.

**Every layer is kept.** One forward returns all of them, so a layer sweep costs
storage rather than compute. The alternative is guessing a layer now and
re-renting the card when the guess is wrong.

**Three positions, and they cannot be added later.** Layers come free with the
forward; positions do not — choosing them is choosing which vectors to write
down. They are three hypotheses about where intent sits:

* `last_prompt` (a) — the last prompt token. The model has read everything and
  is about to speak. The only position that exists *before* generation, so the
  only one an interceptor could use to suppress the turn rather than the action.
* `command` (b) — the first token of the tool call's command string, where the
  path is being written. Closest to the act and therefore latest.
* `generated` (c) — the mean over the generated tokens, in case intent is spread
  across the reasoning rather than sitting on one token.

The thing this module must not get wrong is which vector it stores. An
activation one token to the left of the intended position does not fail loudly:
it trains, reports a number, and the number is about something else. That has
already happened once in this codebase — `HFModel` compared a decoded prefix
with `>=` and returned the token before the command string — so an index that
lands outside its span raises here rather than being clipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from escape_probes.trace import Step

POSITIONS = ("last_prompt", "command", "generated")
"""The three probe positions, in the order they occur within a step."""


@dataclass(frozen=True)
class StepPositions:
    """Where in the token stream each position sits, for one step.

    Indices are absolute over the whole trajectory's stream, because spans nest
    rather than tile: step k's prompt is `(0, n_k)`, so a forward over step k's
    stream indexes the same way.
    """

    last_prompt: int
    command: int | None
    generated: tuple[int, int] | None


def step_positions(step: Step) -> StepPositions:
    """Resolve a step's three positions, refusing indices that cannot be right."""
    prompt_start, prompt_end = step.prompt_span
    gen_start, gen_end = step.gen_span

    if prompt_end <= prompt_start:
        raise ValueError(f"step {step.step_idx}: empty prompt span {step.prompt_span}")

    command = step.tool_start_token_idx
    if command is not None and not gen_start <= command < gen_end:
        # Clipping would store a real vector belonging to a different token, and
        # nothing downstream could tell.
        raise ValueError(
            f"step {step.step_idx}: command token {command} is outside the "
            f"generation span {step.gen_span}"
        )

    return StepPositions(
        last_prompt=prompt_end - 1,
        command=command,
        generated=(gen_start, gen_end) if gen_end > gen_start else None,
    )


def stack_positions(
    hidden: Any,
    positions: StepPositions,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """Gather `(layers, features)` for each available position.

    `hidden` is `(layers, tokens, features)` — every layer's residual stream over
    the step's whole token stream, as one forward returns it.

    A position the step does not have is **absent from the result** rather than
    zero-filled. Zeros are indistinguishable from a genuine activation and would
    join the training set unnoticed; a missing key cannot.

    Stored in fp32, not fp16. The model runs bf16, whose range fp16 does not
    cover: Qwen-family residual streams carry massive-activation outliers well
    past fp16's 65504, and `astype(np.float16)` maps those to `inf` silently —
    `standardise` then turns the whole column to NaN and the probe trains on
    it anyway. bf16 -> fp32 is exact. 9.6 GB for the full set is fine on disk;
    what is not fine is a probe measuring the fp16 cast.
    """
    states = np.asarray(hidden)
    stacked: dict[str, np.ndarray] = {
        "last_prompt": states[:, positions.last_prompt, :].astype(dtype)
    }
    if positions.command is not None:
        stacked["command"] = states[:, positions.command, :].astype(dtype)
    if positions.generated is not None:
        start, end = positions.generated
        stacked["generated"] = states[:, start:end, :].mean(axis=1).astype(dtype)
    return stacked


@dataclass(frozen=True)
class StepSlots:
    """Where one step's vectors sit in a `GatherPlan`'s two matrices."""

    step_idx: int
    last_prompt: int
    """Row in the points matrix."""
    command: int | None
    """Row in the points matrix, or None when the step had no tool call."""
    generated: int | None
    """Row in the spans matrix, or None when the step generated nothing."""


@dataclass(frozen=True)
class GatherPlan:
    """Everything one forward over the full trajectory must gather.

    Spans nest rather than tile, so every step's stream is a prefix of the
    final one and one forward over the final stream contains every step's
    positions — the per-step prefix forwards are redundant with it exactly, not
    approximately: positions are absolute indices, and every layer (full
    attention and gated-delta-net alike) is causal, so token t's residual in
    the full forward equals the prefix forward's.

    The plan is built on CPU and the gather runs inside forward hooks, so the
    only thing that ever leaves the GPU is `(layers, n_points + n_spans,
    features)` — a few MB — rather than the full `(layers, tokens, features)`
    stream, which at 32k tokens is tens of GB and does not fit next to the
    weights.
    """

    length: int
    """Tokens the forward must cover: the last step's `gen_span` end."""
    points: tuple[int, ...]
    """Absolute token indices to gather single vectors at, sorted, unique."""
    spans: tuple[tuple[int, int], ...]
    """Absolute `(start, end)` spans to mean over, in step order."""
    steps: tuple[StepSlots, ...]


def gather_plan(steps: Sequence[Step]) -> GatherPlan:
    """One trajectory's plan. Raises like `step_positions` on impossible spans."""
    if not steps:
        raise ValueError("no steps to plan a gather for")
    resolved = [step_positions(step) for step in steps]

    point_rows: dict[int, int] = {}
    for positions in resolved:
        for index in (positions.last_prompt, positions.command):
            if index is not None and index not in point_rows:
                point_rows[index] = -1
    points = tuple(sorted(point_rows))
    point_rows = {index: row for row, index in enumerate(points)}

    spans: list[tuple[int, int]] = []
    slots = []
    for step, positions in zip(steps, resolved, strict=True):
        generated = None
        if positions.generated is not None:
            generated = len(spans)
            spans.append(positions.generated)
        slots.append(
            StepSlots(
                step_idx=step.step_idx,
                last_prompt=point_rows[positions.last_prompt],
                command=None if positions.command is None else point_rows[positions.command],
                generated=generated,
            )
        )
    return GatherPlan(
        length=max(step.gen_span[1] for step in steps),
        points=points,
        spans=tuple(spans),
        steps=tuple(slots),
    )


def assemble(
    plan: GatherPlan,
    point_matrix: Any,
    span_matrix: Any,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """The npz payload, from what the hooks gathered.

    `point_matrix` is `(layers, len(plan.points), features)`; `span_matrix` is
    `(layers, len(plan.spans), features)`, each span already meaned. Output
    matches the per-step extraction exactly: `{position: (steps, layers,
    features)}` plus `{position}_steps`, with a step's missing position absent
    rather than zero-filled.

    Refuses non-finite values. The model can produce them (bf16 overflow is an
    inf the forward carries along), and a stored inf poisons `standardise`
    into NaN for the whole column — the probe would train, report a number,
    and measure the corruption.
    """
    point_matrix = np.asarray(point_matrix)
    span_matrix = np.asarray(span_matrix)
    if len(plan.points) and point_matrix.shape[1] != len(plan.points):
        raise ValueError(
            f"point matrix has {point_matrix.shape[1]} rows for {len(plan.points)} points"
        )
    if len(plan.spans) and span_matrix.shape[1] != len(plan.spans):
        raise ValueError(f"span matrix has {span_matrix.shape[1]} rows for {len(plan.spans)} spans")

    collected: dict[str, list[np.ndarray]] = {name: [] for name in POSITIONS}
    present: dict[str, list[int]] = {name: [] for name in POSITIONS}
    for slot in plan.steps:
        collected["last_prompt"].append(point_matrix[:, slot.last_prompt])
        present["last_prompt"].append(slot.step_idx)
        if slot.command is not None:
            collected["command"].append(point_matrix[:, slot.command])
            present["command"].append(slot.step_idx)
        if slot.generated is not None:
            collected["generated"].append(span_matrix[:, slot.generated])
            present["generated"].append(slot.step_idx)

    saved: dict[str, np.ndarray] = {}
    for name in POSITIONS:
        if not collected[name]:
            continue
        stacked = np.stack(collected[name]).astype(dtype)
        if not np.isfinite(stacked).all():
            raise ValueError(f"non-finite values in position {name!r}; refusing to store them")
        saved[name] = stacked
        saved[f"{name}_steps"] = np.asarray(present[name], dtype=np.int32)
    return saved


__all__ = [
    "POSITIONS",
    "GatherPlan",
    "StepPositions",
    "StepSlots",
    "assemble",
    "gather_plan",
    "stack_positions",
    "step_positions",
]
