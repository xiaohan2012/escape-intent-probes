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
    dtype: Any = np.float16,
) -> dict[str, np.ndarray]:
    """Gather `(layers, features)` for each available position.

    `hidden` is `(layers, tokens, features)` — every layer's residual stream over
    the step's whole token stream, as one forward returns it.

    A position the step does not have is **absent from the result** rather than
    zero-filled. Zeros are indistinguishable from a genuine activation and would
    join the training set unnoticed; a missing key cannot.

    Stored in half precision. The full set is 96 trajectories x 25 steps x
    65 layers x 3 positions x 5120 features, which is 4.8 GB at fp16 and 9.6 at
    fp32, and the probe is fitted in float64 after loading either way.
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


__all__ = ["POSITIONS", "StepPositions", "stack_positions", "step_positions"]
