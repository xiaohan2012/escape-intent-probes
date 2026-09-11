"""Persisting a trajectory, and reading it back for Pass 2.

Only what cannot be recomputed is stored (D12): the token ids, because sampling
is stochastic; the observations, because execution is stateful; and the run's
identity. Everything else — t*, whether the cue was read, where the probe
positions land — is derived offline by re-reading this record, so changing how a
label is computed never costs a re-rollout.

Layout, one directory per trajectory:

    runs/<run_id>/<instance_id>--<condition>--<seed>/
      meta.json     identity, model, config, outcome
      steps.jsonl   one line per step: spans, tool call, observation
      tokens.npy    the trajectory's full token id sequence (int32)

Token ids are one continuous array rather than per-step slices: Pass 2 runs a
single forward over the whole context and needs absolute indices, and
re-concatenating slices would introduce exactly the mismatch E4 exists to catch.

The stream is the **final** conversation, not a concatenation of per-step
prompts. A chat template is append-only, so every step's prompt is a prefix of
the last one; storing each step's prompt separately would keep the conversation
prefix twenty-five times over, and concatenating them would produce a document
with its own beginning repeated — which Pass 2 would then teacher-force as text
the model never saw in that order.

Step spans therefore **nest** rather than tile: step k's `prompt_span` is
`(0, n_k)` for a growing `n_k`, and its `gen_span` covers the tokens the model
emitted at that point. Pass 2 forwards the stream once and reads each step's
positions out of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from escape_probes.config import Condition
from escape_probes.model import Generation

Outcome = Literal["passed", "failed", "max_steps", "parse_failed", "error"]


class Step(BaseModel):
    """One turn: what the model was shown, what it emitted, what happened."""

    step_idx: int

    prompt_span: tuple[int, int]
    """[0, end) — the context the model saw at this step. Nested, not tiled:
    each step's prompt is a prefix of the next one's."""

    gen_span: tuple[int, int]
    """[start, end) of the tokens the model emitted at this step."""

    tool_start_token_idx: int | None = None
    """Absolute index of the first token of the tool call's command string —
    probe position (b). Stored rather than re-derived: the backend knows the
    boundary for free, and recovering it later means re-parsing JSON through a
    tokenizer, the most fragile link in the chain (D12)."""

    tool_name: str | None = None
    tool_arguments: dict[str, str] = Field(default_factory=dict)
    parse_error: str | None = None

    observation: str = ""
    """Untruncated. What the model saw may have been shortened to fit the
    context, but no analysis should depend on that choice."""

    exit_code: int | None = None

    generate_seconds: float = 0.0
    """Time in the model. Together with `exec_seconds` this splits a step's cost
    between the two things that can be optimised independently — decode speed
    and the container — so a throughput decision rests on a measurement rather
    than an estimate (D13)."""

    exec_seconds: float = 0.0
    """Time in the sandbox: a tool call, or a submission's full test run."""

    @property
    def position_a_index(self) -> int:
        """Probe position (a): the last context token, where the model is about
        to speak. AgentLens's convention (D14)."""
        return self.prompt_span[1] - 1


class TrajectoryMeta(BaseModel):
    """Identity and outcome. The config is echoed in full so a trajectory can be
    interpreted without the file that produced it."""

    run_id: str
    instance_id: str
    condition: Condition
    seed: int

    model_id: str
    temperature: float
    prompt_level: str
    cue_strength: str
    tools: tuple[str, ...]

    image: str
    system_prompt_sha: str

    outcome: Outcome
    n_steps: int
    n_parse_errors: int = 0
    wall_clock_seconds: float = 0.0
    final_diff: str = ""
    """`git diff` at the end — what the agent actually changed."""


class Trajectory(BaseModel):
    meta: TrajectoryMeta
    steps: list[Step]
    token_ids: list[int]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(self.meta.model_dump_json(indent=2))
        with (directory / "steps.jsonl").open("w") as handle:
            for step in self.steps:
                handle.write(step.model_dump_json() + "\n")
        np.save(directory / "tokens.npy", np.asarray(self.token_ids, dtype=np.int32))

    @classmethod
    def load(cls, directory: Path) -> Trajectory:
        meta = TrajectoryMeta.model_validate_json((directory / "meta.json").read_text())
        steps = [
            Step.model_validate_json(line)
            for line in (directory / "steps.jsonl").read_text().splitlines()
            if line.strip()
        ]
        token_ids = np.load(directory / "tokens.npy").tolist()
        return cls(meta=meta, steps=steps, token_ids=token_ids)


class TrajectoryWriter:
    """Accumulates a trajectory while the loop runs.

    The token stream is appended to as the conversation grows, and each step
    records absolute spans into it, so Pass 2 can slice without reconstructing
    anything.
    """

    def __init__(self) -> None:
        self.token_ids: list[int] = []
        self.steps: list[Step] = []
        self._prompt: tuple[int, ...] = ()

    def add_step(self, generation: Generation, **fields: Any) -> Step:
        """Record one step, keeping the stream at the full conversation so far.

        The new prompt must extend the previous one. If it does not, the backend
        re-rendered earlier turns between steps, and Pass 2 would replay a
        sequence the model never actually saw — the same class of failure E4
        checks for, caught here for free.
        """
        prompt = generation.prompt_token_ids
        if prompt[: len(self._prompt)] != self._prompt:
            raise ValueError(
                f"step {len(self.steps)}: this step's prompt does not extend the last one; "
                "the backend re-rendered earlier turns"
            )

        self._prompt = prompt
        self.token_ids = list(prompt) + list(generation.gen_token_ids)

        offset = generation.tool_start_token_idx
        step = Step(
            step_idx=len(self.steps),
            prompt_span=(0, len(prompt)),
            gen_span=(len(prompt), len(self.token_ids)),
            tool_start_token_idx=None if offset is None else len(prompt) + offset,
            **fields,
        )
        self.steps.append(step)
        return step
