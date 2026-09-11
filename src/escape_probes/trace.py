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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from escape_probes.config import DIR_SEPARATOR, Condition, RunConfig

Outcome = Literal["passed", "failed", "gave_up", "max_steps", "parse_failed", "error"]


def trajectory_dirname(instance_id: str, condition: Condition, seed: int) -> str:
    return DIR_SEPARATOR.join([instance_id, condition, str(seed)])


class Step(BaseModel):
    """One turn: what the model was shown, what it emitted, what happened."""

    step_idx: int

    prompt_span: tuple[int, int]
    """[start, end) of this step's context in the token stream."""

    gen_span: tuple[int, int]
    """[start, end) of the model's generation."""

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

    def directory_in(self, config: RunConfig) -> Path:
        return config.trajectory_dir(self.meta.instance_id, self.meta.condition, self.meta.seed)


class TrajectoryWriter:
    """Accumulates a trajectory while the loop runs.

    The token stream is appended to as the conversation grows, and each step
    records absolute spans into it, so Pass 2 can slice without reconstructing
    anything.
    """

    def __init__(self) -> None:
        self.token_ids: list[int] = []
        self.steps: list[Step] = []

    def add_step(
        self,
        prompt_token_ids: tuple[int, ...],
        gen_token_ids: tuple[int, ...],
        tool_start_offset: int | None,
        **fields: Any,
    ) -> Step:
        """Append one step. `tool_start_offset` is relative to the generation;
        it is stored absolute."""
        prompt_start = len(self.token_ids)
        self.token_ids.extend(prompt_token_ids)
        gen_start = len(self.token_ids)
        self.token_ids.extend(gen_token_ids)

        step = Step(
            step_idx=len(self.steps),
            prompt_span=(prompt_start, gen_start),
            gen_span=(gen_start, len(self.token_ids)),
            tool_start_token_idx=None
            if tool_start_offset is None
            else gen_start + tool_start_offset,
            **fields,
        )
        self.steps.append(step)
        return step
