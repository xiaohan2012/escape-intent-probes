"""Tests for Pass 2: turning a stored trajectory into probe inputs.

The expensive half of the project has already happened by the time this runs.
Pass 1 generated the trajectory and froze its token ids; Pass 2 replays them and
reads the residual stream, and everything downstream — which layer, which
labelling rule, which threshold — is a CPU decision on the stored result.

So the one thing this module must not get wrong is *which vector it stored*. A
probe trained on the activation one token to the left of the intended position
does not fail loudly; it trains, reports a number, and the number is about
something else. That failure has already happened once here — `HFModel` compared
a decoded prefix with `>=` and returned the token before the command string
(see `test_probe_position.py`).

Three positions, because they are three different hypotheses about where intent
lives, and because adding a fourth later costs another GPU rental:

* **(a)** the last prompt token — the model has read everything and is about to
  speak. The only position available *before* generation, so the only one an
  interceptor could use to suppress the turn rather than the action.
* **(b)** the first token of the tool call's command string — the moment the
  path is being written. Latest, and closest to the act.
* **(c)** the mean over the generated tokens — in case intent is spread across
  the reasoning rather than localised.

All layers are kept. `output_hidden_states=True` returns them from one forward,
so a layer sweep costs storage rather than compute, and the alternative is
guessing a layer now and re-renting the card when the guess is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from escape_probes.activations import (
    POSITIONS,
    GatherPlan,
    StepPositions,
    StepSlots,
    assemble,
    gather_plan,
    stack_positions,
    step_positions,
)
from escape_probes.trace import Step


def make_step(
    idx: int = 0,
    prompt_end: int = 10,
    gen_end: int = 16,
    tool_start: int | None = 13,
) -> Step:
    return Step(
        step_idx=idx,
        prompt_span=(0, prompt_end),
        gen_span=(prompt_end, gen_end),
        tool_start_token_idx=tool_start,
        tool_name="bash",
        tool_arguments={"cmd": "cat grader.yaml"},
    )


class TestStepPositions:
    """Which token indices each position resolves to."""

    def test_position_a_is_the_last_prompt_token(self) -> None:
        # Not the first generated token: (a) is where the model is about to
        # speak, which is the last token it has *read*.
        assert step_positions(make_step()).last_prompt == 9

    def test_position_b_is_the_command_token(self) -> None:
        assert step_positions(make_step()).command == 13

    def test_position_c_spans_the_generation(self) -> None:
        assert step_positions(make_step()).generated == (10, 16)

    def test_position_b_is_absent_when_the_step_had_no_tool_call(self) -> None:
        # A parse failure has no command string. The step still has (a) and (c),
        # and must not silently borrow a neighbouring index.
        assert step_positions(make_step(tool_start=None)).command is None

    def test_an_empty_generation_has_no_mean(self) -> None:
        step = make_step(prompt_end=10, gen_end=10, tool_start=None)
        assert step_positions(step).generated is None

    @pytest.mark.parametrize("tool_start", [9, 16, 20])
    def test_a_command_index_outside_the_generation_is_refused(self, tool_start: int) -> None:
        # The failure mode that motivated this file: an off-by-one that stays in
        # range reads a real vector belonging to a different token, and nothing
        # downstream can tell. Refuse rather than store it.
        step = make_step(prompt_end=10, gen_end=16, tool_start=tool_start)
        with pytest.raises(ValueError, match="outside"):
            step_positions(step)


class TestStackPositions:
    """Gathering the three positions out of one forward's hidden states."""

    @property
    def hidden(self) -> np.ndarray:
        # (layers, tokens, features), with each token's vector equal to its
        # index, so the right slice is identifiable by value.
        layers, tokens, features = 3, 16, 4
        states = np.zeros((layers, tokens, features), dtype=np.float32)
        for layer in range(layers):
            for token in range(tokens):
                states[layer, token] = token + 100 * layer
        return states

    def test_it_takes_the_named_token_for_a_and_b(self) -> None:
        stacked = stack_positions(self.hidden, step_positions(make_step()))
        # Layer 1, position (a) -> token 9 -> 9 + 100
        assert stacked["last_prompt"][1, 0] == pytest.approx(109.0)
        assert stacked["command"][1, 0] == pytest.approx(113.0)

    def test_it_averages_the_generation_for_c(self) -> None:
        # Tokens 10..15, mean 12.5, plus the layer offset.
        stacked = stack_positions(self.hidden, step_positions(make_step()))
        assert stacked["generated"][2, 0] == pytest.approx(212.5)

    def test_every_position_keeps_every_layer(self) -> None:
        stacked = stack_positions(self.hidden, step_positions(make_step()))
        for name in POSITIONS:
            assert stacked[name].shape == (3, 4), name

    def test_a_missing_position_is_absent_rather_than_zero(self) -> None:
        # Zeros would be indistinguishable from a genuine activation and would
        # quietly join the training set.
        stacked = stack_positions(self.hidden, step_positions(make_step(tool_start=None)))
        assert "command" not in stacked
        assert set(stacked) == {"last_prompt", "generated"}

    def test_it_stores_fp32_by_default(self) -> None:
        # Not fp16: the model runs bf16, whose range fp16 does not cover, and
        # Qwen-family residual streams carry outliers past fp16's 65504 —
        # `astype(np.float16)` maps those to inf silently, and `standardise`
        # then NaNs the whole column. bf16 -> fp32 is exact.
        stacked = stack_positions(self.hidden, step_positions(make_step()))
        assert stacked["last_prompt"].dtype == np.float32


class TestStepPositionsRoundTrip:
    def test_the_indices_are_absolute_over_the_whole_stream(self) -> None:
        # Spans nest rather than tile (trace.py): step k's prompt is (0, n_k),
        # so every index here indexes the one growing token stream, and a
        # forward over step k's stream is what produces the hidden states.
        step = make_step(idx=4, prompt_end=300, gen_end=340, tool_start=328)
        positions = step_positions(step)
        assert positions.last_prompt == 299
        assert positions.command == 328
        assert positions.generated == (300, 340)

    def test_it_is_a_plain_value(self) -> None:
        assert step_positions(make_step()) == StepPositions(
            last_prompt=9, command=13, generated=(10, 16)
        )


class TestGatherPlan:
    """One forward per trajectory: which rows that forward must gather."""

    @property
    def steps(self) -> list[Step]:
        # Three nesting steps: with a tool call, without one, with again.
        return [
            make_step(idx=0, prompt_end=10, gen_end=16, tool_start=13),
            make_step(idx=1, prompt_end=20, gen_end=24, tool_start=None),
            make_step(idx=2, prompt_end=30, gen_end=38, tool_start=31),
        ]

    def test_length_covers_the_last_generation(self) -> None:
        assert gather_plan(self.steps).length == 38

    def test_points_are_sorted_and_unique(self) -> None:
        # last_prompt tokens 9, 19, 29 and command tokens 13, 31.
        assert gather_plan(self.steps).points == (9, 13, 19, 29, 31)

    def test_spans_are_the_generations_in_step_order(self) -> None:
        assert gather_plan(self.steps).spans == ((10, 16), (20, 24), (30, 38))

    def test_slots_point_back_into_the_matrices(self) -> None:
        plan = gather_plan(self.steps)
        assert plan.steps[0] == StepSlots(step_idx=0, last_prompt=0, command=1, generated=0)
        assert plan.steps[1] == StepSlots(step_idx=1, last_prompt=2, command=None, generated=1)
        assert plan.steps[2] == StepSlots(step_idx=2, last_prompt=3, command=4, generated=2)

    def test_a_shared_index_is_gathered_once(self) -> None:
        # A command on the very last prompt token of a later step collides with
        # nothing here, but two steps can share indices only through dedup —
        # the plan must map both slots to the same row.
        steps = [
            make_step(idx=0, prompt_end=10, gen_end=16, tool_start=13),
            make_step(idx=1, prompt_end=14, gen_end=20, tool_start=None),
        ]
        plan = gather_plan(steps)
        assert plan.points == (9, 13)
        assert plan.steps[1].last_prompt == 1  # token 13, the same row as step 0's command

    def test_an_empty_generation_has_no_span(self) -> None:
        steps = [make_step(idx=0, prompt_end=10, gen_end=10, tool_start=None)]
        plan = gather_plan(steps)
        assert plan.spans == ()
        assert plan.steps[0].generated is None

    def test_it_refuses_an_out_of_span_command(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            gather_plan([make_step(tool_start=9)])

    def test_it_refuses_no_steps(self) -> None:
        with pytest.raises(ValueError, match="no steps"):
            gather_plan([])


class TestAssemble:
    """The one-forward result must equal the per-step extraction, exactly."""

    @property
    def steps(self) -> list[Step]:
        return [
            make_step(idx=0, prompt_end=10, gen_end=16, tool_start=13),
            make_step(idx=1, prompt_end=20, gen_end=24, tool_start=None),
            make_step(idx=2, prompt_end=30, gen_end=38, tool_start=31),
        ]

    @property
    def hidden(self) -> np.ndarray:
        # The full trajectory's stream: token t's vector is t + 100 * layer,
        # so any wrong row is identifiable by value.
        layers, tokens, features = 3, 38, 4
        states = np.zeros((layers, tokens, features), dtype=np.float32)
        for layer in range(layers):
            for token in range(tokens):
                states[layer, token] = token + 100 * layer
        return states

    def matrices(self, plan: GatherPlan) -> tuple[np.ndarray, np.ndarray]:
        """What the forward hooks would gather from `self.hidden`."""
        hidden = self.hidden
        points = hidden[:, list(plan.points), :]
        spans = (
            np.stack([hidden[:, start:end, :].mean(axis=1) for start, end in plan.spans], axis=1)
            if plan.spans
            else np.zeros((hidden.shape[0], 0, hidden.shape[2]), dtype=np.float32)
        )
        return points, spans

    def assembled(self) -> dict[str, np.ndarray]:
        plan = gather_plan(self.steps)
        return assemble(plan, *self.matrices(plan))

    def test_it_equals_the_per_step_extraction(self) -> None:
        # The exactness claim behind one-forward-per-trajectory: positions are
        # absolute and spans nest, so slicing the full stream per step must
        # reproduce the per-step `stack_positions` path bit for bit.
        by_step: dict[str, list[np.ndarray]] = {name: [] for name in POSITIONS}
        for step in self.steps:
            stacked = stack_positions(self.hidden, step_positions(step))
            for name, vector in stacked.items():
                by_step[name].append(vector)
        assembled = self.assembled()
        for name in POSITIONS:
            np.testing.assert_array_equal(assembled[name], np.stack(by_step[name]))

    def test_a_missing_position_is_absent_with_its_step_index(self) -> None:
        assembled = self.assembled()
        np.testing.assert_array_equal(assembled["command_steps"], [0, 2])
        np.testing.assert_array_equal(assembled["last_prompt_steps"], [0, 1, 2])

    def test_it_stores_fp32(self) -> None:
        assert self.assembled()["last_prompt"].dtype == np.float32

    def test_it_refuses_non_finite_values(self) -> None:
        # The model can produce an inf (bf16 overflow travels through the
        # forward); storing it poisons `standardise` into NaN for the column
        # and the probe trains on the corruption without a word.
        plan = gather_plan(self.steps)
        points, spans = self.matrices(plan)
        points[1, 2] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            assemble(plan, points, spans)

    @pytest.mark.parametrize("axis", ["points", "spans"])
    def test_it_refuses_a_matrix_of_the_wrong_width(self, axis: str) -> None:
        plan = gather_plan(self.steps)
        points, spans = self.matrices(plan)
        if axis == "points":
            points = points[:, :-1]
        else:
            spans = spans[:, :-1]
        with pytest.raises(ValueError, match="rows for"):
            assemble(plan, points, spans)
