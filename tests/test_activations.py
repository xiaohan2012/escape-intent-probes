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

from escape_probes.activations import POSITIONS, StepPositions, stack_positions, step_positions
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

    def test_it_stores_half_precision(self) -> None:
        # 96 trajectories x 25 steps x 65 layers x 3 positions x 5120 features is
        # 4.8 GB in fp16 and 9.6 in fp32, and the probe is fitted in float64
        # after loading regardless.
        stacked = stack_positions(self.hidden, step_positions(make_step()), dtype=np.float16)
        assert stacked["last_prompt"].dtype == np.float16


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
