"""Tests for the OpenRouter backend (D22).

The screen runs five models from five labs, and the one failure mode that would
be indistinguishable from its result is a formatting failure: if tool calls do
not survive the round trip, every step fails to parse, every trajectory ends in
`parse_failed`, and the batch reports zero crossings — which is exactly the
signal the screen exists to look for. So the conversions in both directions are
tested directly rather than through the loop.

No HTTP here. The backend takes its transport as an argument, which is cheaper
and stricter than patching a client: the tests assert on the payload that would
go over the wire, including the provider pin without which "which model
crossed" is unanswerable.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from escape_probes.api_backend import APIModel, text_from_api_message, to_api_messages
from escape_probes.config import BASH, EDIT, ModelConfig
from escape_probes.model import Generation, Message, render_tool_call
from escape_probes.tools import parse_tool_call


class TestToAPIMessages:
    """Our flat text conversation, back into the API's structured protocol.

    The rollout loop keeps one canonical representation — text, with the tool
    call in Qwen's tag form — because the labels, the persistence layer and the
    fake backend all read it. A model post-trained on the tool protocol expects
    `assistant(tool_calls)` followed by `tool(result)`, so handing it its own
    call back as prose is off-distribution in the way D15 warns about. This
    function undoes the rendering rather than keeping a second history, so the
    backend stays stateless and `generate(messages)` keeps its contract.
    """

    @property
    def call_text(self) -> str:
        return render_tool_call(BASH, {"cmd": "ls -la"})

    def test_plain_turns_pass_through(self) -> None:
        messages = [
            Message(role="system", content="you are an agent"),
            Message(role="user", content="fix the bug"),
        ]
        assert to_api_messages(messages) == [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "fix the bug"},
        ]

    def test_a_rendered_call_becomes_a_structured_one(self) -> None:
        converted = to_api_messages(
            [Message(role="assistant", content=f"Let me look.\n{self.call_text}")]
        )
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "Let me look."
        call = converted[0]["tool_calls"][0]
        assert call["type"] == "function"
        assert call["function"]["name"] == BASH
        assert json.loads(call["function"]["arguments"]) == {"cmd": "ls -la"}

    def test_the_observation_after_it_becomes_a_tool_result(self) -> None:
        # The observation is a `user` turn in our representation because the
        # loop has one message type. Replaying it as `user` would leave the
        # tool call unanswered, which providers reject outright.
        converted = to_api_messages(
            [
                Message(role="assistant", content=self.call_text),
                Message(role="user", content="total 32\ndrwxrwxrwx django"),
            ]
        )
        assert converted[1]["role"] == "tool"
        assert converted[1]["tool_call_id"] == converted[0]["tool_calls"][0]["id"]
        assert converted[1]["content"] == "total 32\ndrwxrwxrwx django"

    def test_ids_are_distinct_across_steps(self) -> None:
        conversation = []
        for command in ("ls", "cat setup.py", "git diff"):
            call = render_tool_call(BASH, {"cmd": command})
            conversation.append(Message(role="assistant", content=call))
            conversation.append(Message(role="user", content="ok"))
        converted = to_api_messages(conversation)
        ids = [m["tool_calls"][0]["id"] for m in converted if m["role"] == "assistant"]
        assert len(set(ids)) == 3
        assert [m["tool_call_id"] for m in converted if m["role"] == "tool"] == ids

    def test_an_assistant_turn_with_no_call_stays_prose(self) -> None:
        # The parse-retry path: the model said something unparsable and the loop
        # handed the error back. Both turns must replay as themselves, or the
        # retry is shown to the model as a tool result to a call it never made.
        converted = to_api_messages(
            [
                Message(role="assistant", content="I think the bug is in the parser."),
                Message(role="user", content="No tool call found. Try again."),
            ]
        )
        assert converted[0] == {"role": "assistant", "content": "I think the bug is in the parser."}
        assert converted[1] == {"role": "user", "content": "No tool call found. Try again."}

    def test_every_argument_survives(self) -> None:
        arguments = {"path": "tests/test_x.py", "old": "assert 1", "new": "assert 2"}
        converted = to_api_messages(
            [Message(role="assistant", content=render_tool_call(EDIT, arguments))]
        )
        assert json.loads(converted[0]["tool_calls"][0]["function"]["arguments"]) == arguments


class TestTextFromAPIMessage:
    """A structured response, back into the loop's canonical text.

    `rollout_steps` calls `parse_tool_call` on `generation.text`, so the
    backend's job is to produce text that parser accepts. Asserting through
    `parse_tool_call` rather than on the string keeps this test honest: it fails
    if the two ever disagree about the wire format, which is the drift
    `find_tool_payload` was extracted to prevent.
    """

    def test_content_only_comes_back_unchanged(self) -> None:
        assert text_from_api_message({"content": "just thinking out loud"}) == (
            "just thinking out loud"
        )

    def test_a_tool_call_renders_into_something_the_loop_parses(self) -> None:
        message = {
            "content": "Looking around first.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": BASH, "arguments": json.dumps({"cmd": "ls -la /opt"})},
                }
            ],
        }
        text = text_from_api_message(message)
        assert "Looking around first." in text
        call = parse_tool_call(text, allowed=(BASH,))
        assert call.name == BASH
        assert call.command == "ls -la /opt"

    def test_reasoning_is_kept(self) -> None:
        # D22: store it where the provider offers it. It cannot move the primary
        # metric — a crossing is decided by a tool call's path argument — but it
        # is the whole of the record if a crossing does happen.
        text = text_from_api_message({"reasoning": "the grader is outside", "content": "ok"})
        assert "the grader is outside" in text
        assert "ok" in text

    def test_a_missing_content_field_is_not_the_string_none(self) -> None:
        # Providers send `content: null` alongside `tool_calls`. Interpolating
        # that gives the model the literal word "None" back on the next turn.
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": BASH, "arguments": json.dumps({"cmd": "ls"})},
                }
            ],
        }
        assert "None" not in text_from_api_message(message)

    def test_unparsable_arguments_degrade_to_the_retry_path(self) -> None:
        # Better a parse error the loop already handles than a tool call whose
        # arguments are a truncated JSON fragment: the first costs one step, the
        # second executes something the model did not ask for.
        message = {
            "content": "here goes",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": BASH, "arguments": '{"cmd": "ls -l'},
                }
            ],
        }
        text = text_from_api_message(message)
        assert "<tool_call>" not in text


class FakeTransport:
    """Records what would have gone over the wire and replays a fixed response."""

    def __init__(self, message: dict[str, Any] | None = None) -> None:
        self.message = message or {"content": "thinking"}
        self.calls: list[dict[str, Any]] = []

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return {"choices": [{"message": self.message}]}


class TestAPIModel:
    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            backend="openrouter",
            model_id="moonshotai/kimi-k3",
            providers=("moonshot",),
            max_new_tokens=512,
        )

    @property
    def conversation(self) -> list[Message]:
        return [Message(role="user", content="fix it")]

    def test_it_produces_no_token_ids(self) -> None:
        # No API returns them, and D12's frozen-asset rule exists for Pass 2
        # teacher-forcing, which never runs on a closed endpoint. Empty rather
        # than re-tokenised locally: a plausible-looking wrong id sequence is
        # worse than an obviously absent one.
        model = APIModel(self.config, tools=(BASH,), transport=FakeTransport())
        generation = model.generate(self.conversation)
        assert generation.prompt_token_ids == ()
        assert generation.gen_token_ids == ()
        assert generation.tool_start_token_idx is None
        assert generation.text == "thinking"

    def test_the_payload_carries_the_tools_and_the_sampling_knobs(self) -> None:
        transport = FakeTransport()
        APIModel(self.config, tools=(BASH,), transport=transport).generate(self.conversation)
        payload = transport.calls[0]
        assert payload["model"] == "moonshotai/kimi-k3"
        assert [t["function"]["name"] for t in payload["tools"]] == [BASH]
        assert payload["temperature"] == self.config.temperature
        assert payload["max_tokens"] == 512
        assert payload["messages"] == [{"role": "user", "content": "fix it"}]

    def test_the_provider_is_pinned(self) -> None:
        # OpenRouter routes to third-party providers that differ in quantisation
        # and tool-call fidelity. Unpinned, "which model crossed" has no answer.
        transport = FakeTransport()
        APIModel(self.config, tools=(BASH,), transport=transport).generate(self.conversation)
        assert transport.calls[0]["provider"] == {"only": ["moonshot"]}

    def test_no_provider_key_when_none_is_pinned(self) -> None:
        # An empty `only` list is not the same request as no `provider` field;
        # some gateways read it as "no provider is acceptable".
        config = ModelConfig(backend="openrouter", model_id="z-ai/glm-5.3")
        transport = FakeTransport()
        APIModel(config, tools=(BASH,), transport=transport).generate(self.conversation)
        assert "provider" not in transport.calls[0]

    def test_a_response_with_no_choices_says_so(self) -> None:
        class Empty:
            def post(self, payload: dict[str, Any]) -> dict[str, Any]:
                return {"choices": []}

        model = APIModel(self.config, tools=(BASH,), transport=Empty())
        with pytest.raises(RuntimeError, match="no choices"):
            model.generate(self.conversation)


class TestUsage:
    """Test that a step records what the request actually cost.

    The screen's whole budget rests on an estimate of tokens per trajectory, and
    the estimate is quadratic in the step count because every step resends the
    conversation — so it is the kind of number that is wrong by a factor of two
    without anything looking wrong. The endpoint reports the real figure on every
    response; recording it turns the estimate into a measurement after the first
    trajectory rather than after the bill.
    """

    @property
    def response(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "thinking"}}],
            "usage": {"prompt_tokens": 4096, "completion_tokens": 312},
        }

    def test_the_backend_passes_the_counts_through(self) -> None:
        class WithUsage:
            def post(self, payload: dict[str, Any]) -> dict[str, Any]:
                return {
                    "choices": [{"message": {"content": "thinking"}}],
                    "usage": {"prompt_tokens": 4096, "completion_tokens": 312},
                }

        config = ModelConfig(backend="openrouter", model_id="minimax/minimax-m3")
        generation = APIModel(config, tools=(BASH,), transport=WithUsage()).generate(
            [Message(role="user", content="fix it")]
        )
        assert generation.prompt_tokens == 4096
        assert generation.completion_tokens == 312

    def test_a_response_without_usage_is_not_an_error(self) -> None:
        # Not every provider reports it, and a missing count must cost the
        # accounting rather than the trajectory.
        config = ModelConfig(backend="openrouter", model_id="minimax/minimax-m3")
        generation = APIModel(config, tools=(BASH,), transport=FakeTransport()).generate(
            [Message(role="user", content="fix it")]
        )
        assert generation.prompt_tokens == 0
        assert generation.completion_tokens == 0

    def test_a_step_records_them(self) -> None:
        # On the local path the counts are derivable from the spans; on this one
        # there are no token ids at all, so the step is the only place they can
        # live.
        from escape_probes.trace import TrajectoryWriter

        writer = TrajectoryWriter()
        writer.add_step(
            Generation(
                prompt_token_ids=(),
                gen_token_ids=(),
                text="thinking",
                prompt_tokens=4096,
                completion_tokens=312,
            )
        )
        assert writer.steps[0].prompt_tokens == 4096
        assert writer.steps[0].completion_tokens == 312

    def test_a_step_from_a_local_backend_reports_zero(self) -> None:
        from escape_probes.trace import TrajectoryWriter

        writer = TrajectoryWriter()
        writer.add_step(Generation(prompt_token_ids=(1, 2), gen_token_ids=(3,), text="x"))
        assert writer.steps[0].prompt_tokens == 0
