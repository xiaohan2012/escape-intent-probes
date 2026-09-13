"""Tests for splitting a reasoning model's generation so the prompt still nests.

Qwen3.8-27B opens every assistant turn with `<think>\\n` in its generation
prompt, and renders a *past* assistant turn from a separate `reasoning_content`
field:

    {%- if message.reasoning_content is string %}
        {%- set reasoning_content = message.reasoning_content %}
    {%- endif %}
    {{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content
        + '\\n</think>\\n\\n' + content }}

Storing the whole emission as `content` therefore renders an *empty* think block
followed by the reasoning as prose, and step 1's prompt stops being a prefix of
step 0's. `TrajectoryWriter.add_step` catches it — this is what E4 is for, and
it fired on the first trajectory of the first real run — but catching it only
turns a silent corruption into a stopped batch. Pass 2 needs the conversation to
nest, because it replays the stored ids.

Splitting is a property of the text, not of the backend, so it is tested here
without a tokenizer or a GPU.
"""

from __future__ import annotations

import pytest

from escape_probes.model import Message, split_reasoning


class TestSplitReasoning:
    def test_it_splits_on_the_closing_tag(self) -> None:
        assert split_reasoning("I should look at grader.yaml.\n</think>\n\nOK") == (
            "I should look at grader.yaml.",
            "OK",
        )

    def test_text_without_a_tag_is_all_content(self) -> None:
        # A non-reasoning model, or a turn that emitted no think block.
        assert split_reasoning("<tool_call>x</tool_call>") == ("", "<tool_call>x</tool_call>")

    def test_an_opening_tag_is_dropped_if_the_model_emits_one(self) -> None:
        # The prompt already ends with `<think>\n`, so the model does not repeat
        # it — but a model that does must not have it folded into the reasoning.
        assert split_reasoning("<think>\nthinking\n</think>\n\nanswer") == ("thinking", "answer")

    def test_only_the_first_closing_tag_splits(self) -> None:
        # `</think>` inside the answer — an agent quoting its own transcript.
        reasoning, content = split_reasoning("plan\n</think>\n\nsee </think> above")
        assert reasoning == "plan"
        assert content == "see </think> above"

    def test_whitespace_is_trimmed_on_both_sides(self) -> None:
        # The template emits `<think>\n` + reasoning + `\n</think>\n\n`, and
        # applies `|trim` to the reasoning. Anything else fails to round-trip.
        assert split_reasoning("  padded  \n</think>\n\n  answer  ") == ("padded", "answer")

    def test_an_empty_reasoning_block(self) -> None:
        assert split_reasoning("\n</think>\n\nanswer") == ("", "answer")


class TestMessageCarriesReasoning:
    def test_the_field_defaults_to_none(self) -> None:
        # None rather than "", so a template that checks `is string` treats an
        # ordinary message as having no reasoning rather than an empty block.
        assert Message(role="user", content="hi").reasoning_content is None

    def test_it_survives_a_dump(self) -> None:
        # render_prompt hands `model_dump()` to the template, so the field has to
        # be there under the name the template reads.
        dumped = Message(role="assistant", content="a", reasoning_content="r").model_dump()
        assert dumped["reasoning_content"] == "r"

    @pytest.mark.parametrize("role", ["user", "system"])
    def test_a_non_assistant_message_dumps_none(self, role: str) -> None:
        assert Message(role=role, content="x").model_dump()["reasoning_content"] is None
