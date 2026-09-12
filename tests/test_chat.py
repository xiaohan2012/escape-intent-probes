"""Tests for the layer both real backends share.

The point of these is comparability. A vLLM trajectory and a HuggingFace
trajectory must be tokenised identically or they are not measuring the same
thing, and the failure would be invisible — two runs that differ only in a
prompt rendering detail look the same in every log we keep. So the rendering and
the position-(b) mapping are tested here once, against a stub tokenizer, rather
than on the GPU against each backend separately.
"""

from typing import Any

import pytest

from escape_probes.chat import (
    ALL_TOOL_SCHEMAS,
    command_token_index,
    render_prompt,
    select_tool_schemas,
)
from escape_probes.config import BASH, EDIT, SUBMIT
from escape_probes.model import Message, render_tool_call


class StubTokenizer:
    """A character-level stand-in with a visible chat template.

    Character level so that a character offset into the text is exactly a token
    index — the same trick `CharTokenizer` uses, and for the same reason: it
    makes the expected position (b) computable by hand.
    """

    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        self.template_calls.append({"conversation": conversation, **kwargs})
        if kwargs.get("broken"):
            return ["not", "a", "string"]
        tools = kwargs.get("tools") or []
        names = ",".join(t["function"]["name"] for t in tools)
        turns = "".join(f"<{m['role']}>{m['content']}" for m in conversation)
        suffix = "<assistant>" if kwargs.get("add_generation_prompt") else ""
        return f"[tools:{names}]{turns}{suffix}"

    def __call__(self, text: str, **kwargs: Any) -> Any:
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        return "".join(chr(i) for i in token_ids)


class TestToolSchemas:
    def test_only_the_enabled_tools_are_shown(self) -> None:
        # The model must never see a tool the loop would reject (Q12).
        schemas = select_tool_schemas((BASH, SUBMIT))
        assert [s["function"]["name"] for s in schemas] == [BASH, SUBMIT]

    def test_the_default_three_all_have_schemas(self) -> None:
        names = {s["function"]["name"] for s in ALL_TOOL_SCHEMAS}
        assert {BASH, EDIT, SUBMIT} <= names

    def test_an_unknown_name_selects_nothing(self) -> None:
        assert select_tool_schemas(("think",)) == []

    def test_edit_declares_the_arguments_the_loop_applies(self) -> None:
        # `apply_edit` reads path/old/new; a schema that named them differently
        # would produce calls the loop cannot execute.
        edit = next(s for s in ALL_TOOL_SCHEMAS if s["function"]["name"] == EDIT)
        assert set(edit["function"]["parameters"]["required"]) == {"path", "old", "new"}


class TestRenderPrompt:
    def test_tools_go_through_the_template(self) -> None:
        # D15: tools are declared to the template, never described in prose.
        tokenizer = StubTokenizer()
        render_prompt(tokenizer, [Message(role="user", content="hi")], select_tool_schemas((BASH,)))
        assert tokenizer.template_calls[0]["tools"][0]["function"]["name"] == BASH

    def test_the_context_ends_where_the_model_speaks(self) -> None:
        # Probe position (a) is read at this point, by AgentLens's convention.
        tokenizer = StubTokenizer()
        ids = render_prompt(tokenizer, [Message(role="user", content="hi")], [])
        assert tokenizer.decode(ids).endswith("<assistant>")

    def test_special_tokens_are_not_added_twice(self) -> None:
        # The template has already put in whatever the model expects.
        calls: list[dict[str, Any]] = []

        class Recording(StubTokenizer):
            def __call__(self, text: str, **kwargs: Any) -> Any:
                calls.append(kwargs)
                return super().__call__(text, **kwargs)

        render_prompt(Recording(), [Message(role="user", content="hi")], [])
        assert calls[0]["add_special_tokens"] is False

    def test_a_template_returning_tokens_is_an_error(self) -> None:
        # `tokenize=False` has changed meaning across transformers versions, and
        # a silently wrong type here would corrupt every stored token id.
        class Broken(StubTokenizer):
            def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
                return [1, 2, 3]

        with pytest.raises(TypeError, match="expected str"):
            render_prompt(Broken(), [Message(role="user", content="hi")], [])


class TestCommandTokenIndex:
    """Probe position (b): the first token of the tool call's command string."""

    def index(self, text: str) -> int | None:
        tokenizer = StubTokenizer()
        ids = [ord(c) for c in text]
        return command_token_index(ids, text, tokenizer.decode)

    def test_points_at_the_first_character_of_the_command(self) -> None:
        text = "I will look around.\n" + render_tool_call(BASH, {"cmd": "ls -la /opt"})
        index = self.index(text)
        assert index is not None
        assert text[index] == "l"

    def test_a_generation_with_no_tool_call_has_no_position(self) -> None:
        assert self.index("Thinking about the bug, no call yet.") is None

    def test_an_unclosed_call_has_no_position(self) -> None:
        # The parser rejects these too; the two must agree or position (b) is
        # silently wrong rather than absent.
        assert self.index("<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n") is None

    def test_the_token_contains_the_offset_it_does_not_precede_it(self) -> None:
        # Regression: a `>=` comparison here returned the token *before* the
        # command, shifting every probe read one token early and silently.
        text = render_tool_call(BASH, {"cmd": "cat /opt/grader/answer.patch"})
        index = self.index(text)
        assert index is not None
        assert text[index] == "c"

    def test_the_first_argument_is_the_one_that_counts(self) -> None:
        # `edit` has three; position (b) is the last moment before the action is
        # named, which is where the first value begins.
        text = render_tool_call(EDIT, {"path": "m.py", "old": "a", "new": "b"})
        index = self.index(text)
        assert index is not None
        assert text[index:].startswith("m.py")
