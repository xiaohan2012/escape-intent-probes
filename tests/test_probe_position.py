"""Tests for probe position (b), which had none.

Position (b) is the first token of the tool call's command string — the last
moment before the action is named, and one of the two places the probe reads.
`HFModel` carried its own copy, never tested; `model.py` computed the same
thing for the fake backend, correctly, and was tested. Two implementations of
one rule with one of them tested is how the HuggingFace copy came to compare
decoded prefix length with `>=`, returning the token *before* the command. The
index stays in range, nothing raises, and the probe trains on the wrong
activation.

There is now one implementation, `model.command_token_index`, taking whatever
`decode` its caller has. These tests drive it through a character-level stub, so
they need neither a GPU nor `transformers`.
"""

from typing import Any

import pytest

from escape_probes.config import BASH, EDIT
from escape_probes.model import render_tool_call


class StubTokenizer:
    """Character-level, so a character offset is exactly a token index.

    The same trick `CharTokenizer` uses, and for the same reason: it makes the
    expected answer computable by hand.
    """

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in token_ids)


@pytest.fixture
def index() -> Any:
    """The shared rule, driven through the stub tokenizer."""
    from escape_probes.model import command_token_index

    def call(text: str) -> int | None:
        return command_token_index([ord(c) for c in text], text, StubTokenizer().decode)

    return call


class TestCommandTokenIndex:
    def test_points_at_the_first_character_of_the_command(self, index: Any) -> None:
        text = "I will look around.\n" + render_tool_call(BASH, {"cmd": "ls -la /opt"})
        found = index(text)
        assert found is not None
        assert text[found] == "l"

    def test_the_token_contains_the_offset_it_does_not_precede_it(self, index: Any) -> None:
        # The regression. A `>=` comparison returned the preceding token.
        text = render_tool_call(BASH, {"cmd": "cat /opt/grader/answer.patch"})
        found = index(text)
        assert found is not None
        assert text[found] == "c"

    def test_a_generation_with_no_tool_call_has_no_position(self, index: Any) -> None:
        assert index("Thinking about the bug, no call yet.") is None

    def test_an_unclosed_call_has_no_position(self, index: Any) -> None:
        # The parser rejects these too; the two must agree, or position (b) is
        # silently wrong rather than absent.
        assert index("<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n") is None

    def test_the_first_argument_is_the_one_that_counts(self, index: Any) -> None:
        text = render_tool_call(EDIT, {"path": "m.py", "old": "a", "new": "b"})
        found = index(text)
        assert found is not None
        assert text[found:].startswith("m.py")

    def test_the_fake_backend_reports_the_same_index(self, index: Any) -> None:
        # The fake backend goes through the same function now, but it reaches it
        # by a different route, and its trajectories are what the persistence
        # and span-arithmetic tests rely on.
        from escape_probes.model import FakeModel, Message, ScriptedStep

        step = ScriptedStep(reasoning="Looking.", tool=BASH, arguments={"cmd": "ls /opt/grader"})
        generation = FakeModel([step]).generate([Message(role="user", content="go")])
        assert generation.tool_start_token_idx == index(generation.text)
